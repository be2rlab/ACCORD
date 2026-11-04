#!/usr/bin/env python3
"""Oxford Spires reference surface: the site's 1 cm laser scan, cropped to one sequence.

    data/spires/scans/christ-church_merged-cloud-1cm.pcd  ->  data/spires/<seq>/accord/pointcloud0/data.ply

As Hilti-Oxford's (scripts/hilti_reference.py, whose helpers this uses): the scan covers the whole site, so it is
cropped to the box around the sequence's reference positions, grown by the evaluation range, and thinned to one point
per voxel. The reference trajectory and the scan are in one frame in this release (the trajectory is lidar scans
registered to the scan); that is checked, not assumed: the reference positions lie near scanned surfaces, none inside
one, and above a scanned floor at a hand-held height. Run after scripts/convert_spires.py.

    python3 scripts/spires_reference.py data/spires/cc03
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from accord.datasets import Spires, read_poses, rig_camera           # noqa: E402
from accord.eval.reference_support import voxel_downsample          # noqa: E402
from accord.mapper.fusion import write_ply                          # noqa: E402
from hilti_reference import (MAX_MEDIAN_SURFACE_DISTANCE_M, crop_box, frame_check, reduce_scans,  # noqa: E402
                             sha256)

PLAN = ROOT / 'config/spires/downloads.json'
PCD_TYPES = {('F', 4): '<f4', ('F', 8): '<f8', ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4', ('I', 1): 'i1',
             ('I', 2): '<i2', ('I', 4): '<i4'}
CHUNK_POINTS = 5_000_000
INSIDE_M = .10              # a reference position this close to a scanned point is inside a surface
FLOOR_RADIUS_M = .30        # the floor under a position: the highest scanned point below it within this radius


def pcd_header(stream):
    """The header of a PCD file as a dict, the stream left at the start of the data."""
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError('PCD header without DATA')
        line = line.decode('ascii', 'replace').strip()
        if not line or line.startswith('#'):
            continue
        key, _, value = line.partition(' ')
        header[key] = value.strip()
        if key == 'DATA':
            return header


def pcd_dtype(header):
    """The numpy record of one point of a binary PCD."""
    fields, sizes, types = header['FIELDS'].split(), header['SIZE'].split(), header['TYPE'].split()
    counts = header.get('COUNT', ' '.join('1' * len(fields))).split()
    if not len(fields) == len(sizes) == len(types) == len(counts):
        raise ValueError(f'inconsistent PCD header: {header}')
    names, formats = [], []
    for k, (name, size, kind, count) in enumerate(zip(fields, sizes, types, counts)):
        base = PCD_TYPES.get((kind, int(size)))
        if base is None:
            raise ValueError(f'unsupported PCD field {name}: TYPE {kind} SIZE {size}')
        names.append(name if name != '_' else f'_pad{k}')
        formats.append(base if int(count) == 1 else (base, (int(count),)))
    return np.dtype(dict(names=names, formats=formats))


def read_pcd_xyz(path, chunk_points=CHUNK_POINTS):
    """The x, y, z of a binary PCD (any other fields skipped) as float64 arrays of at most ``chunk_points`` rows."""
    path = Path(path)
    with path.open('rb') as stream:
        header = pcd_header(stream)
        offset = stream.tell()
    if header['DATA'] != 'binary':
        raise ValueError(f'{path}: DATA {header["DATA"]}; this reader takes binary PCD only')
    dtype = pcd_dtype(header)
    if not {'x', 'y', 'z'} <= set(dtype.names):
        raise ValueError(f'{path}: no x y z fields in {dtype.names}')
    count = int(header['POINTS'])
    if offset + count * dtype.itemsize != path.stat().st_size:
        raise ValueError(f'{path}: {count} points of {dtype.itemsize} bytes do not fill the file')
    points = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))
    for start in range(0, count, chunk_points):
        part = points[start:start + chunk_points]
        yield np.stack([part['x'], part['y'], part['z']], axis=1).astype(float)


def imu_positions(gt_path):
    """(t, the reference positions of the IMU) from gt-tum.txt, through the rig's calibration chain."""
    t, p, q = read_poses(gt_path)
    T_base_imu = np.linalg.inv(Spires.imu_from_base(rig_camera('spires')[1]))
    from scipy.spatial.transform import Rotation
    return t, p + Rotation.from_quat(q).as_matrix() @ T_base_imu[:3, 3]


def free_space_check(cloud, positions, period=1):
    """The positions against the scanned surfaces: how many lie inside one (a scanned point within INSIDE_M), and
    their height above the scanned floor under them."""
    positions = np.asarray(positions, float)[::period]
    coarse = voxel_downsample(cloud, .05)
    inside = cKDTree(coarse).query(positions)[0] < INSIDE_M
    plan = cKDTree(coarse[:, :2])
    heights = []
    for position, near in zip(positions, plan.query_ball_point(positions[:, :2], FLOOR_RADIUS_M)):
        below = coarse[near, 2][coarse[near, 2] < position[2] - .2] if near else np.zeros(0)
        heights.append(position[2] - below.max() if len(below) else np.nan)
    heights = np.asarray(heights)
    found = np.isfinite(heights)
    return dict(positions=int(len(positions)), inside_surface=int(inside.sum()),
                inside_fraction=float(inside.mean()), floor_found_fraction=float(found.mean()),
                height_above_floor_median_m=float(np.median(heights[found])) if found.any() else None,
                height_above_floor_p5_p95_m=[float(x) for x in np.percentile(heights[found], [5, 95])] if found.any() else None)


def write_reference(sequence, scan=None, margin=21., voxel=.01, log=print):
    sequence = Path(sequence).resolve()
    plan = json.loads(PLAN.read_text())
    entry = plan['sequences'].get(sequence.name)
    if entry is None:
        raise SystemExit(f'{sequence.name} is not a sequence of config/spires/downloads.json')
    if not (sequence / 'accord/conversion.json').is_file():
        raise SystemExit(f'{sequence}/accord lacks conversion.json; run scripts/convert_spires.py first')
    site = entry['site']
    scan = Path(scan) if scan else sequence.parent / 'scans' / f'{site}_{Path(plan["scans"][site]).name}'
    if not scan.is_file():
        raise SystemExit(f'{scan} is missing; run scripts/fetch_spires.py --sequences {sequence.name}')
    out = sequence / 'accord/pointcloud0/data.ply'
    if out.exists():
        raise SystemExit(f'{out} exists; a reference is never rebuilt in place')
    _t, positions = imu_positions(sequence / 'gt/gt-tum.txt')
    box = crop_box(positions, margin)
    cloud, read = reduce_scans(read_pcd_xyz(scan), box, voxel)
    if not len(cloud):
        raise SystemExit(f'no scan point lies within {margin} m of the reference: {scan.name} and the ground truth '
                         f'of {sequence.name} are not in one frame')
    sampled = positions[::10]
    check = frame_check(cloud, sampled)
    if check['median_m'] > MAX_MEDIAN_SURFACE_DISTANCE_M:
        raise SystemExit(f'the reference positions are {check["median_m"]:.1f} m (median) from the nearest scanned '
                         f'surface: {scan.name} and the ground truth are not in one frame')
    free = free_space_check(cloud, sampled)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, cloud, np.full((len(cloud), 3), 180, np.uint8))
    report = dict(sequence=sequence.name, name=entry['name'], site=site, scan=scan.name,
                  scan_identity=plan['files'].get(plan['scans'][site]), scan_points_read=int(read),
                  points=int(len(cloud)), voxel_m=float(voxel), margin_m=float(margin),
                  box=[[float(x) for x in box[0]], [float(x) for x in box[1]]], frame_check=check,
                  free_space_check=free, ply_sha256=sha256(out))
    (out.parent / 'reference.json').write_text(json.dumps(report, indent=1) + '\n')
    log(json.dumps(dict(points=report['points'], frame_check=check, free_space_check=free), indent=1))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequences', nargs='+', type=Path)
    p.add_argument('--scan', type=Path, help='the .pcd (default: data/spires/scans/<site>_merged-cloud-1cm.pcd)')
    p.add_argument('--margin-m', type=float, default=21., help='crop this far around the reference: the '
                   'evaluation range (20 m) and some slack')
    p.add_argument('--voxel-m', type=float, default=.01, help='one point per voxel; the scorer thins to 1 cm anyway')
    a = p.parse_args(argv)
    for sequence in a.sequences:
        print(write_reference(sequence, a.scan, a.margin_m, a.voxel_m))
    return 0


if __name__ == '__main__':
    sys.exit(main())
