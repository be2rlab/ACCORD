#!/usr/bin/env python3
"""Hilti-Oxford reference surface: the site's laser scan, cropped to one sequence.

    data/hilti/scans/<site>_full.e57  ->  data/hilti/<seq>/accord/pointcloud0/data.ply

The scan covers the whole site, so it is cropped to the box around the sequence's reference positions, grown by
the evaluation range, and thinned to one point per voxel. The reference trajectory and the scan are in one frame
in this release (the dense poses are lidar scans registered to it); that is checked, not assumed.

    scripts/docker.sh python3 scripts/hilti_reference.py data/hilti/exp14

Reading .e57 needs pye57: `pip install --target thirdparty/py310_extra pye57` (the Makefile does this).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.append(str(ROOT / 'thirdparty/py310_extra'))
from accord.datasets import is_control_points, read_poses           # noqa: E402
from accord.eval.reference_support import voxel_downsample          # noqa: E402
from accord.mapper.fusion import write_ply                          # noqa: E402

PLAN = ROOT / 'config/hilti/downloads.json'
# A hand-held rig walks inside the scanned building, within a few metres of a surface. Beyond this the
# trajectory and the scan are not in one frame.
MAX_MEDIAN_SURFACE_DISTANCE_M = 3.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def crop_box(positions, margin):
    positions = np.asarray(positions, float)
    return positions.min(0) - margin, positions.max(0) + margin


def reduce_scans(scans, box, voxel):
    """Crop and thin an iterable of (N, 3) scans into one cloud, one scan in memory at a time."""
    kept, read = [], 0
    lo, hi = box
    for points in scans:
        points = np.asarray(points, float)
        read += len(points)
        points = points[np.isfinite(points).all(axis=1)]
        inside = points[np.all((points >= lo) & (points <= hi), axis=1)]
        if len(inside):
            kept.append(voxel_downsample(inside, voxel))
    if not kept:
        return np.zeros((0, 3)), read
    return voxel_downsample(np.concatenate(kept), voxel), read


def frame_check(cloud, positions):
    """How far the reference positions are from the scanned surfaces."""
    distance, _ = cKDTree(voxel_downsample(cloud, .10)).query(np.asarray(positions, float))
    return dict(positions=int(len(positions)), median_m=float(np.median(distance)),
                p95_m=float(np.percentile(distance, 95)), max_m=float(distance.max()))


def read_e57(path):
    """Each scan of the file as (N, 3) points in the file's global frame."""
    try:
        import pye57
    except ImportError as exc:
        raise SystemExit('reading .e57 needs pye57: pip install --target thirdparty/py310_extra pye57') from exc
    scan = pye57.E57(str(path))
    for index in range(scan.scan_count):
        data = scan.read_scan(index, ignore_missing_fields=True)
        yield np.stack([data['cartesianX'], data['cartesianY'], data['cartesianZ']], axis=1)


def reference_positions(sequence):
    """Every reference position the release has for the sequence, dense poses and control points."""
    files = sorted((Path(sequence) / 'gt').glob('*.txt'))
    if not files:
        raise SystemExit(f'{sequence}/gt holds no reference; run scripts/fetch_hilti.py')
    tracks = {p.name: read_poses(p) for p in files}
    kinds = {name: 'control_points' if is_control_points(q) else 'dense' for name, (_t, _p, q) in tracks.items()}
    return np.concatenate([p for _t, p, _q in tracks.values()]), kinds


def write_reference(sequence, scan=None, margin=21., voxel=.01, log=print):
    sequence = Path(sequence).resolve()
    plan = json.loads(PLAN.read_text())
    entry = plan['sequences'].get(sequence.name)
    if entry is None:
        raise SystemExit(f'{sequence.name} is not a sequence of config/hilti/downloads.json')
    scan = Path(scan) if scan else sequence.parent / 'scans' / Path(plan['scans'][entry['site']]).name
    if not scan.is_file():
        raise SystemExit(f'{scan} is missing; run scripts/fetch_hilti.py --sequences {sequence.name}')
    out = sequence / 'accord/pointcloud0/data.ply'
    if out.exists():
        raise SystemExit(f'{out} exists; a reference is never rebuilt in place')
    positions, kinds = reference_positions(sequence)
    box = crop_box(positions, margin)
    cloud, read = reduce_scans(read_e57(scan), box, voxel)
    if not len(cloud):
        raise SystemExit(f'no scan point lies within {margin} m of the reference: {scan.name} and the ground truth '
                         f'of {sequence.name} are not in one frame')
    check = frame_check(cloud, positions)
    if check['median_m'] > MAX_MEDIAN_SURFACE_DISTANCE_M:
        raise SystemExit(f'the reference positions are {check["median_m"]:.1f} m (median) from the nearest scanned '
                         f'surface: {scan.name} and the ground truth are not in one frame')
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, cloud, np.full((len(cloud), 3), 180, np.uint8))
    report = dict(sequence=sequence.name, site=entry['site'], scan=scan.name,
                  scan_identity=plan['files'].get(plan['scans'][entry['site']]), scan_points_read=int(read),
                  points=int(len(cloud)), voxel_m=float(voxel), margin_m=float(margin),
                  box=[[float(x) for x in box[0]], [float(x) for x in box[1]]], reference_files=kinds,
                  frame_check=check, ply_sha256=sha256(out))
    (out.parent / 'reference.json').write_text(json.dumps(report, indent=1) + '\n')
    log(json.dumps(dict(points=report['points'], frame_check=check), indent=1))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequences', nargs='+', type=Path)
    p.add_argument('--scan', type=Path, help='the .e57 (default: data/hilti/scans/<site>_full.e57)')
    p.add_argument('--margin-m', type=float, default=21., help='crop this far around the reference: the '
                   'evaluation range (20 m) and some slack')
    p.add_argument('--voxel-m', type=float, default=.01, help='one point per voxel; the scorer thins to 1 cm anyway')
    a = p.parse_args(argv)
    for sequence in a.sequences:
        print(write_reference(sequence, a.scan, a.margin_m, a.voxel_m))
    return 0


if __name__ == '__main__':
    sys.exit(main())
