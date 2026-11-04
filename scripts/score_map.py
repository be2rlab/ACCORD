#!/usr/bin/env python3
"""Score a run's map against the dataset's laser scan.

The map is brought into the scan's frame by the rigid fit of the run's trajectory to the reference
trajectory: nothing is fitted to the surface, and no scale. Accuracy is the distance from each map
point to the scan around the reference cameras; completeness the distance to the map from each scan
point those cameras see (accord/eval/reference_support.py). The F-score is given at 2, 3, 5 and 10 cm.
With --fit, the map is afterwards registered to the scan by a rigid transform or a similarity: the
map's shape scored apart from where the trajectory puts it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from accord.datasets import open_dataset                                              # noqa: E402
from accord.eval import control_points, metrics, reference_support, surface_fit       # noqa: E402
from accord.geometry import quat_to_rot                                               # noqa: E402
from accord.images import depth_view                                                  # noqa: E402
from accord.mapper.fusion import read_ply                             # noqa: E402

THRESHOLDS_M = (.02, .03, .05, .10)
NEAR_M, FAR_M = .15, 20.            # the range of a reference camera
REFERENCE_VOXEL_M = .01             # the scan is thinned to one point per voxel
SUPPORT_PERIOD_S = .1               # one reference camera per this much time
OCCLUSION_TOLERANCE_M = .03


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def world_alignment(run, ds, trajectory_file):
    """Map to reference frame: the rigid fit of the run's trajectory to the reference trajectory, or to the control
    points where those are the only reference (Hilti-Oxford's construction site)."""
    meta = json.loads((run / 'run.json').read_text())
    trajectory = metrics.read_trajectory(run / trajectory_file)
    out = np.eye(4)
    if not getattr(ds, 'dense', True):
        fitted = control_points.fit(trajectory, *ds.control_points())
        if fitted is None:
            raise ValueError('The trajectory covers too few control points to place the map')
        rot, translation, found, estimate = fitted
        aligned = (rot @ estimate[found].T + translation).T
        rmse = np.sqrt(np.mean(np.sum((aligned - ds.control_points()[1][found]) ** 2, axis=1)))
    else:
        est_p, ref_p, _, _, _ = metrics.matched_points(ds, trajectory, meta['interval_start'])
        if est_p is None:
            raise ValueError('No reference support for this trajectory; cannot place the map')
        rot, translation, rmse = metrics.se3_align(est_p, ref_p)
    out[:3, :3], out[:3, 3] = rot, translation.reshape(3)
    return out, float(rmse)


def estimated_cameras(run, ds, trajectory_file, transform, period_s):
    """The run's own cameras in the reference frame: the scan's support where the reference has no orientation (Hilti-
    Oxford's control points). It differs from run to run, so it is never cached."""
    rows = metrics.read_trajectory(run / trajectory_file)
    keep, last = [], -np.inf
    for row in rows:
        if row[0] - last >= period_s - 1e-9:
            keep.append(row)
            last = row[0]
    body_camera = np.eye(4)
    body_camera[:3, :3], body_camera[:3, 3] = ds.R_CtoI, ds.p_IC
    world = np.repeat(np.eye(4)[None], len(keep), axis=0)
    for k, (_t, p, q) in enumerate(keep):
        world[k, :3, :3], world[k, :3, 3] = quat_to_rot(q).T, p
    return transform @ world @ body_camera, np.array([row[0] for row in keep])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True, help='the directory that holds the sequences')
    p.add_argument('--map', type=Path, help='the exported cloud (default RUN/map.ply)')
    p.add_argument('--trajectory', default='map_trajectory_final.tum', help='the stream that places the map')
    p.add_argument('--reference', choices=('dataset', 'openvins'), help='EuRoC: its own reference file, or the corrected '
                   'one OpenVINS distributes (default)')
    p.add_argument('--fit', choices=('none', 'rigid', 'similarity'), default='none')
    p.add_argument('--support-cache', type=Path, default=ROOT / 'runs/_reference_support')
    p.add_argument('--out', type=Path, help='the report (default RUN/surface.json)')
    p.add_argument('--force', action='store_true', help='replace an existing report')
    a = p.parse_args(argv)
    meta = json.loads((a.run / 'run.json').read_text())
    ds = open_dataset(meta['dataset'], a.data, meta['sequence'], a.reference)
    cloud = a.map or (a.run / 'map.ply')
    reference_path = getattr(ds, 'surface_path', ds.root / 'pointcloud0/data.ply')     # the phone rig's: reference/map.ply
    if not reference_path.is_file():
        print(f'{meta["sequence"]} has no reference surface ({reference_path}); the map is not scored')
        return 0
    out = a.out or (a.run / 'surface.json')
    if out.exists() and not a.force:
        raise SystemExit(f'{out} exists; pass --force to replace it')

    transform, fit_rmse = world_alignment(a.run, ds, a.trajectory)
    estimate = read_ply(cloud) @ transform[:3, :3].T + transform[:3, 3]
    reference_sha = sha256_file(reference_path)
    view = depth_view(ds.camera)
    width, height = view.resolution
    options = dict(near=NEAR_M, far=FAR_M, period_s=SUPPORT_PERIOD_S, voxel=REFERENCE_VOXEL_M, tolerance=OCCLUSION_TOLERANCE_M)
    if getattr(ds, 'dense', True):
        reference, crop, seen, support = reference_support.load_or_build(
            a.support_cache, ds, read_ply(reference_path), reference_sha, view.K, (height, width), **options)
    else:
        cameras = estimated_cameras(a.run, ds, a.trajectory, transform, SUPPORT_PERIOD_S)
        reference, crop, seen, support = reference_support.build(
            ds, read_ply(reference_path), reference_sha, view.K, (height, width), cameras=cameras, **options)
    if not seen.any():
        raise SystemExit('No reference point is supported')

    surface_fit_info = None
    if a.fit != 'none':
        fitted, surface_fit_info = surface_fit.fit(estimate, reference[crop], with_scale=a.fit == 'similarity')
        estimate = estimate @ fitted[:3, :3].T + fitted[:3, 3]
        surface_fit_info['transform'] = fitted.tolist()

    # A scan that reaches only so far from its track has nothing to say of points beyond that (the phone rig's RTAB-Map
    # maps): those points are left out of precision, and their share is reported; recall is unchanged.
    reach = getattr(ds, 'surface_reach_m', None)
    beyond_reach = None
    if reach is not None:
        near = cKDTree(ds.groundtruth().p).query(estimate, workers=-1)[0] <= reach
        beyond_reach = dict(reach_m=reach, share_of_points_left_out=float(1. - near.mean()))
        estimate = estimate[near]

    # on all cores: the same distances; one core took hours on a map tens of metres off the scan (Spires CC02)
    if len(estimate):
        accuracy = cKDTree(reference[crop]).query(estimate, workers=-1)[0]
        completeness = cKDTree(estimate).query(reference[seen], workers=-1)[0]
    else:       # every point beyond the scan's reach (a run gone far off): precision and recall are 0
        accuracy, completeness = np.empty(0), np.full(int(seen.sum()), np.inf)

    def summary(d):
        if not len(d) or not np.isfinite(d).all():
            return None
        return dict(mean_m=float(d.mean()), median_m=float(np.median(d)), p95_m=float(np.percentile(d, 95)))
    scores = {}
    for t in THRESHOLDS_M:
        precision, recall = float(np.mean(accuracy <= t)) if len(accuracy) else 0., float(np.mean(completeness <= t))
        scores[f'{t:.2f}'] = dict(precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-30))
    report = dict(
        run=str(a.run), sequence=meta['sequence'], map_ply=str(cloud), map_sha256=sha256_file(cloud),
        reference_ply=str(reference_path), reference_sha256=reference_sha, support=support,
        alignment=dict(source=a.trajectory, fit=a.fit, trajectory_rmse_m=fit_rmse, map_to_reference=transform.tolist()),
        surface_fit=surface_fit_info, beyond_reach=beyond_reach, groundtruth=ds.reference_provenance(),
        reference_points=int(len(reference)), crop_points=int(crop.sum()), visible_points=int(seen.sum()),
        estimate_points=int(len(estimate)), accuracy=summary(accuracy), completeness=summary(completeness),
        chamfer_mean_m=float((accuracy.mean() + completeness.mean()) / 2) if len(estimate) else None, thresholds=scores)
    if beyond_reach is None:        # only the phone rig's reports have it
        del report['beyond_reach']
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('sequence', 'estimate_points', 'accuracy', 'completeness', 'thresholds')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
