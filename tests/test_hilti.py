"""Hilti-Oxford: the rig, the converted sequence as the engine and the scorers read it, and the control points."""
import json

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from accord.datasets import Hilti, densify, open_dataset
from accord.eval import control_points, metrics
from accord.filter import read_camera

T0_NS = 1_649_764_528_000_000_000


def poses(seconds=6., rate=10.):
    """A walk along x that turns about z: (t, p, q_xyzw), world from IMU."""
    t = T0_NS * 1e-9 + .2 + np.arange(0., seconds, 1. / rate)
    p = np.stack([t - t[0] + 100., np.full_like(t, 17.), np.full_like(t, -1.)], axis=1)
    return t, p, Rotation.from_euler('z', .5 * (t - t[0])).as_quat()


def write_poses(path, t, p, q):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{a:.6f} {b[0]} {b[1]} {b[2]} {c[0]} {c[1]} {c[2]} {c[3]}\n' for a, b, c in zip(t, p, q)))


def sequence(root, name='exp99', dense=True, seconds=6.):
    """A sequence as scripts/fetch_hilti.py and scripts/convert_hilti.py leave it."""
    seq = root / name
    (seq / 'accord/cam0/data').mkdir(parents=True)
    (seq / 'accord/imu0').mkdir()
    for i in range(int(seconds * 40)):
        cv2.imwrite(str(seq / f'accord/cam0/data/{T0_NS + i * 25_000_000}.png'), np.full((54, 72), i % 255, np.uint8))
    (seq / 'accord/imu0/data.csv').write_text('#timestamp [ns],w,w,w,a,a,a\n' + ''.join(
        f'{T0_NS - 5_000_000 + i * 2_500_000},0,0,.5,0,0,9.81\n' for i in range(int(seconds * 400) + 8)))
    (seq / 'accord/conversion.json').write_text('{}')
    t, p, q = poses(seconds - .5)
    if dense:
        write_poses(seq / f'gt/{name}_imu.txt', t, p, q)
    pick = np.linspace(1, len(t) - 2, 4).astype(int)
    write_poses(seq / (f'gt/{name}_imu_3dof.txt' if dense else f'gt/{name}.txt'), t[pick], p[pick], np.tile([0., 0., 0., 1.], (4, 1)))
    return seq


def test_the_rig_is_cam0_of_the_dataset_chain_with_its_transform_inverted(root):
    source = yaml.safe_load((root / 'config/hilti/source/calib_3_cam0-1-camchain-imucam.yaml').read_text())['cam0']
    cam = read_camera(root / 'config/hilti/estimator_config.yaml')
    np.testing.assert_allclose(cam['T_imu_cam'], np.linalg.inv(np.asarray(source['T_cam_imu'])), atol=1e-12)
    assert cam['intrinsics'] == source['intrinsics'] and cam['distortion_coeffs'] == source['distortion_coeffs']
    assert cam['distortion_model'] == 'equidistant' and tuple(cam['resolution']) == (720, 540)


def test_a_converted_sequence_gives_its_frames_and_imu_and_the_dense_poses_at_100_hz(tmp_path):
    sequence(tmp_path)
    ds = open_dataset('hilti', tmp_path, 'exp99')
    assert isinstance(ds, Hilti) and len(ds.image_stamps()) == 240 and len(ds.imu()) == 2408
    assert ds.load_filter_image(ds.image_stamps()[3]).dtype == np.uint8
    reference = ds.groundtruth()
    t, p, q = poses(5.5)
    assert np.median(np.diff(reference.t)) == pytest.approx(.01) and reference.orientation_reliable
    k = int(np.argmin(np.abs(reference.t - (t[10] + .05))))           # between two samples of the file
    np.testing.assert_allclose(reference.p[k], p[10] + [reference.t[k] - t[10], 0., 0.], atol=1e-6)
    np.testing.assert_allclose(Rotation.from_matrix(reference.R[k]).as_euler('xyz')[2], .5 * (reference.t[k] - t[0]), atol=1e-6)
    assert ds.reference_provenance()['variant'].startswith('dense poses')
    assert len(ds.control_points()[0]) == 4


def test_the_reference_is_never_interpolated_across_a_gap():
    t = np.array([0., .1, .2, 1.2, 1.3])
    grid, _p, _R = densify(t, np.zeros((5, 3)), np.tile([0., 0., 0., 1.], (5, 1)), 100., .25)
    assert not np.any((grid > .2 + 1e-9) & (grid < 1.2 - 1e-9)) and grid[-1] < 1.3


def test_a_sequence_with_control_points_only_has_them_as_its_reference_without_orientation(tmp_path):
    sequence(tmp_path, 'exp98', dense=False)
    ds = Hilti(tmp_path, 'exp98')
    assert not ds.dense and not ds.groundtruth().orientation_reliable and len(ds.groundtruth().t) == 4


def test_the_dataset_score_at_control_points_after_a_rigid_fit():
    t = np.arange(0., 10., .025)
    truth = np.stack([np.cos(t), np.sin(t), .1 * t], axis=1)
    R = Rotation.from_euler('xyz', [.2, -.1, 1.]).as_matrix()
    trajectory = [(a, R @ b + [5., 1., 2.], None) for a, b in zip(t, truth)]
    when = np.array([1., 3.3, 6., 9.])
    exact = control_points.score(trajectory, when, truth[np.searchsorted(t, when)] + 0.)
    assert exact['rmse_m'] < 1e-6 and exact['score'] == 100. and exact['n_covered'] == 4
    off = truth[np.searchsorted(t, when)] + [[.02, 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    assert 60. < control_points.score(trajectory, when, off)['score'] < 100.
    later = control_points.score(trajectory, np.r_[when, 30.], np.r_[truth[np.searchsorted(t, when)], [[0., 0., 0.]]])
    assert later['n_covered'] == 4 and later['score'] == pytest.approx(80.)         # an uncovered point scores zero


def test_the_gauge_is_the_log_scale_of_the_similarity_positive_when_distances_are_too_long(tmp_path):
    sequence(tmp_path)
    ds = Hilti(tmp_path, 'exp99')
    reference = ds.groundtruth()
    rows = [(t, 1.05 * p, None) for t, p in zip(reference.t[::4], reference.p[::4] - reference.p[0])]
    out = metrics.gauge(ds, rows, reference.t[0])
    assert out['log_scale'] == pytest.approx(np.log(1.05), abs=1e-9) and out['position_m'] < 1e-9


def test_a_hilti_run_is_scored_at_its_control_points(tmp_path, script):
    sequence(tmp_path / 'hilti')
    ds = Hilti(tmp_path / 'hilti', 'exp99')
    run = tmp_path / 'run'
    run.mkdir()
    reference = ds.groundtruth()
    (run / 'run.json').write_text(json.dumps(dict(
        dataset='hilti', sequence='exp99', state='completed', rate=1., interval_start=float(reference.t[0]),
        interval_end=float(reference.t[-1]), windows=dict(mapped=0), loops_closed=0, pose_age_p50_s=.01, pose_age_p95_s=.02)))
    (run / 'trajectory.tum').write_text(''.join(
        f'{t:.9f} {p[0]} {p[1]} {p[2]} {" ".join(map(str, Rotation.from_matrix(R).as_quat()))}\n'
        for t, p, R in zip(reference.t[::4], reference.p[::4], reference.R[::4])))
    (run / 'events.jsonl').write_text('')
    out = script('score_trajectory').evaluate(run, tmp_path / 'hilti')
    assert out['ate']['odometry']['position_m'] < 1e-6 and out['ate']['odometry']['orientation_deg'] < 1e-3
    assert out['control_points']['odometry']['score'] == 100. and out['gauge']['odometry']['log_scale'] == pytest.approx(0., abs=1e-9)


def test_hilti_is_a_dataset_the_runner_takes(root, cfg, script, capsys):
    assert (root / cfg['rigs']['hilti']).is_file()
    with pytest.raises(SystemExit):
        script('run').main(['--dataset', 'hilti', '--data', str(root), '--sequence', 'x', '--out', str(root)])
    err = capsys.readouterr().err
    assert 'invalid choice' not in err and '--out must not exist' in err
