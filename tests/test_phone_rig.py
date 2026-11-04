"""The phone rig dataset (rig_android, rig_iphone): a walk's recordings, its reference and the map scorer's reach."""
import cv2
import numpy as np
import pytest

from accord.datasets import PhoneRig, open_dataset
from accord.filter import read_camera

STAMPS = [1_700_000_000_000_000_000 + i * 33_000_000 for i in range(6)]


def write_ply(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
    path.write_text(header + ''.join(f'{x} {y} {z}\n' for x, y, z in points))


@pytest.fixture
def walk(tmp_path):
    """One walk of the release's layout: an Android recording only, and the reference for both phones."""
    seq = tmp_path / 'release/vio99'
    mav = seq / 'android/mav0'
    (mav / 'cam0/data').mkdir(parents=True)
    for stamp in STAMPS:
        cv2.imwrite(str(mav / f'cam0/data/{stamp}.jpg'), np.full((6, 8, 3), 90, np.uint8))
    (mav / 'imu0').mkdir()
    (mav / 'imu0/data.csv').write_text('#timestamp [ns],w,w,w,a,a,a\n' + ''.join(
        f'{STAMPS[0] - 10_000_000 + i * 5_000_000},0,0,0,0,0,9.81\n' for i in range(60)))
    ref = seq / 'reference'
    ref.mkdir()
    t0 = STAMPS[0] * 1e-9
    # nodes 1 s apart, then a 5 s gap the reference must not be interpolated across
    rows = [(t0 + k, k, .5 * np.sin(k), .2 * k) for k in (0, 1, 2, 7, 8)]
    for device in ('android', 'iphone'):
        (ref / f'trajectory_{device}.tum').write_text(''.join(f'{t:.9f} {x} {y} {z} 0 0 0 1\n' for t, x, y, z in rows))
    write_ply(ref / 'map.ply', [(0., 0., 0.), (1., 0., 0.)])
    return tmp_path / 'release'


def test_a_recording_reads_its_device_and_its_rig(walk, root):
    ds = open_dataset('rig_android', walk, 'vio99')
    assert isinstance(ds, PhoneRig) and ds.rig == 'rig_android' and len(ds.image_stamps()) == len(STAMPS)
    rig = read_camera(root / 'config/rig_android/estimator_config.yaml')
    assert ds.camera.resolution == tuple(rig['resolution']) == (640, 480)
    np.testing.assert_allclose(ds.R_CtoI, np.asarray(rig['T_imu_cam'])[:3, :3])
    assert ds.surface_path == walk / 'vio99/reference/map.ply' and ds.surface_reach_m == PhoneRig.SURFACE_REACH_M
    with pytest.raises(FileNotFoundError, match='no iphone recording'):
        open_dataset('rig_iphone', walk, 'vio99')


def test_the_reference_is_densified_but_never_across_a_long_gap(walk):
    ref = open_dataset('rig_android', walk, 'vio99').groundtruth()
    dt = np.diff(ref.t)
    assert dt.min() == pytest.approx(1. / PhoneRig.DENSE_RATE_HZ, abs=1e-6)
    gap = ref.t[1:][dt > .5] - ref.t[:-1][dt > .5]
    assert len(gap) == 1 and gap[0] > PhoneRig.GAP_S          # the 5 s gap is left open
    np.testing.assert_allclose(ref.p[:, 0], ref.t - ref.t[0], atol=1e-6)   # 1 m/s along x on each side of it


def test_every_phone_rig_dataset_has_its_rig_in_the_config(root, cfg):
    for name in ('rig_android', 'rig_iphone'):
        assert (root / cfg['rigs'][name]).is_file()


def test_map_points_beyond_the_reach_of_the_reference_are_left_out(walk, script, tmp_path):
    """The camera of rig_android looks down the IMU's -z: a wall 2 m below the track is in view and within reach, the
    same wall 10 m to the side is beyond it."""
    import json
    span = np.arange(-.5, 2.51, .05)
    xs, ys = (a.ravel() for a in np.meshgrid(span, np.arange(-.8, .81, .05)))
    wall = np.column_stack((xs, ys, np.full(len(xs), -2.)))
    write_ply(walk / 'vio99/reference/map.ply', wall)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run.json').write_text(json.dumps(dict(dataset='rig_android', sequence='vio99', state='completed',
                                                    interval_start=STAMPS[0] * 1e-9)))
    t0 = STAMPS[0] * 1e-9
    t = np.arange(0., 2.001, .05)                                 # the reference's own path, between its first nodes
    y = np.interp(t, [0, 1, 2], .5 * np.sin([0, 1, 2]))
    (run / 'map_trajectory_final.tum').write_text(''.join(f'{t0 + s:.9f} {s} {v} {.2 * s} 0 0 0 1\n' for s, v in zip(t, y)))
    arguments = ['--run', str(run), '--data', str(walk), '--support-cache', str(tmp_path / 'support')]
    write_ply(run / 'map.ply', np.vstack((wall, wall + [0., 10., 0.])))
    assert script('score_map').main(arguments) == 0
    report = json.loads((run / 'surface.json').read_text())
    assert report['beyond_reach'] == dict(reach_m=PhoneRig.SURFACE_REACH_M, share_of_points_left_out=pytest.approx(.5))
    assert report['estimate_points'] == len(wall) and report['thresholds']['0.05']['precision'] == pytest.approx(1.)
    # a map wholly beyond the reach (a run gone far off) scores 0, not an error
    write_ply(run / 'map.ply', wall + [0., 10., 0.])
    assert script('score_map').main([*arguments, '--force']) == 0
    report = json.loads((run / 'surface.json').read_text())
    assert report['estimate_points'] == 0 and report['thresholds']['0.05']['f1'] == pytest.approx(0.)
    assert report['accuracy'] is None and report['chamfer_mean_m'] is None
