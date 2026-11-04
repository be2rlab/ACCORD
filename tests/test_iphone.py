"""iPhone recordings (scripts/convert_iphone.py): the turned frame's camera, the rig check, and the dataset."""
import importlib.util

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from accord.datasets import Iphone, open_dataset

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('convert_iphone', ROOT / 'scripts/convert_iphone.py')
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)
RIG = np.diag([1., -1., -1.])               # config/iphone's T_imu_cam


def test_a_turned_frame_sees_a_point_where_its_intrinsics_put_it():
    K, (width, height) = np.array([1441.8, 1441.8, 963.9, 717.1]), (1920, 1440)
    X = np.array([.3, -.2, 2.])                                   # in the recorded frame's camera
    u, v = K[0] * X[0] / X[2] + K[2], K[1] * X[1] / X[2] + K[3]
    s = convert.WIDTH / width
    u, v = (u + .5) * s - .5, (v + .5) * s - .5                   # scaled
    expected = (round(height * s) - 1 - v, u)                     # turned clockwise
    fx, fy, cx, cy = convert.turned_intrinsics(K, width, height)
    Y = convert.R_TURNED[90].T @ X                                    # in the turned frame's camera
    np.testing.assert_allclose((fx * Y[0] / Y[2] + cx, fy * Y[1] / Y[2] + cy), expected, atol=1e-9)
    image = np.zeros((height, width, 3), np.uint8)
    image[:30, :40] = 255                                         # the top-left corner goes to the top right
    turned = convert.upright(image, cv2)
    assert turned.shape == (640, 480, 3) and turned[0, -1].min() == 255 and turned[0, 0].max() == 0


@pytest.mark.parametrize('turn', [0, 180])
def test_a_landscape_frame_sees_a_point_where_its_intrinsics_put_it(turn):
    K, (width, height) = np.array([1434.2, 1434.2, 955.1, 721.4]), (1920, 1440)
    X = np.array([.3, -.2, 2.])
    u, v = K[0] * X[0] / X[2] + K[2], K[1] * X[1] / X[2] + K[3]
    s = convert.WIDTH / width
    u, v = (u + .5) * s - .5, (v + .5) * s - .5
    expected = (u, v) if turn == 0 else (convert.WIDTH - 1 - u, round(height * s) - 1 - v)
    fx, fy, cx, cy = convert.turned_intrinsics(K, width, height, turn)
    Y = convert.R_TURNED[turn].T @ X
    np.testing.assert_allclose((fx * Y[0] / Y[2] + cx, fy * Y[1] / Y[2] + cy), expected, atol=1e-9)
    image = np.zeros((height, width, 3), np.uint8)
    image[:30, :40] = 255
    turned = convert.upright(image, cv2, turn)
    assert turned.shape == (480, 640, 3)
    assert (turned[0, 0] if turn == 0 else turned[-1, -1]).min() == 255


@pytest.fixture
def recording():
    """ARKit's orientations and a gyroscope that sees them through the rig, 25 ms late, with gravity's sign."""
    rng = np.random.default_rng(0)
    t = np.arange(0., 6., 1e-3)
    w = np.column_stack([np.sin(2 * np.pi * f * t + p) for f, p in zip(rng.uniform(.3, 1.5, 3), rng.uniform(0, 6, 3))])
    R = [np.eye(3)]
    for k in range(len(t) - 1):
        R.append(R[-1] @ Rotation.from_rotvec(w[k] * 1e-3).as_matrix())
    R_wc = np.asarray(R)                                          # the turned camera to ARKit's world
    frames, imu = np.arange(0, len(t), 33), np.arange(30, len(t) - 30, 10)
    q = Rotation.from_matrix(R_wc[frames] @ convert.R_TURNED[90].T).as_quat()
    up = np.einsum('nji,j->ni', R_wc[imu - 25] @ RIG.T, [0., 1., 0.])
    return dict(t=t[frames], q=q, K=np.tile([1441.8, 1441.8, 963.9, 717.1], (len(frames), 1)), imu_t=t[imu],
                gyro=w[imu - 25] @ RIG.T, accel=-up)


def test_the_check_finds_the_rig_and_the_shift(recording):
    rig = dict(T_imu_cam=np.eye(4), intrinsics=[480.6028, 480.6028, 240.3157, 320.9784])
    rig['T_imu_cam'][:3, :3] = RIG
    with pytest.raises(SystemExit, match='upright'):              # the made-up motion turns the phone every way
        convert.check(recording, rig, (1920, 1440))
    convert.MIN_UPRIGHT, kept = 0., convert.MIN_UPRIGHT
    try:
        out = convert.check(recording, rig, (1920, 1440))
        assert out['rotation_to_rig_deg'] < .5 and out['gyro_residual_rad_s'] < .05
        assert out['imu_shift_s'] == pytest.approx(.025, abs=.0015) and out['gravity_cos'] > .999
        rig['T_imu_cam'][:3, :3] = np.diag([-1., -1., 1.])        # the first version's rig, 180 degrees off
        with pytest.raises(SystemExit, match='rotation'):
            convert.check(recording, rig, (1920, 1440))
    finally:
        convert.MIN_UPRIGHT = kept


def test_an_iphone_sequence_has_the_rig_camera_and_no_reference(tmp_path):
    mav = tmp_path / 'iphone/rec/mav0'
    (mav / 'cam0/data').mkdir(parents=True)
    (mav / 'imu0').mkdir()
    for stamp in (1_000_000_000, 1_033_000_000):
        cv2.imwrite(str(mav / f'cam0/data/{stamp}.jpg'), np.full((640, 480, 3), 90, np.uint8))
    (mav / 'imu0/data.csv').write_text('#t,w,w,w,a,a,a\n' + ''.join(f'{990_000_000 + i * 10_000_000},0,0,0,0,9.81,0\n'
                                                                    for i in range(6)))
    ds = open_dataset('iphone', tmp_path / 'iphone', 'rec')
    assert isinstance(ds, Iphone) and ds.camera.resolution == (480, 640) and ds.load_filter_image(1_000_000_000).ndim == 2
    np.testing.assert_allclose(ds.R_CtoI, RIG)
    with pytest.raises(LookupError):
        ds.groundtruth()
