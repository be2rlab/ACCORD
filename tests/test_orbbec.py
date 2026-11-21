"""Orbbec's nonstandard ROS wire layout and factory IMU->depth->RGB chain."""
from types import SimpleNamespace
import struct

import numpy as np
import pytest


def test_factory_chain_is_inverted_and_millimetres_become_metres(script):
    convert = script('convert_orbbec')
    camera = bytearray(161)
    struct.pack_into('<4f2h', camera, 20, 600., 601., 640., 360., 1280, 720)
    R_cd = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    struct.pack_into('<12f', camera, 112, *R_cd.ravel(), 20., 0., 0.)
    imu = bytearray(2929)
    imu[0] = 1
    struct.pack_into('<2d', imu, 161, .004, .00008)
    struct.pack_into('<2d', imu, 377, .000032, .000002)
    struct.pack_into('<d', imu, 841, .003)
    T_di = np.eye(4)
    T_di[:3, 3] = [1., 2., 3.]
    struct.pack_into('<16d', imu, 849, *T_di.ravel())
    profile = SimpleNamespace(width=1280, height=720, cameraIntrinsic=[600., 601., 640., 360.])
    T_ic, shift, acc, gyro = convert.calibration({4027: camera, 4036: imu}, profile)
    T_ci = np.eye(4)
    T_ci[:3, :3] = R_cd
    T_ci[:3, 3] = (R_cd @ np.array([1., 2., 3.]) + [20., 0., 0.]) * .001
    np.testing.assert_allclose(T_ic @ T_ci, np.eye(4), atol=1e-12)
    assert shift == .003 and acc[:2].tolist() == [.004, .00008] and gyro[1] == .000002
    with pytest.raises(ValueError, match='layout'):
        convert.calibration({4027: camera[:-1], 4036: imu}, profile)


def test_bag_definitions_are_repaired_to_the_actual_sdk_wire_format(script):
    pytest.importorskip('rosbags')
    convert = script('convert_orbbec')
    prop = SimpleNamespace(msgtype='custom_msg/msg/OBProperty', msgdef=SimpleNamespace(data=
        'std_msgs/Header header\nuint8 propertyId\nuint8[] data\nuint32 datasize\n'))
    image = SimpleNamespace(msgtype='sensor_msgs/msg/Image', msgdef=SimpleNamespace(data=
        'std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\n'
        'uint8 is_bigendian\nuint32 step\nuint8[] data\nuint64 number\n'
        'uint64 timestamp_usec\nuint64 timestamp_systemusec\nuint64 timestamp_globalusec\n'
        'uint8[] metadata\nuint32 metadatasize\n'))
    store, names = convert.typestore([prop, image])
    header = struct.pack('<4I', 0, 10, 0, 0)  # seq, seconds, ns, empty frame_id
    raw_property = header + struct.pack('<II3sI', 4036, 3, b'abc', 3)
    msg = store.deserialize_ros1(raw_property, prop.msgtype)
    assert msg.propertyId == 4036 and msg.data.tobytes() == b'abc'
    raw_image = (header + struct.pack('<III4sBII3sf4QII B', 2, 3, 4, b'mjpg', 0, 3, 3, b'jpg',
                                     1., 7, 10_000_000, 10_000_100, 10_000_200, 0, 0, 8))
    msg = store.deserialize_ros1(raw_image, names[image.msgtype])
    assert msg.data.tobytes() == b'jpg' and msg.number == 7
    assert msg.timestamp_usec == 10_000_000 and msg.pixel_bit_size == 8


def test_undistortion_uses_k3_and_the_rational_coefficients(script):
    import cv2
    convert = script('convert_orbbec')
    profile = SimpleNamespace(cameraIntrinsic=[300., 301., 320., 240.], width=640, height=480,
                              cameraDistortion=[.1, -.02, .03, .004, -.005, .006, .001, -.002], distortionModel=4)
    x, y = convert.undistortion(profile)
    K = np.array([[300., 0, 320.], [0, 301., 240.], [0, 0, 1.]])
    point = np.array([[(80. - 320.) / 300., (100. - 240.) / 301., 1.]])
    projected, _ = cv2.projectPoints(point, np.zeros(3), np.zeros(3), K,
                                     np.array([.1, -.02, .001, -.002, .03, .004, -.005, .006]))
    np.testing.assert_allclose([x[100, 80], y[100, 80]], projected.ravel(), atol=1e-4)


def test_orbbec_loader_and_runner_use_the_orbbec_rig(tmp_path, root, cfg, script, capsys):
    import cv2
    from accord.datasets import Orbbec, open_dataset
    from accord.filter import read_camera
    frames = tmp_path / 'walk/mav0/cam0/data'
    frames.mkdir(parents=True)
    for stamp in (1_000_000_000, 1_033_333_000):
        cv2.imwrite(str(frames / f'{stamp}.jpg'), np.zeros((720, 1280, 3), np.uint8))
    ds = open_dataset('orbbec', tmp_path, 'walk')
    assert isinstance(ds, Orbbec) and ds.camera.resolution == (1280, 720)
    assert ds.load_filter_image(ds.stamps[0]).shape == (720, 1280)
    assert np.all(ds.camera.D == 0)
    rig = root / cfg['rigs']['orbbec']
    np.testing.assert_allclose(ds.R_CtoI, np.array(read_camera(rig)['T_imu_cam'])[:3, :3])
    with pytest.raises(SystemExit):
        script('run').main(['--dataset', 'orbbec', '--data', str(tmp_path), '--sequence', 'walk', '--out', str(root)])
    assert '--out must not exist' in capsys.readouterr().err
