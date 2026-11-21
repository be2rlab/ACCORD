#!/usr/bin/env python3
"""OrbbecViewer SDK v2.10.6 bag -> colour + IMU ASL sequence and factory rig.

    pip install rosbags numpy opencv-python-headless PyYAML
    python scripts/convert_orbbec.py --bag 'data/recording.bag' --sequence walk01

No ROS or camera is needed. Uses integer device microseconds, not host arrival
times. MJPG frames are undistorted with the full Brown-Conrady model; the output
is pinhole with zero distortion. The raw calibration properties supply the
IMU->depth->colour transform even when depth was not recorded.

The wire-definition repairs and packed calibration layouts below match SDK
v2.10.6 (src/media/ros/RosbagWriter.cpp, src/shared/InternalTypes.hpp and
3rdparty/rosbag/src/msgs/). Unsupported layouts fail rather than guess.
Factory IMU noise values and camera-IMU time shift are starting values, not a
Kalibr/Allan calibration of this rig.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from accord.filter import write_yaml  # noqa: E402

COLOR = '/cam/sensor_2/frameType_2'
ACCEL = '/cam/sensor_4/frameType_4'
GYRO = '/cam/sensor_5/frameType_7'


def typestore(connections):
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    store = get_typestore(Stores.ROS1_NOETIC)
    # Orbbec reuses standard ROS names with extended, partly incorrect .msg
    # definitions. Register them under private names to avoid standard types.
    names = {'sensor_msgs/msg/Image': 'orbbec/msg/Image',
             'sensor_msgs/msg/Imu': 'orbbec/msg/Imu'}
    for conn in connections:
        definition = conn.msgdef.data.replace('float temperature', 'float32 temperature')
        definition = definition.replace('uint8 propertyId', 'uint32 propertyId')
        if conn.msgtype == 'sensor_msgs/msg/Image':
            definition = definition.replace('uint64 number', 'float32 depth_units\nuint64 number')
            definition = definition.replace('uint32 metadatasize', 'uint32 metadatasize\nuint8 pixel_bit_size')
        store.register(get_types_from_msg(definition, names.get(conn.msgtype, conn.msgtype)))
    return store, names


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, float).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, float) * .001  # SDK translations are mm
    if not np.isfinite(result).all() or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError('invalid factory rotation')
    if not np.isclose(np.linalg.det(result[:3, :3]), 1., atol=1e-5):
        raise ValueError('factory rotation is not right-handed')
    return result


def calibration(properties, profile):
    camera_blob, imu_blob = properties[4027], properties[4036]
    # Packed OBCameraParam: two 20-byte intrinsics, two 36-byte distortions,
    # a 48-byte depth->colour transform and one bool.
    if len(camera_blob) % 161 or len(imu_blob) != 2929 or imu_blob[0] != 1:
        raise ValueError('unsupported packed factory calibration layout')
    matches = []
    for offset in range(0, len(camera_blob), 161):
        intrinsics = struct.unpack_from('<4f2h', camera_blob, offset + 20)
        if intrinsics[4:] == (profile.width, profile.height) and np.allclose(intrinsics[:4], profile.cameraIntrinsic):
            if camera_blob[offset + 160]:
                raise ValueError('mirrored factory camera profile is not supported')
            matches.append(transform(np.frombuffer(camera_blob, '<f4', 9, offset + 112),
                                     np.frombuffer(camera_blob, '<f4', 3, offset + 148)))
    if not matches or not all(np.allclose(t, matches[0]) for t in matches):
        raise ValueError('missing or ambiguous depth->colour calibration')
    # OBIMUCalibrateParams: validNum then three packed 976-byte OBSingleIMUParams.
    # The first IMU contains 16 bytes of name/version, 144 bytes of matrices,
    # acc (216), gyro (192), mag (272), time shift (8), IMU->depth (128).
    acc = np.frombuffer(imu_blob, '<f8', 27, 1 + 160).copy()
    gyro = np.frombuffer(imu_blob, '<f8', 24, 1 + 376).copy()
    time_shift = struct.unpack_from('<d', imu_blob, 1 + 840)[0]
    imu_to_depth = np.frombuffer(imu_blob, '<f8', 16, 1 + 848).reshape(4, 4)
    if not np.allclose(imu_to_depth[3], [0, 0, 0, 1]):
        raise ValueError('invalid IMU->depth transform')
    T_di = transform(imu_to_depth[:3, :3], imu_to_depth[:3, 3])
    return np.linalg.inv(matches[0] @ T_di), time_shift, acc, gyro


def undistortion(profile):
    fx, fy, cx, cy = profile.cameraIntrinsic
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    # Orbbec: k1 k2 k3 k4 k5 k6 p1 p2. OpenCV: k1 k2 p1 p2 k3 k4 k5 k6.
    if profile.distortionModel not in (0, 3, 4):
        raise ValueError(f'unsupported camera distortion model {profile.distortionModel}')
    D = np.asarray(profile.cameraDistortion, float)[[0, 1, 6, 7, 2, 3, 4, 5]]
    if profile.distortionModel == 0:
        D[:] = 0
    return cv2.initUndistortRectifyMap(K, D, None, K, (profile.width, profile.height), cv2.CV_32FC1)


def sorted_samples(rows):
    rows.sort(key=lambda row: row[0])
    ns = np.array([row[0] for row in rows], np.int64)
    values = np.array([row[1] for row in rows], float)
    if len(ns) < 2 or not np.isfinite(values).all():
        raise ValueError('missing or non-finite IMU samples')
    if np.any(np.diff(ns) <= 0):
        raise ValueError('duplicate IMU device timestamps')
    return ns, values


def stream_stats(ns):
    dt = np.diff(np.asarray(ns, np.int64)) * 1e-9
    return dict(samples=len(ns), duration_s=float((ns[-1] - ns[0]) * 1e-9),
                rate_hz=float(1 / np.median(dt)), max_gap_s=float(dt.max()),
                gaps_over_1_5_periods=int(np.sum(dt > 1.5 * np.median(dt))))


def main(argv=None):
    from rosbags.rosbag1 import Reader
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bag', required=True, type=Path)
    p.add_argument('--sequence', required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/orbbec')
    p.add_argument('--rig', type=Path, default=ROOT / 'config/orbbec')
    a = p.parse_args(argv)
    if Path(a.sequence).name != a.sequence or a.sequence in ('.', '..'):
        p.error('--sequence must be one directory name')
    out = a.data / a.sequence
    if out.exists():
        p.error(f'{out} already exists; choose a new --sequence')
    with Reader(a.bag) as reader:
        store, names = typestore(reader.connections)
        present = {c.topic: c for c in reader.connections}
        required = (COLOR, ACCEL, GYRO, '/cam/property', '/cam/streamProfileType_2', '/cam/deviceInfo')
        if missing := set(required) - present.keys():
            raise ValueError(f'missing topics: {sorted(missing)}')

        def messages(topics):
            for conn, _, raw in reader.messages(connections=[present[t] for t in topics]):
                yield conn.topic, store.deserialize_ros1(raw, names.get(conn.msgtype, conn.msgtype))

        properties = {}
        for topic, msg in messages(('/cam/property', '/cam/streamProfileType_2', '/cam/deviceInfo')):
            if topic == '/cam/property':
                properties[msg.propertyId] = msg.data.tobytes()
            elif topic == '/cam/streamProfileType_2':
                profile = msg
            else:
                device = msg
        T_ic, time_shift, acc_calib, gyro_calib = calibration(properties, profile)
        maps = undistortion(profile)
        rows = {ACCEL: [], GYRO: []}
        for topic, msg in messages((ACCEL, GYRO)):
            value = msg.linear_acceleration if topic == ACCEL else msg.angular_velocity
            rows[topic].append((int(msg.timestamp_usec) * 1000, (value.x, value.y, value.z)))
        acc_ns, accel = sorted_samples(rows[ACCEL])
        gyro_ns, gyro = sorted_samples(rows[GYRO])
        inside = (gyro_ns >= acc_ns[0]) & (gyro_ns <= acc_ns[-1])
        imu_ns, gyro = gyro_ns[inside], gyro[inside]
        if len(imu_ns) < 2:
            raise ValueError('accelerometer and gyroscope do not overlap')
        rel = lambda ns: (ns - acc_ns[0]).astype(float)
        accel = np.column_stack([np.interp(rel(imu_ns), rel(acc_ns), accel[:, k]) for k in range(3)])

        camera_cfg = dict(cam0=dict(T_imu_cam=T_ic.tolist(), cam_overlaps=[], camera_model='pinhole',
                                   distortion_coeffs=[0., 0., 0., 0.], distortion_model='radtan',
                                   intrinsics=profile.cameraIntrinsic.astype(float).tolist(),
                                   resolution=[profile.width, profile.height], rostopic=COLOR,
                                   timeshift_cam_imu=float(time_shift)))
        imu_cfg = dict(imu0=dict(T_i_b=np.eye(4).tolist(),
                                accelerometer_noise_density=float(acc_calib[0]),
                                accelerometer_random_walk=float(acc_calib[1]),
                                gyroscope_noise_density=float(gyro_calib[0]),
                                gyroscope_random_walk=float(gyro_calib[1]),
                                rostopic=GYRO, time_offset=0., update_rate=round(1e9 / np.median(np.diff(imu_ns))),
                                model='kalibr', Tw=np.eye(3).tolist(), R_IMUtoGYRO=np.eye(3).tolist(),
                                Ta=np.eye(3).tolist(), R_IMUtoACC=np.eye(3).tolist(), Tg=np.zeros((3, 3)).tolist()))
        from accord.filter import read_yaml
        for name, cfg in (('kalibr_imucam_chain.yaml', camera_cfg), ('kalibr_imu_chain.yaml', imu_cfg)):
            path = a.rig / name
            if path.exists() and read_yaml(path) != cfg:
                raise ValueError(f'{path} differs from this recording; use a separate --rig')

        frames = out / 'mav0/cam0/data'
        frames.mkdir(parents=True)
        stamps = []
        for _, msg in messages((COLOR,)):
            stamp = int(msg.timestamp_usec) * 1000
            # Keep only frames bracketed by IMU. Host/header stamps lose precision.
            if stamp < imu_ns[0] or stamp > imu_ns[-1]:
                continue
            if stamps and stamp <= stamps[-1]:
                raise ValueError('non-increasing RGB device timestamps')
            if msg.encoding.lower() != 'mjpg' or (msg.width, msg.height) != (profile.width, profile.height):
                raise ValueError('expected a constant-size MJPG colour stream')
            image = cv2.imdecode(msg.data, cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (profile.height, profile.width):
                raise ValueError(f'cannot decode RGB frame at {stamp}')
            image = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
            if not cv2.imwrite(str(frames / f'{stamp}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f'cannot write RGB frame at {stamp}')
            stamps.append(stamp)
        if len(stamps) < 2:
            raise ValueError('fewer than two RGB frames overlap the IMU')

    (frames.parent / 'data.csv').write_text('#timestamp [ns],filename\n' + ''.join(f'{t},{t}.jpg\n' for t in stamps))
    imu_path = out / 'mav0/imu0/data.csv'
    imu_path.parent.mkdir()
    with imu_path.open('w') as handle:
        handle.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                     'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        for t, w, acc in zip(imu_ns, gyro, accel):
            handle.write(f'{t},' + ','.join(f'{v:.12g}' for v in (*w, *acc)) + '\n')
    a.rig.mkdir(parents=True, exist_ok=True)
    for name, cfg in (('kalibr_imucam_chain.yaml', camera_cfg), ('kalibr_imu_chain.yaml', imu_cfg)):
        write_yaml(a.rig / name, cfg)
    report = dict(bag=str(a.bag), sequence=a.sequence, serial=device.sn, firmware=device.fwVersion,
                  connection=device.connectionType, timestamp='timestamp_usec * 1000 (device clock)',
                  rgb=stream_stats(stamps), accel=stream_stats(acc_ns), gyro=stream_stats(gyro_ns),
                  resolution=[profile.width, profile.height], imu_samples=len(imu_ns),
                  accel_interpolated_to_gyro=True, imu_units=dict(gyro='rad/s', accel='m/s^2'),
                  undistorted=True, factory_intrinsics=profile.cameraIntrinsic.astype(float).tolist(),
                  factory_distortion=profile.cameraDistortion.astype(float).tolist(),
                  factory_distortion_model=int(profile.distortionModel), rig=str(a.rig),
                  calibration='factory; IMU noise and time shift have not been calibrated with Kalibr/Allan',
                  accel_norm_median=float(np.median(np.linalg.norm(accel, axis=1))))
    (out / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
