#!/usr/bin/env python3
"""A RealSense ROS 2 bag (the colour camera and the IMU) as a Accord sequence: data/realsense/<seq>/mav0, the colour
frames as JPEG under cam0/data/<ns>.jpg and the IMU as an ASL imu0/data.csv, with conversion.json.

Every time is the message's header stamp, the sensor's clock; the camera-IMU time shift Kalibr measured is the rig's
(config/realsense/kalibr_imucam_chain.yaml, timeshift_cam_imu), applied by the filter, not here. The IMU is the
driver's combined topic; where the bag holds the gyroscope and the accelerometer apart, the accelerometer is
interpolated to the gyroscope's stamps. The frame size must be the rig's.

Reads the bag with rosbags (pure Python; ROS 1, ROS 2 sqlite3 and MCAP), from thirdparty/py310_rosbags in the image:

    ACCORD_GPUS=none scripts/docker.sh pip install --no-deps --target thirdparty/py310_rosbags \\
        rosbags ruamel.yaml ruamel.yaml.clib lz4 zstandard
    ACCORD_GPUS=none scripts/docker.sh python3 scripts/convert_realsense.py --bag data/realsense/bags/NAME --sequence NAME
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'thirdparty/py310_rosbags'))     # after the image's own packages
RIG = ROOT / 'config/realsense/kalibr_imucam_chain.yaml'
IMU_TOPICS = ('/camera/camera/imu', '/camera/imu')
GYRO_TOPICS = ('/camera/camera/gyro/sample', '/camera/gyro/sample')
ACCEL_TOPICS = ('/camera/camera/accel/sample', '/camera/accel/sample')


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def first(topics, present):
    return next((t for t in topics if t in present), None)


def image_bgr(msg, cv2):
    """A sensor_msgs Image or CompressedImage as BGR uint8."""
    if hasattr(msg, 'format'):                                   # CompressedImage
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    data = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    image = data.reshape(msg.height, msg.width, 3)
    if msg.encoding.lower() == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding.lower() == 'bgr8':
        return image.copy()
    raise SystemExit(f'image encoding {msg.encoding} is not handled (rgb8, bgr8 or compressed)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bag', type=Path, required=True, help='the bag: a ROS 2 directory, or a .db3/.mcap/.bag file')
    p.add_argument('--sequence', required=True, help='the name it gets under data/realsense')
    p.add_argument('--data', type=Path, default=ROOT / 'data/realsense')
    p.add_argument('--image-topic', default=None, help="default: the rig's (config/realsense)")
    p.add_argument('--jpeg-quality', type=int, default=95)
    a = p.parse_args()
    import cv2
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    rig = yaml.safe_load(RIG.read_text().replace('%YAML:1.0', ''))['cam0']
    image_topic = a.image_topic or rig['rostopic']
    out = a.data / a.sequence / 'mav0'
    if out.exists():
        raise SystemExit(f'{out} exists; remove it or give another --sequence')
    frames = out / 'cam0/data'
    frames.mkdir(parents=True)
    (out / 'imu0').mkdir()

    with AnyReader([a.bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        present = {c.topic: c.msgtype for c in reader.connections}
        print('topics:', json.dumps(present, indent=1))
        if image_topic not in present:
            raise SystemExit(f'no {image_topic} in the bag')
        imu_topic = first(IMU_TOPICS, present)
        gyro_topic, accel_topic = first(GYRO_TOPICS, present), first(ACCEL_TOPICS, present)
        if imu_topic is None and (gyro_topic is None or accel_topic is None):
            raise SystemExit('no IMU topic: neither a combined one nor gyro and accel samples')

        stamps, size = [], None
        for conn, _t, raw in reader.messages(connections=[c for c in reader.connections if c.topic == image_topic]):
            msg = reader.deserialize(raw, conn.msgtype)
            image = image_bgr(msg, cv2)
            if size is None:
                size = [image.shape[1], image.shape[0]]
                if size != list(rig['resolution']):
                    raise SystemExit(f'the frames are {size[0]}x{size[1]}, the rig {rig["resolution"]}: '
                                     'the calibration is not of this stream')
            t = stamp_ns(msg)
            if stamps and t <= stamps[-1]:
                continue                                         # a repeated or out-of-order frame
            cv2.imwrite(str(frames / f'{t}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            stamps.append(t)

        def samples(topic, field):
            """(stamps, int64 ns; values, n x 3), sorted, one per stamp. Stamps stay integers: in float64 a
            nanosecond time of this size is exact only to a few hundred nanoseconds."""
            rows = []
            for conn, _t, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
                msg = reader.deserialize(raw, conn.msgtype)
                v = getattr(msg, field)
                rows.append((stamp_ns(msg), (v.x, v.y, v.z)))
            rows.sort(key=lambda r: r[0])
            ns = np.array([r[0] for r in rows], np.int64)
            keep = np.r_[True, np.diff(ns) > 0]
            return ns[keep], np.array([r[1] for r in rows], float)[keep]

        if imu_topic is not None:
            (imu_ns, gyro), (_, accel) = samples(imu_topic, 'angular_velocity'), samples(imu_topic, 'linear_acceleration')
            source = dict(imu=imu_topic)
        else:
            (imu_ns, gyro), (accel_ns, accel) = (samples(gyro_topic, 'angular_velocity'),
                                                 samples(accel_topic, 'linear_acceleration'))
            inside = (imu_ns >= accel_ns[0]) & (imu_ns <= accel_ns[-1])
            imu_ns, gyro = imu_ns[inside], gyro[inside]
            since = lambda ns: (ns - accel_ns[0]).astype(float)                   # noqa: E731  small: exact in float
            accel = np.column_stack([np.interp(since(imu_ns), since(accel_ns), accel[:, k]) for k in range(3)])
            source = dict(gyro=gyro_topic, accel=accel_topic, accel_interpolated_to_gyro=True)

    with (out / 'cam0/data.csv').open('w') as f:
        f.write('#timestamp [ns],filename\n')
        f.writelines(f'{t},{t}.jpg\n' for t in stamps)
    with (out / 'imu0/data.csv').open('w') as f:
        f.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        f.writelines(f'{t},' + ','.join(f'{x:.9g}' for x in (*w, *acc)) + '\n' for t, w, acc in zip(imu_ns, gyro, accel))
    seconds = lambda ns: int(ns[-1] - ns[0]) * 1e-9    # noqa: E731
    record = dict(bag=str(a.bag), image_topic=image_topic, **source, frames=len(stamps), resolution=size,
                  camera_rate_hz=round((len(stamps) - 1) / seconds(stamps), 2), imu_samples=len(imu_ns),
                  imu_rate_hz=round((len(imu_ns) - 1) / seconds(imu_ns), 1), duration_s=round(seconds(stamps), 2),
                  first_frame_ns=stamps[0], imu_span_ns=[int(imu_ns[0]), int(imu_ns[-1])], jpeg_quality=a.jpeg_quality)
    (out.parent / 'conversion.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
