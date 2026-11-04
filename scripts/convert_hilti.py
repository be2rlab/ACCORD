#!/usr/bin/env python3
"""One deterministic conversion of a Hilti-Oxford ROS 1 bag into Accord's packet layout.

Reads one camera and the IMU; the other four cameras and the lidar are skipped. Writes
    <seq>/accord/cam0/data/<stamp_ns>.png   mono8, lossless, the header stamp as the name
    <seq>/accord/imu0/data.csv              ASL layout: ns, gyro rad/s, accel m/s^2
    <seq>/accord/conversion.json            counts, spans, rates, the bag's identity
Ground truth is not converted; accord.datasets.Hilti reads <seq>/gt/*.txt directly. Run inside Docker
(rosbags and OpenCV live in the image):
    scripts/docker.sh python3 scripts/convert_hilti.py data/hilti/exp14
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CAMERA_TOPIC = '/alphasense/{camera}/image_raw'
IMU_TOPIC = '/alphasense/imu'
IMU_HEADER = ['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
              'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']


def stamp_ns(header):
    return int(header.stamp.sec) * 10**9 + int(header.stamp.nanosec)


def image_array(msg):
    """The mono8 payload as (height, width) uint8; anything else is refused, never reinterpreted."""
    if msg.encoding not in ('mono8', '8UC1'):
        raise ValueError(f'unexpected image encoding {msg.encoding!r}; this converter reads mono8')
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.frombuffer(bytes(msg.data), np.uint8)
    if step < width or data.size != height * step:
        raise ValueError(f'image payload of {data.size} bytes does not match {height}x{step}')
    return np.ascontiguousarray(data.reshape(height, step)[:, :width])


def summarize(stamps):
    stamps = np.asarray(stamps, np.int64)
    steps = np.diff(stamps)
    return dict(count=int(stamps.size), first_ns=int(stamps[0]), last_ns=int(stamps[-1]),
                rate_hz=float(1e9 / np.median(steps)), max_gap_s=float(steps.max() * 1e-9),
                non_monotonic=int(np.count_nonzero(steps <= 0)))


def convert(bag, out, camera='cam0', log=print):
    import cv2
    from rosbags.highlevel import AnyReader
    bag, out = Path(bag), Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists; conversion is never repeated in place')
    (out / 'cam0/data').mkdir(parents=True)
    (out / 'imu0').mkdir()
    camera_topic = CAMERA_TOPIC.format(camera=camera)
    image_stamps, imu_stamps, shape = [], [], None
    with (out / 'imu0/data.csv').open('w', newline='') as handle, AnyReader([bag]) as reader:
        topics = sorted({c.topic for c in reader.connections})
        missing = [t for t in (camera_topic, IMU_TOPIC) if t not in topics]
        if missing:
            raise SystemExit(f'{bag}: no topic {missing}; the bag has {topics}')
        writer = csv.writer(handle)
        writer.writerow(IMU_HEADER)
        wanted = [c for c in reader.connections if c.topic in (camera_topic, IMU_TOPIC)]
        for connection, _received, raw in reader.messages(connections=wanted):
            msg = reader.deserialize(raw, connection.msgtype)
            stamp = stamp_ns(msg.header)
            if connection.topic == IMU_TOPIC:
                g, a = msg.angular_velocity, msg.linear_acceleration
                writer.writerow([stamp, repr(float(g.x)), repr(float(g.y)), repr(float(g.z)),
                                 repr(float(a.x)), repr(float(a.y)), repr(float(a.z))])
                imu_stamps.append(stamp)
                continue
            image = image_array(msg)
            if shape is None:
                shape = image.shape
            elif image.shape != shape:
                raise SystemExit(f'image size changed from {shape} to {image.shape} at {stamp}')
            if not cv2.imwrite(str(out / 'cam0/data' / f'{stamp}.png'), image):
                raise SystemExit(f'could not write frame {stamp}')
            image_stamps.append(stamp)
            if len(image_stamps) % 1000 == 0:
                log(f'  {len(image_stamps)} frames')
    if len(image_stamps) < 2 or len(imu_stamps) < 2:
        raise SystemExit(f'{bag}: {len(image_stamps)} frames and {len(imu_stamps)} IMU samples')
    manifest = bag.parent / 'download_manifest.json'
    identity = json.loads(manifest.read_text())['files'].get(bag.name) if manifest.is_file() else None
    report = dict(bag=bag.name, bag_bytes=bag.stat().st_size, bag_identity=identity,
                  camera=camera, camera_topic=camera_topic, imu_topic=IMU_TOPIC,
                  image_shape=[int(shape[0]), int(shape[1])],
                  images=summarize(image_stamps), imu=summarize(imu_stamps))
    (out / 'conversion.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='the sequence folder, e.g. data/hilti/exp14')
    p.add_argument('--camera', default='cam0', choices=('cam0', 'cam1', 'cam2', 'cam3', 'cam4'),
                   help='config/hilti holds the calibration of cam0 only')
    a = p.parse_args(argv)
    bags = sorted(a.sequence.glob('*.bag'))
    if len(bags) != 1:
        raise SystemExit(f'{a.sequence}: expected one .bag, found {len(bags)}')
    report = convert(bags[0], a.sequence / 'accord', a.camera)
    print(json.dumps(dict(images=report['images'], imu=report['imu']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
