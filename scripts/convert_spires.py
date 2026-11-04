#!/usr/bin/env python3
"""One deterministic conversion of an Oxford Spires sequence (scripts/fetch_spires.py) into Accord's packet layout.

Reads cam0 (the front camera) from images.zip and the IMU from imu.csv; the other two cameras are never read. Writes
    <seq>/accord/cam0/data/<stamp_ns>.jpg   the release's JPEG, byte for byte, named by its stamp in nanoseconds
                                           (the release names it <secs>.<nanoseconds, nine digits>.jpg)
    <seq>/accord/imu0/data.csv              ASL layout: ns, gyro rad/s, accel m/s^2 (the release has secs, nsecs,
                                           accelerometer, gyroscope); the numbers' text is copied, never reprinted
    <seq>/accord/conversion.json            counts, spans, rates, the inputs' identity
The reference is not converted; accord.datasets.Spires reads <seq>/gt/gt-tum.txt directly. The conversion is staged
beside the sequence and renamed into place at the end, so a accord/ folder is always complete.

    python3 scripts/convert_spires.py data/spires/cc03
"""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from convert_hilti import IMU_HEADER, summarize          # noqa: E402

CAMERA = 'cam0'
FRAME = re.compile(r'^(\d+)\.(\d{9})\.jpg$')
IMU_COLUMNS = ('secs', 'nsecs', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'acc_x', 'acc_y', 'acc_z')


def frame_stamp(name):
    """The stamp in nanoseconds of a release frame name, <secs>.<nanoseconds>.jpg; anything else is refused."""
    match = FRAME.match(name)
    if match is None:
        raise ValueError(f'unexpected frame name {name!r}; expected <secs>.<nine-digit nanoseconds>.jpg')
    return int(match.group(1)) * 10**9 + int(match.group(2))


def camera_members(archive, camera=CAMERA):
    """[(stamp_ns, member)] of one camera's frames in the archive, in time order."""
    out = []
    for member in archive.infolist():
        folder, _, name = member.filename.rpartition('/')
        if folder != camera or member.is_dir() or not name:
            continue
        out.append((frame_stamp(name), member))
    out.sort(key=lambda item: item[0])
    stamps = [s for s, _ in out]
    if len(set(stamps)) != len(stamps):
        raise ValueError(f'{camera}: two frames share a stamp')
    return out


def imu_rows(handle):
    """(stamp_ns, [gyro x y z, accel x y z] as the release's text) per row of the release's imu.csv."""
    reader = csv.reader(handle)
    header = [h.strip() for h in next(reader)]
    missing = [c for c in IMU_COLUMNS if c not in header]
    if missing:
        raise ValueError(f'imu.csv lacks the columns {missing}; it has {header}')
    index = [header.index(c) for c in IMU_COLUMNS]
    for row in reader:
        if not row:
            continue
        secs, nsecs, *values = (row[i].strip() for i in index)
        if not 0 <= int(nsecs) < 10**9:
            raise ValueError(f'imu.csv: nanoseconds out of range in {row}')
        for value in values:
            float(value)                        # a number, or the conversion stops
        yield int(secs) * 10**9 + int(nsecs), values


def jpeg_size(data):
    """(width, height) from a JPEG's start-of-frame segment."""
    stream = io.BytesIO(data)
    if stream.read(2) != b'\xff\xd8':
        raise ValueError('not a JPEG')
    while True:
        marker = stream.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            raise ValueError('JPEG without a frame header')
        length = int.from_bytes(stream.read(2), 'big')
        if marker[1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            segment = stream.read(5)
            return int.from_bytes(segment[3:5], 'big'), int.from_bytes(segment[1:3], 'big')
        stream.seek(length - 2, 1)


def convert(sequence, log=print):
    sequence = Path(sequence)
    archive_path, imu_path = sequence / 'images.zip', sequence / 'imu.csv'
    out, stage = sequence / 'accord', sequence / '.accord.partial'
    if out.exists():
        raise SystemExit(f'{out} exists; conversion is never repeated in place')
    for path in (archive_path, imu_path):
        if not path.is_file():
            raise SystemExit(f'{path} is missing; run scripts/fetch_spires.py')
    if stage.exists():
        shutil.rmtree(stage)                    # an interrupted conversion: never used, started again
    (stage / 'cam0/data').mkdir(parents=True)
    (stage / 'imu0').mkdir()
    image_stamps, imu_stamps, size = [], [], None
    with zipfile.ZipFile(archive_path) as archive:
        members = camera_members(archive)
        if len(members) < 2:
            raise SystemExit(f'{archive_path}: {len(members)} frames of {CAMERA}')
        for k, (stamp, member) in enumerate(members):
            data = archive.read(member)         # the CRC is checked while reading
            frame_size = jpeg_size(data)
            if size is None:
                size = frame_size
            elif frame_size != size:
                raise SystemExit(f'frame size changed from {size} to {frame_size} at {stamp}')
            (stage / 'cam0/data' / f'{stamp}.jpg').write_bytes(data)
            image_stamps.append(stamp)
            if (k + 1) % 2000 == 0:
                log(f'  {k + 1} frames', flush=True)
    with imu_path.open(newline='') as source, (stage / 'imu0/data.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(IMU_HEADER)
        for stamp, values in imu_rows(source):
            writer.writerow([stamp, *values])
            imu_stamps.append(stamp)
    if len(imu_stamps) < 2:
        raise SystemExit(f'{imu_path}: {len(imu_stamps)} IMU samples')
    manifest = sequence / 'download_manifest.json'
    identity = json.loads(manifest.read_text())['files'] if manifest.is_file() else None
    report = dict(sequence=sequence.name, camera=CAMERA, image_size=list(size), images=summarize(image_stamps),
                  imu=summarize(imu_stamps), inputs=dict(images=archive_path.name, imu=imu_path.name),
                  identity=identity, note='verbatim JPEG payloads and IMU text; no resampling, no rectification')
    (stage / 'conversion.json').write_text(json.dumps(report, indent=1) + '\n')
    stage.rename(out)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequences', nargs='+', type=Path, help='sequence folders, e.g. data/spires/cc03')
    a = p.parse_args(argv)
    for sequence in a.sequences:
        report = convert(sequence)
        print(json.dumps(dict(sequence=report['sequence'], image_size=report['image_size'], images=report['images'],
                              imu=report['imu']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
