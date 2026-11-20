#!/usr/bin/env python3
"""Fetch EuRoC MAV sequences from the ETH Research Collection, one sequence at a time."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import sys
import urllib.error
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 2 * 1024**3
CHUNK = 8 * 1024**2
AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
BITSTREAMS = 'https://www.research-collection.ethz.ch/server/api/core/bitstreams'
ARCHIVES = {'MH': ('machine_hall', f'{BITSTREAMS}/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content'),
            'V1': ('vicon_room1', f'{BITSTREAMS}/02ecda9a-298f-498b-970c-b7c44334d880/content'),
            'V2': ('vicon_room2', f'{BITSTREAMS}/ea12bc01-3677-4b4c-853d-87c7870b8c44/content')}
SEQUENCES = ('MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult',
             'V1_01_easy', 'V1_02_medium', 'V1_03_difficult', 'V2_01_easy', 'V2_02_medium', 'V2_03_difficult')
FORMER_ROOMS = ('machine_hall', 'vicon_room')       # where the former fetcher (fetch_euroc.sh) left the sequences


def open_range(url, start, end):
    """Bytes start..end (inclusive) of the file at url, as a stream, and the file's size."""
    request = urllib.request.Request(url, headers={'User-Agent': AGENT, 'Range': f'bytes={start}-{end}'})
    response = urllib.request.urlopen(request, timeout=120)
    if response.status != 206:
        response.close()
        raise RuntimeError(f'{url}: the server ignored the range request')
    return response, int(response.headers['Content-Range'].rsplit('/', 1)[1])


def read_range(url, start, end):
    stream, _ = open_range(url, start, end)
    with stream:
        return stream.read()


class Remote:
    """A remote file read by range, enough for zipfile to read the central directory."""

    def __init__(self, url):
        self.url, self.position = url, 0
        stream, self.size = open_range(url, 0, 0)
        stream.close()

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (offset, self.position + offset, self.size + offset)[whence]
        return self.position

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.position + n, self.size)
        if end <= self.position:
            return b''
        data = read_range(self.url, self.position, end - 1)
        self.position += len(data)
        return data


def require_space(path, size):
    if shutil.disk_usage(path).free < size + RESERVE:
        raise RuntimeError(f'not enough disk at {path}: {(size + RESERVE) / 1e9:.1f} GB needed')


def complete(sequence):
    mav = Path(sequence) / 'mav0'
    return (mav / 'imu0/data.csv').is_file() and len(list((mav / 'cam0/data').glob('*.png'))) >= 2


def transfer(url, start, size, part):
    """Resume the archive's bytes start..start+size into part."""
    have = part.stat().st_size if part.exists() else 0
    if have < size:
        stream, _ = open_range(url, start + have, start + size - 1)
        with stream, part.open('ab') as out:
            while True:
                block = stream.read(CHUNK)
                if not block:
                    break
                out.write(block)
                have += len(block)
                print(f'\r  {have / 1e9:.2f} of {size / 1e9:.2f} GB', end='', flush=True)
        print()
    if part.stat().st_size != size:
        raise RuntimeError(f'{part}: {part.stat().st_size} bytes, not {size}; rerun to resume')


def inflate(part, info, out):
    """The member's bytes, as the archive stores them, into out; checked against the archive's size and CRC."""
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError(f'{info.filename}: compression {info.compress_type} is not supported')
    inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    crc, size = 0, 0
    try:
        with part.open('rb') as source, out.open('wb') as sink:
            for block in iter(lambda: source.read(CHUNK), b''):
                data = inflater.decompress(block) if inflater else block
                crc, size = zlib.crc32(data, crc), size + len(data)
                sink.write(data)
            if inflater:
                tail = inflater.flush()
                crc, size = zlib.crc32(tail, crc), size + len(tail)
                sink.write(tail)
    except zlib.error as exc:
        raise ValueError(f'{info.filename}: the transfer is corrupt ({exc})') from exc
    if (crc, size) != (info.CRC, info.file_size):
        raise ValueError(f'{info.filename}: the transfer is corrupt (CRC or size differs from the archive\'s)')


def extract(asl, destination, provenance):
    """The mav0/ of an ASL zip into destination, through a staging directory, so a sequence is there whole or not."""
    stage = destination.with_name(f'.{destination.name}.partial')
    shutil.rmtree(stage, ignore_errors=True)
    with zipfile.ZipFile(asl) as archive:
        members = [m for m in archive.infolist() if m.filename.startswith('mav0/')]
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError(f'{asl}: unsafe member {member.filename}')
        archive.extractall(stage, members)        # zipfile checks each file's CRC as it reads it
    if not complete(stage):
        shutil.rmtree(stage)
        raise ValueError(f'{asl}: no mav0/cam0/data frames or mav0/imu0/data.csv in it')
    record = dict(provenance, files=sum(not m.is_dir() for m in members), bytes=sum(m.file_size for m in members),
                  fetched=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (stage / 'download_manifest.json').write_text(json.dumps(record, indent=1) + '\n')
    stage.rename(destination)


def fetch(root, seq):
    destination = root / seq
    if complete(destination):
        print(f'already there: {destination}')
        return
    for room in FORMER_ROOMS:
        former = root / room / seq
        if complete(former) and not destination.exists():
            former.rename(destination)
            print(f'moved {former} to {destination}')
            return
    if destination.exists():
        raise RuntimeError(f'{destination} exists but holds no complete sequence; remove it to refetch')
    category, url = ARCHIVES[seq[:2]]
    member = f'{category}/{seq}/{seq}.zip'
    info = zipfile.ZipFile(Remote(url)).getinfo(member)
    header = read_range(url, info.header_offset, info.header_offset + 29)
    if header[:4] != b'PK\x03\x04':
        raise ValueError(f'{url}: no local header for {member}')
    name_length, extra_length = struct.unpack('<HH', header[26:30])
    start = info.header_offset + 30 + name_length + extra_length
    cache = root / '.download_cache'
    cache.mkdir(parents=True, exist_ok=True)
    asl, part = cache / f'{seq}.zip', cache / f'{seq}.zip.deflated.part'
    if not asl.exists():
        require_space(cache, info.compress_size - (part.stat().st_size if part.exists() else 0) + info.file_size)
        print(f'fetching {seq}: {info.compress_size / 1e9:.2f} GB of the {category} archive')
        transfer(url, start, info.compress_size, part)
        inflated = cache / f'{seq}.zip.part'
        try:
            inflate(part, info, inflated)
        except ValueError:
            part.unlink()                         # a corrupt transfer is not resumed
            inflated.unlink(missing_ok=True)
            raise
        inflated.rename(asl)
        part.unlink()
    require_space(root, info.file_size)
    print(f'extracting {seq} into {destination}')
    extract(asl, destination, dict(sequence=seq, source=url, member=member, member_bytes=info.file_size,
                                   member_crc32=f'{info.CRC:08x}'))
    asl.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sequences', nargs='+', default=['V1_01_easy'], help=' '.join(SEQUENCES))
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    a = p.parse_args(argv)
    sequences = list(dict.fromkeys(a.sequences))
    unknown = [s for s in sequences if s not in SEQUENCES]
    if unknown:
        p.error(f'unknown sequence {unknown}; known: {" ".join(SEQUENCES)}')
    root = a.data_root.resolve() / 'euroc'
    root.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        fetch(root, seq)
    print(f'done. Run with:  ./accord run {sequences[0]} --viz')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, zipfile.BadZipFile, urllib.error.URLError) as exc:
        print(f'fetch_euroc stopped: {exc}', file=sys.stderr)
        raise SystemExit(2)
