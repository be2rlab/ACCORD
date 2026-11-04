#!/usr/bin/env python3
"""Fetch Oxford Spires (Tao et al., IJRR 2025) from its Hugging Face release: the Christ Church sequences
CC02, CC03 and CC05, the calibration, and the site's laser scan.

Everything is pinned in config/spires/downloads.json: the revision, and the size and hash of every file (sha256 for
the LFS files, the git blob id for the small text files). A transfer resumes where it stopped, lands under a .part
name, and is renamed only after its hash matches, so a file with its final name is always complete.

    python3 scripts/fetch_spires.py --plan                         # what would be fetched, and sizes
    python3 scripts/fetch_spires.py --sequences cc03               # images, IMU, reference, calibration, scan
    python3 scripts/fetch_spires.py --data-root /data --sequences cc02 cc05 --no-scan

Layout written under <data-root>/spires:
    <seq>/images.zip                 the three cameras' JPEG frames (scripts/convert_spires.py reads cam0)
    <seq>/imu.csv                    the IMU: secs, nsecs, accelerometer, gyroscope
    <seq>/gt/gt-tum.txt              the reference trajectory: the base frame in the scan's frame
    <seq>/download_manifest.json     what was fetched, from which revision
    calibration/*                    the release's calibration (cam0.yaml, cam-lidar-imu.yaml, imu.yaml, README.md)
    scans/christ-church_merged-cloud-1cm.pcd   the terrestrial laser scan of the site
Stdlib only; the transfer is scripts/fetch_hilti.py's (aria2c when installed, urllib otherwise).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_hilti import CHUNK, RESERVE, download, sha256          # noqa: E402

PLAN = ROOT / 'config/spires/downloads.json'


def load_plan(path=PLAN):
    plan = json.loads(Path(path).read_text())
    for key in ('source', 'revision', 'sequences', 'scans', 'calibration', 'files'):
        if key not in plan:
            raise ValueError(f'{path}: missing {key!r}')
    return plan


def git_oid(path):
    """The git blob id of a file: what the release records for its small, non-LFS files."""
    path = Path(path)
    digest = hashlib.sha1(b'blob %d\0' % path.stat().st_size)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    path = Path(path)
    if path.stat().st_size != expected['size']:
        return f'size {path.stat().st_size} != {expected["size"]}'
    if expected.get('sha256') and sha256(path) != expected['sha256']:
        return 'sha256 mismatch'
    if expected.get('git_oid') and git_oid(path) != expected['git_oid']:
        return 'git blob id mismatch'
    return None


def scan_name(plan, site):
    return f'{site}_{Path(plan["scans"][site]).name}'


def targets(plan, sequences, scan=True, calibration=True):
    """[(release path, destination relative to <data-root>/spires)] for the request."""
    out = []
    for key in sequences:
        entry = plan['sequences'][key]
        out += [(entry['images'], f'{key}/images.zip'), (entry['imu'], f'{key}/imu.csv'),
                (entry['ground_truth'], f'{key}/gt/{Path(entry["ground_truth"]).name}')]
        if scan:
            item = (plan['scans'][entry['site']], f'scans/{scan_name(plan, entry["site"])}')
            if item not in out:
                out.append(item)
    if calibration:
        out += [(p, f'calibration/{Path(p).name}') for p in plan['calibration']]
    return out


def url(plan, path):
    return f'{plan["source"]}/resolve/{plan["revision"]}/{path}'


def fetch(plan, source_path, destination, use_aria2c=False, log=print, max_rate=None):
    expected = plan['files'][source_path]
    destination = Path(destination)
    if destination.is_file():
        problem = verify(destination, expected)
        if problem is None:
            log(f'ok       {destination}')
            return False
        raise RuntimeError(f'{destination} exists but is wrong ({problem}); remove it to refetch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + '.part')
    have = part.stat().st_size if part.exists() else 0
    if shutil.disk_usage(destination.parent).free < expected['size'] - have + RESERVE:
        raise RuntimeError(f'not enough disk for {destination} ({expected["size"] / 1e9:.1f} GB)')
    log(f'fetching {destination} ({expected["size"] / 1e9:.2f} GB)', flush=True)
    download(url(plan, source_path), part, expected['size'], use_aria2c, max_rate)
    problem = verify(part, expected)
    if problem is not None:
        raise RuntimeError(f'{part}: {problem} after the transfer; rerun to resume, or delete it')
    part.rename(destination)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sequences', nargs='+', default=['cc03'], help='cc02 cc03 cc05, or all')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--no-scan', action='store_true', help='skip the site laser scan (2.7 GB)')
    p.add_argument('--plan', action='store_true', help='list the files and sizes, fetch nothing')
    p.add_argument('--no-aria2c', action='store_true', help='use urllib even when aria2c is installed')
    p.add_argument('--max-rate', help='cap the transfer, e.g. 20M (bytes per second; aria2c only)')
    a = p.parse_args(argv)
    plan = load_plan()
    sequences = sorted(plan['sequences']) if a.sequences == ['all'] else list(dict.fromkeys(a.sequences))
    unknown = [s for s in sequences if s not in plan['sequences']]
    if unknown:
        p.error(f'unknown sequence {unknown}; known: {sorted(plan["sequences"])} or all')
    wanted = targets(plan, sequences, scan=not a.no_scan)
    root = a.data_root.resolve() / 'spires'
    total = sum(plan['files'][s]['size'] for s, _ in wanted)
    if a.plan:
        for source_path, relative in wanted:
            print(f'{plan["files"][source_path]["size"] / 1e9:8.3f} GB  {relative}')
        print(f'{total / 1e9:8.3f} GB  total, into {root} (revision {plan["revision"][:12]})')
        return 0
    use_aria2c = not a.no_aria2c and shutil.which('aria2c') is not None
    for source_path, relative in wanted:
        fetch(plan, source_path, root / relative, use_aria2c, max_rate=a.max_rate)
    for key in sequences:
        entry = plan['sequences'][key]
        paths = (entry['images'], entry['imu'], entry['ground_truth'])
        manifest = dict(sequence=key, name=entry['name'], site=entry['site'], source=plan['source'],
                        revision=plan['revision'], license=plan.get('license'),
                        fetched=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        files={s: plan['files'][s] for s in paths})
        (root / key / 'download_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'done: {", ".join(sequences)} under {root}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'fetch_spires stopped: {exc}', file=sys.stderr)
        raise SystemExit(2)
