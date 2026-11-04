#!/usr/bin/env python3
"""Fetch Hilti-Oxford (Hilti SLAM Challenge 2022) from its Hugging Face release.

Everything is pinned in config/hilti/downloads.json: the revision, and the size and sha256 of every
file. A transfer resumes where it stopped, lands under a .part name, and is renamed only after its
sha256 matches, so a file with its final name is always complete.

    python3 scripts/fetch_hilti.py --plan                      # what would be fetched, and sizes
    python3 scripts/fetch_hilti.py --sequences exp14 exp18     # bags, ground truth, calibration, scans
    python3 scripts/fetch_hilti.py --sequences P0 --no-scans   # a priority group, without the .e57

Layout written under <data-root>/hilti:
    <seq>/<name>.bag                 the ROS 1 bag (five cameras, IMU, lidar)
    <seq>/gt/*.txt                   control points, and dense poses where the release has them
    <seq>/download_manifest.json     what was fetched, from which revision
    calibration/*.yaml               the release's Kalibr chains
    scans/<site>_full.e57            the terrestrial laser scan of the site
Stdlib only; uses aria2c for the transfer when it is installed (parallel ranges), urllib otherwise.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/hilti/downloads.json'
RESERVE = 2 * 1024**3
CHUNK = 8 * 1024**2


def load_plan(path=PLAN):
    plan = json.loads(Path(path).read_text())
    for key in ('source', 'revision', 'sequences', 'scans', 'calibration', 'files', 'priority'):
        if key not in plan:
            raise ValueError(f'{path}: missing {key!r}')
    return plan


def expand(plan, tokens):
    """Sequence keys from keys, priority groups (P0, P1, P2) or 'all', in the order given."""
    out = []
    for token in tokens:
        if token == 'all':
            group = sorted(plan['sequences'])
        elif token in plan['priority']:
            group = plan['priority'][token]
        elif token in plan['sequences']:
            group = [token]
        else:
            raise ValueError(f'unknown sequence or group {token!r}; known: '
                             f'{sorted(plan["sequences"])} and {sorted(plan["priority"])}')
        out += [s for s in group if s not in out]
    return out


def targets(plan, sequences, scans=True, calibration=True):
    """[(release path, destination relative to <data-root>/hilti)] for the request."""
    out = []
    for key in sequences:
        entry = plan['sequences'][key]
        out.append((entry['bag'], f'{key}/{Path(entry["bag"]).name}'))
        out += [(p, f'{key}/gt/{Path(p).name}') for p in entry['ground_truth']]
        if scans:
            scan = plan['scans'][entry['site']]
            if (scan, f'scans/{Path(scan).name}') not in out:
                out.append((scan, f'scans/{Path(scan).name}'))
    if calibration:
        out += [(p, f'calibration/{Path(p).name}') for p in plan['calibration']]
    return out


def url(plan, path):
    return f'{plan["source"]}/resolve/{plan["revision"]}/{path}'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    """Size always; sha256 where the release records one (LFS files). Small text files have none."""
    path = Path(path)
    if path.stat().st_size != expected['size']:
        return f'size {path.stat().st_size} != {expected["size"]}'
    if expected.get('sha256') and sha256(path) != expected['sha256']:
        return 'sha256 mismatch'
    return None


def download(source, part, size, use_aria2c, max_rate=None):
    """Resume `source` into `part` until it holds `size` bytes."""
    if use_aria2c:
        limit = [f'--max-overall-download-limit={max_rate}'] if max_rate else []
        subprocess.run(['aria2c', *limit, '--continue=true', '--max-connection-per-server=8', '--split=8',
                        '--min-split-size=64M', '--file-allocation=none', '--summary-interval=60',
                        '--console-log-level=warn', '--dir', str(part.parent), '--out', part.name,
                        source], check=True)
        return
    have = part.stat().st_size if part.exists() else 0
    if have >= size:
        return
    request = urllib.request.Request(source, headers={'Range': f'bytes={have}-'} if have else {})
    with urllib.request.urlopen(request, timeout=120) as response, part.open('ab' if have else 'wb') as out:
        if have and response.status != 206:
            raise RuntimeError(f'{source}: the server ignored the range request; delete {part}')
        shutil.copyfileobj(response, out, CHUNK)


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
    p.add_argument('--sequences', nargs='+', default=['P0'],
                   help='sequence keys (exp14 ...), priority groups (P0 P1 P2) or all')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--no-scans', action='store_true', help='skip the site laser scans (.e57)')
    p.add_argument('--plan', action='store_true', help='list the files and sizes, fetch nothing')
    p.add_argument('--no-aria2c', action='store_true', help='use urllib even when aria2c is installed')
    p.add_argument('--max-rate', help='cap the transfer, e.g. 20M (bytes per second; aria2c only)')
    a = p.parse_args(argv)
    plan = load_plan()
    sequences = expand(plan, a.sequences)
    wanted = targets(plan, sequences, scans=not a.no_scans)
    root = a.data_root.resolve() / 'hilti'
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
        manifest = dict(sequence=key, name=entry['name'], site=entry['site'], source=plan['source'],
                        revision=plan['revision'], license=plan.get('license'),
                        fetched=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        files={Path(s).name: plan['files'][s] for s in [entry['bag']] + entry['ground_truth']})
        (root / key / 'download_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'done: {", ".join(sequences)} under {root}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'fetch_hilti stopped: {exc}', file=sys.stderr)
        raise SystemExit(2)
