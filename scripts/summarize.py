#!/usr/bin/env python3
"""A table of scored runs (docs/SCORING.md): per group of runs (their parent directory, e.g. an ablation arm) and dataset,
the map's F-score at 5 cm as built, corrected and live, and the trajectory's ATE (cm), each the mean over the sequences
of the mean over a sequence's runs. Failures follow the paper's rule: a run that did not complete, or whose trajectory is
more than FAILED_M off, has failed; its map counts 0 and its trajectory is left out, and each cell says how many.

    python3 scripts/summarize.py runs/ablation [RUN or DIRECTORY ...] [--data data]
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

COLUMNS = (('built', 'surface_fused.json'), ('corrected', 'surface.json'), ('live', 'surface_online.json'))
STREAMS = ('odometry', 'final')
FAILED_M = 10.                      # a trajectory more than this far off has failed (the placement's or the ATE's RMSE)
SCANS = ('accord/pointcloud0/data.ply', 'mav0/pointcloud0/data.ply', 'reference/map.ply')   # a sequence's scan, under data/DATASET/SEQ


def read(path, *keys):
    try:
        value = json.loads(Path(path).read_text())
        for key in keys:
            value = value[key]
        return float(value)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def runs(paths):
    for path in map(Path, paths):
        yield from ([path] if (path / 'run.json').is_file() else sorted(p.parent for p in path.rglob('run.json')))


def record(run, meta):
    """What the table needs of one run: its map scores, its trajectory errors (m) and whether it failed."""
    rec = {name: read(run / report, 'thresholds', '0.05', 'f1') for name, report in COLUMNS}
    rec.update({f'ATE {s}': read(run / 'evaluation.json', 'ate', s, 'position_m') for s in STREAMS})
    placement = read(run / 'surface.json', 'alignment', 'trajectory_rmse_m')
    rec['failed'] = meta.get('state', 'completed') != 'completed' or (placement or 0.) > FAILED_M
    return rec


def cell(sequences, name, scanned):
    """The mean over sequences of the mean over a sequence's runs, and how many runs failed. A column no run has (the
    live map of runs without the online correction) is empty; a sequence counts in a map column when it has a scan or a
    run has the score (so a map never made counts 0), in a trajectory column when a run has the score (a reference)."""
    means, failed = [], 0
    if not any(r[name] is not None for recs in sequences.values() for r in recs):
        return means, failed
    for seq, recs in sequences.items():
        if not (any(r[name] is not None for r in recs) or (not name.startswith('ATE') and seq in scanned)):
            continue
        if name.startswith('ATE'):           # a failed trajectory is left out
            kept = [r[name] for r in recs if not r['failed'] and r[name] is not None and r[name] <= FAILED_M]
            failed += len(recs) - len(kept)
            if kept:
                means.append(100. * np.mean(kept))
        else:                                # a failed run's map counts 0
            values = []
            for r in recs:
                bad = r['failed'] or r['corrected'] is None
                failed += bad
                if bad or r[name] is not None:
                    values.append(0. if bad else r[name])
            if values:
                means.append(np.mean(values))
    return means, failed


def main(argv):
    data = Path(__file__).resolve().parents[1] / 'data'
    if '--data' in argv:
        i = argv.index('--data')
        data, argv = Path(argv[i + 1]), argv[:i] + argv[i + 2:]
    table = defaultdict(lambda: defaultdict(list))     # (group, dataset) -> sequence -> the records of its runs
    for run in runs(argv or ['runs']):
        meta = json.loads((run / 'run.json').read_text())
        table[(str(run.parent), meta['dataset'])][meta['sequence']].append(record(run, meta))
    names = [name for name, _ in COLUMNS] + [f'ATE {s}' for s in STREAMS]
    print('| runs | dataset | sequences | ' + ' | '.join(names) + ' |')
    print('|' + '---|' * (3 + len(names)))
    for (group, dataset), sequences in sorted(table.items()):
        cells = []
        scanned = {s for s in sequences if any((data / dataset / s / path).is_file() for path in SCANS)}
        for name in names:
            means, failed = cell(sequences, name, scanned)
            text = '-' if not means else f'{np.mean(means):.1f}' if name.startswith('ATE') else f'{np.mean(means):.3f}'
            cells.append(text + (f' ({failed} failed)' if failed else ''))
        print(f'| {group} | {dataset} | {len(sequences)} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(sys.argv[1:])
