#!/usr/bin/env python3
"""Fuse a run's submaps into one point cloud (RUN/map.ply). Each image's scale is corrected by its agreement with the
other windows that see the same surface (the corrections go beside it, as <name>.correction.json); --no-correction
exports the map as fused, --online the correction the run itself held at its end."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from accord.mapper import revisit, sim3                     # noqa: E402
from accord.mapper.fusion import fuse_map, write_ply        # noqa: E402
from accord.mapper.mapper import global_offset              # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, help='the .ply to write (default RUN/map.ply, RUN/map_online.ply with --online)')
    p.add_argument('--no-correction', action='store_true', help='the map as fused, without the correction per image')
    p.add_argument('--online', action='store_true', help='the correction the run held at its end (mapping.online_correction)')
    a = p.parse_args()
    out = a.out or a.run / ('map_online.ply' if a.online else 'map.ply')
    if out.exists():
        p.error(f'{out} exists')
    cfg = json.loads((a.run / 'run.json').read_text())['config']['mapping']
    index = json.loads((a.run / 'map/map_index.json').read_text())
    submaps, nodes = [], {}
    for key, entry in index['submaps'].items():
        with np.load(a.run / 'map' / entry['file'], allow_pickle=False) as z:
            sm = {k: z[k].copy() for k in z.files}
        sm.update(id=key, frame_ids=entry['frame_ids'])
        submaps.append(sm)
        nodes[key] = np.asarray(entry['T_map_submap'])
    if cfg.get('anchor_mode', 'window') == 'global':        # the global-anchor arm: one offset for the whole map
        offset = global_offset(submaps, cfg)
        for sm in submaps:
            sm['depth'] = sm['depth'] * np.float32(np.exp(-offset))
        out.with_suffix('.global_anchor.json').write_text(json.dumps(dict(
            log_offset=offset, targets=int(sum(len(sm.get('anchor_residuals', ())) for sm in submaps)))) + '\n')
    if a.online:
        revisit.attach_stored(submaps, index)
    elif not a.no_correction:
        placed = [nodes[sm['id']] for sm in submaps]
        applied, info = revisit.corrections(submaps, [sim3.pose(T) for T in placed], [sim3.scale(T) for T in placed], cfg)
        for sm, c in zip(submaps, applied):
            sm['depth'] = sm['depth'] * np.exp(-c)[:, None, None].astype(sm['depth'].dtype)
        out.with_suffix('.correction.json').write_text(json.dumps(dict(
            info, log_corrections={sm['id']: c.tolist() for sm, c in zip(submaps, applied)}), indent=1) + '\n')
        print(json.dumps(info))
    xyz, rgb = fuse_map(submaps, nodes, voxel_size=cfg['voxel_m'], pixel_step=cfg['pixel_step'])
    write_ply(out, xyz, rgb)


if __name__ == '__main__':
    main()
