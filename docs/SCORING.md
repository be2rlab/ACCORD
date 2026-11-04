# Scoring

```bash
scripts/score.sh runs/V1_01_easy runs/r01            # score runs, print the table
python3 scripts/summarize.py runs/ablation           # the table of every run under a directory
```

`scripts/score.sh` writes into each run:

| file | what |
|---|---|
| `evaluation.json` | the trajectories (`scripts/score_trajectory.py`) |
| `surface_fused.json` | the map as built: the submaps where the graph put them (`map_fused.ply`) |
| `surface.json` | the map after the export-time correction (`map.ply`) |
| `surface_online.json` | the map as corrected while it was built, for runs with `mapping.online_correction=true` (`map_online.ply`) |

## Trajectory

The absolute trajectory error (ATE, RMSE of positions) after a rigid fit to the reference: rotation and translation, no
scale, so a scale error counts. Three streams: `odometry` (the filter, `trajectory.tum`), `live` (the graph while running,
`map_trajectory.tum`) and `final` (the graph at the end, `map_trajectory_final.tum`). The report also gives the scale error
(the log scale of a similarity fit; positive when the trajectory is too long). On Hilti-Oxford, the dataset's own score at
its surveyed control points.

References: EuRoC, the corrected file OpenVINS distributes; TUM-VI, the motion capture; ORI, Hilti-Oxford and Oxford
Spires, the poses registered to the laser scan; the phone rig, RTAB-Map's loop-closed graph from a LiDAR phone carried
with the others (about 1 Hz, interpolated to 100 Hz, never across a gap over 3.5 s), on each phone's clock.

## Map

The map is placed in the scan's frame by the rigid fit of the run's own trajectory: nothing is fitted to the surface. Then,
on the scan points the reference cameras saw (completeness) and the map points within reach of the reference cameras
(accuracy):

- accuracy: share of map points within `d` of the scan;
- completeness: share of visible scan points within `d` of the map;
- F-score: their harmonic mean, at `d` = 2, 3, 5 and 10 cm. The paper reports F@5 cm.

The phone rig's reference map is RTAB-Map's, which reaches only about 3.5 m from its track: map points more than 3 m from
the reference track are left out of accuracy, and their share is reported (`beyond_reach`); completeness is unchanged. A map
with no point within that reach scores 0.

`scripts/score_map.py --fit rigid|similarity` also registers the map to the scan first, as a second protocol.

## Table

`scripts/summarize.py` groups runs by their parent directory and dataset. Each cell is the mean over sequences of the mean
over a sequence's runs: F@5 as built, corrected and live, and ATE odometry and final in cm.

Failures follow the paper's rule. A run that did not complete, or whose trajectory is more than 10 m off (the rigid fit that
places its map, or its ATE), has failed: its map counts 0 and its trajectory is left out of the mean, and the cell says how
many runs failed. A map never made counts 0 on a sequence with a scan (`data/DATASET/SEQ/accord/pointcloud0`,
`mav0/pointcloud0` or, on the phone rig, `reference/map.ply`; `--data` points elsewhere).
