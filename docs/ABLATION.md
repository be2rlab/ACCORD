# Ablation

```bash
scripts/ablation.sh none anchor system                          # the cumulative ablation
scripts/ablation.sh online global videpth noloops refposes      # the other arms
scripts/ablation.sh base large giant                            # the depth model
SEQS='euroc:V1_01_easy ori:r01' RUNS=0 scripts/ablation.sh system   # a quick check
```

Each arm runs live on every sequence (default: the six EuRoC Vicon rooms and ORI r01–r05, three runs each; Hilti-Oxford
exp18, exp16, exp21, one run), at sensor rate (ORI at 0.3), with DA3-Giant. Runs go to `runs/ablation/ARM/SEQ_rN`, are
scored by `scripts/score.sh` (docs/SCORING.md), and the table is printed at the end. A run that has its scores is skipped,
so the script resumes where it stopped.

| arm | what changes | settings |
|---|---|---|
| `none` | depth at the model's own scale, no context keyframes | `mapping.context_frames=0`, `mapping.anchor_min_targets=1000000000` |
| `anchor` | + the metric anchor | `mapping.context_frames=0` |
| `system` | + context keyframes: the full system | (defaults) |
| `online` | the map corrected while it is built | `mapping.online_correction=true` |
| `global` | one scale and offset for the whole map | `mapping.anchor_mode=global` |
| `videpth` | per-image affine anchor, depth not conditioned on the filter | `mapping.anchor_mode=image_affine`, `mapping.depth_conditioning=false` |
| `noloops` | no loop closures | `mapping.loops_enabled=false` |
| `refposes` | the mapper on the reference camera poses (EuRoC, ORI) | `--poses` from `scripts/reference_poses.py` |
| `base`, `large`, `giant` | the depth model | `--weights thirdparty/da3_weights/MODEL` |

Read the table by column: *built* is the map as the graph placed it, *corrected* after the export-time correction, *live* the
online correction (arm `online`), and the ATE of the filter and of the final graph.

Live runs depend on timing (which keyframes the mapper gets to), so runs of one arm differ a little; that is why three.
