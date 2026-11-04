#!/usr/bin/env bash
# Score runs (docs/SCORING.md): the trajectories, and the map as built, corrected and, when the run corrected it online,
# live. Exports what is missing; reports go into each run (evaluation.json, surface*.json).
#   scripts/score.sh RUN [RUN ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
in_image() { ACCORD_GPUS=none scripts/docker.sh "$@"; }
for run in "$@"; do
  run=${run%/}
  [ -f "$run/run.json" ] || { echo "$run: not a run" >&2; continue; }
  data="data/$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["dataset"])' "$run/run.json")"
  in_image python3 scripts/score_trajectory.py --run "$run" --data "$data" > /dev/null
  [ -f "$run/map/map_index.json" ] || continue
  [ -f "$run/map.ply" ] || in_image python3 scripts/export_map.py --run "$run" > /dev/null
  [ -f "$run/map_fused.ply" ] || in_image python3 scripts/export_map.py --run "$run" --no-correction --out "$run/map_fused.ply"
  maps=(map:surface map_fused:surface_fused)
  if grep -q '"log_corrections"' "$run/map/map_index.json"; then
    [ -f "$run/map_online.ply" ] || in_image python3 scripts/export_map.py --run "$run" --online
    maps+=(map_online:surface_online)
  fi
  for pair in "${maps[@]}"; do
    in_image python3 scripts/score_map.py --run "$run" --data "$data" --map "$run/${pair%%:*}.ply" \
      --out "$run/${pair#*:}.json" --force > /dev/null
  done
done
python3 scripts/summarize.py "$@"
