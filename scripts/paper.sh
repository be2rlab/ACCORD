#!/usr/bin/env bash
# The paper's main results (README, "Reproducing the paper"): the system with DA3-Giant on every sequence of the public
# datasets, three runs each, exported, scored, then summarized.
#   scripts/paper.sh [DATASET ...]      DATASET: euroc tumvi ori hilti spires rig_android rig_iphone (default: all)
# RUNS (default 0 1 2), WEIGHTS (default giant), OUT (default runs/paper). A run that has its scores is skipped, so the
# script resumes where it stopped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUNS=${RUNS:-0 1 2}
WEIGHTS=${WEIGHTS:-thirdparty/da3_weights/giant}
OUT=${OUT:-runs/paper}

sequences() {  # the sequences the paper reports, per dataset
  case "$1" in
    euroc)  echo V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult ;;
    tumvi)  echo room1 room2 room3 room4 room5 room6 ;;
    ori)    echo r01 r02 r03 r04 r05 ;;
    hilti)  echo exp01 exp04 exp05 exp07 exp11 exp14 exp15 exp16 exp18 exp21 ;;   # maps scored where a scan matches
    spires) echo cc02 cc03 cc05 ;;
    rig_android) echo vio01 vio02 vio04 vio05 vio06 vio07 vio08 vio11 vio12 vio13 ;;   # the phone rig: no Note 9 vio03
    rig_iphone)  echo vio01 vio02 vio03 vio04 vio05 vio06 vio07 vio08 vio11 vio12 vio13 ;;
    *) echo "unknown dataset $1" >&2; exit 2 ;;
  esac
}

datasets=("$@"); [ ${#datasets[@]} -gt 0 ] || datasets=(euroc tumvi ori hilti spires rig_android rig_iphone)
for ds in "${datasets[@]}"; do
  rate=1; [ "$ds" = ori ] && rate=0.3          # ORI's two-megapixel frames: the filter keeps up at 0.3 of real time
  for run in $RUNS; do
    for seq in $(sequences "$ds"); do
      out=$OUT/$ds/${seq}_r$run
      [ -f "$out/evaluation.json" ] && continue
      rm -rf "$out"; mkdir -p "$(dirname "$out")"
      ./accord run "$seq" --dataset "$ds" --out "$out" --weights "$WEIGHTS" --rate "$rate" || echo "$out: failed" >&2
    done
  done
done
python3 scripts/summarize.py "$OUT"
