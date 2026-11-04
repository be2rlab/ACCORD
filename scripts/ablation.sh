#!/usr/bin/env bash
# The ablation (docs/ABLATION.md): each ARM run live on each sequence, exported and scored, then summarized.
#   scripts/ablation.sh ARM [ARM ...]
# ARM: none anchor system online global videpth noloops refposes base large giant
# SEQS (dataset:sequence ...; default the paper's), RUNS (default 0 1 2; Hilti one run), WEIGHTS (default giant; the
# base/large/giant arms set their own), OUT (default runs/ablation).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOMS="V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
SEQS=${SEQS:-"$(for s in $ROOMS; do printf 'euroc:%s ' "$s"; done)ori:r01 ori:r02 ori:r03 ori:r04 ori:r05 hilti:exp18 hilti:exp16 hilti:exp21"}
RUNS=${RUNS:-0 1 2}
WEIGHTS=${WEIGHTS:-thirdparty/da3_weights/giant}
OUT=${OUT:-runs/ablation}

settings() {  # the options of scripts/run.py that make an arm
  case "$1" in
    none)    echo "--set mapping.context_frames=0 --set mapping.anchor_min_targets=1000000000" ;;  # no anchor, no context
    anchor)  echo "--set mapping.context_frames=0" ;;                       # the anchor, no context keyframes
    system|refposes|base|large|giant) echo "" ;;                           # the system
    online)  echo "--set mapping.online_correction=true" ;;                 # the correction while the map is built
    global)  echo "--set mapping.anchor_mode=global" ;;                     # one offset for the whole map
    videpth) echo "--set mapping.anchor_mode=image_affine --set mapping.depth_conditioning=false" ;;
    noloops) echo "--set mapping.loops_enabled=false" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || { sed -n 2,6p "$0"; exit 2; }
for arm in "$@"; do
  weights=$WEIGHTS
  case "$arm" in base|large|giant) weights=thirdparty/da3_weights/$arm ;; esac
  for run in $RUNS; do
    for item in $SEQS; do
      ds=${item%%:*}; seq=${item#*:}; out=$OUT/$arm/${seq}_r$run
      [ "$ds" = hilti ] && [ "$run" != "${RUNS%% *}" ] && continue   # Hilti's scanned sequences: one run
      [ "$ds" = hilti ] && [ "$arm" = refposes ] && continue           # no reference trajectory, only control points
      [ -f "$out/surface.json" ] && continue
      rm -rf "$out"; mkdir -p "$(dirname "$out")"
      rate=1; [ "$ds" = ori ] && rate=0.3          # ORI's two-megapixel frames: the filter keeps up at 0.3 of real time
      extra=()
      if [ "$arm" = refposes ]; then                # the mapper on the reference's camera poses
        poses=$OUT/refposes/$seq.tum
        [ -s "$poses" ] || ACCORD_GPUS=none scripts/docker.sh python3 scripts/reference_poses.py --dataset "$ds" \
          --data "data/$ds" --sequence "$seq" --out "$poses"
        extra=(--poses "$poses")
      fi
      # shellcheck disable=SC2046
      ./accord run "$seq" --dataset "$ds" --out "$out" --weights "$weights" --rate "$rate" \
        $(settings "$arm") "${extra[@]}" || echo "$out: failed" >&2
      scripts/score.sh "$out" > /dev/null || true
    done
  done
done
python3 scripts/summarize.py "$OUT"
