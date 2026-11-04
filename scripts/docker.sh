#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTY=(); [ -t 0 ] && TTY=(-it)
if [ "${ACCORD_GPUS:-all}" = none ]; then
  GPU=(--runtime runc -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=)
else
  GPU=(--gpus "${ACCORD_GPUS:-all}")
fi
MOUNTS=(-v "$ROOT:/workspace")
# data/, thirdparty/ or runs/ may link elsewhere, as may what they hold (a dataset, the weights, a build): each link's
# target is mounted at its own path, so the link resolves inside.
while IFS= read -r link; do
  target="$(readlink -f "$link")"; [ -e "$target" ] && MOUNTS+=(-v "$target:$target")
done < <(find "$ROOT/data" "$ROOT/thirdparty" "$ROOT/runs" -maxdepth 3 -type l 2>/dev/null)
# shellcheck disable=SC2206
EXTRA=(${ACCORD_DOCKER_ARGS:-})
exec docker run --rm "${TTY[@]}" "${GPU[@]}" "${EXTRA[@]}" --ipc host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=2 \
  -e PYTHONPATH=/workspace/src:/workspace/thirdparty/Depth-Anything-3/src \
  "${MOUNTS[@]}" -w /workspace "${ACCORD_IMAGE:-accord/dense-gpu:latest}" \
  bash -c 'source /opt/ros/humble/setup.bash; exec "$@"' -- "${@:-bash}"
