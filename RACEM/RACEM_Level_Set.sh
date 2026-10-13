#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# Extract and save condition maps offline; online extraction is controlled by the ControlNet trainer.
RACEM_FEATURES=(all)                 # (all) / (edge) / (depth) / (mask) / (edge depth)
RACEM_IMAGES="dataset/porcelain_DIOR/images"
RACEM_LABELS="dataset/porcelain_DIOR/labels"
RACEM_OUTPUT="dataset/porcelain_DIOR/conditions"
RACEM_REGION="clip"                 # full: entire image; bbox: inside labeled boxes; clip: inside refined masks.
RACEM_BBOX_TYPE="obb"                # obb / hbb
RACEM_DEVICE="cuda:0"                # cpu / cuda:N
RACEM_OVERWRITE=0                    # 0: skip existing outputs; 1: overwrite outputs.

# Execution code below needs no editing; relative paths are resolved from the project root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Offline processing runs one image at a time. Online extraction uses the trainer's RACEM_MODE=batch_wise and batch_size.
# HED, SAM2 and MiDaS checkpoints are loaded from ckpts/racem/ by default.
command -v "$PYTHON_BIN" >/dev/null || {
  printf '找不到 Python：%s，请先激活环境。\n' "$PYTHON_BIN" >&2
  exit 1
}
[[ "$RACEM_OVERWRITE" == 0 || "$RACEM_OVERWRITE" == 1 ]] || {
  printf 'RACEM_OVERWRITE 只能为 0 或 1。\n' >&2
  exit 1
}
CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/RACEM_Level_Set.py"
  --features "${RACEM_FEATURES[@]}"
  --images "$RACEM_IMAGES"
  --output "$RACEM_OUTPUT"
  --region "$RACEM_REGION"
  --bbox-type "$RACEM_BBOX_TYPE"
  --device "$RACEM_DEVICE"
)

# Full-image edge/depth extraction needs no labels or SAM2; mask/bbox/clip extraction requires labels.
NEEDS_LABELS=0
[[ "$RACEM_REGION" != full ]] && NEEDS_LABELS=1
for feature in "${RACEM_FEATURES[@]}"; do
  [[ "$feature" == all || "$feature" == mask ]] && NEEDS_LABELS=1
done
if [[ "$NEEDS_LABELS" == 1 ]]; then
  CMD+=(--labels "$RACEM_LABELS")
fi
if [[ "$RACEM_OVERWRITE" == 1 ]]; then
  CMD+=(--overwrite)
fi

printf 'RACEM 离线提取：features=%s，region=%s，device=%s\n' \
  "${RACEM_FEATURES[*]}" "$RACEM_REGION" "$RACEM_DEVICE"
"${CMD[@]}"
