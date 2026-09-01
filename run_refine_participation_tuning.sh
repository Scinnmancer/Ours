#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/ours/configs/brats2020.yaml}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/ours/runs/dual_swin_zernike_brats2020/final.pt}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/ours/runs/dual_swin_zernike_brats2020/refine_participation_tuning}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT}"

echo "[one-click] target atomic top-1 change rate: 2%"
echo "[one-click] selecting on source_val only; external centers are final evaluation only"
echo "[one-click] output=${OUTPUT}"
"${PYTHON_BIN}" -m ours.tune_refine_participation \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  "$@" 2>&1 | tee "${OUTPUT}/participation_tuning.log"
