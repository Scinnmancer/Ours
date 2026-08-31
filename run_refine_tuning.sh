#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/ours/configs/brats2020.yaml}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/ours/runs/dual_swin_zernike_brats2020/final.pt}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/ours/runs/dual_swin_zernike_brats2020/refine_tuning}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

echo "[one-click] comparing legacy and local-excess-confidence refinement on source_val"
exec "${PYTHON_BIN}" -m ours.tune_refine \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  "$@"
