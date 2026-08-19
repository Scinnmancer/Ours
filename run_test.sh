#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/ours/configs/brats2020.yaml}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/ours/runs/dual_swin_zernike_brats2020/final.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m ours.test --config "${CONFIG}" --checkpoint "${CHECKPOINT}" "$@"

