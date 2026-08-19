#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/ours/configs/brats2020.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cd "${ROOT_DIR}"
echo "Training preflight will generate the configured split file when it is missing."
exec "${PYTHON_BIN}" -m ours.train --config "${CONFIG}" --stage all "$@"
