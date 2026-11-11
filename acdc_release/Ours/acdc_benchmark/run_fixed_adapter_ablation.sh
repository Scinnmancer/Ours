#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: bash acdc_benchmark/run_fixed_adapter_ablation.sh ACDC_ROOT DATA_ROOT ORIGINAL_RUN MARGIN_RUN OUTPUT_DIR" >&2
    exit 2
fi
acdc=$(realpath "$1")
data=$(realpath "$2")
original=$(realpath "$3")
margin_run=$(realpath -m "$4")
out=$(realpath -m "$5")
python=${PYTHON_BIN:-python}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}" CUDA_DEVICE_ORDER=PCI_BUS_ID
exec 9>"$(dirname "$data")/.gpu-${GPU:-0}.lock"
echo "Waiting for the ACDC GPU lock"
flock 9

variant=feature_noise_head1_20
online="$margin_run/test/$variant/tta"
checkpoint="$margin_run/train/$variant/dual_swin_zernike_acdc/final.pt"
test -s "$margin_run/test_table.csv"
test -s "$online/protocol.json"
test -s "$checkpoint"
"$python" -u -m acdc_benchmark evaluate --acdc-root "$acdc" --data-root "$data" \
    --method ours --split test --checkpoint "$checkpoint" --directional-zernike \
    --perturbation "$variant" --reference "$original/error_auroc/test/baseline" \
    --output-dir "$out" --extended-metrics --error-auroc --fixed-adapter-mc --save-head-segmentations
"$python" -m acdc_benchmark.perturbation_report "$out" test --online-tta-dir "$online"
