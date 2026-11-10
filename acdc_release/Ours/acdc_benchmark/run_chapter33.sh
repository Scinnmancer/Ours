#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: bash acdc_benchmark/run_chapter33.sh ACDC_ROOT DATA_ROOT ORIGINAL_RUN PERTURBATION_RUN OUTPUT_DIR" >&2
    exit 2
fi
acdc=$(realpath "$1")
data=$(realpath "$2")
original=$(realpath "$3")
trained=$(realpath "$4")
out=$(realpath "$5")
python=${PYTHON_BIN:-python}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}" CUDA_DEVICE_ORDER=PCI_BUS_ID
export ACDC_DATA_ROOT="$data"
exec >> "$out/chapter33.log" 2>&1
exec 9>"$(dirname "$data")/.gpu-${GPU:-0}.lock"
flock 9
common=(--acdc-root "$acdc" --data-root "$data" --extended-metrics --ueo --error-auroc)
for variant in dropout feature_noise cct_feature_dropout feature_cutout; do
    checkpoint="$trained/train/$variant/dual_swin_zernike_acdc/final.pt"
    if [[ "$variant" == dropout ]]; then checkpoint="$original/checkpoints/ours.pt"; fi
    if [[ "$variant" == feature_cutout ]]; then
        checkpoint="$out/train/$variant/dual_swin_zernike_acdc/final.pt"
    fi
    test -s "$checkpoint" || { echo "Missing frozen checkpoint: $checkpoint" >&2; exit 1; }
done
for split in val test; do
    for variant in dropout feature_noise cct_feature_dropout feature_cutout; do
        case "$variant" in
            dropout) checkpoint="$original/checkpoints/ours.pt" ;;
            feature_cutout) checkpoint="$out/train/$variant/dual_swin_zernike_acdc/final.pt" ;;
            *) checkpoint="$trained/train/$variant/dual_swin_zernike_acdc/final.pt" ;;
        esac
        for mode in frozen tta; do
            flags=()
            if [[ "$mode" == tta ]]; then flags+=(--tta); fi
            if [[ "$split" == test ]]; then
                flags+=(--ueo-validation "$out/val/$variant/$mode/protocol.json")
            fi
            echo "Evaluating $split $variant $mode"
            "$python" -u -m acdc_benchmark evaluate "${common[@]}" --method ours \
                --perturbation "$variant" --checkpoint "$checkpoint" --split "$split" \
                --reference "$original/error_auroc/$split/baseline" \
                --output-dir "$out/$split/$variant/$mode" "${flags[@]}"
        done
    done
    "$python" -m acdc_benchmark.perturbation_report "$out" "$split"
done
