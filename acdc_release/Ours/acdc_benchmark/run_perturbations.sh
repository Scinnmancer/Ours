#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "Usage: bash acdc_benchmark/run_perturbations.sh ACDC_ROOT DATA_ROOT ORIGINAL_RUN OUTPUT_DIR" >&2
    exit 2
fi
acdc=$(realpath "$1")
data=$(realpath "$2")
original=$(realpath "$3")
out=$(realpath -m "$4")
python=${PYTHON_BIN:-python}
gpu=${GPU:-0}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
export ACDC_DATA_ROOT="$data"
split_json="$acdc/baseline/jsons/acdc_domain_shift.json"
reference_root="$original/error_auroc"
common=(--acdc-root "$acdc" --data-root "$data" --split-json "$split_json")
exec 9>"$(dirname "$data")/.gpu-$gpu.lock"
flock -n 9
test -s "$original/checkpoints/baseline.pt"
test -s "$original/checkpoints/ours.pt"
for split in val test; do test -s "$reference_root/$split/baseline/protocol.json"; done
mkdir "$out"
exec >> "$out/run.log" 2>&1
"$python" -u -m acdc_benchmark.server_checks "${common[@]}"
"$python" -u -m acdc_benchmark.perturbation_smoke "${common[@]}" \
    --checkpoint "$original/checkpoints/ours.pt" --reference "$reference_root/val/baseline"
for variant in feature_noise cct_feature_dropout feature_cutout; do
    (cd "$acdc"; "$python" -u -m ours.train --stage all --config "$acdc/ours/configs/acdc.yaml" \
        --set "model.head_perturbation=$variant" --set "paths.data_root=$data" \
        --set "paths.split_json=$split_json" --set "paths.baseline_checkpoint=$original/checkpoints/baseline.pt" \
        --set "paths.run_root=$out/train/$variant")
done
for split in val test; do
    for variant in dropout feature_noise cct_feature_dropout feature_cutout; do
        checkpoint="$out/train/$variant/dual_swin_zernike_acdc/final.pt"
        if [[ "$variant" == dropout ]]; then checkpoint="$original/checkpoints/ours.pt"; fi
        for mode in frozen tta; do
            flags=()
            if [[ "$mode" == tta ]]; then flags+=(--tta); fi
            if [[ "$split" == test ]]; then
                flags+=(--ueo-validation "$out/val/$variant/$mode/protocol.json")
            fi
            "$python" -u -m acdc_benchmark evaluate "${common[@]}" --method ours \
                --perturbation "$variant" --checkpoint "$checkpoint" --split "$split" \
                --reference "$reference_root/$split/baseline" --extended-metrics --ueo --error-auroc \
                --output-dir "$out/$split/$variant/$mode" "${flags[@]}"
        done
    done
    "$python" -m acdc_benchmark.perturbation_report "$out" "$split"
done
