#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "Usage: bash acdc_benchmark/run_paper_aligned.sh ACDC_ROOT DATA_ROOT ORIGINAL_RUN OUTPUT_DIR [feature_noise_equal20|feature_noise_head1_20|dropout|none] [COMPLETED_NOISE_RUN]" >&2
    exit 2
fi
variant=${5:-feature_noise_equal20}
case "$variant" in
    feature_noise_equal20|feature_noise_head1_20|dropout|none) ;;
    *) echo "Unknown perturbation: $variant" >&2; exit 2 ;;
esac
acdc=$(realpath "$1")
data=$(realpath "$2")
original=$(realpath "$3")
out=$(realpath -m "$4")
noise_results=
if [[ $# == 6 ]]; then
    if [[ "$variant" != none ]]; then
        echo "COMPLETED_NOISE_RUN requires the none variant" >&2
        exit 2
    fi
    noise_results="$(realpath "$6")/test/feature_noise_equal20"
    for mode in frozen tta; do
        test -s "$noise_results/$mode/protocol.json"
        test -s "$noise_results/$mode/patient_metrics.csv"
    done
fi
python=${PYTHON_BIN:-python}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}" CUDA_DEVICE_ORDER=PCI_BUS_ID ACDC_DATA_ROOT="$data"
exec 9>"$(dirname "$data")/.gpu-${GPU:-0}.lock"
if [[ "${ACDC_WAIT_FOR_GPU:-0}" == 1 ]]; then
    echo "Waiting for the ACDC GPU lock"
    flock 9
else
    flock -n 9
fi

reference="$original/error_auroc/test/baseline"
split_json="$acdc/baseline/jsons/acdc_domain_shift.json"
test -s "$reference/protocol.json"
train_stage=all
train_flags=()
margin_weight=${ACDC_MARGIN_WEIGHT:-0.03}
if [[ -n "${ACDC_CALIBRATION_SOURCE_RUN:-}" ]]; then
    if [[ "${ACDC_RESUME_WARMUP:-0}" == 1 ]]; then
        echo "A calibration source and warmup resume are mutually exclusive" >&2
        exit 2
    fi
    source_run=$(realpath "$ACDC_CALIBRATION_SOURCE_RUN")
    stats="$source_run/train/$variant/dual_swin_zernike_acdc/stats_fitted.pt"
    test -s "$source_run/test_table.csv"
    test -s "$stats"
    train_stage=calibration
    train_flags+=(--checkpoint "$stats")
fi
if [[ "${ACDC_RESUME_WARMUP:-0}" == 1 ]]; then
    test -s "$out/train/$variant/dual_swin_zernike_acdc/last_warmup.pt"
    test -s "$out/train/$variant/dual_swin_zernike_acdc/best_seg.pt"
    test ! -e "$out/test"
    train_flags+=(--resume-warmup)
else
    mkdir "$out"
fi
exec >> "$out/run.log" 2>&1
"$python" -u -m acdc_benchmark.server_checks --acdc-root "$acdc" --data-root "$data" --split-json "$split_json"

echo "Training $variant on 1.5T, stage=$train_stage, margin_weight=$margin_weight"
(cd "$acdc"; "$python" -u -m ours.train --stage "$train_stage" --config "$acdc/ours/configs/acdc.yaml" "${train_flags[@]}" \
    --set "paths.data_root=$data" --set "paths.split_json=$split_json" \
    --set "paths.run_root=$out/train/$variant" --set "model.head_perturbation=$variant" \
    --set model.encoder_warm_start=false --set training.paper_aligned=true \
    --set zernike.descriptor=directional_coefficients --set reproducibility.seed=2026 \
    --set uncertainty.calibration_fusion.xi=6.0 --set uncertainty.calibration_fusion.bias=-5.5 \
    --set "uncertainty.margin_gradient.weight=$margin_weight" --set uncertainty.margin_gradient.uncertainty_power=1.5 \
    --set monitoring.uncertainty_map_every=0)
checkpoint="$out/train/$variant/dual_swin_zernike_acdc/final.pt"
test -s "$checkpoint"
for mode in tta frozen; do
    flags=()
    if [[ "$mode" == tta ]]; then flags+=(--tta); fi
    echo "Evaluating 3T $variant $mode"
    "$python" -u -m acdc_benchmark evaluate --acdc-root "$acdc" --data-root "$data" \
        --split-json "$split_json" --split test --method ours --checkpoint "$checkpoint" \
        --directional-zernike --perturbation "$variant" --reference "$reference" \
        --output-dir "$out/test/$variant/$mode" --extended-metrics --error-auroc \
        --save-head-segmentations "${flags[@]}"
done
if [[ -n "$noise_results" ]]; then
    ln -s "$noise_results" "$out/test/feature_noise_equal20"
    "$python" -m acdc_benchmark.perturbation_report "$out" test none feature_noise_equal20 --control none
else
    "$python" -m acdc_benchmark.perturbation_report "$out" test "$variant"
fi
