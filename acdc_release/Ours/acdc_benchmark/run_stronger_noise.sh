#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: bash acdc_benchmark/run_stronger_noise.sh ACDC_ROOT DATA_ROOT ORIGINAL_RUN CHAPTER33_RUN OUTPUT_DIR" >&2
    exit 2
fi
acdc=$(realpath "$1")
data=$(realpath "$2")
original=$(realpath "$3")
previous=$(realpath "$4")
out=$(realpath -m "$5")
python=${PYTHON_BIN:-python}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}" CUDA_DEVICE_ORDER=PCI_BUS_ID ACDC_DATA_ROOT="$data"
exec 9>"$(dirname "$data")/.gpu-${GPU:-0}.lock"
flock -n 9
variant=feature_noise_strong
for split in val test; do
    test -s "$original/error_auroc/$split/baseline/protocol.json"
    test -s "$previous/${split}_table.csv"
done
mkdir "$out"
exec >> "$out/run.log" 2>&1
(cd "$acdc"; "$python" -u -m ours.train --stage all --config "$acdc/ours/configs/acdc.yaml" \
    --set "model.head_perturbation=$variant" --set "paths.data_root=$data" \
    --set "paths.baseline_checkpoint=$original/checkpoints/baseline.pt" \
    --set "paths.run_root=$out/train")
checkpoint="$out/train/dual_swin_zernike_acdc/final.pt"
for split in val test; do
    for mode in frozen tta; do
        flags=()
        if [[ "$mode" == tta ]]; then flags+=(--tta); fi
        if [[ "$split" == test ]]; then flags+=(--ueo-validation "$out/val/$mode/protocol.json"); fi
        "$python" -u -m acdc_benchmark evaluate --acdc-root "$acdc" --data-root "$data" \
            --method ours --perturbation "$variant" --split "$split" --checkpoint "$checkpoint" \
            --reference "$original/error_auroc/$split/baseline" --output-dir "$out/$split/$mode" \
            --extended-metrics --error-auroc --ueo "${flags[@]}"
    done
    "$python" - "$out" "$previous" "$split" <<'PY'
import csv
import json
from pathlib import Path
import sys
from acdc_benchmark.evaluate import write_csv
from acdc_benchmark.perturbation_report import paired_auc

out, previous = map(Path, sys.argv[1:3])
split = sys.argv[3]

def load(path, name, mode):
    saved = json.loads((path / "protocol.json").read_text())
    comparison = saved["comparison"].copy()
    assert comparison.pop("head_perturbation") == name
    assert comparison["tta"] == (mode == "tta")
    with (path / "patient_metrics.csv").open() as file:
        patients = {row["patient_id"]: row for row in csv.DictReader(file)}
    row = dict(variant=name, mode=mode, **saved["domain"], ueo_threshold=saved["ueo_threshold"])
    identity = comparison, saved["reference_checkpoint_sha256"], saved["frames"]
    return row, patients, identity

rows, effects = [], []
for mode in ("frozen", "tta"):
    strong, patients, identity = load(out / split / mode, "feature_noise_strong", mode)
    for variant in ("dropout", "feature_noise"):
        row, control, control_identity = load(previous / split / variant / mode, variant, mode)
        if control_identity != identity:
            raise ValueError("Incompatible evaluation reference or TTA protocol")
        rows.append(row)
        effects.append(dict(mode=mode, reference=variant, **paired_auc(patients, control)))
    rows.append(strong)
write_csv(out / f"{split}_table.csv", rows)
write_csv(out / f"{split}_paired_auc.csv", effects)
PY
done
