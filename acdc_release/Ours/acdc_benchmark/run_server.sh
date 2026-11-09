#!/usr/bin/env bash
set -euo pipefail

resume=${2:-}
if [[ $# -lt 1 || $# -gt 2 || ! "${1:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ( $# -eq 2 && "$resume" != --resume && "$resume" != --validation-only ) ]]; then
    echo "Usage: bash acdc_benchmark/run_server.sh RUN_ID [--resume | --validation-only]" >&2
    exit 2
fi

ours=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
acdc="$(dirname "$ours")/ACDC"
disk=/autodl-fs/data
storage="$disk/acdc_benchmark"
data="$storage/database"
out="$storage/runs/$1"
split="$acdc/baseline/jsons/acdc_domain_shift.json"
python=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu=${GPU:-0}

mount_target=$(findmnt -T "$disk" -n -o TARGET)
if [[ -z "$mount_target" || "$mount_target" == / || ! -d "$disk" || ! -w "$disk" ]]; then
    echo "Mount the writable data disk at $disk before starting" >&2
    exit 1
fi
disk=$(realpath "$disk")
for path in "$storage" "$storage/runs" "$data"; do
    case "$(realpath -m "$path")" in
        "$disk"/*) ;;
        *) echo "Path resolves outside the data disk: $path" >&2; exit 1 ;;
    esac
done
df -h "$disk"
mkdir -p "$storage/runs"
disk_device=$(stat -c %d "$disk")
test "$(stat -c %d "$storage/runs")" = "$disk_device"
exec 9>"$storage/.gpu-$gpu.lock"
if [[ "$resume" == --validation-only ]]; then
    echo "Waiting for GPU $gpu to run the uniform 1.5T validation"
    flock 9
else
    flock -n 9
fi
if [[ "$resume" ]]; then
    test -d "$out/checkpoints" && test -f "$out/run.log" || exit 1
else
    mkdir "$out"
    mkdir "$out/checkpoints"
fi
echo "Logging to $out/run.log"
exec >> "$out/run.log" 2>&1
trap 'echo "Run failed at line $LINENO. See $out/run.log" >&2' ERR
if [[ "$resume" == --resume ]]; then
    printf '\nResume %s\n' "$(date -Is)"
elif [[ "$resume" == --validation-only ]]; then
    printf '\nUniform source validation %s\n' "$(date -Is)"
fi

cd "$ours"
common=(--acdc-root "$acdc" --data-root "$data" --split-json "$split")
"$python" -u -m acdc_benchmark smoke --output-dir "$out/smoke"

# Stage the existing server dataset once. The lock prevents concurrent transfers.
(
    flock 8
    if [[ -d "$data" ]]; then
        test "$(stat -c %d "$data")" = "$disk_device"
        "$python" -u -m acdc_benchmark.server_checks "${common[@]}"
    else
        source_data=${ACDC_DATA_SOURCE:-/root/autodl-tmp/database}
        test -d "$source_data" || { echo "ACDC data source missing: $source_data" >&2; exit 1; }
        if [[ -e "$data" || -L "$data" || -L "$storage/.database.incoming" ]]; then
            echo "Cannot safely stage the dataset at $data" >&2
            exit 1
        fi
        mkdir -p "$storage/.database.incoming"
        rsync -a --no-owner --no-group "$source_data/" "$storage/.database.incoming/"
        "$python" -u -m acdc_benchmark.server_checks --acdc-root "$acdc" \
            --data-root "$storage/.database.incoming" --split-json "$split"
        mv -T "$storage/.database.incoming" "$data"
    fi
) 8>"$storage/.data.lock"

# Legacy trainers set CUDA_VISIBLE_DEVICES themselves from --gpu. The benchmark
# and Ours instead see only the selected GPU and address it as cuda:0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ACDC_DATA_ROOT="$data"
gpu_memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( gpu_memory >= 500 )); then
    echo "GPU $gpu is busy ($gpu_memory MiB in use)" >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES="$gpu" "$python" -c 'import monai, nibabel, scipy, SimpleITK, torch, yaml; torch.manual_seed(42); x = torch.randn(8, 8, device="cuda"); assert torch.isfinite(x @ x).all().item(); print(torch.cuda.get_device_name())'

choose_checkpoint() {
    local method=$1 explicit=$2 historical=$3 selected= train_dir="$out/train/$1"
    local destination="$out/checkpoints/$method.pt" resume_args=()
    if [[ "$resume" && -s "$destination" ]]; then
        echo "Reusing frozen $method checkpoint"
        return
    fi
    if [[ -n "$explicit" ]]; then
        "$python" -u -m acdc_benchmark.server_checks "${common[@]}" --method "$method" --checkpoint "$explicit"
        selected=$explicit
    elif [[ -s "$historical" ]] && "$python" -u -m acdc_benchmark.server_checks \
            "${common[@]}" --method "$method" --checkpoint "$historical"; then
        selected=$historical
    fi
    if [[ -z "$selected" ]]; then
        if [[ "$resume" && -d "$train_dir" ]]; then
            if [[ "$method" != devis || ! -s "$train_dir/model_latest.pt" ]]; then
                echo "Review unfinished $method directory before resuming: $train_dir" >&2
                return 1
            fi
            resume_args=(--resume "$train_dir/model_latest.pt")
        fi
        echo "Training $method on the source domain"
        if [[ "$method" == ours ]]; then
            (
                cd "$acdc"
                CUDA_VISIBLE_DEVICES="$gpu" "$python" -u -m ours.train --stage all \
                    --config "$acdc/ours/configs/acdc.yaml" \
                    --set "paths.data_root=$data" --set "paths.split_json=$split" \
                    --set "paths.baseline_checkpoint=$out/checkpoints/baseline.pt" \
                    --set "paths.run_root=$train_dir"
            )
            selected="$train_dir/dual_swin_zernike_acdc/final.pt"
        else
            local entry="$acdc/sota/$method/main.py"
            [[ "$method" != baseline ]] || entry="$acdc/baseline/main.py"
            "$python" -u "$entry" --data-dir "$data" --json-list "$split" --logdir "$train_dir" --gpu "$gpu" \
                ${resume_args[@]+"${resume_args[@]}"}
            selected="$train_dir/model_best.pt"
        fi
        "$python" -u -m acdc_benchmark.server_checks "${common[@]}" --method "$method" --checkpoint "$selected"
    fi
    cp --reflink=auto -- "$selected" "$destination.incoming"
    mv -T "$destination.incoming" "$destination"
    printf '%s\t%s\n' "$method" "$selected" >> "$out/checkpoint_sources.tsv"
}

evaluate() {
    local method=$1 split_name=$2
    local reference=()
    if [[ -s "$out/$split_name/$method/protocol.json" ]]; then
        echo "Reusing completed $split_name evaluation for $method"
        return
    fi
    if [[ "$method" != baseline ]]; then
        reference=(--reference "$out/$split_name/baseline")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -u -m acdc_benchmark evaluate "${common[@]}" \
        --method "$method" --split "$split_name" --checkpoint "$out/checkpoints/$method.pt" \
        --output-dir "$out/$split_name/$method" ${reference[@]+"${reference[@]}"}
}

evaluate_split() {
    local split_name=$1 method
    for method in baseline nacl ml1_ace mbls edl devis fgrm ours; do
        test -s "$out/checkpoints/$method.pt"
    done
    for method in baseline nacl ml1_ace mbls edl devis fgrm ours; do
        evaluate "$method" "$split_name"
    done
    "$python" -m acdc_benchmark table --results-dir "$out/$split_name" --output "$out/${split_name}_table.csv"
    echo "Completed: $out/${split_name}_table.csv"
}

if [[ "$resume" == --validation-only ]]; then
    evaluate_split val
    exit 0
fi

choose_checkpoint baseline "${BASELINE_CKPT:-}" /home/baseline/runs/acdc_monai_swinunetr_200ep/model_best.pt
evaluate baseline val
for method in nacl mbls edl; do
    if [[ "$resume" && -s "$out/train/$method/training_protocol.json" && -s "$out/train/$method/model_best.pt" ]]; then
        if [[ ! -e "$out/checkpoints/$method.pt" ]]; then
            ln "$out/train/$method/model_best.pt" "$out/checkpoints/$method.pt"
        fi
        echo "Reusing frozen $method checkpoint"
        continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -u -m acdc_benchmark train "${common[@]}" \
        --method "$method" --reference "$out/val/baseline" --output-dir "$out/train/$method"
    ln "$out/train/$method/model_best.pt" "$out/checkpoints/$method.pt"
done
choose_checkpoint ml1_ace "${ML1_ACE_CKPT:-}" /home/sota_experiments/ml1_ace/runs/acdc_200ep/model_best.pt
choose_checkpoint devis "${DEVIS_CKPT:-}" /home/sota_experiments/devis/runs/acdc_200ep/model_best.pt
choose_checkpoint ours "${OURS_CKPT:-}" /home/ours/runs/dual_swin_zernike_acdc/final.pt
choose_checkpoint fgrm "${FGRM_CKPT:-}" /home/sota_experiments/fgrm/runs/acdc_200ep/model_best.pt

evaluate_split test
evaluate_split val
