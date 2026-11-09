#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || "$1" == -* ]]; then
    echo "Usage: bash acdc_benchmark/sync_server.sh SSH_TARGET" >&2
    exit 2
fi

target=$1
ours=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
acdc=${ACDC_SOURCE:-"$(dirname "$ours")/ACDC"}
remote=${ACDC_DEPLOY_ROOT:-/home/acdc-benchmark}
[[ "$remote" =~ ^/home/[A-Za-z0-9_-]+$ ]]
for path in baseline ours sota/ml1_ace sota/devis sota/fgrm; do
    test -d "$acdc/$path"
done

ssh "$target" "mkdir -p $remote/Ours $remote/ACDC"
filters=(--exclude=.git --exclude=.DS_Store --exclude=__pycache__ --exclude=.pytest_cache
         --exclude=implement-stage --exclude=runs --exclude=outputs --exclude='*.log'
         --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.nii.gz')
(
    cd "$ours"
    rsync -azR "${filters[@]}" acdc_benchmark nacl \
        directional_zernike_project/ours/tools/fixed_ece.py "$target:$remote/Ours/"
)
(
    cd "$acdc"
    rsync -azR "${filters[@]}" baseline ours sota/ml1_ace sota/devis sota/fgrm \
        "$target:$remote/ACDC/"
)
