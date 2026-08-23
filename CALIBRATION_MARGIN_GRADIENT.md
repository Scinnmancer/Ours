# Geometric-uncertainty weighted margin calibration

## Objective

Calibration keeps the existing geometric uncertainty path unchanged:

```text
zernike_disagreement = Zernike(head1_atomic, head2_atomic, fitted_statistics)
u = sigmoid(bias + softplus(raw_xi) * zernike_disagreement)
```

The calibration stage detaches `u` and uses it only as a voxel weight. It does
not refit the uncertainty fusion. Only `head1` and `head2` are trainable; the
encoder, Zernike statistics, fusion, and label-transfer module remain frozen.
Because the objective is explicitly `stopgrad(u^beta)`, calibration computes
the unchanged Zernike disagreement and fusion formula under `torch.no_grad()`
from detached atomic probabilities. This avoids retaining the multi-scale 3D
convolution graph; it does not change `u`, the loss, or the head gradients.

For the base four-class atomic probability `p`, define atomic logits
`a = log(p + eps)` and detached top class `k* = argmax(a)`. On the union `M` of
predicted and true tumor voxels, calibration minimizes

```text
L = Dice(head1) + Dice(head2) + weight * L_margin

L_margin = mean_M u^beta * sum(valid k != k*) relu(a[k*] - a[k] - margin)
```

Only excessive class gaps are reduced. Once a gap is within the positive
margin, that pair contributes no calibration gradient. Risk Brier, Risk ECE,
and geometric disagreement remain diagnostics and do not participate in
backward. Exact structural-zero classes created by the nested-region bridge
are excluded from the competitor set because they have no usable local
gradient; this prevents the penalty from reducing only the winning class.

## Configuration

```yaml
uncertainty:
  margin_gradient:
    enabled: true
    weight: 0.01
    uncertainty_power: 2.0
    margin: 1.0
```

Legacy configurations without `margin_gradient` leave the margin objective
disabled. The feature adds no model parameter and does not change `state_dict`.

The BraTS 2020 configuration uses `zernike.chunk_depth: 16` with the existing
halo-preserving chunk implementation and `training.sw_batch_size: 1`. These
settings reduce training and validation peak VRAM respectively without changing
the geometric-moment definition or calibration objective.

## Checkpoint selection

A calibration checkpoint is eligible only when

```text
mean_dice >= reference_dice - training.dice_tolerance
```

Eligible checkpoints are ranked only by ordinary four-class segmentation
`ece`; equal ECE keeps the earlier checkpoint. Dice is an eligibility guard,
not a ranking or tie-break metric. Risk ECE and Risk Brier are diagnostic only.
If no checkpoint is eligible, `last_calibration.pt` is retained for diagnosis,
the stage fails, and neither `best_calibrated.pt` nor `final.pt` is produced by
that run.

## Start directly from warmup

```bash
python -m ours.train \
  --config ours/configs/brats2020.yaml \
  --stage calibration \
  --checkpoint ours/runs/dual_swin_zernike_brats2020/best_seg.pt
```

The warmup checkpoint is loaded with `strict=True`, then a fresh head-only
optimizer, scheduler, and GradScaler are created. Warmup epochs and warmup
optimizer state are not resumed. If Zernike statistics are absent, the existing
statistics fitting pass writes `stats_fitted.pt` before calibration starts. If
checkpoint metadata has no finite `mean_dice`, the loaded warmup model is
validated once to establish the Dice eligibility reference.
