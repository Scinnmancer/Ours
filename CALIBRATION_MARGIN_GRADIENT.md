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
The default BraTS calibration run does apply a fixed mapping override after the
warmup/statistics checkpoint has been loaded, then freezes that overridden
mapping for the full calibration stage. The Zernike disagreement and fusion
formula themselves are unchanged.
Because the objective uses detached geometric uncertainty weights, calibration computes
the unchanged Zernike disagreement and fusion formula under `torch.no_grad()`
from detached atomic probabilities. This avoids retaining the multi-scale 3D
convolution graph; it does not change `u`, the loss, or the head gradients.

For the base four-class atomic probability `p`, define atomic logits
`a = log(p + eps)` and detached top class `k* = argmax(a)`. On the union `M` of
predicted and true tumor voxels, calibration minimizes

```text
L = Dice(head1) + Dice(head2) + weight * L_margin

raw_weight[i] = eps + stopgrad(u[i])^beta
weight[i] = raw_weight[i] / (mean_M(raw_weight) + eps)
L_margin = mean_M weight[i] * sum(valid k != k*) relu(a[k*] - a[k] - margin)
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
  calibration_fusion:
    enabled: true
    xi: 6.0
    bias: -5.5
  margin_gradient:
    enabled: true
    weight: 0.03
    uncertainty_power: 1.5
    margin: 1.0

training:
  calibration_epochs: 50
  warmup_validation_every: 5
  validation_every: 1
```

The default focused setting uses ROI-normalized `eps + u^1.5`. Its mean weight
is approximately one, while high-uncertainty voxels receive a larger relative
gradient than low-uncertainty voxels. The margin coefficient is therefore not
silently reduced when the absolute uncertainty level is low. The two decoder
heads use asymmetric training-time `Dropout3d` rates `[0.2, 0.3]`; dropout is
disabled automatically during validation and inference.

Legacy configurations without `margin_gradient` leave the margin objective
disabled. Legacy configurations without `calibration_fusion` preserve the
fusion values loaded from their checkpoint. The override adds no model
parameter and does not change `state_dict`, so old warmup checkpoints still
load with `strict=True`.

The recommended mapping is `u = sigmoid(6Z - 5.5)` and places the sigmoid
midpoint at disagreement `Z=0.917`: `u(0)=0.004`, `u(0.5)=0.076`,
`u(1.0)=0.622`, and `u(1.5)=0.971`. This keeps low-disagreement voxels near
zero while assigning high uncertainty to large geometric disagreement. With
representative correct/error disagreements `0.77/1.10`, the `u^1.5` weights
have a relative ratio of about four before ROI mean normalization. These are
diagnostic expectations rather than checkpoint eligibility conditions.

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
Before the first calibration update, the loaded model with the fixed fusion
override is evaluated by the same validation path as epoch 0. If it passes the
Dice guard, it becomes the initial ECE candidate. A training epoch replaces it
only when its ordinary ECE is strictly lower, so calibration cannot discard a
better starting model merely because all 50 training epochs are worse.
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
validated once to establish the Dice eligibility reference. After any fitted
statistics checkpoint is reloaded, `xi=6.0` and `bias=-5.5` are applied when
`calibration_fusion.enabled=true`; this ordering prevents checkpoint values
from replacing the configured calibration mapping.
