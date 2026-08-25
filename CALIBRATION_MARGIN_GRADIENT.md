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
Because the objective is explicitly `stopgrad(u^beta)`, calibration computes
the unchanged Zernike disagreement and fusion formula under `torch.no_grad()`
from detached atomic probabilities. This avoids retaining the multi-scale 3D
convolution graph; it does not change `u`, the loss, or the head gradients.

For the base four-class atomic probability `p`, define atomic logits
`a = log(p + eps)` and detached top class `k* = argmax(a)`. Within the union of
predicted and true tumor voxels, convert detached `u` to its per-case percentile
rank `r`. The active set contains only currently incorrect voxels in the highest
uncertainty quantile. Calibration minimizes

```text
L = Dice(head1) + Dice(head2) + weight * L_margin

M = {i: prediction_i != target_i and r_i >= uncertainty_quantile}
L_margin = (1 / |ROI|) * sum_(i in M) r_i^beta
           * sum(valid k != k*) relu(a[k*] - a[k] - margin)
```

The raw geometric uncertainty remains unchanged for inference, telemetry, and
maps; percentile rank is used only as a detached gradient weight. This keeps
the emphasis on relatively high-risk voxels when the absolute `u` scale drifts
as the two heads change. Correct predictions receive no calibration gradient.
Normalization remains over the original ROI, so selection does not silently
amplify the configured margin weight.
For an active error, only excessive class gaps are reduced. Once a gap is
within the positive margin, that pair contributes no calibration gradient.
Risk Brier, Risk ECE, and geometric disagreement remain diagnostics and do not
participate in backward. Exact structural-zero classes created by the
nested-region bridge are excluded from the competitor set because they have no
usable local gradient; this prevents the penalty from reducing only the
winning class.

## Configuration

```yaml
uncertainty:
  calibration_fusion:
    enabled: true
    xi: 8.0
    bias: -4.8
  margin_gradient:
    enabled: true
    weight: 0.02
    uncertainty_power: 1.0
    margin: 1.0
    error_selective: true
    uncertainty_quantile: 0.7
    percentile_weighting: true

training:
  calibration_epochs: 10
  warmup_validation_every: 5
  validation_every: 1
```

Legacy configurations without `margin_gradient` leave the margin objective
disabled. Legacy configurations without `calibration_fusion` preserve the
fusion values loaded from their checkpoint. The override adds no model
parameter and does not change `state_dict`, so old warmup checkpoints still
load with `strict=True`.

The recommended mapping places the sigmoid midpoint at disagreement `0.6`:
`u(0)=0.008`, `u(0.6)=0.5`, and `u(1.0)=0.961`. An offline remapping of the
current epoch-10 validation maps estimates mean uncertainty near `0.528` for
correct tumor voxels and `0.772` for error voxels, increasing their numerical
gap while reducing saturation among correct voxels. These are diagnostic
expectations rather than checkpoint eligibility conditions.

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
better starting model merely because all 10 training epochs are worse.
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
statistics checkpoint is reloaded, `xi=8.0` and `bias=-4.8` are applied when
`calibration_fusion.enabled=true`; this ordering prevents checkpoint values
from replacing the configured calibration mapping.
