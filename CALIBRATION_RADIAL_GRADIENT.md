# Uncertainty-weighted radial calibration gradient

## Purpose

The calibration stage assumes that the fused Zernike uncertainty `u` is
monotonically aligned with voxel error probability. High-uncertainty,
high-confidence predictions therefore receive an additional inward logit
gradient so their probabilities become less extreme. Warm-up, inference, and
label transfer are unchanged.

The existing calibration objectives remain active:

- each decoder retains its sigmoid Dice loss;
- fused uncertainty retains voxel-wise Risk Brier supervision;
- the radial intervention is added directly during backward and is not an
  additional scalar loss.

## Gradient definition

Each decoder emits independent `[TC, WT, ET]` sigmoid logits. For voxel `i` and
head `h`, let

```text
d[h,i] = z[h,i] / (||z[h,i]||_2 + eps)
w[i]   = stopgrad(u[i]^beta * c[i]^gamma)
g[h,i] = lambda * w[i] * d[h,i]
```

`c` is the maximum probability of the four-class atomic base distribution.
Because these are independent sigmoid logits, zero is the uninformative point;
the implementation contracts the raw logit vector rather than subtracting a
channel mean. Gradient descent subtracts `g`, moving high-risk logits toward
zero while leaving the forward graph and checkpoint state layout unchanged.

The same detached `u` and `c` weight is used for both decoder heads. The
intervention is limited to the union of predicted and ground-truth tumor voxels,
matching the mask used by calibration validation. A zero-logit voxel produces a
zero radial gradient.

## Configuration

```yaml
uncertainty:
  lambda_u: 1.0
  radial_gradient:
    enabled: true
    weight: 0.01
    uncertainty_power: 2.0
    confidence_power: 2.0
```

- `enabled`: activates the intervention only in `stage=calibration`.
- `weight`: radial gradient coefficient `lambda`.
- `uncertainty_power`: exponent `beta` applied to `u`.
- `confidence_power`: exponent `gamma` applied to atomic confidence.

Legacy configurations without `radial_gradient` default to disabled. All
values can be overridden with `--set`, for example:

```bash
python -m ours.train \
  --config ours/configs/brats2020.yaml \
  --stage calibration \
  --checkpoint /path/to/best_seg.pt \
  --set uncertainty.radial_gradient.weight=0.02
```

## AMP and trainable parameters

The radial vector is detached before injection. When AMP is enabled, the hook
multiplies it by the current GradScaler scale; the normal optimizer unscale then
restores the configured effective magnitude. This prevents the intervention
from disappearing under a large dynamic scale.

With the default `training.calibration_trainable_scope=heads`, only the two
decoder heads and active fusion parameters are trainable. The radial gradient
reaches the heads, Risk Brier continues to fit `raw_xi` and `bias`, and the
shared encoder remains frozen. No model parameter or inference operation is
added.

Training metrics include:

- `radial_gradient_enabled`;
- `radial_gradient_weight`;
- `radial_gradient_l2`;
- `radial_gradient_abs_mean`;
- `radial_uncertainty_power` and `radial_confidence_power`.

## Checkpoint selection

Calibration checkpoints must first satisfy

```text
mean_dice >= reference_dice - training.dice_tolerance
```

Eligible checkpoints are ordered by `(ece, -mean_dice)`. Ordinary top-label
segmentation ECE is the selection metric; Dice breaks exact ECE ties. `risk_ece`
and `risk_brier` remain diagnostic metrics and do not participate in selection.

## Starting directly from a warm-up checkpoint

The intervention adds no state-dict entries, so existing same-architecture
warm-up checkpoints continue to load with `strict=True`:

```bash
python -m ours.train \
  --config ours/configs/brats2020.yaml \
  --stage calibration \
  --checkpoint /path/to/best_seg.pt
```

This command does not execute warm-up. It creates a fresh calibration optimizer
and scheduler. If the checkpoint does not contain fitted Zernike statistics,
the training entry point fits them from the configured statistics loader,
writes `stats_fitted.pt`, and then begins the requested calibration epochs. A
checkpoint that already contains fitted statistics proceeds directly to
calibration training.
