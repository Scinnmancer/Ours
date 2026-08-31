# Local Excess-Confidence Directional Refinement

## Scope

This change is refine-only. It keeps the trained heads, Zernike descriptors,
geometric uncertainty mapping, reliable-neighbor kernel, support term,
`alpha_max`, uncertainty power, and three iterations unchanged. It adds no
trainable parameter or state-dict entry, so existing warmup/calibration
checkpoints still load with `strict=True`.

## Method

Let `p_i` be the original four-class atomic distribution and `r_i^(t)` the
existing reliability-weighted neighbor distribution at iteration `t`. The
original top-1 class is

```text
k_i* = argmax_k p_i,k .
```

Its center confidence and reliable-neighbor support are

```text
c_i = p_i,k_i* ,      n_i^(t) = r_i,k_i*^(t) .
```

The new parameter-free local excess-confidence coefficient is

```text
rho_i^(t) = clip((c_i - n_i^(t)) / (1 - n_i^(t) + eps), 0, 1).
```

The established complementary correction is retained:

```text
a_i^(t) = Normalize((1 - p_i) * r_i^(t)).
```

Only the direction is changed:

```text
t_i^(t) = (1 - rho_i^(t)) r_i^(t) + rho_i^(t) a_i^(t)
alpha_i = alpha_max * u_i^beta * support_i
q_i^(t+1) = (1 - alpha_i) p_i + alpha_i t_i^(t).
```

All three iterations recompute `r` and `rho`, but each update remains anchored
to the original `p`. There is no hard consensus threshold or top-1 freeze.

When a confident center is contradicted by reliable neighbors, `rho` approaches
one and preserves the stronger corrective behavior of the best prior version.
When the neighborhood supports the center at least as strongly as the center
itself, `rho=0`; ordinary neighbor propagation avoids needlessly depressing a
well-supported class. This directly targets the observed failure in which the
legacy complement direction improved overconfident domains but slightly harmed
the already well-calibrated TMC domain.

This refine does not claim to recover spatially coherent shared errors that are
both low-uncertainty and supported by their neighborhood. Such errors provide
neither a geometric-uncertainty trigger nor a contradictory local direction;
addressing them would require a larger change to the uncertainty estimator or
an image-conditioned refiner and is outside this minimal update.

## One-command experiment

From the directory containing the `ours` package:

```bash
bash ours/run_refine_tuning.sh
```

Server paths can be overridden without editing the script:

```bash
CHECKPOINT=/home/ours/runs/dual_swin_zernike_brats2020/final.pt \
OUTPUT=/home/ours/runs/dual_swin_zernike_brats2020/refine_tuning \
bash /home/ours/run_refine_tuning.sh
```

The runner performs a source-validation A/B comparison against
`legacy_complement`, applies a `0.001` absolute Dice guardrail, selects by
ordinary ECE, and then evaluates the best source-validation candidate from each
direction once on the held-out centers. Those held-out results provide a fair
ablation but do not feed back into selection. It records candidate results in
`candidates.csv`, the decision in `selection.json`, final case/domain metrics
under `final__*/`, and
the joined external comparison in `final_comparison.csv`.

## Reviewer-facing claims and acceptance

Neighbor-aware calibration, uncertainty-guided refinement, and local label
propagation are established ideas and are not claimed as new. The narrow claim
is the parameter-free continuous interpolation between reliable-neighbor
diffusion and complementary correction, driven by local excess confidence and
scaled by the existing geometry-derived uncertainty.

The method should be accepted empirically only if source validation selects it
without violating the Dice guardrail, held-out mean ECE improves over the
legacy direction, the TMC ECE regression is removed or materially reduced, and
case-level paired confidence intervals do not reveal a clinically important
Dice loss. Report ordinary ECE together with Brier, multiple ECE bin counts,
per-case deltas, and the worst-decile high-confidence error cases; do not tune
on held-out centers.

Related precedents include spatially varying label smoothing, neighbor-aware
calibration, and uncertainty-guided graph refinement:

- Islam and Glocker, *Spatially Varying Label Smoothing* (2021),
  <https://arxiv.org/abs/2104.05788>
- Murugesan et al., *Neighbor-Aware Calibration of Segmentation Networks with
  Penalty-Based Constraints* (2024), <https://arxiv.org/abs/2401.14487>
- Soberanis-Mukul et al., *Uncertainty-based Graph Convolutional Networks for
  Organ Segmentation Refinement* (2020),
  <https://proceedings.mlr.press/v121/soberanis-mukul20a.html>
