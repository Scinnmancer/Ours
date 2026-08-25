from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class UncertaintyFusion(nn.Module):
    def __init__(self, eta: float = 1.0, xi: float = 1.0, bias: float = -2.0):
        super().__init__()
        self.raw_eta = nn.Parameter(torch.tensor(_inverse_softplus(eta), dtype=torch.float32))
        self.raw_xi = nn.Parameter(torch.tensor(_inverse_softplus(xi), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.raw_eta)

    @property
    def xi(self) -> torch.Tensor:
        return F.softplus(self.raw_xi)

    def set_xi_bias(self, xi: float, bias: float) -> None:
        """Set the effective geometric mapping without changing state layout."""
        xi = float(xi)
        bias = float(bias)
        if not math.isfinite(xi) or xi <= 0.0:
            raise ValueError("xi must be finite and positive")
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        with torch.no_grad():
            self.raw_xi.copy_(
                torch.as_tensor(
                    _inverse_softplus(xi),
                    dtype=self.raw_xi.dtype,
                    device=self.raw_xi.device,
                )
            )
            self.bias.copy_(
                torch.as_tensor(bias, dtype=self.bias.dtype, device=self.bias.device)
            )

    def forward(
        self,
        zernike_disagreement: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate risk from Zernike geometric disagreement only.

        ``raw_eta`` remains in the state layout solely so existing warm-up and
        probability-calibration checkpoints continue to load strictly. It does
        not contribute to the active risk logit.
        """
        return torch.sigmoid(self.bias + self.xi * zernike_disagreement)


def uncertainty_components(
    zernike_disagreement: torch.Tensor,
    fusion: UncertaintyFusion,
):
    uncertainty = fusion(zernike_disagreement)
    return zernike_disagreement, uncertainty


def uncertainty_weighted_margin_loss(
    atomic_probability: torch.Tensor,
    uncertainty: torch.Tensor,
    mask: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    error_selective: bool = False,
    uncertainty_quantile: float = 0.0,
    percentile_weighting: bool = False,
    uncertainty_power: float = 2.0,
    margin: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize excessive atomic-logit gaps at selected geometric-risk voxels.

    ``log(p + eps)`` is used as the four-class atomic logit bridge.  The
    current top class and the geometric-moment uncertainty are detached, so
    this objective can only update the prediction heads.  Positive ``margin``
    leaves a non-zero class gap instead of continuously pushing predictions
    toward a uniform distribution.  Optional error selection leaves correct
    predictions untouched, while percentile weighting keeps the relative
    high-risk emphasis stable when the absolute uncertainty scale drifts.
    """
    if atomic_probability.ndim < 3 or atomic_probability.shape[1] != 4:
        raise ValueError(
            "Expected four-class [B,4,...] atomic probabilities, got "
            f"{tuple(atomic_probability.shape)}"
        )
    expected_scalar_shape = (
        atomic_probability.shape[0],
        1,
        *atomic_probability.shape[2:],
    )
    if tuple(uncertainty.shape) != expected_scalar_shape:
        raise ValueError(
            "uncertainty must have shape "
            f"{expected_scalar_shape}, got {tuple(uncertainty.shape)}"
        )
    if not math.isfinite(float(uncertainty_power)) or float(uncertainty_power) < 0.0:
        raise ValueError("uncertainty_power must be finite and non-negative")
    if (
        not math.isfinite(float(uncertainty_quantile))
        or float(uncertainty_quantile) < 0.0
        or float(uncertainty_quantile) > 1.0
    ):
        raise ValueError("uncertainty_quantile must be finite and in [0, 1]")
    if not isinstance(error_selective, bool):
        raise ValueError("error_selective must be a boolean")
    if not isinstance(percentile_weighting, bool):
        raise ValueError("percentile_weighting must be a boolean")
    if not math.isfinite(float(margin)) or float(margin) <= 0.0:
        raise ValueError("margin must be finite and positive")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    atomic_logits = atomic_probability.float().clamp_min(float(eps)).log()
    top_index = atomic_logits.detach().argmax(dim=1, keepdim=True)
    top_logit = atomic_logits.gather(1, top_index)
    excessive_gap = F.relu(top_logit - atomic_logits - float(margin))
    competitor = torch.ones_like(excessive_gap, dtype=torch.bool)
    competitor.scatter_(1, top_index, False)
    # Nested-region closure can create exact structural zeros.  Those classes
    # have no usable local gradient through the atomic bridge, so including
    # them would only push down the winner and could collapse the segmentation.
    competitor &= atomic_probability.detach() > float(eps)
    weighted_gap = excessive_gap * competitor.to(dtype=excessive_gap.dtype)
    detached_uncertainty = uncertainty.detach().float().clamp(0.0, 1.0)

    if mask is not None:
        detached_mask = mask.detach()
        if detached_mask.ndim == atomic_probability.ndim - 1:
            detached_mask = detached_mask.unsqueeze(1)
        if tuple(detached_mask.shape) != expected_scalar_shape:
            raise ValueError(
                f"mask must have shape {expected_scalar_shape} or omit the channel dimension"
            )
        voxel_mask = detached_mask.to(dtype=torch.bool)
    else:
        voxel_mask = torch.ones_like(detached_uncertainty, dtype=torch.bool)

    ranking_mask = voxel_mask.clone()

    percentile_rank = torch.zeros_like(detached_uncertainty)
    for batch_index in range(detached_uncertainty.shape[0]):
        sample_mask = ranking_mask[batch_index]
        sample_values = detached_uncertainty[batch_index][sample_mask]
        if sample_values.numel() == 0:
            continue
        sorted_values = sample_values.sort().values
        sample_rank = torch.searchsorted(sorted_values, sample_values, right=True).to(
            detached_uncertainty.dtype
        ) / float(sample_values.numel())
        percentile_rank[batch_index][sample_mask] = sample_rank

    if error_selective:
        if target is None:
            raise ValueError("target is required when error_selective=True")
        detached_target = target.detach()
        if detached_target.ndim == atomic_probability.ndim - 1:
            detached_target = detached_target.unsqueeze(1)
        if tuple(detached_target.shape) != expected_scalar_shape:
            raise ValueError(
                f"target must have shape {expected_scalar_shape} or omit the channel dimension"
            )
        voxel_mask &= top_index != detached_target.long()

    if float(uncertainty_quantile) > 0.0:
        voxel_mask &= percentile_rank >= float(uncertainty_quantile)
    weight_source = percentile_rank if percentile_weighting else detached_uncertainty
    weight = weight_source.pow(float(uncertainty_power))

    # Preserve the original ROI normalization so enabling selection changes
    # where gradients act without silently amplifying each selected voxel.
    denominator = ranking_mask.sum()
    if not bool(denominator > 0):
        return atomic_probability.sum() * 0.0
    return (
        weighted_gap * weight * voxel_mask.to(dtype=weighted_gap.dtype)
    ).sum() / denominator
