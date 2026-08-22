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


def uncertainty_weighted_radial_gradient(
    logits: torch.Tensor,
    uncertainty: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor | None = None,
    uncertainty_power: float = 2.0,
    confidence_power: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a detached inward gradient for independent-sigmoid logits.

    The model emits independent ``[TC, WT, ET]`` logits, so zero—not a
    channel-wise mean—is the uninformative point. Gradient descent subtracts
    the returned vector and therefore contracts high-risk logits toward zero
    without introducing parameters or changing the forward pass.
    """
    if logits.ndim < 3 or logits.shape[1] != 3:
        raise ValueError(f"Expected [B,3,...] logits, got {tuple(logits.shape)}")
    expected_scalar_shape = (logits.shape[0], 1, *logits.shape[2:])
    if tuple(uncertainty.shape) != expected_scalar_shape:
        raise ValueError(
            "uncertainty must have shape "
            f"{expected_scalar_shape}, got {tuple(uncertainty.shape)}"
        )
    if tuple(confidence.shape) != expected_scalar_shape:
        raise ValueError(
            "confidence must have shape "
            f"{expected_scalar_shape}, got {tuple(confidence.shape)}"
        )
    if not math.isfinite(float(uncertainty_power)) or float(uncertainty_power) < 0.0:
        raise ValueError("uncertainty_power must be finite and non-negative")
    if not math.isfinite(float(confidence_power)) or float(confidence_power) < 0.0:
        raise ValueError("confidence_power must be finite and non-negative")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    work_logits = logits.detach().float()
    norm = torch.linalg.vector_norm(work_logits, dim=1, keepdim=True)
    direction = work_logits / (norm + float(eps))
    weight = (
        uncertainty.detach().float().clamp(0.0, 1.0).pow(float(uncertainty_power))
        * confidence.detach().float().clamp(0.0, 1.0).pow(float(confidence_power))
    )
    if mask is not None:
        detached_mask = mask.detach()
        if detached_mask.ndim == logits.ndim - 1:
            detached_mask = detached_mask.unsqueeze(1)
        if tuple(detached_mask.shape) != expected_scalar_shape:
            raise ValueError(
                f"mask must have shape {expected_scalar_shape} or omit the channel dimension"
            )
        weight = weight * detached_mask.to(dtype=weight.dtype)
    return (weight * direction).to(device=logits.device, dtype=logits.dtype)
