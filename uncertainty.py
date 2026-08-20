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
