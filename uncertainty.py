from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .probability import jensen_shannon_divergence


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

    def forward(self, probability_disagreement: torch.Tensor, zernike_disagreement: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bias + self.eta * probability_disagreement + self.xi * zernike_disagreement)


def uncertainty_components(head1_atomic: torch.Tensor, head2_atomic: torch.Tensor, zernike_disagreement: torch.Tensor, fusion: UncertaintyFusion):
    probability_disagreement = jensen_shannon_divergence(head1_atomic, head2_atomic)
    uncertainty = fusion(probability_disagreement, zernike_disagreement)
    return probability_disagreement, zernike_disagreement, uncertainty

