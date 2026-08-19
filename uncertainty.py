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

    def forward(
        self,
        probability_disagreement: torch.Tensor,
        zernike_disagreement: torch.Tensor | None = None,
        probability_disagreement_scale: float = 1.0,
    ) -> torch.Tensor:
        """Calibrate risk from probability disagreement only.

        ``zernike_disagreement`` and ``raw_xi`` remain part of the public/state
        layout solely so legacy warm-up checkpoints continue to load strictly.
        They intentionally do not contribute to the risk logit.
        """
        del zernike_disagreement
        scaled = probability_disagreement * float(probability_disagreement_scale)
        return torch.sigmoid(self.bias + self.eta * scaled)


def uncertainty_components(
    head1_atomic: torch.Tensor,
    head2_atomic: torch.Tensor,
    fusion: UncertaintyFusion,
    probability_disagreement_scale: float = 1.0,
):
    probability_disagreement = jensen_shannon_divergence(head1_atomic, head2_atomic)
    uncertainty = fusion(
        probability_disagreement,
        probability_disagreement_scale=probability_disagreement_scale,
    )
    return probability_disagreement, uncertainty
