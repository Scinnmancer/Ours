from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionDiceLoss(nn.Module):
    """Baseline-equivalent sigmoid Dice loss for [TC, WT, ET] logits."""

    def __init__(self):
        super().__init__()
        from monai.losses import DiceLoss

        self.loss = DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target.float())


def balanced_pairwise_ranking_loss(
    disagreement: torch.Tensor,
    error: torch.Tensor,
    max_pairs: int = 32768,
) -> torch.Tensor:
    """Rank error voxels above correct voxels using balanced random pairs."""
    if max_pairs <= 0:
        raise ValueError("max_pairs must be a positive integer")
    disagreement = disagreement.reshape(-1)
    error = error.to(dtype=disagreement.dtype).reshape(-1)
    positive = torch.nonzero(error > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(error <= 0.5, as_tuple=False).flatten()
    if positive.numel() == 0 or negative.numel() == 0:
        return disagreement.sum() * 0.0
    count = min(positive.numel(), negative.numel(), max_pairs)
    pos_idx = positive[torch.randperm(positive.numel(), device=positive.device)[:count]]
    neg_idx = negative[torch.randperm(negative.numel(), device=negative.device)[:count]]
    # softplus(d_correct - d_error) directly optimizes the desired ordering
    # without treating the disagreement score as a calibrated probability.
    return F.softplus(disagreement[neg_idx] - disagreement[pos_idx]).mean()
