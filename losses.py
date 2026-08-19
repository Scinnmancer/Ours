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


def balanced_brier_loss(uncertainty: torch.Tensor, error: torch.Tensor, max_samples: int = 65536) -> torch.Tensor:
    uncertainty = uncertainty.reshape(-1)
    error = error.to(dtype=uncertainty.dtype).reshape(-1)
    positive = torch.nonzero(error > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(error <= 0.5, as_tuple=False).flatten()
    if positive.numel() == 0 or negative.numel() == 0:
        return F.mse_loss(uncertainty, error)
    count = min(positive.numel(), negative.numel(), max_samples // 2)
    pos_idx = positive[torch.randperm(positive.numel(), device=positive.device)[:count]]
    neg_idx = negative[torch.randperm(negative.numel(), device=negative.device)[:count]]
    indices = torch.cat((pos_idx, neg_idx))
    return F.mse_loss(uncertainty[indices], error[indices])
