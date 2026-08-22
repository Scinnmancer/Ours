from __future__ import annotations

import torch
import torch.nn as nn


class RegionDiceLoss(nn.Module):
    """Baseline-equivalent sigmoid Dice loss for [TC, WT, ET] logits."""

    def __init__(self):
        super().__init__()
        from monai.losses import DiceLoss

        self.loss = DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target.float())


def risk_brier_loss(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """Fit uncertainty as the voxel-wise probability of segmentation error."""
    if uncertainty.shape != error.shape:
        raise ValueError("uncertainty and error must have identical shapes")
    uncertainty = uncertainty.reshape(-1)
    error = error.to(dtype=uncertainty.dtype).reshape(-1)
    if uncertainty.numel() == 0:
        return uncertainty.sum() * 0.0
    return (uncertainty - error).square().mean()
