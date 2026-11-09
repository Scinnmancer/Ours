from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDiceLoss(nn.Module):
    """The adaptive multiclass Dice term used by the ACDC baseline."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = logits.shape[1]
        probability = F.softmax(logits, dim=1)
        target_one_hot = torch.zeros_like(logits).scatter_(1, target, 1.0)
        false_positive = probability * (1.0 - target_one_hot)
        false_negative = (1.0 - probability) * target_one_hot
        smooth = torch.full((classes,), 1e-5, device=logits.device)
        fp = false_positive.transpose(0, 1).reshape(classes, -1).sum(1)
        fn = false_negative.transpose(0, 1).reshape(classes, -1).sum(1)
        alpha = torch.clamp(fp / (fp + fn + smooth), min=0.2, max=0.8)
        true_positive = (probability * target_one_hot).transpose(0, 1).reshape(classes, -1).sum(1)
        denominator = true_positive + alpha * fp + (1.0 - alpha) * fn
        return (1.0 - true_positive / (denominator + smooth)).mean()


class ACDCSegmentationLoss(nn.Module):
    """Baseline-identical weighted cross entropy plus adaptive Dice."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.tensor([0.5, 1.0, 1.0, 1.0], dtype=torch.float32))
        self.dice = AdaptiveDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        if target.ndim != logits.ndim or target.shape[1] != 1:
            raise ValueError(f"Expected ACDC target [B,1,...], got {tuple(target.shape)}")
        cross_entropy = F.cross_entropy(logits, target[:, 0], weight=self.weights)
        return cross_entropy + self.dice(logits, target)


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
