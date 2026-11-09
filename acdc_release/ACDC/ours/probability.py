from __future__ import annotations

import torch
import torch.nn.functional as F


CLASS_NAMES = ("BG", "RV", "MYO", "LV")
FOREGROUND_NAMES = CLASS_NAMES[1:]


def scalar_to_class(label: torch.Tensor) -> torch.Tensor:
    """Validate ACDC scalar labels and return contiguous class indices."""
    if label.ndim >= 2 and label.shape[1] == 1:
        label = label[:, 0]
    label = label.long()
    valid = (label >= 0) & (label <= 3)
    if not bool(valid.all()):
        invalid = torch.unique(label[~valid]).detach().cpu().tolist()
        raise ValueError(f"Invalid ACDC labels: {invalid}")
    return label


def class_to_scalar(label: torch.Tensor) -> torch.Tensor:
    return scalar_to_class(label)


def class_one_hot(label: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return F.one_hot(scalar_to_class(label), num_classes=4).movedim(-1, 1).to(dtype=dtype)


def logits_to_class_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 3 or logits.shape[1] != 4:
        raise ValueError(f"Expected ACDC [B,4,...] logits, got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=1)


def class_probability_to_foreground(probability: torch.Tensor) -> torch.Tensor:
    if probability.ndim < 3 or probability.shape[1] != 4:
        raise ValueError(f"Expected ACDC [B,4,...] probability, got {tuple(probability.shape)}")
    return probability[:, 1:]


def class_label_to_foreground(label: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return class_one_hot(label, dtype=dtype)[:, 1:]


def jensen_shannon_divergence(p1: torch.Tensor, p2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if p1.shape != p2.shape or p1.shape[1] != 4:
        raise ValueError("JS inputs must have equal ACDC [B,4,...] shapes")
    p1 = p1.clamp_min(eps)
    p2 = p2.clamp_min(eps)
    p1 = p1 / p1.sum(dim=1, keepdim=True)
    p2 = p2 / p2.sum(dim=1, keepdim=True)
    mean = 0.5 * (p1 + p2)
    entropy_mean = -(mean * mean.log()).sum(dim=1, keepdim=True)
    entropy_heads = -0.5 * (
        (p1 * p1.log()).sum(dim=1, keepdim=True)
        + (p2 * p2.log()).sum(dim=1, keepdim=True)
    )
    return (entropy_mean - entropy_heads).clamp_min(0.0)
