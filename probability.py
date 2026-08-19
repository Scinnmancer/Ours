from __future__ import annotations

import torch
import torch.nn.functional as F


REGION_NAMES = ("TC", "WT", "ET")
ATOMIC_NAMES = ("BG", "NCR_NET", "ED", "ET")


def scalar_to_atomic(label: torch.Tensor) -> torch.Tensor:
    """Map BraTS scalar labels {0,1,2,4} to contiguous {0,1,2,3}."""
    if label.ndim >= 2 and label.shape[1] == 1:
        label = label[:, 0]
    label = label.long()
    valid = (label == 0) | (label == 1) | (label == 2) | (label == 4)
    if not bool(valid.all()):
        invalid = torch.unique(label[~valid]).detach().cpu().tolist()
        raise ValueError(f"Invalid BraTS labels: {invalid}")
    result = label.clone()
    result[label == 4] = 3
    return result


def atomic_label_to_scalar(label: torch.Tensor) -> torch.Tensor:
    label = label.long()
    if not bool(((label >= 0) & (label <= 3)).all()):
        raise ValueError("Atomic labels must be in {0,1,2,3}")
    result = label.clone()
    result[label == 3] = 4
    return result


def scalar_to_regions(label: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    atomic = scalar_to_atomic(label)
    tc = (atomic == 1) | (atomic == 3)
    wt = atomic > 0
    et = atomic == 3
    return torch.stack((tc, wt, et), dim=1).to(dtype=dtype)


def independent_logits_to_regions(logits: torch.Tensor) -> torch.Tensor:
    """Match the baseline output convention: independent sigmoid [TC, WT, ET]."""
    if logits.ndim < 3 or logits.shape[1] != 3:
        raise ValueError(f"Expected [B,3,...] logits, got {tuple(logits.shape)}")
    return torch.sigmoid(logits)


def nested_region_closure(regions: torch.Tensor) -> torch.Tensor:
    """Create a nested copy for the internal atomic-probability bridge.

    External segmentation probabilities remain the unmodified independent
    baseline outputs. Internally, positive evidence for an inner BraTS region
    is propagated outwards so ET <= TC <= WT before conversion to four
    mutually exclusive classes.
    """
    if regions.ndim < 3 or regions.shape[1] != 3:
        raise ValueError(f"Expected [B,3,...] regions, got {tuple(regions.shape)}")
    tc, wt, et = regions[:, 0:1], regions[:, 1:2], regions[:, 2:3]
    nested_tc = torch.maximum(tc, et)
    nested_wt = torch.maximum(wt, nested_tc)
    return torch.cat((nested_tc, nested_wt, et), dim=1)


def independent_logits_to_atomic(logits: torch.Tensor) -> torch.Tensor:
    regions = nested_region_closure(independent_logits_to_regions(logits))
    return regions_to_atomic(regions)


def hierarchical_logits_to_regions(logits: torch.Tensor) -> torch.Tensor:
    """Convert [WT, TC|WT, ET|TC] logits to external [TC, WT, ET] probabilities."""
    if logits.ndim < 3 or logits.shape[1] != 3:
        raise ValueError(f"Expected [B,3,...] logits, got {tuple(logits.shape)}")
    wt = torch.sigmoid(logits[:, 0:1])
    tc = wt * torch.sigmoid(logits[:, 1:2])
    et = tc * torch.sigmoid(logits[:, 2:3])
    return torch.cat((tc, wt, et), dim=1)


def regions_to_atomic(regions: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if regions.shape[1] != 3:
        raise ValueError(f"Expected [TC,WT,ET] channels, got {tuple(regions.shape)}")
    tc, wt, et = regions[:, 0:1], regions[:, 1:2], regions[:, 2:3]
    if torch.any(et > tc + 1e-5) or torch.any(tc > wt + 1e-5):
        raise ValueError("Region probabilities violate ET <= TC <= WT")
    atomic = torch.cat((1.0 - wt, tc - et, wt - tc, et), dim=1).clamp_min(0.0)
    return atomic / atomic.sum(dim=1, keepdim=True).clamp_min(eps)


def hierarchical_logits_to_atomic(logits: torch.Tensor) -> torch.Tensor:
    return regions_to_atomic(hierarchical_logits_to_regions(logits))


def atomic_to_regions(atomic: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if atomic.shape[1] != 4:
        raise ValueError(f"Expected four atomic channels, got {tuple(atomic.shape)}")
    atomic = atomic.clamp_min(0.0)
    atomic = atomic / atomic.sum(dim=1, keepdim=True).clamp_min(eps)
    ncr, edema, et = atomic[:, 1:2], atomic[:, 2:3], atomic[:, 3:4]
    return torch.cat((ncr + et, ncr + edema + et, et), dim=1)


def atomic_one_hot(label: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    atomic = scalar_to_atomic(label)
    return F.one_hot(atomic, num_classes=4).movedim(-1, 1).to(dtype=dtype)


def jensen_shannon_divergence(p1: torch.Tensor, p2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if p1.shape != p2.shape or p1.shape[1] != 4:
        raise ValueError("JS inputs must have equal [B,4,...] shapes")
    p1 = p1.clamp_min(eps)
    p2 = p2.clamp_min(eps)
    p1 = p1 / p1.sum(dim=1, keepdim=True)
    p2 = p2 / p2.sum(dim=1, keepdim=True)
    mean = 0.5 * (p1 + p2)
    entropy_mean = -(mean * mean.log()).sum(dim=1, keepdim=True)
    entropy_heads = -0.5 * ((p1 * p1.log()).sum(dim=1, keepdim=True) + (p2 * p2.log()).sum(dim=1, keepdim=True))
    return (entropy_mean - entropy_heads).clamp_min(0.0)
