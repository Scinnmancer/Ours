from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from .probability import (
    ATOMIC_NAMES,
    REGION_NAMES,
    atomic_to_regions,
    scalar_to_atomic,
    scalar_to_regions,
)


ATOMIC_CHANGE_CODES = {
    0: "outside_roi",
    1: "class_unchanged",
    2: "corrected",
    3: "corrupted",
    4: "wrong_to_wrong",
}


@dataclass
class ProbabilitySnapshot:
    atomic_prediction: torch.Tensor
    region_prediction: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor


@dataclass
class RefineAudit:
    metrics: dict[str, Any]
    atomic_transitions: list[dict[str, Any]]
    region_transitions: list[dict[str, Any]]
    atomic_change_map: torch.Tensor
    confidence_delta_map: torch.Tensor
    region_change_maps: dict[str, torch.Tensor]


def probability_snapshot(
    probability: torch.Tensor, eps: float = 1e-7
) -> ProbabilitySnapshot:
    """Compact the information needed to compare base and refined probabilities."""
    value = probability.detach().to(device="cpu", dtype=torch.float32)
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError("Refine audit expects a batch size of one")
        value = value[0]
    if value.ndim != 4 or value.shape[0] != 4:
        raise ValueError(
            "Atomic probability must have shape [4,D,H,W] or [1,4,D,H,W]"
        )
    value = value.clamp_min(0.0)
    value = value / value.sum(dim=0, keepdim=True).clamp_min(eps)
    confidence, atomic_prediction = value.max(dim=0)
    entropy = -(value.clamp_min(eps) * value.clamp_min(eps).log()).sum(dim=0)
    region_prediction = (atomic_to_regions(value.unsqueeze(0))[0] >= 0.5).to(
        torch.uint8
    )
    return ProbabilitySnapshot(
        atomic_prediction=atomic_prediction.to(torch.uint8),
        region_prediction=region_prediction,
        confidence=confidence,
        entropy=entropy,
    )


def _target_tensors(scalar_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    atomic = scalar_to_atomic(scalar_target.detach().cpu())
    regions = scalar_to_regions(scalar_target.detach().cpu(), dtype=torch.uint8)
    if atomic.ndim == 4:
        if atomic.shape[0] != 1:
            raise ValueError("Refine audit expects a batch size of one")
        atomic = atomic[0]
    if regions.ndim == 5:
        if regions.shape[0] != 1:
            raise ValueError("Refine audit expects a batch size of one")
        regions = regions[0]
    return atomic.to(torch.uint8), regions.to(torch.uint8)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    return float(values[mask].float().mean()) if bool(mask.any()) else None


def _transition_rows(
    target: torch.Tensor,
    base: torch.Tensor,
    refined: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_class in range(4):
        for base_class in range(4):
            for refined_class in range(4):
                count = int(
                    (
                        mask
                        & (target == target_class)
                        & (base == base_class)
                        & (refined == refined_class)
                    ).sum()
                )
                if count:
                    rows.append(
                        {
                            "target_class": target_class,
                            "target_name": ATOMIC_NAMES[target_class],
                            "base_class": base_class,
                            "base_name": ATOMIC_NAMES[base_class],
                            "refined_class": refined_class,
                            "refined_name": ATOMIC_NAMES[refined_class],
                            "count": count,
                        }
                    )
    return rows


def audit_refinement(
    base: ProbabilitySnapshot,
    refined: ProbabilitySnapshot,
    scalar_target: torch.Tensor,
    confidence_epsilon: float = 1e-7,
) -> RefineAudit:
    """Describe where refinement changed predictions and whether changes were correct."""
    target, region_target = _target_tensors(scalar_target)
    if (
        base.atomic_prediction.shape != target.shape
        or refined.atomic_prediction.shape != target.shape
    ):
        raise ValueError("Base, refined, and target atomic labels must have identical shapes")
    if (
        base.region_prediction.shape != region_target.shape
        or refined.region_prediction.shape != region_target.shape
    ):
        raise ValueError("Base, refined, and target region labels must have identical shapes")

    base_prediction = base.atomic_prediction
    refined_prediction = refined.atomic_prediction
    roi = (target > 0) | (base_prediction > 0) | (refined_prediction > 0)
    if not bool(roi.any()):
        roi = torch.ones_like(target, dtype=torch.bool)
    changed = roi & (base_prediction != refined_prediction)
    base_correct = base_prediction == target
    refined_correct = refined_prediction == target
    corrected = changed & ~base_correct & refined_correct
    corrupted = changed & base_correct & ~refined_correct
    wrong_to_wrong = changed & ~base_correct & ~refined_correct
    unchanged = roi & ~changed
    base_error = roi & ~base_correct
    base_correct_roi = roi & base_correct

    confidence_delta = refined.confidence - base.confidence
    confidence_decreased = confidence_delta < -confidence_epsilon
    confidence_increased = confidence_delta > confidence_epsilon
    confidence_unchanged = ~(confidence_decreased | confidence_increased)
    entropy_delta = refined.entropy - base.entropy

    roi_count = int(roi.sum())
    changed_count = int(changed.sum())
    corrected_count = int(corrected.sum())
    corrupted_count = int(corrupted.sum())
    wrong_to_wrong_count = int(wrong_to_wrong.sum())
    base_error_count = int(base_error.sum())
    base_correct_count = int(base_correct_roi.sum())
    refined_error_count = int((roi & ~refined_correct).sum())
    wrong_touched = corrected_count + wrong_to_wrong_count

    metrics: dict[str, Any] = {
        "roi_voxels": roi_count,
        "atomic_unchanged_voxels": int(unchanged.sum()),
        "atomic_changed_voxels": changed_count,
        "atomic_corrected_voxels": corrected_count,
        "atomic_corrupted_voxels": corrupted_count,
        "atomic_wrong_to_wrong_voxels": wrong_to_wrong_count,
        "atomic_net_corrected_voxels": corrected_count - corrupted_count,
        "base_error_voxels": base_error_count,
        "refined_error_voxels": refined_error_count,
        "base_correct_voxels": base_correct_count,
        "atomic_change_rate": _safe_rate(changed_count, roi_count),
        "atomic_top1_preservation_rate": _safe_rate(int(unchanged.sum()), roi_count),
        "atomic_correction_precision": _safe_rate(corrected_count, changed_count),
        "atomic_error_repair_rate": _safe_rate(corrected_count, base_error_count),
        "atomic_error_change_coverage": _safe_rate(wrong_touched, base_error_count),
        "atomic_error_introduction_rate": _safe_rate(
            corrupted_count, base_correct_count
        ),
        "base_accuracy": _safe_rate(base_correct_count, roi_count),
        "refined_accuracy": _safe_rate(roi_count - refined_error_count, roi_count),
        "base_mean_confidence": _mean(base.confidence, roi),
        "refined_mean_confidence": _mean(refined.confidence, roi),
        "mean_confidence_delta": _mean(confidence_delta, roi),
        "base_mean_entropy": _mean(base.entropy, roi),
        "refined_mean_entropy": _mean(refined.entropy, roi),
        "mean_entropy_delta": _mean(entropy_delta, roi),
        "base_error_confidence_decrease_rate": _safe_rate(
            int((base_error & confidence_decreased).sum()), base_error_count
        ),
        "base_error_confidence_increase_rate": _safe_rate(
            int((base_error & confidence_increased).sum()), base_error_count
        ),
        "base_correct_confidence_decrease_rate": _safe_rate(
            int((base_correct_roi & confidence_decreased).sum()), base_correct_count
        ),
        "base_correct_confidence_increase_rate": _safe_rate(
            int((base_correct_roi & confidence_increased).sum()), base_correct_count
        ),
        "confidence_unchanged_voxels": int((roi & confidence_unchanged).sum()),
    }

    atomic_change_map = torch.zeros_like(target, dtype=torch.uint8)
    atomic_change_map[unchanged] = 1
    atomic_change_map[corrected] = 2
    atomic_change_map[corrupted] = 3
    atomic_change_map[wrong_to_wrong] = 4

    region_change_maps: dict[str, torch.Tensor] = {}
    region_transitions: list[dict[str, Any]] = []
    for channel, region_name in enumerate(REGION_NAMES):
        target_region = region_target[channel].bool()
        base_region = base.region_prediction[channel].bool()
        refined_region = refined.region_prediction[channel].bool()
        region_roi = target_region | base_region | refined_region
        region_changed = region_roi & (base_region != refined_region)
        region_base_correct = base_region == target_region
        region_refined_correct = refined_region == target_region
        region_corrected = region_changed & ~region_base_correct & region_refined_correct
        region_corrupted = region_changed & region_base_correct & ~region_refined_correct

        change_map = torch.zeros_like(target, dtype=torch.uint8)
        change_map[region_roi & ~region_changed] = 1
        change_map[region_corrected] = 2
        change_map[region_corrupted] = 3
        region_change_maps[region_name] = change_map

        for target_value in (0, 1):
            for base_value in (0, 1):
                for refined_value in (0, 1):
                    count = int(
                        (
                            region_roi
                            & (target_region == bool(target_value))
                            & (base_region == bool(base_value))
                            & (refined_region == bool(refined_value))
                        ).sum()
                    )
                    if count:
                        region_transitions.append(
                            {
                                "region": region_name,
                                "target": target_value,
                                "base": base_value,
                                "refined": refined_value,
                                "count": count,
                            }
                        )

        for prefix, prediction in (("base", base_region), ("refined", refined_region)):
            metrics[f"{region_name}_{prefix}_tp"] = int((prediction & target_region).sum())
            metrics[f"{region_name}_{prefix}_fp"] = int((prediction & ~target_region).sum())
            metrics[f"{region_name}_{prefix}_fn"] = int((~prediction & target_region).sum())
        metrics[f"{region_name}_corrected_voxels"] = int(region_corrected.sum())
        metrics[f"{region_name}_corrupted_voxels"] = int(region_corrupted.sum())
        metrics[f"{region_name}_net_corrected_voxels"] = int(
            region_corrected.sum() - region_corrupted.sum()
        )

    return RefineAudit(
        metrics=metrics,
        atomic_transitions=_transition_rows(
            target, base_prediction, refined_prediction, roi
        ),
        region_transitions=region_transitions,
        atomic_change_map=atomic_change_map,
        confidence_delta_map=torch.where(
            roi, confidence_delta, torch.zeros_like(confidence_delta)
        ),
        region_change_maps=region_change_maps,
    )


_COUNT_SUFFIXES = ("_voxels", "_tp", "_fp", "_fn")


def summarize_audits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce domain and overall summaries with pooled counts and case-macro means."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["domain"]), []).append(row)
    if rows:
        groups["overall"] = rows
    summaries: list[dict[str, Any]] = []
    excluded = {"domain", "case"}
    for domain, group in groups.items():
        result: dict[str, Any] = {"domain": domain, "n_cases": len(group)}
        for key in group[0]:
            if key in excluded:
                continue
            values = [item.get(key) for item in group]
            finite = [float(value) for value in values if value is not None]
            if key.endswith(_COUNT_SUFFIXES):
                result[key] = int(sum(finite))
            else:
                result[f"case_macro_{key}"] = sum(finite) / len(finite) if finite else None

        roi = int(result.get("roi_voxels", 0))
        changed = int(result.get("atomic_changed_voxels", 0))
        corrected = int(result.get("atomic_corrected_voxels", 0))
        corrupted = int(result.get("atomic_corrupted_voxels", 0))
        wrong_to_wrong = int(result.get("atomic_wrong_to_wrong_voxels", 0))
        base_error = int(result.get("base_error_voxels", 0))
        base_correct = int(result.get("base_correct_voxels", 0))
        refined_error = int(result.get("refined_error_voxels", 0))
        result.update(
            {
                "pooled_atomic_change_rate": _safe_rate(changed, roi),
                "pooled_atomic_top1_preservation_rate": _safe_rate(
                    roi - changed, roi
                ),
                "pooled_atomic_correction_precision": _safe_rate(corrected, changed),
                "pooled_atomic_error_repair_rate": _safe_rate(corrected, base_error),
                "pooled_atomic_error_change_coverage": _safe_rate(
                    corrected + wrong_to_wrong, base_error
                ),
                "pooled_atomic_error_introduction_rate": _safe_rate(
                    corrupted, base_correct
                ),
                "pooled_base_accuracy": _safe_rate(base_correct, roi),
                "pooled_refined_accuracy": _safe_rate(roi - refined_error, roi),
            }
        )
        summaries.append(result)
    return summaries


def save_audit_nifti(
    audit: RefineAudit,
    affine: np.ndarray,
    output_dir: Path,
    case: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(audit.atomic_change_map.numpy().astype(np.uint8), affine),
        output_dir / f"{case}__atomic_change_type.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(audit.confidence_delta_map.numpy().astype(np.float32), affine),
        output_dir / f"{case}__confidence_delta.nii.gz",
    )
    for region_name, change_map in audit.region_change_maps.items():
        nib.save(
            nib.Nifti1Image(change_map.numpy().astype(np.uint8), affine),
            output_dir / f"{case}__{region_name}_change_type.nii.gz",
        )
