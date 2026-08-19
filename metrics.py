from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .probability import atomic_to_regions, scalar_to_atomic, scalar_to_regions


class MetricSampleReservoir:
    """Keep a deterministic, uniformly sampled, bounded set of aligned metrics."""

    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._priorities = np.empty(0, dtype=np.float64)
        self._values: dict[str, np.ndarray] = {}

    def update(self, **values: np.ndarray) -> None:
        if not values:
            return
        arrays = {name: np.asarray(value).reshape(-1) for name, value in values.items()}
        sizes = {array.size for array in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("All metric sample arrays must have the same size")
        count = sizes.pop()
        if count == 0:
            return
        if self._values and arrays.keys() != self._values.keys():
            raise ValueError("Metric sample fields must remain unchanged between updates")

        priorities = self._rng.random(count)
        # Reduce a large incoming case before merging so temporary storage is
        # bounded by roughly twice the configured reservoir size.
        if count > self.capacity:
            selected = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            priorities = priorities[selected]
            arrays = {name: array[selected] for name, array in arrays.items()}

        if self._priorities.size:
            priorities = np.concatenate((self._priorities, priorities))
            arrays = {
                name: np.concatenate((self._values[name], array))
                for name, array in arrays.items()
            }
        if priorities.size > self.capacity:
            selected = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            priorities = priorities[selected]
            arrays = {name: array[selected] for name, array in arrays.items()}
        self._priorities = priorities
        self._values = arrays

    def values(self, name: str) -> np.ndarray:
        if name not in self._values:
            return np.empty(0, dtype=np.float32)
        return self._values[name]

    def __len__(self) -> int:
        return int(self._priorities.size)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def dice_per_region(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    values = []
    for channel in range(3):
        pred = prediction[channel].astype(bool)
        truth = target[channel].astype(bool)
        denominator = pred.sum() + truth.sum()
        values.append(float("nan") if denominator == 0 else float(2.0 * np.logical_and(pred, truth).sum() / denominator))
    return values


def hd95_per_region(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    from monai.metrics.hausdorff_distance import compute_hausdorff_distance

    pred = torch.as_tensor(prediction[None].astype(np.uint8))
    truth = torch.as_tensor(target[None].astype(np.uint8))
    values = compute_hausdorff_distance(pred, truth, include_background=True, percentile=95)[0]
    return [float(item) if math.isfinite(float(item)) else float("nan") for item in values]


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if confidence.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & (confidence <= upper if upper == 1.0 else confidence < upper)
        if selected.any():
            value += float(selected.mean() * abs(correct[selected].mean() - confidence[selected].mean()))
    return value


def risk_calibration_metrics(
    uncertainty: np.ndarray,
    correct: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    """Evaluate u as an estimated probability of voxel-wise prediction error."""
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if uncertainty.shape != correct.shape:
        raise ValueError("uncertainty and correct must have identical shapes")
    if uncertainty.size == 0:
        return {"risk_ece": float("nan"), "risk_brier": float("nan")}
    uncertainty = np.clip(uncertainty, 0.0, 1.0)
    error = 1.0 - correct
    return {
        "risk_ece": expected_calibration_error(1.0 - uncertainty, correct, bins),
        "risk_brier": float(np.mean((uncertainty - error) ** 2)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1.0 - sorted_labels)
    recall = np.concatenate(([0.0], tp / positives))
    precision_values = tp / np.maximum(tp + fp, 1e-12)
    precision = np.concatenate(([precision_values[0]], precision_values))
    return _trapezoid(precision, recall)


def _rejection_auc(confidence: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(confidence, kind="mergesort")
    sorted_correct = correct[order].astype(np.float64)
    count = sorted_correct.size
    remaining = np.cumsum(sorted_correct[::-1])[::-1] / np.arange(count, 0, -1)
    rejection = np.arange(count, dtype=np.float64) / max(count, 1)
    return _trapezoid(remaining, rejection)


def prediction_rejection_ratio(confidence: np.ndarray, correct: np.ndarray) -> float:
    accuracy = float(correct.mean())
    if confidence.size < 2 or not 0.0 < accuracy < 1.0:
        return float("nan")
    rejection = np.arange(correct.size, dtype=np.float64) / correct.size
    random_auc = _trapezoid(np.full(correct.size, accuracy), rejection)
    denominator = _rejection_auc(correct, correct) - random_auc
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((_rejection_auc(confidence, correct) - random_auc) / denominator * 100.0)


def evaluate_case(
    atomic_probability: torch.Tensor,
    uncertainty: torch.Tensor,
    scalar_target: torch.Tensor,
    region_probability: torch.Tensor | None = None,
    risk_reference_probability: torch.Tensor | None = None,
    bins: int = 15,
    max_voxels: int = 200000,
    high_confidence_threshold: float = 0.95,
) -> dict[str, Any]:
    probability = atomic_probability.detach().float().cpu()
    if probability.ndim == 5:
        probability = probability[0]
    scalar = scalar_target.detach().cpu()
    if scalar.ndim == 5:
        scalar = scalar[0]
    if scalar.ndim == 4 and scalar.shape[0] == 1:
        scalar = scalar[0]
    atomic_target = scalar_to_atomic(scalar.unsqueeze(0).unsqueeze(0))[0]
    region_target = scalar_to_regions(scalar.unsqueeze(0).unsqueeze(0))[0].numpy().astype(np.uint8)
    atomic_prediction = probability.argmax(dim=0)
    if region_probability is None:
        region_probability = atomic_to_regions(probability.unsqueeze(0))[0]
    else:
        region_probability = region_probability.detach().float().cpu()
        if region_probability.ndim == 5:
            region_probability = region_probability[0]
        if region_probability.ndim != 4 or region_probability.shape[0] != 3:
            raise ValueError("region_probability must have shape [3,D,H,W] or [1,3,D,H,W]")
    region_prediction = (region_probability >= 0.5).numpy().astype(np.uint8)
    dice = dice_per_region(region_prediction, region_target)
    hd95 = hd95_per_region(region_prediction, region_target)
    mask = (atomic_target > 0) | (atomic_prediction > 0)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)
    confidence = probability.max(dim=0).values[mask].numpy()
    correct = (atomic_prediction[mask] == atomic_target[mask]).numpy().astype(np.float32)
    error = 1.0 - correct
    risk_probability = atomic_probability if risk_reference_probability is None else risk_reference_probability
    risk_probability = risk_probability.detach().float().cpu()
    if risk_probability.ndim == 5:
        risk_probability = risk_probability[0]
    risk_prediction = risk_probability.argmax(dim=0)
    risk_mask = (atomic_target > 0) | (risk_prediction > 0)
    if not bool(risk_mask.any()):
        risk_mask = torch.ones_like(risk_mask, dtype=torch.bool)
    risk_correct = (risk_prediction[risk_mask] == atomic_target[risk_mask]).numpy().astype(np.float32)
    uncertainty_values = uncertainty.detach().float().cpu()
    if uncertainty_values.ndim == 5:
        uncertainty_values = uncertainty_values[0, 0]
    elif uncertainty_values.ndim == 4:
        uncertainty_values = uncertainty_values[0]
    uncertainty_values = uncertainty_values[risk_mask].numpy()
    risk_brier = risk_calibration_metrics(uncertainty_values, risk_correct, bins)["risk_brier"]
    target_one_hot = torch.nn.functional.one_hot(atomic_target.long(), num_classes=4).movedim(-1, 0).float()
    brier = float(((probability[:, mask] - target_one_hot[:, mask]) ** 2).sum(dim=0).mean())
    high_confidence_error = float(np.mean((confidence >= high_confidence_threshold) & (error > 0.5)))
    if confidence.size > max_voxels:
        indices = np.linspace(0, confidence.size - 1, max_voxels).astype(np.int64)
        confidence, correct, error = (values[indices] for values in (confidence, correct, error))
    if uncertainty_values.size > max_voxels:
        risk_indices = np.linspace(0, uncertainty_values.size - 1, max_voxels).astype(np.int64)
        uncertainty_values = uncertainty_values[risk_indices]
        risk_correct = risk_correct[risk_indices]
    risk_error = 1.0 - risk_correct
    risk_metrics = risk_calibration_metrics(uncertainty_values, risk_correct, bins)
    risk_metrics["risk_brier"] = risk_brier
    result: dict[str, Any] = {
        "mean_dice": float(np.nanmean(dice)),
        "mean_hd95": float(np.nanmean(hd95)),
        "ece": expected_calibration_error(confidence, correct, bins),
        "brier": brier,
        "auroc_error": auroc(risk_error, uncertainty_values),
        "aupr_error": aupr(risk_error, uncertainty_values),
        "prr": prediction_rejection_ratio(confidence, correct),
        "high_confidence_error_rate": high_confidence_error,
        **risk_metrics,
    }
    for name, value in zip(("TC", "WT", "ET"), dice):
        result[f"dice_{name}"] = value
    for name, value in zip(("TC", "WT", "ET"), hd95):
        result[f"hd95_{name}"] = value
    return result
