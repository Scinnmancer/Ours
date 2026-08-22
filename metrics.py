from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .probability import atomic_to_regions


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
    """Compute symmetric HD95 without retaining dense distances for all regions.

    MONAI's batched implementation is convenient, but a pathological full-volume
    prediction can make its temporary distance tensors the evaluation memory
    high-water mark. Crop each region to the union of prediction and target and
    compute the two directed surface distances sequentially instead.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("HD95 inputs must have equal [C,D,H,W] shapes")

    structure = generate_binary_structure(3, 1)
    values: list[float] = []
    for channel in range(prediction.shape[0]):
        pred_source = np.asarray(prediction[channel])
        truth_source = np.asarray(target[channel])
        if not pred_source.any() or not truth_source.any():
            values.append(float("nan"))
            continue

        union = (pred_source != 0) | (truth_source != 0)
        bounds: list[slice] = []
        for axis in range(3):
            occupied = np.any(union, axis=tuple(item for item in range(3) if item != axis))
            indices = np.flatnonzero(occupied)
            bounds.append(slice(int(indices[0]), int(indices[-1]) + 1))
        crop = tuple(bounds)
        pred = pred_source[crop].astype(bool, copy=False)
        truth = truth_source[crop].astype(bool, copy=False)
        del union

        pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
        truth_surface = truth ^ binary_erosion(truth, structure=structure, border_value=0)
        del pred, truth

        distance = distance_transform_edt(~truth_surface)
        pred_to_truth = distance[pred_surface]
        del distance
        distance = distance_transform_edt(~pred_surface)
        truth_to_pred = distance[truth_surface]
        del distance, pred_surface, truth_surface

        # Match MONAI's symmetric Hausdorff definition: take the percentile
        # independently in each direction, then the larger directed value.
        values.append(float(max(np.percentile(pred_to_truth, 95), np.percentile(truth_to_pred, 95))))
    return values


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


def _sample_mask_indices(mask: torch.Tensor, max_voxels: int) -> torch.Tensor:
    """Return deterministic flat indices while bounding downstream metric arrays."""
    if max_voxels <= 0:
        raise ValueError("max_voxels must be positive")
    flat = mask.reshape(-1)
    count = int(flat.sum())
    if count <= max_voxels:
        return torch.nonzero(flat, as_tuple=False).squeeze(1)

    # Select evenly spaced foreground ranks without first materializing every
    # foreground index (which alone can be ~68 MiB for a full BraTS volume).
    ranks = torch.linspace(0, count - 1, max_voxels, dtype=torch.float64).long()
    selected: list[torch.Tensor] = []
    cursor = 0
    seen = 0
    chunk_size = 1_000_000
    for start in range(0, flat.numel(), chunk_size):
        local = torch.nonzero(flat[start : start + chunk_size], as_tuple=False).squeeze(1)
        next_seen = seen + local.numel()
        next_cursor = int(torch.searchsorted(ranks, torch.tensor(next_seen), right=False))
        if next_cursor > cursor:
            selected.append(local[ranks[cursor:next_cursor] - seen] + start)
            cursor = next_cursor
        seen = next_seen
        if cursor == max_voxels:
            break
    return torch.cat(selected)


def _full_probability_aggregates(
    probability: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    high_confidence_threshold: float,
) -> tuple[float, float]:
    """Compute exact Brier/high-confidence metrics with bounded scratch space."""
    probability = probability.reshape(4, -1)
    target = target.reshape(-1)
    prediction = prediction.reshape(-1)
    mask = mask.reshape(-1)
    brier_sum = 0.0
    high_confidence_errors = 0
    count = 0
    chunk_size = 1_000_000
    for start in range(0, mask.numel(), chunk_size):
        selected = mask[start : start + chunk_size]
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        chunk_probability = probability[:, start : start + chunk_size][:, selected]
        chunk_target = target[start : start + chunk_size][selected]
        chunk_prediction = prediction[start : start + chunk_size][selected]
        brier_values = torch.zeros(selected_count, dtype=torch.float32)
        for channel in range(4):
            expected = (chunk_target == channel).to(torch.float32)
            brier_values.add_((chunk_probability[channel] - expected).square())
        confidence = torch.amax(chunk_probability, dim=0)
        high_confidence_errors += int(
            ((confidence >= high_confidence_threshold) & (chunk_prediction != chunk_target)).sum()
        )
        brier_sum += float(brier_values.sum())
        count += selected_count
    return brier_sum / count, high_confidence_errors / count


def _full_risk_brier(
    uncertainty: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute exact risk Brier score without full-size selected arrays."""
    uncertainty = uncertainty.reshape(-1)
    target = target.reshape(-1)
    prediction = prediction.reshape(-1)
    mask = mask.reshape(-1)
    squared_error_sum = 0.0
    count = 0
    chunk_size = 1_000_000
    for start in range(0, mask.numel(), chunk_size):
        selected = mask[start : start + chunk_size]
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        chunk_uncertainty = uncertainty[start : start + chunk_size][selected].float().clamp(0.0, 1.0)
        error = (
            prediction[start : start + chunk_size][selected]
            != target[start : start + chunk_size][selected]
        ).to(torch.float32)
        squared_error_sum += float((chunk_uncertainty - error).square().sum())
        count += selected_count
    return squared_error_sum / count


def evaluate_case(
    atomic_probability: torch.Tensor,
    uncertainty: torch.Tensor,
    scalar_target: torch.Tensor,
    region_probability: torch.Tensor | None = None,
    risk_reference_probability: torch.Tensor | None = None,
    atomic_prediction: torch.Tensor | np.ndarray | None = None,
    region_prediction: torch.Tensor | np.ndarray | None = None,
    risk_reference_prediction: torch.Tensor | np.ndarray | None = None,
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
    scalar_values = scalar.numpy()
    valid = np.isin(scalar_values, (0, 1, 2, 4))
    if not bool(valid.all()):
        raise ValueError(f"Invalid BraTS labels: {np.unique(scalar_values[~valid]).tolist()}")
    scalar_array = scalar_values.astype(np.uint8, copy=False)
    atomic_target_array = scalar_array.copy()
    atomic_target_array[atomic_target_array == 4] = 3
    region_target = np.stack(
        (
            (atomic_target_array == 1) | (atomic_target_array == 3),
            atomic_target_array > 0,
            atomic_target_array == 3,
        )
    ).astype(np.uint8, copy=False)

    if atomic_prediction is None:
        atomic_prediction_array = probability.argmax(dim=0).to(torch.uint8).numpy()
    elif isinstance(atomic_prediction, torch.Tensor):
        atomic_prediction_array = atomic_prediction.detach().cpu().numpy().astype(np.uint8, copy=False)
    else:
        atomic_prediction_array = np.asarray(atomic_prediction, dtype=np.uint8)
    if atomic_prediction_array.shape != atomic_target_array.shape:
        raise ValueError("atomic_prediction must match the scalar target shape")

    if region_prediction is None:
        if region_probability is None:
            region_probability = atomic_to_regions(probability.unsqueeze(0))[0]
        else:
            region_probability = region_probability.detach().cpu()
            if region_probability.ndim == 5:
                region_probability = region_probability[0]
            if region_probability.ndim != 4 or region_probability.shape[0] != 3:
                raise ValueError("region_probability must have shape [3,D,H,W] or [1,3,D,H,W]")
        region_prediction_array = (region_probability >= 0.5).numpy().astype(np.uint8)
    elif isinstance(region_prediction, torch.Tensor):
        region_prediction_array = region_prediction.detach().cpu().numpy().astype(np.uint8, copy=False)
    else:
        region_prediction_array = np.asarray(region_prediction, dtype=np.uint8)
    if region_prediction_array.shape != region_target.shape:
        raise ValueError("region_prediction must match the [3,D,H,W] region target shape")

    dice = dice_per_region(region_prediction_array, region_target)
    hd95 = hd95_per_region(region_prediction_array, region_target)
    atomic_target = torch.as_tensor(atomic_target_array, dtype=torch.uint8)
    atomic_prediction_tensor = torch.as_tensor(atomic_prediction_array, dtype=torch.uint8)
    mask = (atomic_target > 0) | (atomic_prediction_tensor > 0)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)
    metric_indices = _sample_mask_indices(mask, max_voxels)
    sampled_probability = probability.reshape(4, -1)[:, metric_indices]
    sampled_target = atomic_target.reshape(-1)[metric_indices]
    sampled_prediction = atomic_prediction_tensor.reshape(-1)[metric_indices]
    confidence = torch.amax(sampled_probability, dim=0).numpy()
    correct = (sampled_prediction == sampled_target).numpy().astype(np.float32)
    error = 1.0 - correct
    brier, high_confidence_error = _full_probability_aggregates(
        probability,
        atomic_target,
        atomic_prediction_tensor,
        mask,
        high_confidence_threshold,
    )
    if risk_reference_prediction is not None:
        if isinstance(risk_reference_prediction, torch.Tensor):
            risk_prediction = risk_reference_prediction.detach().cpu().to(torch.uint8)
        else:
            risk_prediction = torch.as_tensor(risk_reference_prediction, dtype=torch.uint8)
    else:
        risk_probability = atomic_probability if risk_reference_probability is None else risk_reference_probability
        risk_probability = risk_probability.detach().float().cpu()
        if risk_probability.ndim == 5:
            risk_probability = risk_probability[0]
        risk_prediction = risk_probability.argmax(dim=0).to(torch.uint8)
    if risk_prediction.shape != atomic_target.shape:
        raise ValueError("risk_reference_prediction must match the scalar target shape")
    risk_mask = (atomic_target > 0) | (risk_prediction > 0)
    if not bool(risk_mask.any()):
        risk_mask = torch.ones_like(risk_mask, dtype=torch.bool)
    uncertainty_map = uncertainty.detach().cpu()
    if uncertainty_map.ndim == 5:
        uncertainty_map = uncertainty_map[0, 0]
    elif uncertainty_map.ndim == 4:
        uncertainty_map = uncertainty_map[0]
    risk_brier = _full_risk_brier(uncertainty_map, atomic_target, risk_prediction, risk_mask)
    risk_indices = _sample_mask_indices(risk_mask, max_voxels)
    risk_correct = (
        risk_prediction.reshape(-1)[risk_indices] == atomic_target.reshape(-1)[risk_indices]
    ).numpy().astype(np.float32)
    uncertainty_values = uncertainty_map.reshape(-1)[risk_indices].float().numpy()
    risk_error = 1.0 - risk_correct
    risk_metrics = risk_calibration_metrics(uncertainty_values, risk_correct, bins)
    risk_metrics["risk_brier"] = risk_brier
    basic_ece = expected_calibration_error(confidence, correct, bins)
    result: dict[str, Any] = {
        "mean_dice": float(np.nanmean(dice)),
        "mean_hd95": float(np.nanmean(hd95)),
        # Ordinary top-label multiclass ECE of the segmentation probabilities.
        # Keep ``ece`` as a compatibility alias for existing result readers;
        # ``basic_ece`` makes the distinction from uncertainty risk ECE explicit.
        "basic_ece": basic_ece,
        "ece": basic_ece,
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
