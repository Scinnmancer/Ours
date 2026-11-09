"""ACDC metrics using the current Ours calibration implementation."""
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ECE_SOURCE = (Path(__file__).resolve().parents[1]
              / "directional_zernike_project/ours/tools/fixed_ece.py")
ECE_SOURCE_SHA256 = "bc204faea3a31169fb4611bd9106b85d470992c5f24d44e6c94869b864348ed1"
if hashlib.sha256(ECE_SOURCE.read_bytes()).hexdigest() != ECE_SOURCE_SHA256:
    raise RuntimeError(f"ECE source differs from the locked 2026-11-06 implementation: {ECE_SOURCE}")

_spec = spec_from_file_location("acdc_ours_fixed_ece", ECE_SOURCE)
_fixed = module_from_spec(_spec)
_spec.loader.exec_module(_fixed)

FixedECEAccumulator = _fixed.FixedECEAccumulator
support_sha256 = _fixed.support_sha256
target_sha256 = _fixed.target_sha256
CLASSES = ("RV", "MYO", "LV")
METRICS = ("Dice", "dice_RV", "dice_MYO", "dice_LV", "ECE",
           "fixed_foreground_NLL", "fixed_foreground_Brier", "high_confidence_error_fraction")
EXTRA_METRICS = ("HD95", *(f"hd95_{name}" for name in CLASSES), "AURC",
                 *(f"miss_{name}" for name in CLASSES))
UEO_THRESHOLDS = np.arange(76, dtype=np.float64) / 100


def error_auroc(probability, labels, support):
    p = probability[:, support]
    return rank_auroc(-p.max(axis=0), p.argmax(axis=0) != labels[support])


def rank_auroc(scores, error):
    from scipy.stats import rankdata

    if not np.isfinite(scores).all():
        raise ValueError("AUROC requires finite scores")
    positives = int(error.sum())
    negatives = len(error) - positives
    if not positives or not negatives:
        return None
    # Ranking negative confidence is equivalent to ranking 1 - confidence,
    # without rounding two distinct confidences into a tie.
    ranks = rankdata(scores, method="average")
    return float((ranks[error].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def uncertainty_error_overlap(probability, labels, support, thresholds):
    p = probability[:, support]
    uncertainty = 1.0 - p.max(axis=0).astype(np.float64)
    error = p.argmax(axis=0) != labels[support]
    bins = np.searchsorted(thresholds, uncertainty, side="right")
    counts = np.bincount(bins, minlength=len(thresholds) + 1)
    hits = np.bincount(bins[error], minlength=len(thresholds) + 1)
    denominator = counts[:0:-1].cumsum()[::-1] + error.sum()
    overlap = 2 * hits[:0:-1].cumsum()[::-1]
    return np.divide(overlap, denominator, out=np.ones(len(thresholds)), where=denominator > 0)


def risk_coverage_area(probability, labels, support):
    p = probability[:, support]
    confidence = p.max(axis=0)
    error = p.argmax(axis=0) != labels[support]
    order = np.argsort(-confidence)
    ranked = confidence[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, ranked.size]
    weights = np.diff(ends, prepend=0) / ranked.size
    return float(weights @ (np.cumsum(error[order])[ends - 1] / ends))


def extended_score(probability, labels, valid, foreground, record, roi, spacing):
    import SimpleITK as sitk
    from ours.data import restore_prediction_to_original
    from ours.metrics import hd95_per_region

    restored = restore_prediction_to_original(
        probability.argmax(0), record["image"], record["label"], roi, spacing)
    truth = sitk.ReadImage(record["label"])
    if not restored.IsSameImageGeometryAs(truth):
        raise ValueError("HD95 requires matching native prediction and label geometry")
    classes = np.arange(1, 4)[:, None, None, None]
    predicted = sitk.GetArrayFromImage(restored) == classes
    target = sitk.GetArrayFromImage(truth) == classes
    hd95 = hd95_per_region(predicted, target, tuple(reversed(truth.GetSpacing())))
    result = {}
    for name, distance, pred, label in zip(CLASSES, hd95, predicted, target):
        present = bool(label.any())
        detected = bool(pred.any())
        result[f"hd95_{name}"] = distance if present and detected else None
        result[f"miss_{name}"] = int(not detected) if present else None
    result["HD95"] = average(result[f"hd95_{name}"] for name in CLASSES)
    result["AURC"] = risk_coverage_area(probability, labels, fixed_support(labels, valid, foreground))
    return result


def fixed_support(labels, valid, reference_foreground):
    if (valid.shape != labels.shape or reference_foreground.shape != labels.shape
            or valid.dtype != np.bool_ or reference_foreground.dtype != np.bool_):
        raise ValueError("Labels, valid image field and frozen foreground must share a grid")
    if np.any((labels > 0) & ~valid):
        raise ValueError("Ground-truth foreground falls outside the valid image field")
    return valid & ((labels > 0) | reference_foreground)


def average(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def score(probability, labels, valid, reference_foreground):
    if labels.ndim != 3 or probability.shape != (4, *labels.shape):
        raise ValueError("Expected ACDC four-class probabilities on a 3D label grid")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("ACDC ground-truth classes must be integer labels")
    support = fixed_support(labels, valid, reference_foreground)
    accumulator = FixedECEAccumulator()
    for start in range(0, labels.shape[0], 16):
        slab = slice(start, start + 16)
        accumulator.add(probability[:, slab], labels[slab], support[slab])
    prediction = probability.argmax(0)
    dice = {}
    for index, name in enumerate(CLASSES, start=1):
        predicted = valid & (prediction == index)
        truth = labels == index
        denominator = int(predicted.sum()) + int(truth.sum())
        dice[f"dice_{name}"] = 2 * int((predicted & truth).sum()) / denominator if denominator else None
    calibration = accumulator.result()
    calibration["ece_support"] = "valid_image & (GT foreground | frozen baseline atomic foreground)"
    return {"Dice": average(dice.values()), **dice, **calibration,
            "ece_support_sha256": support_sha256(support),
            "ece_target_sha256": target_sha256(labels)}
