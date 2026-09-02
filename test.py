from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config
from .data import build_loader
from .inference import infer_volume
from .metrics import evaluate_case
from .model import DualHeadSwinUNETR
from .probability import atomic_label_to_scalar
from .refine_audit import (
    ATOMIC_CHANGE_CODES,
    audit_refinement,
    probability_snapshot,
    save_audit_nifti,
    summarize_audits,
)
from .reproducibility import save_run_metadata, set_reproducibility


def _configure_test_time_refinement(
    model: DualHeadSwinUNETR, config: dict[str, Any]
) -> dict[str, float]:
    """Increase label-transfer strength for evaluation without changing checkpoints."""
    base_alpha_max = float(model.label_transfer.alpha_max)
    strength_scale = float(config["evaluation"].get("refine_strength_scale", 1.0))
    effective_alpha_max = base_alpha_max * strength_scale
    if (
        not math.isfinite(strength_scale)
        or strength_scale <= 0.0
        or not 0.0 <= effective_alpha_max < 1.0
    ):
        raise ValueError(
            "Effective test-time refine alpha_max must be in [0, 1); "
            f"received {base_alpha_max} * {strength_scale} = {effective_alpha_max}"
        )
    model.label_transfer.alpha_max = effective_alpha_max
    neighborhood_radius = int(
        config["evaluation"].get(
            "refine_neighborhood_radius", model.label_transfer.radius
        )
    )
    neighborhood_sigma = float(
        config["evaluation"].get(
            "refine_neighborhood_sigma", model.label_transfer.sigma
        )
    )
    model.label_transfer.set_neighborhood(neighborhood_radius, neighborhood_sigma)
    neighbor_reliability_power = float(
        config["evaluation"].get("refine_neighbor_reliability_power", 1.0)
    )
    model.label_transfer.set_neighbor_reliability_power(neighbor_reliability_power)
    return {
        "refine_base_alpha_max": base_alpha_max,
        "refine_strength_scale": strength_scale,
        "refine_effective_alpha_max": effective_alpha_max,
        "refine_neighborhood_radius": neighborhood_radius,
        "refine_neighborhood_sigma": neighborhood_sigma,
        "refine_neighbor_reliability_power": neighbor_reliability_power,
    }


def _case_name(batch: dict[str, Any], index: int) -> str:
    subject = batch.get("subject_id")
    if isinstance(subject, (list, tuple)):
        return str(subject[0])
    if subject is not None:
        return str(subject)
    return f"case_{index:04d}"


def _affine(batch: dict[str, Any]) -> np.ndarray:
    metadata = batch.get("image_meta_dict", {})
    for key in ("affine", "original_affine"):
        value = metadata.get(key)
        if value is not None:
            value = value[0] if getattr(value, "ndim", 0) == 3 else value
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    image = batch.get("image")
    if hasattr(image, "meta"):
        value = image.meta.get("affine", image.meta.get("original_affine"))
        if value is not None:
            value = value[0] if getattr(value, "ndim", 0) == 3 else value
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return np.eye(4)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["domain"], row["prediction"]), []).append(row)
    summaries = []
    excluded = {"domain", "prediction", "case"}
    for (domain, prediction), group in groups.items():
        summary: dict[str, Any] = {"domain": domain, "prediction": prediction, "n_cases": len(group)}
        for key in group[0]:
            if key in excluded:
                continue
            values = np.asarray([item.get(key, np.nan) for item in group], dtype=float)
            summary[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        summaries.append(summary)
    base_by_domain = {item["domain"]: item for item in summaries if item["prediction"] == "base"}
    for item in summaries:
        if item["prediction"] != "refined" or item["domain"] not in base_by_domain:
            continue
        base = base_by_domain[item["domain"]]
        for key in (
            "mean_dice",
            "mean_hd95",
            "basic_ece",
            "ece",
            "brier",
            "risk_ece",
            "risk_brier",
            "auroc_error",
            "aupr_error",
            "prr",
            "high_confidence_error_rate",
        ):
            if key in item and key in base:
                item[f"delta_{key}"] = item[key] - base[key]
    return summaries


def _required_cpu_tensor(value: torch.Tensor | None, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"Evaluation output is missing {name}")
    return value.detach().to(device="cpu", dtype=dtype)


def _evaluate_batch(
    model: DualHeadSwinUNETR,
    batch: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
    output: Path,
    domain: str,
    index: int,
    batch_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    evaluation = config["evaluation"]
    image = batch["image"].to(device, non_blocking=True)
    result = infer_volume(model, image, config, compute_uncertainty=True, refine=True)
    case = _case_name(batch, index)
    audit_config = evaluation.get("refine_audit", {})
    audit_enabled = bool(audit_config.get("enabled", False))
    audit_save_nifti = audit_enabled and bool(audit_config.get("save_nifti", False))
    affine = (
        _affine(batch)
        if bool(evaluation.get("save_nifti", False)) or audit_save_nifti
        else None
    )

    # Evaluate base before transferring refined probabilities. This prevents
    # two full four-channel volumes from residing in host memory together.
    base_region_prediction = _required_cpu_tensor(
        result.base_region_probability >= 0.5, "base region prediction", torch.uint8
    )[0]
    base_atomic_probability = _required_cpu_tensor(
        result.base_atomic_probability, "base atomic probability"
    )
    base_snapshot = probability_snapshot(base_atomic_probability)
    uncertainty = _required_cpu_tensor(result.uncertainty, "uncertainty")
    base_atomic_prediction = base_snapshot.atomic_prediction
    refined_atomic_device = result.refined_atomic_probability
    if refined_atomic_device is None:
        raise RuntimeError("Evaluation output is missing refined atomic probability")
    del result, image
    if device.type == "cuda" and bool(evaluation.get("release_cuda_cache", True)):
        torch.cuda.empty_cache()

    metric_options = {
        "bins": int(evaluation.get("calibration_bins", 15)),
        "max_voxels": int(evaluation.get("max_metric_voxels", 200000)),
        "high_confidence_threshold": float(evaluation.get("high_confidence_threshold", 0.95)),
    }
    base_metrics = evaluate_case(
        base_atomic_probability,
        uncertainty,
        batch["label_scalar"],
        atomic_prediction=base_atomic_prediction,
        region_prediction=base_region_prediction,
        risk_reference_prediction=base_atomic_prediction,
        **metric_options,
    )
    case_rows: list[dict[str, Any]] = [
        {"domain": domain, "prediction": "base", "case": case, **base_metrics}
    ]
    print(
        f"{domain} {index + 1}/{batch_count} {case} "
        f"base dice={base_metrics['mean_dice']:.4f} "
        f"basic_ece={base_metrics['basic_ece']:.4f}"
    )
    del base_atomic_probability, base_region_prediction, base_metrics

    refined_atomic_probability = _required_cpu_tensor(
        refined_atomic_device, "refined atomic probability"
    )
    del refined_atomic_device
    refined_snapshot = probability_snapshot(refined_atomic_probability)
    refined_atomic_prediction = refined_snapshot.atomic_prediction
    refined_region_prediction = refined_snapshot.region_prediction
    if device.type == "cuda" and bool(evaluation.get("release_cuda_cache", True)):
        torch.cuda.empty_cache()
    refined_metrics = evaluate_case(
        refined_atomic_probability,
        uncertainty,
        batch["label_scalar"],
        atomic_prediction=refined_atomic_prediction,
        region_prediction=refined_region_prediction,
        risk_reference_prediction=base_atomic_prediction,
        **metric_options,
    )
    case_rows.append(
        {"domain": domain, "prediction": "refined", "case": case, **refined_metrics}
    )
    print(
        f"{domain} {index + 1}/{batch_count} {case} "
        f"refined dice={refined_metrics['mean_dice']:.4f} "
        f"basic_ece={refined_metrics['basic_ece']:.4f}"
    )
    if bool(evaluation.get("save_nifti", False)):
        scalar = atomic_label_to_scalar(refined_atomic_prediction).numpy().astype(np.uint8)
        prediction_dir = output / "nifti" / domain
        prediction_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(scalar, affine), prediction_dir / f"{case}.nii.gz")
    audit_rows: list[dict[str, Any]] = []
    atomic_transition_rows: list[dict[str, Any]] = []
    region_transition_rows: list[dict[str, Any]] = []
    if audit_enabled:
        audit = audit_refinement(base_snapshot, refined_snapshot, batch["label_scalar"])
        audit_rows.append({"domain": domain, "case": case, **audit.metrics})
        atomic_transition_rows.extend(
            {"domain": domain, "case": case, **row} for row in audit.atomic_transitions
        )
        region_transition_rows.extend(
            {"domain": domain, "case": case, **row} for row in audit.region_transitions
        )
        if audit_save_nifti:
            audit_dir = output / "refine_audit_maps" / domain
            save_audit_nifti(audit, affine, audit_dir, case)
    return case_rows, audit_rows, atomic_transition_rows, region_transition_rows


@torch.inference_mode()
def run_evaluation(config: dict[str, Any], checkpoint: str, output_dir: str | None = None) -> Path:
    seed = int(config["reproducibility"]["seed"])
    set_reproducibility(seed, bool(config["reproducibility"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadSwinUNETR(config).to(device)
    payload = load_checkpoint(checkpoint, model, strict=True)
    if not model.zernike_stats.fitted:
        raise RuntimeError(
            "Checkpoint does not contain fitted Zernike source statistics; "
            "run the stats or calibration stage first"
        )
    refinement = _configure_test_time_refinement(model, config)
    model.eval()
    output = Path(output_dir) if output_dir else Path(checkpoint).resolve().parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output, config)
    with (output / "checkpoint_metadata.json").open("w") as stream:
        json.dump(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "stage": payload.get("stage"),
                "epoch": payload.get("epoch"),
                "uncertainty_source": "zernike_disagreement",
                "evaluation_calibration_metric": "basic_ece",
                "refine_audit_enabled": bool(
                    config["evaluation"].get("refine_audit", {}).get("enabled", False)
                ),
                "refine_audit_change_codes": ATOMIC_CHANGE_CODES,
                **refinement,
            },
            stream,
            indent=2,
        )
    del payload
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    atomic_transition_rows: list[dict[str, Any]] = []
    region_transition_rows: list[dict[str, Any]] = []
    evaluation = config["evaluation"]
    for split in evaluation["splits"]:
        domain = "source_val" if split == "val" else split.removeprefix("test_")
        loader = build_loader(config, split, "test")
        for index, batch in enumerate(loader):
            try:
                (
                    batch_rows,
                    batch_audits,
                    batch_atomic_transitions,
                    batch_region_transitions,
                ) = _evaluate_batch(
                    model,
                    batch,
                    device,
                    config,
                    output,
                    domain,
                    index,
                    len(loader),
                )
                rows.extend(batch_rows)
                audit_rows.extend(batch_audits)
                atomic_transition_rows.extend(batch_atomic_transitions)
                region_transition_rows.extend(batch_region_transitions)
            finally:
                del batch
                if device.type == "cuda" and bool(evaluation.get("release_cuda_cache", True)):
                    torch.cuda.empty_cache()
    summary = _summary(rows)
    _write_csv(output / "case_metrics.csv", rows)
    _write_csv(output / "summary_metrics.csv", summary)
    with (output / "summary_metrics.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
    if audit_rows:
        audit_summary = summarize_audits(audit_rows)
        _write_csv(output / "refine_audit_case.csv", audit_rows)
        _write_csv(output / "refine_audit_summary.csv", audit_summary)
        _write_csv(output / "refine_atomic_transitions.csv", atomic_transition_rows)
        _write_csv(output / "refine_region_transitions.csv", region_transition_rows)
        with (output / "refine_audit_summary.json").open("w") as stream:
            json.dump(audit_summary, stream, indent=2, allow_nan=False)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base and refined dual-head BraTS predictions.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("configs") / "brats2020.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    print(run_evaluation(config, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
