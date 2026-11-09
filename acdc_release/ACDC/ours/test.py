from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

from .checkpoint import load_checkpoint
from .config import load_config
from .data import build_loader, restore_prediction_to_original
from .inference import infer_volume
from .metrics import evaluate_case
from .model import DualHeadSwinUNETR
from .reproducibility import save_run_metadata, set_reproducibility


def _configure_test_time_refinement(
    model: DualHeadSwinUNETR, config: dict[str, Any]
) -> dict[str, float]:
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
    return {
        "refine_base_alpha_max": base_alpha_max,
        "refine_strength_scale": strength_scale,
        "refine_effective_alpha_max": effective_alpha_max,
    }


def _scalar(batch: dict[str, Any], key: str, default: str = "") -> str:
    value = batch.get(key, default)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: list[Any]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def aggregate_patient_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["prediction"], row["patient_id"])].append(row)
    output: list[dict[str, Any]] = []
    excluded = {"split", "domain", "scanner_domain", "prediction", "case", "patient_id", "frame", "group"}
    for (domain, prediction, patient_id), group_rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "domain": domain,
            "scanner_domain": group_rows[0]["scanner_domain"],
            "prediction": prediction,
            "patient_id": patient_id,
            "group": group_rows[0]["group"],
            "frames": "+".join(sorted(row["frame"] for row in group_rows)),
            "n_frames": len(group_rows),
        }
        for key in group_rows[0]:
            if key not in excluded:
                try:
                    result[key] = _finite_mean([row[key] for row in group_rows])
                except (TypeError, ValueError):
                    pass
        output.append(result)
    return output


def aggregate_domain_rows(patient_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in patient_rows:
        grouped[(row["domain"], row["prediction"])].append(row)
    output: list[dict[str, Any]] = []
    excluded = {"domain", "scanner_domain", "prediction", "patient_id", "group", "frames", "n_frames"}
    for (domain, prediction), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "domain": domain,
            "scanner_domain": rows[0]["scanner_domain"],
            "prediction": prediction,
            "n_patients": len(rows),
        }
        for key in rows[0]:
            if key not in excluded:
                try:
                    result[key] = _finite_mean([row[key] for row in rows])
                except (TypeError, ValueError):
                    pass
        output.append(result)
    base_by_domain = {row["domain"]: row for row in output if row["prediction"] == "base"}
    for row in output:
        if row["prediction"] != "refined" or row["domain"] not in base_by_domain:
            continue
        base = base_by_domain[row["domain"]]
        for key, value in list(row.items()):
            if key in base and isinstance(value, (int, float)) and key != "n_patients":
                row[f"delta_{key}"] = float(value) - float(base[key])
    return output


def aggregate_source_target_comparison(domain_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair domain-equal summaries and report the target-minus-source shift."""
    by_prediction: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in domain_rows:
        by_prediction[str(row["prediction"])][str(row["domain"])] = row
    output: list[dict[str, Any]] = []
    excluded = {"domain", "scanner_domain", "prediction", "n_patients"}
    for prediction, domains in sorted(by_prediction.items()):
        source = domains.get("source_val")
        target = domains.get("target_3T")
        if source is None or target is None:
            continue
        comparison: dict[str, Any] = {
            "prediction": prediction,
            "source_domain": "source_val",
            "target_domain": "target_3T",
            "source_n_patients": source["n_patients"],
            "target_n_patients": target["n_patients"],
        }
        for key, source_value in source.items():
            target_value = target.get(key)
            if key in excluded or not isinstance(source_value, (int, float)) or not isinstance(
                target_value, (int, float)
            ):
                continue
            comparison[f"source_{key}"] = source_value
            comparison[f"target_{key}"] = target_value
            comparison[f"target_minus_source_{key}"] = float(target_value) - float(source_value)
        output.append(comparison)
    return output


def _required_cpu_tensor(
    value: torch.Tensor | None,
    name: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"Evaluation output is missing {name}")
    return value.detach().to(device="cpu", dtype=dtype)


def _class_prediction(probability: torch.Tensor) -> torch.Tensor:
    if probability.ndim != 5 or probability.shape[:2] != (1, 4):
        raise ValueError("ACDC class probability must have shape [1,4,D,H,W]")
    return probability.argmax(dim=1)[0].to(torch.uint8)


def _evaluate_batch(
    model: DualHeadSwinUNETR,
    batch: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
    output: Path,
    split: str,
    index: int,
    batch_count: int,
) -> list[dict[str, Any]]:
    evaluation = config["evaluation"]
    image = batch["image"].to(device, non_blocking=True)
    result = infer_volume(model, image, config, compute_uncertainty=True, refine=True)
    case = _scalar(batch, "subject_id", f"case_{index:04d}")
    domain = "source_val" if split == "val" else "target_3T"
    metadata = {
        "split": split,
        "domain": domain,
        "scanner_domain": _scalar(batch, "domain"),
        "case": case,
        "patient_id": _scalar(batch, "patient_id"),
        "frame": _scalar(batch, "frame"),
        "group": _scalar(batch, "group"),
    }
    spacing = tuple(float(value) for value in batch["spacing"][0].numpy())
    metric_options = {
        "spacing": spacing,
        "bins": int(evaluation.get("calibration_bins", 15)),
        "max_voxels": int(evaluation.get("max_metric_voxels", 200000)),
        "high_confidence_threshold": float(evaluation.get("high_confidence_threshold", 0.95)),
    }

    base_probability = _required_cpu_tensor(result.base_class_probability, "base class probability")
    uncertainty = _required_cpu_tensor(result.uncertainty, "uncertainty")
    base_prediction = _class_prediction(base_probability)
    refined_device = result.refined_class_probability
    if refined_device is None:
        raise RuntimeError("Evaluation output is missing refined class probability")
    base_metrics = evaluate_case(
        base_probability,
        uncertainty,
        batch["label"],
        class_prediction=base_prediction,
        risk_reference_prediction=base_prediction,
        **metric_options,
    )
    rows = [{**metadata, "prediction": "base", **base_metrics}]
    print(
        f"{domain} {index + 1}/{batch_count} {case} base "
        f"dice={base_metrics['mean_dice']:.4f} ece={base_metrics['basic_ece']:.4f}"
    )

    refined_probability = _required_cpu_tensor(refined_device, "refined class probability")
    refined_prediction = _class_prediction(refined_probability)
    refined_metrics = evaluate_case(
        refined_probability,
        uncertainty,
        batch["label"],
        class_prediction=refined_prediction,
        risk_reference_prediction=base_prediction,
        **metric_options,
    )
    rows.append({**metadata, "prediction": "refined", **refined_metrics})
    print(
        f"{domain} {index + 1}/{batch_count} {case} refined "
        f"dice={refined_metrics['mean_dice']:.4f} ece={refined_metrics['basic_ece']:.4f}"
    )

    if bool(evaluation.get("save_nifti", False)):
        restored = restore_prediction_to_original(
            refined_prediction.numpy(),
            _scalar(batch, "image_path"),
            _scalar(batch, "label_path"),
            tuple(int(value) for value in config["data"]["roi_size"]),
            tuple(float(value) for value in config["data"]["target_spacing"]),
        )
        prediction_dir = output / "nifti" / domain
        prediction_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(restored, str(prediction_dir / f"{case}.nii.gz"), True)
    return rows


@torch.inference_mode()
def run_evaluation(config: dict[str, Any], checkpoint: str, output_dir: str | None = None) -> Path:
    seed = int(config["reproducibility"]["seed"])
    set_reproducibility(seed, bool(config["reproducibility"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadSwinUNETR(config).to(device)
    payload = load_checkpoint(checkpoint, model, strict=True)
    if not model.zernike_stats.fitted:
        raise RuntimeError("Checkpoint lacks fitted source-train Zernike statistics")
    refinement = _configure_test_time_refinement(model, config)
    model.eval()
    output = Path(output_dir) if output_dir else Path(checkpoint).resolve().parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output, config)
    (output / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "stage": payload.get("stage"),
                "epoch": payload.get("epoch"),
                "classes": ["BG", "RV", "MYO", "LV"],
                "uncertainty_source": "zernike_disagreement",
                "evaluation_calibration_metric": "basic_ece",
                **refinement,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    evaluation = config["evaluation"]
    for split in evaluation["splits"]:
        loader = build_loader(config, split, "test")
        for index, batch in enumerate(loader):
            rows.extend(_evaluate_batch(model, batch, device, config, output, split, index, len(loader)))
            if device.type == "cuda" and bool(evaluation.get("release_cuda_cache", True)):
                torch.cuda.empty_cache()
    patient_rows = aggregate_patient_rows(rows)
    domain_rows = aggregate_domain_rows(patient_rows)
    comparison_rows = aggregate_source_target_comparison(domain_rows)
    _write_csv(output / "frame_metrics.csv", rows)
    _write_csv(output / "patient_metrics.csv", patient_rows)
    _write_csv(output / "domain_metrics.csv", domain_rows)
    _write_csv(output / "source_target_comparison.csv", comparison_rows)
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "n_frames": len({(row["case"], row["domain"]) for row in rows}),
        "n_patients": len({(row["patient_id"], row["domain"]) for row in patient_rows}),
        "domains": domain_rows,
        "source_target_comparison": comparison_rows,
    }
    (output / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output / "source_target_comparison.json").write_text(
        json.dumps(
            {"checkpoint": summary["checkpoint"], "comparisons": comparison_rows},
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base and refined dual-head ACDC predictions.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("configs") / "acdc.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    print(run_evaluation(config, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
