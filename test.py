from __future__ import annotations

import argparse
import csv
import json
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
from .reproducibility import save_run_metadata, set_reproducibility


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


@torch.inference_mode()
def run_evaluation(config: dict[str, Any], checkpoint: str, output_dir: str | None = None) -> Path:
    seed = int(config["reproducibility"]["seed"])
    set_reproducibility(seed, bool(config["reproducibility"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadSwinUNETR(config).to(device)
    payload = load_checkpoint(checkpoint, model, strict=True)
    if not model.zernike_stats.fitted:
        raise RuntimeError("Checkpoint does not contain fitted Zernike source statistics")
    model.eval()
    output = Path(output_dir) if output_dir else Path(checkpoint).resolve().parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output, config)
    with (output / "checkpoint_metadata.json").open("w") as stream:
        json.dump(
            {"checkpoint": str(Path(checkpoint).resolve()), "stage": payload.get("stage"), "epoch": payload.get("epoch")},
            stream,
            indent=2,
        )
    rows: list[dict[str, Any]] = []
    evaluation = config["evaluation"]
    for split in evaluation["splits"]:
        domain = "source_val" if split == "val" else split.removeprefix("test_")
        loader = build_loader(config, split, "eval")
        for index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            result = infer_volume(model, image, config, compute_uncertainty=True, refine=True)
            case = _case_name(batch, index)
            for prediction_name, probability, region_probability in (
                ("base", result.base_atomic_probability, result.base_region_probability),
                ("refined", result.refined_atomic_probability, result.refined_region_probability),
            ):
                metrics = evaluate_case(
                    probability,
                    result.uncertainty,
                    batch["label_scalar"],
                    region_probability=region_probability,
                    risk_reference_probability=result.base_atomic_probability,
                    bins=int(evaluation.get("calibration_bins", 15)),
                    max_voxels=int(evaluation.get("max_metric_voxels", 200000)),
                    high_confidence_threshold=float(evaluation.get("high_confidence_threshold", 0.95)),
                )
                rows.append({"domain": domain, "prediction": prediction_name, "case": case, **metrics})
                print(f"{domain} {index + 1}/{len(loader)} {case} {prediction_name} dice={metrics['mean_dice']:.4f}")
            if bool(evaluation.get("save_nifti", False)):
                label = result.refined_atomic_probability.argmax(dim=1)[0].cpu()
                scalar = atomic_label_to_scalar(label).numpy().astype(np.uint8)
                prediction_dir = output / "nifti" / domain
                prediction_dir.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(scalar, _affine(batch)), prediction_dir / f"{case}.nii.gz")
            del metrics, probability, region_probability, result, image, batch
    summary = _summary(rows)
    _write_csv(output / "case_metrics.csv", rows)
    _write_csv(output / "summary_metrics.csv", summary)
    with (output / "summary_metrics.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
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
