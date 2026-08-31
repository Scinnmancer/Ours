from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config, validate_config
from .data import build_loader
from .inference import infer_volume
from .metrics import (
    _sample_mask_indices,
    dice_per_region,
    expected_calibration_error,
)
from .model import DualHeadSwinUNETR
from .reproducibility import save_run_metadata, set_reproducibility
from .test import (
    _atomic_prediction,
    _case_name,
    _region_prediction_from_atomic,
    _required_cpu_tensor,
    _summary,
    _write_csv,
    run_evaluation,
)


_SWEEP_SIGNATURE = "refine_sweep_input.json"
_SWEEP_VERSION = 2


def candidate_name(beta: float, strength_scale: float) -> str:
    beta_text = f"{beta:g}".replace(".", "p")
    scale_text = f"{strength_scale:g}".replace(".", "p")
    return f"beta_{beta_text}__scale_{scale_text}"


def build_candidates(config: dict[str, Any]) -> list[dict[str, float | str]]:
    tuning = config["evaluation"].get("refine_tuning", {})
    beta_values = [float(value) for value in tuning.get("beta_values", [1.0, 1.5, 2.0])]
    strength_scales = [
        float(value)
        for value in tuning.get("strength_scales", [1.0, 1.5, 2.0, 2.4])
    ]
    alpha_max = float(config["label_transfer"].get("alpha_max", 0.35))
    candidates: list[dict[str, float | str]] = []
    seen: set[tuple[float, float]] = set()
    for beta in beta_values:
        for strength_scale in strength_scales:
            key = (beta, strength_scale)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "name": candidate_name(beta, strength_scale),
                    "beta": beta,
                    "refine_strength_scale": strength_scale,
                    "effective_alpha_max": alpha_max * strength_scale,
                }
            )
    return candidates


def select_candidate(
    candidates: list[dict[str, Any]],
    dice_tolerance: float,
    ece_tie_tolerance: float = 0.001,
) -> dict[str, Any] | None:
    """Minimize ordinary ECE after a Dice guardrail, with a stable ECE tie band."""
    best: dict[str, Any] | None = None
    for candidate in candidates:
        base_dice = float(candidate["base_mean_dice"])
        refined_dice = float(candidate["mean_dice"])
        ece = float(candidate["basic_ece"])
        eligible = (
            math.isfinite(base_dice)
            and math.isfinite(refined_dice)
            and math.isfinite(ece)
            and refined_dice >= base_dice - dice_tolerance
        )
        candidate["eligible"] = eligible
        if not eligible:
            continue
        if best is None:
            best = candidate
            continue
        best_ece = float(best["basic_ece"])
        best_dice = float(best["mean_dice"])
        if ece < best_ece - ece_tie_tolerance or (
            abs(ece - best_ece) <= ece_tie_tolerance
            and refined_dice > best_dice
        ):
            best = candidate
    return best


def _evaluate_selection_case(
    atomic_probability: torch.Tensor,
    scalar_target: torch.Tensor,
    atomic_prediction: torch.Tensor,
    region_prediction: torch.Tensor,
    *,
    bins: int,
    max_voxels: int,
    high_confidence_threshold: float = 0.95,
    reference_prediction: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute selection metrics and confidence diagnostics for one case."""
    probability = atomic_probability.detach().float().cpu()
    if probability.ndim == 5:
        probability = probability[0]
    scalar = scalar_target.detach().cpu()
    if scalar.ndim == 5:
        scalar = scalar[0]
    if scalar.ndim == 4 and scalar.shape[0] == 1:
        scalar = scalar[0]
    atomic_target_array = scalar.numpy().astype(np.uint8, copy=True)
    if not bool(np.isin(atomic_target_array, (0, 1, 2, 4)).all()):
        raise ValueError("Source validation contains invalid BraTS labels")
    atomic_target_array[atomic_target_array == 4] = 3
    region_target = np.stack(
        (
            (atomic_target_array == 1) | (atomic_target_array == 3),
            atomic_target_array > 0,
            atomic_target_array == 3,
        )
    ).astype(np.uint8, copy=False)
    prediction = atomic_prediction.detach().cpu().to(torch.uint8)
    regions = region_prediction.detach().cpu().numpy().astype(np.uint8, copy=False)
    dice = dice_per_region(regions, region_target)
    atomic_target = torch.as_tensor(atomic_target_array, dtype=torch.uint8)
    mask = (atomic_target > 0) | (prediction > 0)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)
    indices = _sample_mask_indices(mask, max_voxels)
    sampled_probability = probability.reshape(4, -1)[:, indices]
    sampled_target = atomic_target.reshape(-1)[indices]
    sampled_prediction = prediction.reshape(-1)[indices]
    confidence_tensor = torch.amax(sampled_probability, dim=0)
    confidence = confidence_tensor.numpy()
    correct_tensor = sampled_prediction == sampled_target
    correct = correct_tensor.numpy().astype(np.float32)
    basic_ece = expected_calibration_error(confidence, correct, bins)
    entropy = -(
        sampled_probability.clamp_min(torch.finfo(sampled_probability.dtype).eps)
        * sampled_probability.clamp_min(torch.finfo(sampled_probability.dtype).eps).log()
    ).sum(dim=0)
    errors = ~correct_tensor
    error_top_confidence = (
        float(confidence_tensor[errors].mean()) if bool(errors.any()) else float("nan")
    )
    if reference_prediction is None:
        top1_flip_rate = 0.0
    else:
        reference = reference_prediction.detach().cpu().to(torch.uint8)
        if reference.ndim == 4 and reference.shape[0] == 1:
            reference = reference[0]
        if reference.shape != prediction.shape:
            raise ValueError("reference_prediction must match atomic_prediction")
        sampled_reference = reference.reshape(-1)[indices]
        top1_flip_rate = float((sampled_prediction != sampled_reference).float().mean())
    return {
        "mean_dice": float(np.nanmean(dice)),
        "basic_ece": basic_ece,
        "ece": basic_ece,
        "mean_top_confidence": float(confidence_tensor.mean()),
        "p95_top_confidence": float(torch.quantile(confidence_tensor, 0.95)),
        "mean_error_top_confidence": error_top_confidence,
        "mean_entropy": float(entropy.mean()),
        "high_confidence_error_rate": float(
            ((confidence_tensor >= high_confidence_threshold) & errors).float().mean()
        ),
        "top1_flip_rate": top1_flip_rate,
        **{name: value for name, value in zip(("dice_TC", "dice_WT", "dice_ET"), dice)},
    }


def _write_json(path: Path, value: Any, *, allow_nan: bool = False) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=allow_nan)


def _write_candidate_result(
    output: Path,
    config: dict[str, Any],
    checkpoint: str,
    candidate: dict[str, Any],
    base_rows: list[dict[str, Any]],
    refined_rows: list[dict[str, Any]],
) -> None:
    candidate_output = output / "source_val" / str(candidate["name"])
    candidate_output.mkdir(parents=True, exist_ok=True)
    rows = [*base_rows, *refined_rows]
    summaries = _summary(rows)
    _write_csv(candidate_output / "case_metrics.csv", rows)
    _write_csv(candidate_output / "summary_metrics.csv", summaries)
    _write_json(candidate_output / "summary_metrics.json", summaries, allow_nan=True)
    _write_json(
        candidate_output / "refine_metadata.json",
        {
            "checkpoint": str(Path(checkpoint).resolve()),
            "selection_split": "source_val",
            "selection_metric": "basic_ece",
            "beta": candidate["beta"],
            "refine_strength_scale": candidate["refine_strength_scale"],
            "base_alpha_max": float(config["label_transfer"].get("alpha_max", 0.35)),
            "effective_alpha_max": candidate["effective_alpha_max"],
            "iterations": int(config["label_transfer"].get("iterations", 3)),
        },
    )


@torch.inference_mode()
def run_source_sweep(
    config: dict[str, Any],
    checkpoint: str,
    output: Path,
    candidates: list[dict[str, Any]],
) -> None:
    """Infer source validation once, then apply every refine parameter candidate."""
    seed = int(config["reproducibility"]["seed"])
    set_reproducibility(
        seed, bool(config["reproducibility"].get("deterministic", True))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadSwinUNETR(config).to(device)
    payload = load_checkpoint(checkpoint, model, strict=True)
    if not model.zernike_stats.fitted:
        raise RuntimeError(
            "Checkpoint does not contain fitted Zernike source statistics; "
            "run the stats or calibration stage first"
        )
    model.eval()
    del payload

    base_alpha_max = float(model.label_transfer.alpha_max)
    evaluation = config["evaluation"]
    bins = int(evaluation.get("calibration_bins", 15))
    max_voxels = int(evaluation.get("max_metric_voxels", 200000))
    high_confidence_threshold = float(
        evaluation.get("high_confidence_threshold", 0.95)
    )
    loader = build_loader(config, "val", "test")
    base_rows: list[dict[str, Any]] = []
    refined_rows: dict[str, list[dict[str, Any]]] = {
        str(candidate["name"]): [] for candidate in candidates
    }

    for index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        result = infer_volume(
            model, image, config, compute_uncertainty=True, refine=False
        )
        case = _case_name(batch, index)
        base_atomic_device = result.base_atomic_probability
        uncertainty_device = result.uncertainty
        if base_atomic_device is None or uncertainty_device is None:
            raise RuntimeError("Source sweep requires base atomic probability and uncertainty")

        base_region_prediction = _required_cpu_tensor(
            result.base_region_probability >= 0.5,
            "base region prediction",
            torch.uint8,
        )[0]
        base_atomic_probability = _required_cpu_tensor(
            base_atomic_device, "base atomic probability"
        )
        base_atomic_prediction = _atomic_prediction(base_atomic_probability)
        base_metrics = _evaluate_selection_case(
            base_atomic_probability,
            batch["label_scalar"],
            base_atomic_prediction,
            base_region_prediction,
            bins=bins,
            max_voxels=max_voxels,
            high_confidence_threshold=high_confidence_threshold,
        )
        base_rows.append(
            {
                "domain": "source_val",
                "prediction": "base",
                "case": case,
                **base_metrics,
            }
        )

        for candidate in candidates:
            name = str(candidate["name"])
            model.label_transfer.beta = float(candidate["beta"])
            model.label_transfer.alpha_max = (
                base_alpha_max * float(candidate["refine_strength_scale"])
            )
            refined_device = model.label_transfer(
                base_atomic_device, uncertainty_device
            )
            refined_probability = _required_cpu_tensor(
                refined_device, "refined atomic probability"
            )
            refined_atomic_prediction = _atomic_prediction(refined_probability)
            refined_region_prediction = _region_prediction_from_atomic(
                refined_probability
            )
            refined_metrics = _evaluate_selection_case(
                refined_probability,
                batch["label_scalar"],
                refined_atomic_prediction,
                refined_region_prediction,
                bins=bins,
                max_voxels=max_voxels,
                high_confidence_threshold=high_confidence_threshold,
                reference_prediction=base_atomic_prediction,
            )
            refined_rows[name].append(
                {
                    "domain": "source_val",
                    "prediction": "refined",
                    "case": case,
                    **refined_metrics,
                }
            )
            del refined_device, refined_probability

        print(
            f"[refine-tuning] source_val {index + 1}/{len(loader)} {case} "
            f"evaluated {len(candidates)} candidates",
            flush=True,
        )
        del result, image, base_atomic_device, uncertainty_device
        del base_atomic_probability, base_region_prediction
        if device.type == "cuda" and bool(evaluation.get("release_cuda_cache", True)):
            torch.cuda.empty_cache()

    model.label_transfer.alpha_max = base_alpha_max
    for candidate in candidates:
        _write_candidate_result(
            output,
            config,
            checkpoint,
            candidate,
            base_rows,
            refined_rows[str(candidate["name"])],
        )


def _source_rows(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with summary_path.open() as stream:
        rows = json.load(stream)
    source = {
        row["prediction"]: row
        for row in rows
        if row.get("domain") == "source_val"
    }
    if "base" not in source or "refined" not in source:
        raise RuntimeError(f"Missing source_val base/refined rows in {summary_path}")
    return source["base"], source["refined"]


def _collect_candidate_metrics(
    output: Path, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for candidate in candidates:
        summary_path = (
            output / "source_val" / str(candidate["name"]) / "summary_metrics.json"
        )
        base, refined = _source_rows(summary_path)
        collected.append(
            {
                **candidate,
                "base_mean_dice": float(base["mean_dice"]),
                "base_basic_ece": float(base["basic_ece"]),
                "mean_dice": float(refined["mean_dice"]),
                "basic_ece": float(refined["basic_ece"]),
                "delta_mean_dice": float(refined["mean_dice"])
                - float(base["mean_dice"]),
                "delta_basic_ece": float(refined["basic_ece"])
                - float(base["basic_ece"]),
                "mean_top_confidence": float(refined["mean_top_confidence"]),
                "delta_mean_top_confidence": float(refined["mean_top_confidence"])
                - float(base["mean_top_confidence"]),
                "p95_top_confidence": float(refined["p95_top_confidence"]),
                "delta_p95_top_confidence": float(refined["p95_top_confidence"])
                - float(base["p95_top_confidence"]),
                "mean_error_top_confidence": float(
                    refined["mean_error_top_confidence"]
                ),
                "delta_mean_error_top_confidence": float(
                    refined["mean_error_top_confidence"]
                )
                - float(base["mean_error_top_confidence"]),
                "mean_entropy": float(refined["mean_entropy"]),
                "delta_mean_entropy": float(refined["mean_entropy"])
                - float(base["mean_entropy"]),
                "high_confidence_error_rate": float(
                    refined["high_confidence_error_rate"]
                ),
                "delta_high_confidence_error_rate": float(
                    refined["high_confidence_error_rate"]
                )
                - float(base["high_confidence_error_rate"]),
                "top1_flip_rate": float(refined["top1_flip_rate"]),
                "output": str(summary_path.parent),
            }
        )
    return collected


def _run_signature(
    config: dict[str, Any], checkpoint: str, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_stat = checkpoint_path.stat()
    payload = {
        "version": _SWEEP_VERSION,
        "config": config,
        "candidates": candidates,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": _SWEEP_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _can_reuse_sweep(
    output: Path,
    candidates: list[dict[str, Any]],
    signature: dict[str, Any],
    force: bool,
) -> bool:
    if force:
        return False
    signature_path = output / _SWEEP_SIGNATURE
    if not signature_path.is_file():
        return False
    try:
        if json.loads(signature_path.read_text()) != signature:
            return False
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (
            output
            / "source_val"
            / str(candidate["name"])
            / "summary_metrics.json"
        ).is_file()
        for candidate in candidates
    )


def _write_candidates_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "beta",
        "refine_strength_scale",
        "effective_alpha_max",
        "base_mean_dice",
        "base_basic_ece",
        "mean_dice",
        "basic_ece",
        "delta_mean_dice",
        "delta_basic_ece",
        "mean_top_confidence",
        "delta_mean_top_confidence",
        "p95_top_confidence",
        "delta_p95_top_confidence",
        "mean_error_top_confidence",
        "delta_mean_error_top_confidence",
        "mean_entropy",
        "delta_mean_entropy",
        "high_confidence_error_rate",
        "delta_high_confidence_error_rate",
        "top1_flip_rate",
        "selection_tier",
        "eligible",
        "output",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: candidate.get(key) for key in fieldnames}
            for candidate in candidates
        )


def run_tuning(
    config_path: str,
    checkpoint: str,
    output_dir: str,
    *,
    run_final: bool = True,
    force: bool = False,
) -> Path:
    config = load_config(config_path)
    candidates = build_candidates(config)
    tuning = config["evaluation"].get("refine_tuning", {})
    dice_tolerance = float(tuning.get("dice_tolerance", 0.0))
    fallback_dice_tolerance = float(tuning.get("fallback_dice_tolerance", 0.0002))
    ece_tie_tolerance = float(tuning.get("ece_tie_tolerance", 0.001))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output, config)

    signature = _run_signature(config, checkpoint, candidates)
    if _can_reuse_sweep(output, candidates, signature, force):
        print("[refine-tuning] reusing completed source-validation sweep", flush=True)
    else:
        print(
            f"[refine-tuning] inferring source validation once for "
            f"{len(candidates)} refine candidates",
            flush=True,
        )
        run_source_sweep(config, checkpoint, output, candidates)
        _write_json(output / _SWEEP_SIGNATURE, signature)

    collected = _collect_candidate_metrics(output, candidates)
    best = select_candidate(collected, dice_tolerance, ece_tie_tolerance)
    selection_tier = "strict"
    if best is None and fallback_dice_tolerance > dice_tolerance:
        best = select_candidate(
            collected, fallback_dice_tolerance, ece_tie_tolerance
        )
        selection_tier = "diagnostic_fallback"
    for candidate in collected:
        candidate["selection_tier"] = (
            selection_tier if candidate is best else "not_selected"
        )
    _write_candidates_csv(output / "candidates.csv", collected)
    _write_json(output / "candidates.json", collected, allow_nan=False)
    final_output = output / f"final__{best['name']}" if best is not None else None
    selection = {
        "selection_split": "source_val",
        "selection_metric": "basic_ece",
        "selection_rule": (
            "minimum basic_ece after Dice guardrail; higher Dice breaks an ECE "
            "difference within ece_tie_tolerance"
        ),
        "dice_tolerance": dice_tolerance,
        "fallback_dice_tolerance": fallback_dice_tolerance,
        "ece_tie_tolerance": ece_tie_tolerance,
        "selection_tier": selection_tier if best is not None else "none",
        "checkpoint": str(Path(checkpoint).resolve()),
        "candidates": collected,
        "best": best,
        "final_output": str(final_output) if final_output is not None else None,
        "final_evaluation_requested": bool(run_final),
    }
    selection_path = output / "selection.json"
    _write_json(selection_path, selection)
    if best is None:
        raise RuntimeError(
            f"No refine candidate passed either Dice guardrail; see {selection_path}"
        )

    print(
        "[refine-tuning] selected "
        f"beta={float(best['beta']):g}, "
        f"scale={float(best['refine_strength_scale']):g}, "
        f"source_val ECE={float(best['basic_ece']):.6f}, "
        f"Dice={float(best['mean_dice']):.6f}, "
        f"tier={selection_tier}",
        flush=True,
    )
    if run_final and final_output is not None:
        final_config = copy.deepcopy(config)
        final_config["label_transfer"]["beta"] = float(best["beta"])
        final_config["evaluation"]["refine_strength_scale"] = float(
            best["refine_strength_scale"]
        )
        validate_config(final_config)
        final_signature = _run_signature(final_config, checkpoint, [best])
        final_signature_path = final_output / _SWEEP_SIGNATURE
        reuse_final = False
        if not force and (final_output / "summary_metrics.json").is_file():
            try:
                reuse_final = (
                    json.loads(final_signature_path.read_text()) == final_signature
                )
            except (OSError, json.JSONDecodeError):
                reuse_final = False
        if reuse_final:
            print("[refine-tuning] reusing completed final evaluation", flush=True)
        else:
            print(
                "[refine-tuning] evaluating the selected parameters on configured splits",
                flush=True,
            )
            run_evaluation(final_config, checkpoint, str(final_output))
            _write_json(final_signature_path, final_signature)
        selection["final_output"] = str(final_output)
        _write_json(selection_path, selection)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search refine-only beta/strength parameters on source validation, "
            "then evaluate the selected setting once on all configured splits."
        )
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("configs") / "brats2020.yaml"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore reusable results and rerun the source sweep and final evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        run_tuning(
            args.config,
            args.checkpoint,
            args.output,
            run_final=not args.skip_final,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
