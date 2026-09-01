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
from .metrics import _sample_mask_indices, dice_per_region, expected_calibration_error
from .model import DualHeadSwinUNETR
from .refine_audit import audit_refinement, probability_snapshot, summarize_audits
from .reproducibility import save_run_metadata, set_reproducibility
from .test import (
    _case_name,
    _required_cpu_tensor,
    _summary,
    _write_csv,
    run_evaluation,
)


_SIGNATURE_FILE = "participation_sweep_input.json"
_SWEEP_VERSION = 1
_REFINE_FORMULA = "complement_neighbor_uncertainty_gain_v1"


def candidate_name(iterations: int, beta: float, uncertainty_gain: float) -> str:
    beta_text = f"{beta:g}".replace(".", "p")
    gain_text = f"{uncertainty_gain:g}".replace(".", "p")
    return f"iter_{iterations}__beta_{beta_text}__gain_{gain_text}"


def build_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    tuning = config["evaluation"].get("refine_participation_tuning", {})
    iterations = [int(value) for value in tuning.get("iteration_values", [4, 5, 6])]
    betas = [float(value) for value in tuning.get("beta_values", [0.5, 0.75, 1.0])]
    gains = [
        float(value)
        for value in tuning.get("uncertainty_gain_values", [1.5, 1.75, 2.0, 2.25])
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, float, float]] = set()
    for iteration in iterations:
        for beta in betas:
            for gain in gains:
                key = (iteration, beta, gain)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "name": candidate_name(iteration, beta, gain),
                        "iterations": iteration,
                        "beta": beta,
                        "uncertainty_gain": gain,
                    }
                )
    return candidates


def _finite(candidate: dict[str, Any], *keys: str) -> bool:
    return all(math.isfinite(float(candidate[key])) for key in keys)


def _eligible(
    candidate: dict[str, Any],
    *,
    dice_tolerance: float,
    change_rate_min: float,
    change_rate_max: float,
    min_correction_precision: float,
    max_et_dice_drop: float,
) -> bool:
    if not _finite(
        candidate,
        "base_mean_dice",
        "mean_dice",
        "base_dice_ET",
        "dice_ET",
        "basic_ece",
        "atomic_change_rate",
        "atomic_correction_precision",
    ):
        return False
    return bool(
        change_rate_min
        <= float(candidate["atomic_change_rate"])
        <= change_rate_max
        and int(candidate["atomic_corrected_voxels"])
        > int(candidate["atomic_corrupted_voxels"])
        and float(candidate["atomic_correction_precision"])
        >= min_correction_precision
        and float(candidate["mean_dice"])
        >= float(candidate["base_mean_dice"]) - dice_tolerance
        and float(candidate["dice_ET"])
        >= float(candidate["base_dice_ET"]) - max_et_dice_drop
    )


def select_candidate(
    candidates: list[dict[str, Any]],
    *,
    target_change_rate: float = 0.02,
    change_rate_min: float = 0.015,
    change_rate_max: float = 0.025,
    min_correction_precision: float = 0.55,
    max_et_dice_drop: float = 0.0005,
    dice_tolerance: float = 0.0,
) -> dict[str, Any] | None:
    """Apply safety constraints, then prefer Dice, ECE, and target proximity."""
    eligible: list[tuple[int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        candidate["eligible"] = _eligible(
            candidate,
            dice_tolerance=dice_tolerance,
            change_rate_min=change_rate_min,
            change_rate_max=change_rate_max,
            min_correction_precision=min_correction_precision,
            max_et_dice_drop=max_et_dice_drop,
        )
        if candidate["eligible"]:
            eligible.append((index, candidate))
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            -float(item[1]["mean_dice"]),
            float(item[1]["basic_ece"]),
            abs(float(item[1]["atomic_change_rate"]) - target_change_rate),
            item[0],
        ),
    )[1]


def select_with_guardrails(
    candidates: list[dict[str, Any]], tuning: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    common = {
        "target_change_rate": float(tuning.get("target_change_rate", 0.02)),
        "change_rate_min": float(tuning.get("change_rate_min", 0.015)),
        "change_rate_max": float(tuning.get("change_rate_max", 0.025)),
        "min_correction_precision": float(
            tuning.get("min_correction_precision", 0.55)
        ),
        "max_et_dice_drop": float(tuning.get("max_et_dice_drop", 0.0005)),
    }
    strict_tolerance = float(tuning.get("dice_tolerance", 0.0))
    best = select_candidate(
        candidates, dice_tolerance=strict_tolerance, **common
    )
    if best is not None:
        return best, "strict"
    fallback = float(tuning.get("fallback_dice_tolerance", 0.0001))
    if fallback > strict_tolerance:
        best = select_candidate(candidates, dice_tolerance=fallback, **common)
        if best is not None:
            return best, "fallback"
    return None, None


def _evaluate_selection_case(
    probability: torch.Tensor,
    scalar_target: torch.Tensor,
    atomic_prediction: torch.Tensor,
    region_prediction: torch.Tensor,
    *,
    bins: int,
    max_voxels: int,
) -> dict[str, float]:
    value = probability.detach().float().cpu()
    if value.ndim == 5:
        value = value[0]
    scalar = scalar_target.detach().cpu()
    if scalar.ndim == 5:
        scalar = scalar[0]
    if scalar.ndim == 4 and scalar.shape[0] == 1:
        scalar = scalar[0]
    target = scalar.numpy().astype(np.uint8, copy=True)
    if not bool(np.isin(target, (0, 1, 2, 4)).all()):
        raise ValueError("Source validation contains invalid BraTS labels")
    target[target == 4] = 3
    region_target = np.stack(
        ((target == 1) | (target == 3), target > 0, target == 3)
    ).astype(np.uint8, copy=False)
    prediction = atomic_prediction.detach().cpu().to(torch.uint8)
    regions = region_prediction.detach().cpu().numpy().astype(np.uint8, copy=False)
    dice = dice_per_region(regions, region_target)
    atomic_target = torch.as_tensor(target, dtype=torch.uint8)
    mask = (atomic_target > 0) | (prediction > 0)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)
    indices = _sample_mask_indices(mask, max_voxels)
    confidence = torch.amax(value.reshape(4, -1)[:, indices], dim=0).numpy()
    correct = (
        prediction.reshape(-1)[indices] == atomic_target.reshape(-1)[indices]
    ).numpy().astype(np.float32)
    ece = expected_calibration_error(confidence, correct, bins)
    return {
        "mean_dice": float(np.nanmean(dice)),
        "basic_ece": float(ece),
        "ece": float(ece),
        **{
            name: float(metric)
            for name, metric in zip(("dice_TC", "dice_WT", "dice_ET"), dice)
        },
    }


def _json(path: Path, value: Any, *, allow_nan: bool = False) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=allow_nan)


def _overall_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_audits(rows)
    return next(row for row in summary if row["domain"] == "overall")


def _write_candidate_result(
    output: Path,
    candidate: dict[str, Any],
    base_rows: list[dict[str, Any]],
    refined_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> None:
    destination = output / "source_val" / str(candidate["name"])
    destination.mkdir(parents=True, exist_ok=True)
    summaries = _summary([*base_rows, *refined_rows])
    audits = summarize_audits(audit_rows)
    _write_csv(destination / "case_metrics.csv", [*base_rows, *refined_rows])
    _write_csv(destination / "summary_metrics.csv", summaries)
    _write_csv(destination / "refine_audit_case.csv", audit_rows)
    _write_csv(destination / "refine_audit_summary.csv", audits)
    _json(destination / "summary_metrics.json", summaries, allow_nan=True)
    _json(destination / "refine_audit_summary.json", audits)
    _json(
        destination / "refine_metadata.json",
        {
            "refine_formula": _REFINE_FORMULA,
            "selection_split": "source_val",
            "external_splits_used_for_selection": False,
            "iterations": int(candidate["iterations"]),
            "beta": float(candidate["beta"]),
            "uncertainty_gain": float(candidate["uncertainty_gain"]),
        },
    )


@torch.inference_mode()
def run_source_sweep(
    config: dict[str, Any],
    checkpoint: str,
    output: Path,
    candidates: list[dict[str, Any]],
) -> None:
    seed = int(config["reproducibility"]["seed"])
    set_reproducibility(
        seed, bool(config["reproducibility"].get("deterministic", True))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadSwinUNETR(config).to(device)
    payload = load_checkpoint(checkpoint, model, strict=True)
    if not model.zernike_stats.fitted:
        raise RuntimeError(
            "Checkpoint lacks fitted Zernike statistics; run stats/calibration first"
        )
    del payload
    model.eval()
    evaluation = config["evaluation"]
    bins = int(evaluation.get("calibration_bins", 15))
    max_voxels = int(evaluation.get("max_metric_voxels", 200000))
    loader = build_loader(config, "val", "test")
    base_rows: list[dict[str, Any]] = []
    refined_rows = {str(item["name"]): [] for item in candidates}
    audit_rows = {str(item["name"]): [] for item in candidates}
    original = (
        model.label_transfer.iterations,
        model.label_transfer.beta,
        model.label_transfer.uncertainty_gain,
    )

    for index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        result = infer_volume(
            model, image, config, compute_uncertainty=True, refine=False
        )
        case = _case_name(batch, index)
        base_device = result.base_atomic_probability
        uncertainty = result.uncertainty
        if base_device is None or uncertainty is None:
            raise RuntimeError("Sweep requires atomic probability and uncertainty")
        base_probability = _required_cpu_tensor(base_device, "base atomic probability")
        base_snapshot = probability_snapshot(base_probability)
        base_region = _required_cpu_tensor(
            result.base_region_probability >= 0.5,
            "base region prediction",
            torch.uint8,
        )[0]
        base_rows.append(
            {
                "domain": "source_val",
                "prediction": "base",
                "case": case,
                **_evaluate_selection_case(
                    base_probability,
                    batch["label_scalar"],
                    base_snapshot.atomic_prediction,
                    base_region,
                    bins=bins,
                    max_voxels=max_voxels,
                ),
            }
        )
        for candidate in candidates:
            name = str(candidate["name"])
            model.label_transfer.iterations = int(candidate["iterations"])
            model.label_transfer.beta = float(candidate["beta"])
            model.label_transfer.uncertainty_gain = float(
                candidate["uncertainty_gain"]
            )
            refined_device = model.label_transfer(base_device, uncertainty)
            refined_probability = _required_cpu_tensor(
                refined_device, "refined atomic probability"
            )
            refined_snapshot = probability_snapshot(refined_probability)
            refined_rows[name].append(
                {
                    "domain": "source_val",
                    "prediction": "refined",
                    "case": case,
                    **_evaluate_selection_case(
                        refined_probability,
                        batch["label_scalar"],
                        refined_snapshot.atomic_prediction,
                        refined_snapshot.region_prediction,
                        bins=bins,
                        max_voxels=max_voxels,
                    ),
                }
            )
            audit = audit_refinement(
                base_snapshot, refined_snapshot, batch["label_scalar"]
            )
            audit_rows[name].append(
                {"domain": "source_val", "case": case, **audit.metrics}
            )
            del refined_device, refined_probability, refined_snapshot, audit
        print(
            f"[participation-tuning] source_val {index + 1}/{len(loader)} "
            f"{case}: {len(candidates)} candidates",
            flush=True,
        )
        del result, image, base_device, uncertainty, base_probability, base_snapshot
        if device.type == "cuda" and bool(evaluation.get("release_cuda_cache", True)):
            torch.cuda.empty_cache()

    (
        model.label_transfer.iterations,
        model.label_transfer.beta,
        model.label_transfer.uncertainty_gain,
    ) = original
    for candidate in candidates:
        name = str(candidate["name"])
        _write_candidate_result(
            output, candidate, base_rows, refined_rows[name], audit_rows[name]
        )


def _source_metric_rows(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = json.loads(path.read_text())
    by_prediction = {
        row["prediction"]: row
        for row in rows
        if row.get("domain") == "source_val"
    }
    return by_prediction["base"], by_prediction["refined"]


def _collect(output: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for candidate in candidates:
        directory = output / "source_val" / str(candidate["name"])
        base, refined = _source_metric_rows(directory / "summary_metrics.json")
        audits = json.loads((directory / "refine_audit_summary.json").read_text())
        audit = next(row for row in audits if row["domain"] == "overall")
        collected.append(
            {
                **candidate,
                "base_mean_dice": float(base["mean_dice"]),
                "base_dice_ET": float(base["dice_ET"]),
                "base_basic_ece": float(base["basic_ece"]),
                "mean_dice": float(refined["mean_dice"]),
                "dice_ET": float(refined["dice_ET"]),
                "basic_ece": float(refined["basic_ece"]),
                "delta_mean_dice": float(refined["mean_dice"])
                - float(base["mean_dice"]),
                "delta_dice_ET": float(refined["dice_ET"])
                - float(base["dice_ET"]),
                "delta_basic_ece": float(refined["basic_ece"])
                - float(base["basic_ece"]),
                "atomic_change_rate": float(audit["pooled_atomic_change_rate"]),
                "atomic_correction_precision": float(
                    audit["pooled_atomic_correction_precision"]
                ),
                "atomic_corrected_voxels": int(audit["atomic_corrected_voxels"]),
                "atomic_corrupted_voxels": int(audit["atomic_corrupted_voxels"]),
                "atomic_net_corrected_voxels": int(
                    audit["atomic_net_corrected_voxels"]
                ),
                "output": str(directory),
            }
        )
    return collected


def _signature(
    config: dict[str, Any], checkpoint: str, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    stat = checkpoint_path.stat()
    encoded = json.dumps(
        {
            "version": _SWEEP_VERSION,
            "formula": _REFINE_FORMULA,
            "config": config,
            "candidates": candidates,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "version": _SWEEP_VERSION,
        "formula": _REFINE_FORMULA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _can_reuse(
    output: Path,
    candidates: list[dict[str, Any]],
    signature: dict[str, Any],
    force: bool,
) -> bool:
    path = output / _SIGNATURE_FILE
    if force or not path.is_file():
        return False
    try:
        if json.loads(path.read_text()) != signature:
            return False
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (output / "source_val" / str(item["name"]) / "refine_audit_summary.json").is_file()
        and (output / "source_val" / str(item["name"]) / "summary_metrics.json").is_file()
        for item in candidates
    )


def _write_candidates(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = list(candidates[0]) if candidates else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)


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
    tuning = config["evaluation"].get("refine_participation_tuning", {})
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output, config)
    signature = _signature(config, checkpoint, candidates)
    if _can_reuse(output, candidates, signature, force):
        print("[participation-tuning] reusing source-validation sweep", flush=True)
    else:
        print(
            f"[participation-tuning] evaluating {len(candidates)} candidates "
            "from one source-validation inference pass",
            flush=True,
        )
        run_source_sweep(config, checkpoint, output, candidates)
        _json(output / _SIGNATURE_FILE, signature)

    collected = _collect(output, candidates)
    best, tier = select_with_guardrails(collected, tuning)
    for candidate in collected:
        candidate["selection_tier"] = tier if candidate is best else "not_selected"
    _write_candidates(output / "candidates.csv", collected)
    _json(output / "candidates.json", collected)
    final_output = output / f"final__{best['name']}" if best is not None else None
    selection = {
        "selection_split": "source_val",
        "external_splits_used_for_selection": False,
        "selection_rule": (
            "change-rate band + positive net correction + correction precision + "
            "ET/Dice guardrails; then maximum Dice, minimum ECE, nearest 2% rate"
        ),
        "selection_tier": tier,
        "target_change_rate": float(tuning.get("target_change_rate", 0.02)),
        "guardrails": {
            key: tuning.get(key)
            for key in (
                "change_rate_min",
                "change_rate_max",
                "min_correction_precision",
                "max_et_dice_drop",
                "dice_tolerance",
                "fallback_dice_tolerance",
            )
        },
        "checkpoint": str(Path(checkpoint).resolve()),
        "candidates": collected,
        "best": best,
        "final_output": str(final_output) if final_output else None,
        "final_evaluation_requested": bool(run_final),
    }
    selection_path = output / "selection.json"
    _json(selection_path, selection)
    if best is None:
        raise RuntimeError(
            "No candidate met the participation and safety guardrails; "
            f"inspect {selection_path}. External evaluation was not run."
        )

    print(
        "[participation-tuning] selected "
        f"iterations={best['iterations']}, beta={best['beta']:g}, "
        f"gain={best['uncertainty_gain']:g}, "
        f"change={best['atomic_change_rate']:.3%}, "
        f"Dice={best['mean_dice']:.6f}, ECE={best['basic_ece']:.6f}",
        flush=True,
    )
    if run_final and final_output is not None:
        final_config = copy.deepcopy(config)
        final_config["label_transfer"]["iterations"] = int(best["iterations"])
        final_config["label_transfer"]["beta"] = float(best["beta"])
        final_config["label_transfer"]["uncertainty_gain"] = float(
            best["uncertainty_gain"]
        )
        validate_config(final_config)
        final_signature = _signature(final_config, checkpoint, [best])
        final_signature_path = final_output / _SIGNATURE_FILE
        reuse_final = False
        if not force and (final_output / "summary_metrics.json").is_file():
            try:
                reuse_final = json.loads(final_signature_path.read_text()) == final_signature
            except (OSError, json.JSONDecodeError):
                reuse_final = False
        if not reuse_final:
            run_evaluation(final_config, checkpoint, str(final_output))
            _json(final_signature_path, final_signature)
        else:
            print("[participation-tuning] reusing final evaluation", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune refine iterations, beta, and uncertainty gain on source validation "
            "for about 2% top-1 participation, then evaluate the guarded winner."
        )
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("configs") / "brats2020.yaml"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--force", action="store_true")
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
