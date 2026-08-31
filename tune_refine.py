from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config, validate_config
from .test import run_evaluation


_INPUT_METADATA = "tuning_input.json"


def _candidate_name(
    beta: float,
    strength_scale: float,
    direction_mode: str = "local_excess_confidence",
) -> str:
    beta_text = f"{beta:g}".replace(".", "p")
    scale_text = f"{strength_scale:g}".replace(".", "p")
    return f"{direction_mode}__beta_{beta_text}__scale_{scale_text}"


def _source_rows(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with summary_path.open() as stream:
        rows = json.load(stream)
    source_rows = {
        row["prediction"]: row
        for row in rows
        if row.get("domain") == "source_val"
    }
    if "base" not in source_rows or "refined" not in source_rows:
        raise RuntimeError(f"Missing source_val base/refined rows in {summary_path}")
    return source_rows["base"], source_rows["refined"]


def select_candidate(
    candidates: list[dict[str, Any]], dice_tolerance: float
) -> dict[str, Any] | None:
    """Select the earliest minimum-ECE candidate passing the Dice guardrail."""
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
        if eligible and (best is None or ece < float(best["basic_ece"])):
            best = candidate
    return best


def _write_candidates_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "direction_mode",
        "beta",
        "refine_strength_scale",
        "effective_alpha_max",
        "base_mean_dice",
        "mean_dice",
        "basic_ece",
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


def _write_final_comparison(
    path: Path,
    final_outputs: dict[str, dict[str, str]],
) -> None:
    rows: list[dict[str, Any]] = []
    for direction_mode, item in final_outputs.items():
        with (Path(item["output"]) / "summary_metrics.json").open() as stream:
            summaries = json.load(stream)
        rows.extend(
            {
                "direction_mode": direction_mode,
                "candidate": item["candidate"],
                **summary,
            }
            for summary in summaries
        )
    with (path / "final_comparison.json").open("w") as stream:
        json.dump(rows, stream, indent=2, allow_nan=True)
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (path / "final_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _best_by_direction(
    candidates: list[dict[str, Any]],
    direction_modes: list[str],
    dice_tolerance: float,
) -> list[dict[str, Any]]:
    finalists: list[dict[str, Any]] = []
    for direction_mode in direction_modes:
        best = select_candidate(
            [
                candidate
                for candidate in candidates
                if candidate["direction_mode"] == direction_mode
            ],
            dice_tolerance,
        )
        if best is not None:
            finalists.append(best)
    return finalists


def _run_signature(config: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_stat = checkpoint_path.stat()
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
    }


def _can_reuse_result(output: Path, signature: dict[str, Any], force: bool) -> bool:
    summary_path = output / "summary_metrics.json"
    metadata_path = output / _INPUT_METADATA
    if force or not summary_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open() as stream:
            return json.load(stream) == signature
    except (OSError, json.JSONDecodeError):
        return False


def _write_run_signature(output: Path, signature: dict[str, Any]) -> None:
    with (output / _INPUT_METADATA).open("w") as stream:
        json.dump(signature, stream, indent=2)


def run_tuning(
    config_path: str,
    checkpoint: str,
    output_dir: str,
    run_final: bool = True,
    force: bool = False,
) -> Path:
    config = load_config(config_path)
    evaluation = config["evaluation"]
    tuning = evaluation.get("refine_tuning", {})
    direction_modes = [
        str(value)
        for value in tuning.get(
            "direction_modes",
            ["legacy_complement", "local_excess_confidence"],
        )
    ]
    beta_values = [float(value) for value in tuning.get("beta_values", [2.0, 2.5])]
    strength_scales = [
        float(value) for value in tuning.get("strength_scales", [1.8, 2.0, 2.2])
    ]
    dice_tolerance = float(tuning.get("dice_tolerance", 0.01))
    valid_direction_modes = {"legacy_complement", "local_excess_confidence"}
    if not direction_modes or any(
        value not in valid_direction_modes for value in direction_modes
    ):
        raise ValueError(
            "evaluation.refine_tuning.direction_modes must contain only supported modes"
        )
    if not beta_values or not strength_scales:
        raise ValueError("evaluation.refine_tuning parameter lists must not be empty")
    if not math.isfinite(dice_tolerance) or dice_tolerance < 0.0:
        raise ValueError("evaluation.refine_tuning.dice_tolerance must be non-negative")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for direction_mode in direction_modes:
        for beta in beta_values:
            for strength_scale in strength_scales:
                candidate_config = copy.deepcopy(config)
                candidate_config["label_transfer"]["direction_mode"] = direction_mode
                candidate_config["label_transfer"]["beta"] = beta
                candidate_config["evaluation"]["refine_strength_scale"] = strength_scale
                candidate_config["evaluation"]["splits"] = ["val"]
                validate_config(candidate_config)
                name = _candidate_name(beta, strength_scale, direction_mode)
                candidate_output = output / "source_val" / name
                summary_path = candidate_output / "summary_metrics.json"
                signature = _run_signature(candidate_config, checkpoint)
                if _can_reuse_result(candidate_output, signature, force):
                    print(f"[refine-tuning] reusing completed {name}", flush=True)
                else:
                    print(f"[refine-tuning] evaluating {name} on source_val", flush=True)
                    run_evaluation(candidate_config, checkpoint, str(candidate_output))
                    _write_run_signature(candidate_output, signature)
                base, refined = _source_rows(summary_path)
                candidates.append(
                    {
                        "name": name,
                        "direction_mode": direction_mode,
                        "beta": beta,
                        "refine_strength_scale": strength_scale,
                        "effective_alpha_max": (
                            float(candidate_config["label_transfer"]["alpha_max"])
                            * strength_scale
                        ),
                        "base_mean_dice": float(base["mean_dice"]),
                        "mean_dice": float(refined["mean_dice"]),
                        "basic_ece": float(refined["basic_ece"]),
                        "output": str(candidate_output),
                    }
                )

    best = select_candidate(candidates, dice_tolerance)
    _write_candidates_csv(output / "candidates.csv", candidates)
    final_output = output / f"final__{best['name']}" if best is not None else None
    selection = {
        "selection_split": "source_val",
        "selection_metric": "basic_ece",
        "dice_tolerance": dice_tolerance,
        "checkpoint": str(Path(checkpoint).resolve()),
        "candidates": candidates,
        "best": best,
        "final_output": str(final_output) if final_output is not None else None,
        "final_outputs": {},
        "final_evaluation_requested": bool(run_final),
    }
    selection_path = output / "selection.json"
    with selection_path.open("w") as stream:
        json.dump(selection, stream, indent=2, allow_nan=False)
    if best is None:
        raise RuntimeError(
            f"No refine candidate passed the Dice guardrail; see {selection_path}"
        )

    print(
        "[refine-tuning] selected "
        f"mode={best['direction_mode']}, beta={best['beta']:g}, "
        f"scale={best['refine_strength_scale']:g}, "
        f"source_val ECE={best['basic_ece']:.6f}, Dice={best['mean_dice']:.6f}",
        flush=True,
    )
    if run_final:
        final_outputs: dict[str, dict[str, str]] = {}
        finalists = _best_by_direction(candidates, direction_modes, dice_tolerance)
        for finalist in finalists:
            mode = str(finalist["direction_mode"])
            mode_output = output / f"final__{finalist['name']}"
            final_config = copy.deepcopy(config)
            final_config["label_transfer"]["direction_mode"] = mode
            final_config["label_transfer"]["beta"] = finalist["beta"]
            final_config["evaluation"]["refine_strength_scale"] = finalist[
                "refine_strength_scale"
            ]
            validate_config(final_config)
            final_signature = _run_signature(final_config, checkpoint)
            if _can_reuse_result(mode_output, final_signature, force):
                print(
                    f"[refine-tuning] reusing completed final {mode} evaluation",
                    flush=True,
                )
            else:
                print(
                    f"[refine-tuning] running final {mode} evaluation on configured splits",
                    flush=True,
                )
                run_evaluation(final_config, checkpoint, str(mode_output))
                _write_run_signature(mode_output, final_signature)
            final_outputs[mode] = {
                "candidate": str(finalist["name"]),
                "output": str(mode_output),
            }
        selection["final_outputs"] = final_outputs
        _write_final_comparison(output, final_outputs)
        with selection_path.open("w") as stream:
            json.dump(selection, stream, indent=2, allow_nan=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune refine parameters on source validation, then evaluate the selected "
            "setting once on the configured splits."
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
        help="Rerun candidates and final evaluation even when result files exist.",
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
