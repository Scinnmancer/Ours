from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def parse_overrides(overrides: Iterable[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw = item.split("=", 1)
        _set_nested(parsed, key, yaml.safe_load(raw))
    return parsed


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_paths(config: dict[str, Any], config_path: Path) -> None:
    paths = config.setdefault("paths", {})
    data_root = paths.get("data_root") or "/root/autodl-tmp/archive"
    paths["data_root"] = str(Path(_expand(data_root)).resolve())
    for key in ("split_json", "run_root", "baseline_checkpoint"):
        value = paths.get(key)
        if not value:
            continue
        value_path = Path(_expand(value))
        if not value_path.is_absolute():
            value_path = (config_path.parent / value_path).resolve()
        paths[key] = str(value_path)


def validate_config(config: dict[str, Any]) -> None:
    required = ("paths", "data", "model", "zernike", "uncertainty", "label_transfer", "training", "evaluation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    windows = config["zernike"].get("windows", [])
    if not windows or any(int(size) < 3 or int(size) % 2 == 0 for size in windows):
        raise ValueError("zernike.windows must contain odd integers >= 3")
    roi = config["data"].get("roi_size", [])
    if len(roi) != 3 or any(int(size) <= 0 for size in roi):
        raise ValueError("data.roi_size must contain three positive integers")
    head_dropout_rates = config["model"].get("head_dropout_rates", [0.2, 0.3])
    if (
        not isinstance(head_dropout_rates, (list, tuple))
        or len(head_dropout_rates) != 2
        or any(not 0.0 <= float(rate) < 1.0 for rate in head_dropout_rates)
    ):
        raise ValueError("model.head_dropout_rates must contain two values in [0, 1)")
    alpha = float(config["label_transfer"].get("alpha_max", 0.35))
    if not 0.0 <= alpha < 1.0:
        raise ValueError("label_transfer.alpha_max must be in [0, 1)")
    refine_scale = float(config["evaluation"].get("refine_strength_scale", 1.0))
    if not math.isfinite(refine_scale) or refine_scale <= 0.0:
        raise ValueError("evaluation.refine_strength_scale must be finite and greater than 0")
    if alpha * refine_scale >= 1.0:
        raise ValueError(
            "label_transfer.alpha_max * evaluation.refine_strength_scale must be less than 1"
        )
    refine_tuning = config["evaluation"].get("refine_tuning", {})
    if not isinstance(refine_tuning, dict):
        raise ValueError("evaluation.refine_tuning must be a mapping")
    iteration_values = refine_tuning.get("iteration_values", [1, 2, 3, 4, 5])
    beta_values = refine_tuning.get("beta_values", [1.75, 2.0, 2.25])
    strength_scales = refine_tuning.get(
        "strength_scales", [2.4, 2.5, 2.6, 2.7, 2.8]
    )
    if not isinstance(iteration_values, (list, tuple)) or not iteration_values:
        raise ValueError(
            "evaluation.refine_tuning.iteration_values must be a non-empty list"
        )
    if not isinstance(beta_values, (list, tuple)) or not beta_values:
        raise ValueError("evaluation.refine_tuning.beta_values must be a non-empty list")
    if not isinstance(strength_scales, (list, tuple)) or not strength_scales:
        raise ValueError(
            "evaluation.refine_tuning.strength_scales must be a non-empty list"
        )
    if any(
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) != int(value)
        or int(value) <= 0
        for value in iteration_values
    ):
        raise ValueError(
            "evaluation.refine_tuning.iteration_values must contain positive integers"
        )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in beta_values):
        raise ValueError(
            "evaluation.refine_tuning.beta_values must contain finite non-negative values"
        )
    if any(
        not math.isfinite(float(value))
        or float(value) <= 0.0
        or alpha * float(value) >= 1.0
        for value in strength_scales
    ):
        raise ValueError(
            "evaluation.refine_tuning.strength_scales must contain finite positive "
            "values whose effective alpha_max is less than 1"
        )
    refine_dice_tolerance = float(refine_tuning.get("dice_tolerance", 0.0))
    if not math.isfinite(refine_dice_tolerance) or refine_dice_tolerance < 0.0:
        raise ValueError("evaluation.refine_tuning.dice_tolerance must be non-negative")
    fallback_dice_tolerance = float(
        refine_tuning.get("fallback_dice_tolerance", 0.0001)
    )
    if (
        not math.isfinite(fallback_dice_tolerance)
        or fallback_dice_tolerance < refine_dice_tolerance
    ):
        raise ValueError(
            "evaluation.refine_tuning.fallback_dice_tolerance must be finite and "
            "at least dice_tolerance"
        )
    ece_tie_tolerance = float(refine_tuning.get("ece_tie_tolerance", 0.0002))
    if not math.isfinite(ece_tie_tolerance) or ece_tie_tolerance < 0.0:
        raise ValueError(
            "evaluation.refine_tuning.ece_tie_tolerance must be non-negative"
        )
    for key in ("warmup_epochs", "calibration_epochs", "validation_every"):
        if int(config["training"].get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if int(
        config["training"].get(
            "warmup_validation_every",
            config["training"].get("validation_every", 0),
        )
    ) <= 0:
        raise ValueError("training.warmup_validation_every must be a positive integer")
    calibration_weight = float(config["uncertainty"].get("lambda_u", 1.0))
    if not math.isfinite(calibration_weight) or calibration_weight < 0.0:
        raise ValueError("uncertainty.lambda_u must be finite and non-negative")
    calibration_fusion = config["uncertainty"].get("calibration_fusion", {})
    if not isinstance(calibration_fusion, dict):
        raise ValueError("uncertainty.calibration_fusion must be a mapping")
    fusion_override_enabled = calibration_fusion.get("enabled", False)
    if not isinstance(fusion_override_enabled, bool):
        raise ValueError("uncertainty.calibration_fusion.enabled must be a boolean")
    fusion_xi = float(calibration_fusion.get("xi", 8.0))
    if not math.isfinite(fusion_xi) or fusion_xi <= 0.0:
        raise ValueError("uncertainty.calibration_fusion.xi must be finite and positive")
    fusion_bias = float(calibration_fusion.get("bias", -4.8))
    if not math.isfinite(fusion_bias):
        raise ValueError("uncertainty.calibration_fusion.bias must be finite")
    margin_gradient = config["uncertainty"].get("margin_gradient", {})
    if not isinstance(margin_gradient, dict):
        raise ValueError("uncertainty.margin_gradient must be a mapping")
    margin_enabled = margin_gradient.get("enabled", False)
    if not isinstance(margin_enabled, bool):
        raise ValueError("uncertainty.margin_gradient.enabled must be a boolean")
    margin_weight = float(margin_gradient.get("weight", 0.01))
    if not math.isfinite(margin_weight) or margin_weight < 0.0:
        raise ValueError("uncertainty.margin_gradient.weight must be finite and non-negative")
    uncertainty_power = float(margin_gradient.get("uncertainty_power", 1.0))
    if not math.isfinite(uncertainty_power) or uncertainty_power < 0.0:
        raise ValueError(
            "uncertainty.margin_gradient.uncertainty_power must be finite and non-negative"
        )
    margin = float(margin_gradient.get("margin", 1.0))
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("uncertainty.margin_gradient.margin must be finite and positive")
    freeze_calibration = config["training"].get("freeze_segmentation_during_calibration", True)
    if not isinstance(freeze_calibration, bool):
        raise ValueError("training.freeze_segmentation_during_calibration must be a boolean")
    calibration_scope = config["training"].get("calibration_trainable_scope")
    if calibration_scope is not None and calibration_scope != "heads":
        raise ValueError(
            "training.calibration_trainable_scope must be heads for margin calibration"
        )
    if int(config["data"].get("workers", 8)) < 0:
        raise ValueError("data.workers must be a non-negative integer")
    if int(config["evaluation"].get("workers", 0)) < 0:
        raise ValueError("evaluation.workers must be a non-negative integer")
    if int(config["evaluation"].get("max_metric_voxels", 200000)) <= 0:
        raise ValueError("evaluation.max_metric_voxels must be a positive integer")
    monitoring = config.get("monitoring", {})
    if int(monitoring.get("batch_interval", 0)) < 0:
        raise ValueError("monitoring.batch_interval must be a non-negative integer")
    if int(monitoring.get("gradient_interval", 10)) <= 0:
        raise ValueError("monitoring.gradient_interval must be a positive integer")
    if int(monitoring.get("uncertainty_map_every", 0)) < 0:
        raise ValueError("monitoring.uncertainty_map_every must be a non-negative integer")
    if str(monitoring.get("uncertainty_map_dtype", "float16")) not in ("float16", "float32"):
        raise ValueError("monitoring.uncertainty_map_dtype must be float16 or float32")
    if int(monitoring.get("plateau_patience_validations", 10)) < 0:
        raise ValueError("monitoring.plateau_patience_validations must be a non-negative integer")
    if float(monitoring.get("memory_multiplier_limit", 3.0)) <= 1.0:
        raise ValueError("monitoring.memory_multiplier_limit must be greater than 1")


def load_config(path: str | os.PathLike[str], overrides: Iterable[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream) or {}
    if os.environ.get("BRATS_DATA_ROOT"):
        _set_nested(config, "paths.data_root", os.environ["BRATS_DATA_ROOT"])
    config = _merge(config, parse_overrides(overrides))
    config = _expand(config)
    _resolve_paths(config, config_path)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def save_resolved_config(config: dict[str, Any], output_path: str | os.PathLike[str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
