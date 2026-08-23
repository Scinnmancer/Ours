from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .data import brain_mask, build_loader, generate_splits
from .inference import autocast_context, infer_volume
from .losses import RegionDiceLoss, risk_brier_loss
from .metrics import MetricSampleReservoir, aupr, auroc, expected_calibration_error, risk_calibration_metrics
from .model import DualHeadOutput, DualHeadSwinUNETR
from .monitoring import TrainingTelemetry, gradient_statistics
from .reproducibility import save_run_metadata, set_reproducibility
from .uncertainty import uncertainty_weighted_margin_loss
from .zernike import WelfordAccumulator


def _run_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["run_root"]) / config["experiment"]


def _append_metrics(run_dir: Path, payload: dict[str, Any]) -> None:
    with (run_dir / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _case_name(batch: dict[str, Any], index: int, sample_index: int = 0) -> str:
    subject = batch.get("subject_id")
    if isinstance(subject, (list, tuple)):
        subject = subject[sample_index]
    if subject is None:
        subject = f"case_{index:04d}_{sample_index:02d}"
    value = str(subject)
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _affine(batch: dict[str, Any], sample_index: int = 0) -> np.ndarray:
    metadata = batch.get("image_meta_dict", {})
    for key in ("affine", "original_affine"):
        value = metadata.get(key)
        if value is not None:
            if getattr(value, "ndim", 0) == 3:
                value = value[sample_index]
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    image = batch.get("image")
    if hasattr(image, "meta"):
        value = image.meta.get("affine", image.meta.get("original_affine"))
        if value is not None:
            if getattr(value, "ndim", 0) == 3:
                value = value[sample_index]
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return np.eye(4, dtype=np.float64)


def _save_uncertainty_batch(
    output: DualHeadOutput,
    batch: dict[str, Any],
    target_atomic: torch.Tensor,
    destination: Path,
    batch_index: int,
    epoch: int,
    dtype: str,
    save_components: bool,
) -> list[Path]:
    if output.uncertainty is None:
        raise RuntimeError("Cannot save uncertainty maps before uncertainty has been computed")
    if save_components and output.zernike_disagreement is None:
        raise RuntimeError("Zernike disagreement has not been computed")
    numpy_dtype = np.float16 if dtype == "float16" else np.float32
    destination.mkdir(parents=True, exist_ok=True)
    prediction = output.base_atomic_probability.argmax(dim=1)
    saved: list[Path] = []
    for sample_index in range(output.uncertainty.shape[0]):
        case = _case_name(batch, batch_index, sample_index)
        target = target_atomic[sample_index].detach().cpu().numpy().astype(np.uint8)
        predicted = prediction[sample_index].detach().cpu().numpy().astype(np.uint8)
        payload: dict[str, np.ndarray] = {
            "uncertainty": output.uncertainty[sample_index, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(numpy_dtype),
            "error": (predicted != target).astype(np.uint8),
            "base_prediction": predicted,
            "target_atomic": target,
            "affine": _affine(batch, sample_index).astype(np.float64),
            "epoch": np.asarray(epoch, dtype=np.int32),
        }
        if save_components:
            payload["zernike_disagreement"] = (
                output.zernike_disagreement[sample_index, 0]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(numpy_dtype)
            )
        path = destination / f"{case}.npz"
        np.savez_compressed(path, **payload)
        saved.append(path)
    return saved


def _scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def segmentation_state_sha256(model: DualHeadSwinUNETR) -> str:
    """Hash only the shared encoder and decoder parameters/buffers."""
    prefixes = (*model.ENCODER_PREFIXES, "head1.", "head2.")
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if not key.startswith(prefixes):
            continue
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encoder_state_sha256(model: DualHeadSwinUNETR) -> str:
    """Hash the shared encoder state so head-only calibration can keep it frozen."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if not key.startswith(model.ENCODER_PREFIXES):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_calibration_state_sha256(model: DualHeadSwinUNETR) -> str:
    """Hash every state entry that head-only calibration must not change."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(("head1.", "head2.")):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def calibration_trainable_scope(config: dict[str, Any]) -> str:
    """Resolve explicit scopes while preserving legacy boolean configuration."""
    scope = config["training"].get("calibration_trainable_scope")
    if scope is not None:
        return str(scope)
    return "heads" if bool(config["training"].get("freeze_segmentation_during_calibration", True)) else "full"


def calibration_checkpoint_key(metrics: dict[str, Any]) -> float:
    """Return the sole ranking metric for Dice-eligible checkpoints."""
    return float(metrics.get("basic_ece", metrics.get("ece", math.inf)))


def configure_calibration_trainability(
    model: DualHeadSwinUNETR,
    scope: str | bool,
) -> list[torch.nn.Parameter]:
    """Select the two prediction heads and freeze every other model state."""
    if isinstance(scope, bool):
        scope = "heads" if scope else "full"
    if scope != "heads":
        raise ValueError("Margin calibration supports only calibration_trainable_scope=heads")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("head1.", "head2.")))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Calibration has no trainable parameters")
    return parameters


def train_epoch(
    model: DualHeadSwinUNETR,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    loss_function: RegionDiceLoss,
    config: dict[str, Any],
    lambda_u: float = 0.0,
    telemetry: TrainingTelemetry | None = None,
    stage: str = "warmup",
    epoch: int = 0,
    freeze_segmentation: bool = False,
    calibration_scope: str | None = None,
) -> dict[str, Any]:
    scope = calibration_scope or ("heads" if freeze_segmentation else "full")
    margin_config = config["uncertainty"].get("margin_gradient", {})
    margin_gradient_weight = float(margin_config.get("weight", 0.01))
    margin_gradient_enabled = (
        stage == "calibration"
        and bool(margin_config.get("enabled", False))
        and margin_gradient_weight > 0.0
    )
    margin_uncertainty_power = float(margin_config.get("uncertainty_power", 2.0))
    margin_value = float(margin_config.get("margin", 1.0))
    if stage == "calibration":
        # Keep the frozen encoder deterministic while enabling each decoder's
        # independent Dropout3d during calibration.
        model.eval()
        model.head1.train()
        model.head2.train()
    else:
        model.train()
    started = time.perf_counter()
    totals = {
        "loss": 0.0,
        "segmentation": 0.0,
        "head1_loss": 0.0,
        "head2_loss": 0.0,
        "calibration_brier": 0.0,
        "weighted_calibration_brier": 0.0,
        "margin_calibration": 0.0,
        "weighted_margin_calibration": 0.0,
        "head_region_l1": 0.0,
        "gradient_l2": 0.0,
        "gradient_abs_max": 0.0,
        "gradient_nonfinite": 0.0,
        "gradient_samples": 0.0,
        "amp_skipped_steps": 0.0,
        "zernike_disagreement": 0.0,
        "uncertainty_mean": 0.0,
    }
    intersections = {name: torch.zeros(3, dtype=torch.float64) for name in ("head1", "head2", "base")}
    denominators = {name: torch.zeros(3, dtype=torch.float64) for name in ("head1", "head2", "base")}
    predicted_positive = {name: torch.zeros(3, dtype=torch.float64) for name in ("head1", "head2", "base")}
    target_positive = torch.zeros(3, dtype=torch.float64)
    voxel_count = 0
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    batch_interval = int(config.get("monitoring", {}).get("batch_interval", 0))
    gradient_interval = max(int(config.get("monitoring", {}).get("gradient_interval", 10)), 1)
    if telemetry is not None:
        telemetry.reset_peak_memory()
    for index, batch in enumerate(loader):
        batch_started = time.perf_counter()
        if stage == "calibration":
            image1 = batch["image_view1"].to(device, non_blocking=True)
            image2 = batch["image_view2"].to(device, non_blocking=True)
        else:
            image1 = batch["image_view1"].to(device, non_blocking=True)
            image2 = batch["image_view2"].to(device, non_blocking=True)
        target = batch["label_regions"].to(device, non_blocking=True)
        atomic_target = batch["label_atomic"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            compute_uncertainty = stage == "calibration" or lambda_u > 0.0
            output = model(image1, image2, compute_uncertainty=compute_uncertainty)
            head_losses = [loss_function(logits, target) for logits in output.head_logits]
            segmentation = sum(head_losses)
            calibration_brier = torch.zeros((), device=device)
            if compute_uncertainty:
                base_prediction = output.base_atomic_probability.argmax(dim=1)
                error = (base_prediction != atomic_target).to(output.zernike_disagreement.dtype)
                calibration_mask = (atomic_target > 0) | (base_prediction > 0)
                brier_mask = calibration_mask
                if not bool(brier_mask.any()):
                    brier_mask = torch.ones_like(brier_mask, dtype=torch.bool)
                calibration_brier = risk_brier_loss(
                    output.uncertainty[:, 0].detach()[brier_mask],
                    error[brier_mask],
                )
            margin_calibration = torch.zeros((), device=device)
            if margin_gradient_enabled:
                margin_calibration = uncertainty_weighted_margin_loss(
                    output.base_atomic_probability,
                    output.uncertainty,
                    mask=calibration_mask,
                    uncertainty_power=margin_uncertainty_power,
                    margin=margin_value,
                )
            weighted_margin_calibration = margin_gradient_weight * margin_calibration
            if stage == "calibration":
                # Risk Brier is diagnostic only.  Detached geometric risk
                # weights the margin objective without changing its computation.
                weighted_calibration_brier = torch.zeros_like(calibration_brier)
            else:
                weighted_calibration_brier = lambda_u * calibration_brier
            loss = segmentation + weighted_calibration_brier + weighted_margin_calibration
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        sample_gradient = (index + 1) % gradient_interval == 0 or index + 1 == len(loader)
        if sample_gradient:
            scaler.unscale_(optimizer)
            gradient = gradient_statistics(model.parameters())
        else:
            gradient = None
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        totals["loss"] += float(loss.detach())
        totals["segmentation"] += float(segmentation.detach())
        totals["head1_loss"] += float(head_losses[0].detach())
        totals["head2_loss"] += float(head_losses[1].detach())
        totals["calibration_brier"] += float(calibration_brier.detach())
        totals["weighted_calibration_brier"] += float(weighted_calibration_brier.detach())
        totals["margin_calibration"] += float(margin_calibration.detach())
        totals["weighted_margin_calibration"] += float(weighted_margin_calibration.detach())
        if gradient is not None:
            totals["gradient_l2"] += float(gradient["gradient_l2"])
            totals["gradient_abs_max"] = max(totals["gradient_abs_max"], float(gradient["gradient_abs_max"]))
            totals["gradient_nonfinite"] += float(gradient["gradient_nonfinite"])
            totals["gradient_samples"] += 1.0
        totals["amp_skipped_steps"] += float(scale_after < scale_before)
        with torch.no_grad():
            probabilities = {
                "head1": output.head_region_probabilities[0],
                "head2": output.head_region_probabilities[1],
                "base": output.base_region_probability,
            }
            target_bool = target.bool()
            reduction_dims = (0, *range(2, target.ndim))
            target_count = target_bool.sum(dim=reduction_dims).double().cpu()
            target_positive += target_count
            spatial_voxels = target.shape[0] * math.prod(target.shape[2:])
            voxel_count += spatial_voxels
            for name, probability in probabilities.items():
                prediction = probability >= 0.5
                intersection = (prediction & target_bool).sum(dim=reduction_dims).double().cpu()
                prediction_count = prediction.sum(dim=reduction_dims).double().cpu()
                intersections[name] += intersection
                denominators[name] += prediction_count + target_count
                predicted_positive[name] += prediction_count
            totals["head_region_l1"] += float(
                (output.head_region_probabilities[0] - output.head_region_probabilities[1]).abs().mean()
            )
            if compute_uncertainty:
                totals["zernike_disagreement"] += float(output.zernike_disagreement.mean())
                totals["uncertainty_mean"] += float(output.uncertainty.mean())
        if telemetry is not None and batch_interval > 0 and (index + 1) % batch_interval == 0:
            telemetry.event(
                "batch_finished",
                stage=stage,
                epoch=epoch,
                batch=index + 1,
                batches=len(loader),
                duration_seconds=time.perf_counter() - batch_started,
                loss=float(loss.detach()),
                segmentation=float(segmentation.detach()),
                calibration_brier=float(calibration_brier.detach()),
                weighted_calibration_brier=float(weighted_calibration_brier.detach()),
                margin_calibration=float(margin_calibration.detach()),
                weighted_margin_calibration=float(weighted_margin_calibration.detach()),
                gradient=gradient,
                amp_scale=scale_after,
                amp_step_skipped=scale_after < scale_before,
            )
        print(
            f"batch {index + 1}/{len(loader)} loss={float(loss.detach()):.5f} "
            f"seg={float(segmentation.detach()):.5f} "
            f"calibration_brier={float(calibration_brier.detach()):.5f} "
            f"weighted_calibration_brier={float(weighted_calibration_brier.detach()):.5f} "
            f"margin_calibration={float(margin_calibration.detach()):.5f} "
            f"weighted_margin_calibration={float(weighted_margin_calibration.detach()):.5f}",
            flush=True,
        )
    batches = max(len(loader), 1)
    metrics = {
        key: (
            value
            if key in ("gradient_abs_max", "gradient_nonfinite", "gradient_samples", "amp_skipped_steps")
            else value / batches
        )
        for key, value in totals.items()
    }
    metrics["gradient_l2"] = totals["gradient_l2"] / max(totals["gradient_samples"], 1.0)
    for region_index, region_name in enumerate(("TC", "WT", "ET")):
        metrics[f"target_positive_fraction_{region_name}"] = float(target_positive[region_index] / max(voxel_count, 1))
        for prediction_name in ("head1", "head2", "base"):
            denominator = denominators[prediction_name][region_index]
            metrics[f"{prediction_name}_dice_{region_name}"] = (
                float(2.0 * intersections[prediction_name][region_index] / denominator)
                if denominator > 0
                else float("nan")
            )
            metrics[f"{prediction_name}_positive_fraction_{region_name}"] = float(
                predicted_positive[prediction_name][region_index] / max(voxel_count, 1)
            )
    metrics["duration_seconds"] = time.perf_counter() - started
    metrics["lambda_u"] = float(lambda_u)
    metrics["configured_risk_brier_weight"] = float(lambda_u)
    metrics["calibration_loss_weight"] = 0.0 if stage == "calibration" else float(lambda_u)
    metrics["freeze_segmentation"] = 0.0
    metrics["calibration_trainable_scope"] = scope
    metrics["risk_brier_backprop"] = stage != "calibration" and lambda_u > 0.0
    metrics["margin_gradient_enabled"] = margin_gradient_enabled
    metrics["margin_gradient_weight"] = margin_gradient_weight if margin_gradient_enabled else 0.0
    metrics["margin_uncertainty_power"] = margin_uncertainty_power
    metrics["margin"] = margin_value
    if compute_uncertainty:
        metrics["fusion_xi"] = float(model.fusion.xi.detach())
        metrics["fusion_bias"] = float(model.fusion.bias.detach())
    for index, group in enumerate(optimizer.param_groups):
        metrics[f"learning_rate_group_{index}"] = float(group["lr"])
    return metrics


@torch.no_grad()
def validate(
    model: DualHeadSwinUNETR,
    loader,
    device: torch.device,
    config: dict[str, Any],
    calibrated: bool,
    uncertainty_output_dir: Path | None = None,
    epoch: int = 0,
) -> dict[str, Any]:
    from monai.metrics import DiceMetric
    from monai.utils.enums import MetricReduction

    model.eval()
    dice_metrics = {
        name: DiceMetric(
            include_background=True,
            reduction=MetricReduction.MEAN_BATCH,
            get_not_nans=True,
        )
        for name in ("head1", "head2", "base")
    }
    metric_samples = MetricSampleReservoir(
        capacity=int(config["evaluation"].get("max_metric_voxels", 200000)),
        seed=int(config["reproducibility"]["seed"]),
    )
    positive_counts = {name: torch.zeros(3, dtype=torch.float64) for name in ("target", "head1", "head2", "base")}
    voxel_count = 0
    head_l1_sum = 0.0
    head_l1_count = 0
    nesting_violation_count = 0
    saved_uncertainty_maps: list[Path] = []
    monitoring = config.get("monitoring", {})
    if uncertainty_output_dir is not None and not calibrated:
        raise ValueError("Uncertainty maps can only be saved during calibration validation")
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        output = infer_volume(model, image, config, compute_uncertainty=calibrated)
        atomic = output.base_atomic_probability
        target_regions = batch["label_regions"].to(device)
        region_probabilities = {
            "head1": output.head_region_probabilities[0],
            "head2": output.head_region_probabilities[1],
            "base": output.base_region_probability,
        }
        reduction_dims = (0, *range(2, target_regions.ndim))
        positive_counts["target"] += target_regions.bool().sum(dim=reduction_dims).double().cpu()
        voxel_count += target_regions.shape[0] * math.prod(target_regions.shape[2:])
        for name, probability in region_probabilities.items():
            prediction = probability >= 0.5
            dice_metrics[name](y_pred=prediction, y=target_regions)
            positive_counts[name] += prediction.sum(dim=reduction_dims).double().cpu()
        base_prediction = region_probabilities["base"] >= 0.5
        nesting_violation_count += int(
            ((base_prediction[:, 2] & ~base_prediction[:, 0]) | (base_prediction[:, 0] & ~base_prediction[:, 1])).sum()
        )
        head_difference = (region_probabilities["head1"] - region_probabilities["head2"]).abs()
        head_l1_sum += float(head_difference.sum())
        head_l1_count += head_difference.numel()
        target_atomic = batch["label_atomic"].to(device).long()
        if uncertainty_output_dir is not None:
            saved_uncertainty_maps.extend(
                _save_uncertainty_batch(
                    output,
                    batch,
                    target_atomic,
                    uncertainty_output_dir,
                    batch_index,
                    epoch,
                    str(monitoring.get("uncertainty_map_dtype", "float16")),
                    bool(monitoring.get("uncertainty_map_components", True)),
                )
            )
        mask = (target_atomic > 0) | (atomic.argmax(dim=1) > 0)
        if not bool(mask.any()):
            mask = torch.ones_like(mask, dtype=torch.bool)
        samples = {
            "confidence": atomic.max(dim=1).values[mask].float().cpu().numpy(),
            "correct": (atomic.argmax(dim=1)[mask] == target_atomic[mask]).float().cpu().numpy(),
        }
        if calibrated:
            samples.update(
                uncertainty=output.uncertainty[:, 0][mask].float().cpu().numpy(),
                zernike_disagreement=output.zernike_disagreement[:, 0][mask].float().cpu().numpy(),
            )
        metric_samples.update(**samples)
        del (
            samples,
            mask,
            target_atomic,
            target_regions,
            atomic,
            prediction,
            probability,
            base_prediction,
            head_difference,
            region_probabilities,
            output,
            image,
            batch,
        )
        if device.type == "cuda" and bool(config["evaluation"].get("release_cuda_cache", True)):
            torch.cuda.empty_cache()
    confidence = metric_samples.values("confidence")
    correct = metric_samples.values("correct")
    aggregated_dice: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, metric in dice_metrics.items():
        dice_by_region, dice_not_nans = metric.aggregate()
        metric.reset()
        aggregated_dice[name] = (torch.where(dice_not_nans > 0, dice_by_region, torch.nan), dice_not_nans)
    reported_dice, dice_not_nans = aggregated_dice["base"]
    metrics = {
        "mean_dice": float(torch.nanmean(reported_dice)),
        "ece": expected_calibration_error(confidence, correct, int(config["evaluation"].get("calibration_bins", 15))),
        "head_region_l1": head_l1_sum / max(head_l1_count, 1),
        "nesting_violation_fraction": nesting_violation_count / max(voxel_count, 1),
    }
    for index, name in enumerate(("TC", "WT", "ET")):
        metrics[f"dice_{name}"] = float(reported_dice[index])
        metrics[f"dice_not_nans_{name}"] = float(dice_not_nans[index])
        metrics[f"target_positive_fraction_{name}"] = float(positive_counts["target"][index] / max(voxel_count, 1))
        for prediction_name in ("head1", "head2", "base"):
            dice_values, _ = aggregated_dice[prediction_name]
            metrics[f"{prediction_name}_dice_{name}"] = float(dice_values[index])
            metrics[f"{prediction_name}_positive_fraction_{name}"] = float(
                positive_counts[prediction_name][index] / max(voxel_count, 1)
            )
    if calibrated:
        uncertainty = metric_samples.values("uncertainty")
        zernike_disagreement = metric_samples.values("zernike_disagreement")
        metrics.update(
            risk_calibration_metrics(
                uncertainty,
                correct,
                int(config["evaluation"].get("calibration_bins", 15)),
            )
        )
        error = 1.0 - correct
        metrics.update(
            {
                "uncertainty_auroc": auroc(error, uncertainty),
                "uncertainty_aupr": aupr(error, uncertainty),
                "zernike_disagreement_auroc": auroc(error, zernike_disagreement),
                "zernike_disagreement_aupr": aupr(error, zernike_disagreement),
                "uncertainty_mean": float(np.mean(uncertainty)) if uncertainty.size else float("nan"),
                "uncertainty_std": float(np.std(uncertainty)) if uncertainty.size else float("nan"),
                "uncertainty_q05": float(np.quantile(uncertainty, 0.05)) if uncertainty.size else float("nan"),
                "uncertainty_q50": float(np.quantile(uncertainty, 0.50)) if uncertainty.size else float("nan"),
                "uncertainty_q95": float(np.quantile(uncertainty, 0.95)) if uncertainty.size else float("nan"),
                "uncertainty_correct_mean": float(np.mean(uncertainty[correct > 0.5]))
                if bool(np.any(correct > 0.5))
                else float("nan"),
                "uncertainty_error_mean": float(np.mean(uncertainty[correct <= 0.5]))
                if bool(np.any(correct <= 0.5))
                else float("nan"),
                "zernike_disagreement_mean": float(np.mean(zernike_disagreement))
                if zernike_disagreement.size
                else float("nan"),
                "zernike_disagreement_correct_mean": float(np.mean(zernike_disagreement[correct > 0.5]))
                if bool(np.any(correct > 0.5))
                else float("nan"),
                "zernike_disagreement_error_mean": float(np.mean(zernike_disagreement[correct <= 0.5]))
                if bool(np.any(correct <= 0.5))
                else float("nan"),
                "zernike_disagreement_error_gap": (
                    float(np.mean(zernike_disagreement[correct <= 0.5]))
                    - float(np.mean(zernike_disagreement[correct > 0.5]))
                )
                if bool(np.any(correct <= 0.5)) and bool(np.any(correct > 0.5))
                else float("nan"),
                "fusion_xi": float(model.fusion.xi.detach()),
                "fusion_bias": float(model.fusion.bias.detach()),
                "uncertainty_source": "zernike_disagreement",
            }
        )
    if uncertainty_output_dir is not None:
        save_components = bool(monitoring.get("uncertainty_map_components", True))
        keys = ["uncertainty", "error", "base_prediction", "target_atomic", "affine", "epoch"]
        if save_components:
            keys.append("zernike_disagreement")
        manifest = {
            "stage": "calibration",
            "epoch": epoch,
            "split": "val",
            "uncertainty_source": "zernike_disagreement",
            "case_count": len(saved_uncertainty_maps),
            "dtype": str(monitoring.get("uncertainty_map_dtype", "float16")),
            "components_saved": save_components,
            "keys": keys,
            "files": [path.name for path in saved_uncertainty_maps],
        }
        with (uncertainty_output_dir / "manifest.json").open("w") as stream:
            json.dump(manifest, stream, indent=2)
        metrics["uncertainty_maps_saved"] = float(len(saved_uncertainty_maps))
    return metrics


@torch.no_grad()
def fit_zernike_statistics(
    model: DualHeadSwinUNETR, loader, device: torch.device, config: dict[str, Any]
) -> dict[str, float]:
    """Fit source-domain descriptor statistics required by geometric risk."""
    model.eval()
    accumulator = WelfordAccumulator(tuple(model.zernike_stats.mean.shape))
    max_batches = int(config["zernike"].get("stats_max_batches", 0))
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        output = infer_volume(model, image, config, compute_uncertainty=False)
        mask = brain_mask(image)
        for probability in output.head_atomic_probabilities:
            for start, stop, descriptor in model.zernike.iter_descriptors(probability):
                accumulator.update(descriptor, mask[:, :, start:stop])
        print(f"zernike statistics batch {index + 1}/{len(loader)}", flush=True)
        del mask, output, image, batch
    mean, std, count = accumulator.finalize()
    model.zernike_stats.set_values(mean.to(device), std.to(device), count.to(device))
    if not model.zernike_stats.fitted:
        raise RuntimeError("Zernike statistics fitting did not collect at least two samples per component")
    return {
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "count_min": float(count.min()),
        "count_max": float(count.max()),
        "mean_nonfinite": float((~torch.isfinite(mean)).sum()),
        "std_nonfinite": float((~torch.isfinite(std)).sum()),
        "zero_or_negative_std": float((std <= 0).sum()),
    }


@torch.no_grad()
def fit_z0(model: DualHeadSwinUNETR, loader, device: torch.device, config: dict[str, Any]) -> float:
    model.eval()
    values = MetricSampleReservoir(
        capacity=int(config["evaluation"].get("max_metric_voxels", 200000)),
        seed=int(config["reproducibility"]["seed"]),
    )
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        output = infer_volume(model, image, config, compute_uncertainty=True)
        evidence = model.label_transfer.evidence(output.uncertainty)
        mask = brain_mask(image)
        selected = evidence[mask]
        values.update(evidence=selected.float().cpu().numpy())
        del selected, mask, evidence, output, image, batch
    evidence_values = values.values("evidence")
    if evidence_values.size == 0:
        raise RuntimeError("Cannot fit Z0 from an empty validation loader")
    quantile = float(config["label_transfer"].get("z0_quantile", 0.5))
    z0 = max(float(np.quantile(evidence_values, quantile)), 1e-6)
    model.label_transfer.set_z0(z0)
    return z0


def _load_required(path: Path, model: DualHeadSwinUNETR) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint does not exist: {path}")
    return load_checkpoint(path, model, strict=True)


def run(config: dict[str, Any], stage: str, checkpoint: str | None = None) -> Path:
    seed = int(config["reproducibility"]["seed"])
    set_reproducibility(seed, bool(config["reproducibility"].get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(run_dir, config)
    split_path = Path(config["paths"]["split_json"])
    if not split_path.is_file():
        print(f"Split file not found; generating {split_path}", flush=True)
        generate_splits(
            config["paths"]["data_root"],
            str(split_path),
            seed=seed,
            train_center=config["data"].get("train_center", "TCIA"),
            val_fraction=float(config["data"].get("val_fraction", 0.2)),
        )
    telemetry = TrainingTelemetry(run_dir, config, device)
    model = DualHeadSwinUNETR(config).to(device)
    loss_function = RegionDiceLoss()
    train_loader = build_loader(config, "train", "train")
    stats_loader = build_loader(config, "train", "stats")
    val_loader = build_loader(config, "val", "eval")
    telemetry.event(
        "dataloaders_ready",
        train_batches=len(train_loader),
        stats_batches=len(stats_loader),
        validation_batches=len(val_loader),
    )
    best_seg_path = run_dir / "best_seg.pt"
    stats_path = run_dir / "stats_fitted.pt"
    calibrated_path = run_dir / "best_calibrated.pt"

    if stage in ("warmup", "all"):
        telemetry.event("stage_started", stage="warmup")
        if checkpoint:
            payload = load_checkpoint(checkpoint, model, strict=True)
            telemetry.event(
                "checkpoint_loaded",
                purpose="warmup_start",
                path=str(Path(checkpoint).resolve()),
                source_stage=payload.get("stage"),
                source_epoch=payload.get("epoch"),
            )
        elif bool(config["model"].get("encoder_warm_start", True)):
            baseline = config["paths"].get("baseline_checkpoint")
            if baseline and Path(baseline).is_file():
                report = model.load_baseline_encoder(baseline)
                with (run_dir / "baseline_encoder_load.json").open("w") as stream:
                    json.dump(report, stream, indent=2)
                print(json.dumps(report, indent=2))
                telemetry.event("baseline_encoder_loaded", report=report)
            else:
                print(f"Baseline checkpoint not found; training from scratch: {baseline}")
                telemetry.event("anomaly", severity="ADVISORY", type="baseline_checkpoint_missing", path=baseline)
        epochs = int(config["training"]["warmup_epochs"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = _scheduler(optimizer, epochs, int(config["training"].get("scheduler_warmup_epochs", 20)))
        scaler = _make_scaler(bool(config["training"].get("amp", True)) and device.type == "cuda")
        best_dice = -math.inf
        for epoch in range(epochs):
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                loss_function,
                config,
                telemetry=telemetry,
                stage="warmup",
                epoch=epoch + 1,
            )
            scheduler.step()
            telemetry.epoch_finished(
                "warmup", epoch + 1, train_metrics, optimizer, model, train_metrics["duration_seconds"]
            )
            if (epoch + 1) % int(config["training"].get("validation_every", 5)) == 0 or epoch + 1 == epochs:
                metrics = validate(model, val_loader, device, config, calibrated=False)
                print(f"warmup epoch={epoch + 1} train={train_metrics} val={metrics}")
                telemetry.validation_finished("warmup", epoch + 1, metrics)
                _append_metrics(
                    run_dir,
                    {
                        "run_id": telemetry.run_id,
                        "stage": "warmup",
                        "epoch": epoch + 1,
                        "train": train_metrics,
                        "val": metrics,
                    },
                )
                save_checkpoint(run_dir / "last_warmup.pt", model, "warmup", epoch, config, optimizer, scheduler, metrics)
                if metrics["mean_dice"] > best_dice:
                    best_dice = metrics["mean_dice"]
                    save_checkpoint(best_seg_path, model, "warmup", epoch, config, optimizer, scheduler, metrics)
                    telemetry.event(
                        "best_checkpoint_saved",
                        stage="warmup",
                        epoch=epoch + 1,
                        metric="mean_dice",
                        value=best_dice,
                        path=str(best_seg_path),
                    )
        telemetry.event("stage_finished", stage="warmup", best_mean_dice=best_dice)

    if stage in ("stats", "all"):
        source = Path(checkpoint) if checkpoint and stage == "stats" else best_seg_path
        source_payload = _load_required(source, model)
        telemetry.event(
            "stage_started",
            stage="stats",
            source_checkpoint=str(source.resolve()),
            source_stage=source_payload.get("stage"),
            source_epoch=source_payload.get("epoch"),
        )
        zernike_health = fit_zernike_statistics(model, stats_loader, device, config)
        torch.save(
            {
                "mean": model.zernike_stats.mean.detach().cpu(),
                "std": model.zernike_stats.std.detach().cpu(),
                "count": model.zernike_stats.count.detach().cpu(),
            },
            run_dir / "zernike_statistics.pt",
        )
        save_checkpoint(
            stats_path,
            model,
            "stats",
            int(source_payload.get("epoch", -1)),
            config,
            metrics=source_payload.get("metrics"),
        )
        telemetry.event(
            "stage_finished",
            stage="stats",
            source_checkpoint=str(source.resolve()),
            statistics=zernike_health,
            checkpoint=str(stats_path),
        )

    if stage in ("calibration", "all"):
        source = Path(checkpoint) if checkpoint and stage == "calibration" else stats_path
        if not source.is_file() and stage == "calibration":
            source = best_seg_path
        source_payload = _load_required(source, model)
        if not model.zernike_stats.fitted:
            telemetry.event(
                "stage_started",
                stage="stats",
                source_checkpoint=str(source.resolve()),
                automatic=True,
            )
            zernike_health = fit_zernike_statistics(model, stats_loader, device, config)
            torch.save(
                {
                    "mean": model.zernike_stats.mean.detach().cpu(),
                    "std": model.zernike_stats.std.detach().cpu(),
                    "count": model.zernike_stats.count.detach().cpu(),
                },
                run_dir / "zernike_statistics.pt",
            )
            save_checkpoint(
                stats_path,
                model,
                "stats",
                int(source_payload.get("epoch", -1)),
                config,
                metrics=source_payload.get("metrics"),
            )
            telemetry.event(
                "stage_finished",
                stage="stats",
                automatic=True,
                statistics=zernike_health,
                checkpoint=str(stats_path),
            )
            source = stats_path
            source_payload = _load_required(source, model)
        reference_dice = float(source_payload.get("metrics", {}).get("mean_dice", math.nan))
        if not math.isfinite(reference_dice):
            reference_metrics = validate(model, val_loader, device, config, calibrated=False)
            reference_dice = float(reference_metrics["mean_dice"])
            if not math.isfinite(reference_dice):
                raise RuntimeError("Warmup checkpoint validation produced a non-finite reference Dice")
            telemetry.event(
                "reference_dice_fitted",
                source_checkpoint=str(source.resolve()),
                mean_dice=reference_dice,
                reason="checkpoint_metrics_missing_or_nonfinite",
            )
        calibration_scope = calibration_trainable_scope(config)
        margin_gradient_config = config["uncertainty"].get("margin_gradient", {})
        calibration_objective = (
            "geometric_uncertainty_weighted_atomic_margin"
            if bool(margin_gradient_config.get("enabled", False))
            and float(margin_gradient_config.get("weight", 0.01)) > 0.0
            else "disabled"
        )
        freeze_segmentation = False
        segmentation_sha256_before = segmentation_state_sha256(model)
        encoder_sha256_before = encoder_state_sha256(model)
        frozen_calibration_sha256_before = frozen_calibration_state_sha256(model)
        calibration_parameters = configure_calibration_trainability(model, calibration_scope)
        telemetry.event(
            "stage_started",
            stage="calibration",
            source_checkpoint=str(source.resolve()),
            source_stage=source_payload.get("stage"),
            source_epoch=source_payload.get("epoch"),
            uncertainty_source="zernike_disagreement",
            uncertainty_objective="geometric_uncertainty_weighted_atomic_margin",
            risk_brier_mode="diagnostic_only",
            calibration_trainable_scope=calibration_scope,
            freeze_segmentation=freeze_segmentation,
            fusion_frozen=True,
            calibration_input="independent_augmented_views_with_head_dropout",
            calibration_objective=calibration_objective,
            margin_gradient=margin_gradient_config,
            segmentation_sha256_before=segmentation_sha256_before,
            encoder_sha256_before=encoder_sha256_before,
            frozen_calibration_sha256_before=frozen_calibration_sha256_before,
        )
        epochs = int(config["training"]["calibration_epochs"])
        optimizer = torch.optim.AdamW(
            calibration_parameters,
            lr=float(config["training"]["calibration_learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = _scheduler(optimizer, epochs, min(5, epochs))
        scaler = _make_scaler(bool(config["training"].get("amp", True)) and device.type == "cuda")
        best_calibration_key: float | None = None
        selected_this_run = False
        last_path = run_dir / "last_calibration.pt"
        target_lambda = float(config["uncertainty"].get("lambda_u", 1.0))
        for epoch in range(epochs):
            lambda_u = target_lambda
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                loss_function,
                config,
                lambda_u,
                telemetry=telemetry,
                stage="calibration",
                epoch=epoch + 1,
                freeze_segmentation=freeze_segmentation,
                calibration_scope=calibration_scope,
            )
            scheduler.step()
            telemetry.epoch_finished(
                "calibration", epoch + 1, train_metrics, optimizer, model, train_metrics["duration_seconds"]
            )
            if (epoch + 1) % int(config["training"].get("validation_every", 5)) == 0 or epoch + 1 == epochs:
                map_every = int(config.get("monitoring", {}).get("uncertainty_map_every", 0))
                save_maps = map_every > 0 and (epoch + 1) % map_every == 0
                map_dir = (
                    run_dir / "uncertainty_maps" / "calibration" / f"epoch_{epoch + 1:04d}"
                    if save_maps
                    else None
                )
                metrics = validate(
                    model,
                    val_loader,
                    device,
                    config,
                    calibrated=True,
                    uncertainty_output_dir=map_dir,
                    epoch=epoch + 1,
                )
                metrics["lambda_u"] = lambda_u
                metrics["configured_risk_brier_weight"] = lambda_u
                metrics["calibration_loss_weight"] = 0.0
                metrics["freeze_segmentation"] = freeze_segmentation
                metrics["calibration_trainable_scope"] = calibration_scope
                metrics["calibration_input"] = "independent_augmented_views_with_head_dropout"
                metrics["uncertainty_objective"] = "geometric_uncertainty_weighted_atomic_margin"
                metrics["risk_brier_mode"] = "diagnostic_only"
                metrics["calibration_objective"] = calibration_objective
                metrics["margin_gradient"] = margin_gradient_config
                metrics["fusion_frozen"] = True
                metrics["segmentation_sha256_before"] = segmentation_sha256_before
                metrics["encoder_sha256_before"] = encoder_sha256_before
                metrics["frozen_calibration_sha256_before"] = frozen_calibration_sha256_before
                print(f"calibration epoch={epoch + 1} train={train_metrics} val={metrics}")
                telemetry.validation_finished("calibration", epoch + 1, metrics)
                if map_dir is not None:
                    telemetry.event(
                        "uncertainty_maps_saved",
                        stage="calibration",
                        epoch=epoch + 1,
                        path=str(map_dir),
                        case_count=int(metrics.get("uncertainty_maps_saved", 0)),
                    )
                _append_metrics(
                    run_dir,
                    {
                        "run_id": telemetry.run_id,
                        "stage": "calibration",
                        "epoch": epoch + 1,
                        "train": train_metrics,
                        "val": metrics,
                    },
                )
                save_checkpoint(last_path, model, "calibration", epoch, config, optimizer, scheduler, metrics)
                eligible = metrics["mean_dice"] >= reference_dice - float(config["training"].get("dice_tolerance", 0.01))
                candidate_key = calibration_checkpoint_key(metrics)
                finite_candidate = math.isfinite(candidate_key)
                if eligible and finite_candidate and (
                    best_calibration_key is None or candidate_key < best_calibration_key
                ):
                    best_calibration_key = candidate_key
                    selected_this_run = True
                    save_checkpoint(calibrated_path, model, "calibration", epoch, config, optimizer, scheduler, metrics)
                    telemetry.event(
                        "best_checkpoint_saved",
                        stage="calibration",
                        epoch=epoch + 1,
                        metric="ece",
                        value=float(metrics["ece"]),
                        risk_ece=float(metrics["risk_ece"]),
                        risk_brier=float(metrics["risk_brier"]),
                        mean_dice=metrics["mean_dice"],
                        path=str(calibrated_path),
                    )
        if not selected_this_run:
            telemetry.event(
                "anomaly",
                severity="RED_FLAG",
                type="no_dice_eligible_calibration_checkpoint",
                reference_dice=reference_dice,
                dice_tolerance=float(config["training"].get("dice_tolerance", 0.01)),
                diagnostic_checkpoint=str(last_path),
            )
            telemetry.close("failed", artifact=str(last_path))
            raise RuntimeError(
                "No calibration checkpoint satisfied the warmup Dice tolerance; "
                f"diagnostic checkpoint retained at {last_path}"
            )
        calibrated_payload = _load_required(calibrated_path, model)
        if "optimizer" in calibrated_payload:
            optimizer.load_state_dict(calibrated_payload["optimizer"])
        if "scheduler" in calibrated_payload:
            scheduler.load_state_dict(calibrated_payload["scheduler"])
        segmentation_sha256_after = segmentation_state_sha256(model)
        encoder_sha256_after = encoder_state_sha256(model)
        frozen_calibration_sha256_after = frozen_calibration_state_sha256(model)
        if freeze_segmentation and segmentation_sha256_after != segmentation_sha256_before:
            raise RuntimeError(
                "Frozen segmentation weights changed during calibration: "
                f"{segmentation_sha256_before} != {segmentation_sha256_after}"
            )
        if calibration_scope == "heads" and encoder_sha256_after != encoder_sha256_before:
            raise RuntimeError(
                "Frozen encoder weights changed during head-only calibration: "
                f"{encoder_sha256_before} != {encoder_sha256_after}"
            )
        if frozen_calibration_sha256_after != frozen_calibration_sha256_before:
            raise RuntimeError(
                "Frozen non-head model state changed during calibration: "
                f"{frozen_calibration_sha256_before} != {frozen_calibration_sha256_after}"
            )
        z0 = fit_z0(model, val_loader, device, config)
        telemetry.event("z0_fitted", value=z0, source="validation")
        final_path = run_dir / "final.pt"
        final_metrics = dict(calibrated_payload.get("metrics", {}))
        final_metrics.update(
            {
                "z0": z0,
                "uncertainty_source": "zernike_disagreement",
                "uncertainty_objective": "geometric_uncertainty_weighted_atomic_margin",
                "risk_brier_mode": "diagnostic_only",
                "fusion_frozen": True,
                "calibration_input": "independent_augmented_views_with_head_dropout",
                "calibration_objective": calibration_objective,
                "margin_gradient": margin_gradient_config,
                "freeze_segmentation": freeze_segmentation,
                "calibration_trainable_scope": calibration_scope,
                "segmentation_sha256_before": segmentation_sha256_before,
                "segmentation_sha256_after": segmentation_sha256_after,
                "encoder_sha256_before": encoder_sha256_before,
                "encoder_sha256_after": encoder_sha256_after,
                "frozen_calibration_sha256_before": frozen_calibration_sha256_before,
                "frozen_calibration_sha256_after": frozen_calibration_sha256_after,
            }
        )
        save_checkpoint(
            final_path,
            model,
            "postprocess",
            int(calibrated_payload.get("epoch", epochs - 1)),
            config,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=final_metrics,
        )
        print(f"fitted Z0={z0:.6f}; final checkpoint={final_path}")
        telemetry.event(
            "stage_finished",
            stage="calibration",
            best_ece=best_calibration_key,
            calibration_trainable_scope=calibration_scope,
            segmentation_sha256_before=segmentation_sha256_before,
            segmentation_sha256_after=segmentation_sha256_after,
            segmentation_unchanged=segmentation_sha256_before == segmentation_sha256_after,
            encoder_sha256_before=encoder_sha256_before,
            encoder_sha256_after=encoder_sha256_after,
            encoder_unchanged=encoder_sha256_before == encoder_sha256_after,
            frozen_calibration_sha256_before=frozen_calibration_sha256_before,
            frozen_calibration_sha256_after=frozen_calibration_sha256_after,
            frozen_calibration_unchanged=(
                frozen_calibration_sha256_before == frozen_calibration_sha256_after
            ),
        )
        telemetry.close("completed", artifact=str(final_path))
        return final_path
    artifact = stats_path if stage == "stats" else best_seg_path
    telemetry.close("completed", artifact=str(artifact))
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-head Swin UNETR with Zernike-risk calibration.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("configs") / "brats2020.yaml"))
    parser.add_argument("--stage", choices=("warmup", "stats", "calibration", "all"), default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    print(run(config, args.stage, args.checkpoint))


if __name__ == "__main__":
    main()
