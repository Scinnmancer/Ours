from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from .model import DualHeadOutput, DualHeadSwinUNETR


def autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def infer_volume(
    model: DualHeadSwinUNETR,
    image: torch.Tensor,
    config: dict[str, Any],
    compute_uncertainty: bool,
    refine: bool = False,
) -> DualHeadOutput:
    if refine and not compute_uncertainty:
        raise ValueError("refine=True requires compute_uncertainty=True")
    from monai.inferers import sliding_window_inference

    roi = tuple(int(value) for value in config["data"]["roi_size"])
    training = config["training"]
    device = image.device
    with autocast_context(device, bool(training.get("amp", True))):
        stacked = sliding_window_inference(
            image,
            roi_size=roi,
            sw_batch_size=int(training.get("sw_batch_size", 4)),
            predictor=model.stacked_logits,
            overlap=float(training.get("infer_overlap", 0.5)),
        )
        logits1, logits2 = stacked[:, :3], stacked[:, 3:]
        return model.output_from_logits(
            logits1,
            logits2,
            compute_uncertainty=compute_uncertainty,
            refine=refine,
        )
