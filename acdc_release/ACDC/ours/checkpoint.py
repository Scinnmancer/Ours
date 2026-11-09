from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def safe_torch_load(path: str | os.PathLike[str]) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    stage: str,
    epoch: int,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    metrics: dict[str, float] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    payload: dict[str, Any] = {
        "stage": stage,
        "epoch": int(epoch),
        "state_dict": raw_model.state_dict(),
        "config": config,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, output)


def load_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    strict: bool = True,
) -> dict[str, Any]:
    payload = safe_torch_load(path)
    state = payload.get("state_dict", payload)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
