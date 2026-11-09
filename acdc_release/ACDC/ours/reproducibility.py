from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    try:
        from monai.utils import set_determinism

        set_determinism(seed=seed)
    except ImportError:
        pass


def environment_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    for package in ("monai", "nibabel", "scipy", "yaml"):
        try:
            module = __import__(package)
            result[package] = getattr(module, "__version__", "unknown")
        except Exception as error:
            result[f"{package}_error"] = str(error)
    return result


def save_run_metadata(run_dir: str | os.PathLike[str], config: dict[str, Any]) -> None:
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "environment.json").open("w") as stream:
        json.dump(environment_info(), stream, indent=2, sort_keys=True)
    with (output / "resolved_config.json").open("w") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
