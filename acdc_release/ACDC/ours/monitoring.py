from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(float(value.detach().cpu()))
        return [_json_safe(item) for item in value.detach().cpu().tolist()]
    return str(value)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(package_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(package_root.glob("*.py")) + sorted((package_root / "configs").glob("*.yaml"))
    for path in files:
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _split_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text())
    split_names = [key for key, value in payload.items() if isinstance(value, list)]
    identifiers = {
        split: {str(item.get("subject_id", item.get("label", ""))) for item in payload[split]}
        for split in split_names
    }
    disjoint = {}
    for first in split_names:
        for second in split_names:
            if first >= second:
                continue
            if first in ("test", "center_ood") or second in ("test", "center_ood"):
                continue
            overlap = identifiers[first] & identifiers[second]
            disjoint[f"{first}__{second}"] = len(overlap)
    public_metadata = {
        key: value for key, value in payload.get("metadata", {}).items() if key not in ("skipped",)
    }
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "counts": {split: len(payload[split]) for split in split_names},
        "pairwise_overlap_counts": disjoint,
        "metadata": public_metadata,
    }


def _device_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result["gpu"] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
        }
    return result


def resource_snapshot(device: torch.device) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB and macOS reports bytes. Training targets Linux.
    rss_mb = float(usage.ru_maxrss) / (1024.0 if platform.system() != "Darwin" else 1024.0**2)
    result: dict[str, Any] = {
        "max_rss_mb": rss_mb,
        "cpu_user_seconds": float(usage.ru_utime),
        "cpu_system_seconds": float(usage.ru_stime),
    }
    if hasattr(os, "getloadavg"):
        result["load_average"] = list(os.getloadavg())
    if device.type == "cuda":
        result.update(
            {
                "gpu_allocated_mb": torch.cuda.memory_allocated(device) / 1024.0**2,
                "gpu_reserved_mb": torch.cuda.memory_reserved(device) / 1024.0**2,
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024.0**2,
                "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024.0**2,
            }
        )
    return result


def gradient_statistics(parameters: Iterable[torch.nn.Parameter]) -> dict[str, float | int]:
    squared_norm = 0.0
    maximum = 0.0
    nonfinite = 0
    tensors = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach().float()
        finite = torch.isfinite(detached)
        nonfinite += int((~finite).sum().item())
        if bool(finite.any()):
            values = detached[finite]
            squared_norm += float(torch.sum(values * values))
            maximum = max(maximum, float(values.abs().max()))
        tensors += 1
    return {
        "gradient_l2": math.sqrt(squared_norm),
        "gradient_abs_max": maximum,
        "gradient_nonfinite": nonfinite,
        "gradient_tensors": tensors,
    }


def parameter_statistics(model: torch.nn.Module) -> dict[str, float | int]:
    squared_norm = 0.0
    nonfinite = 0
    count = 0
    for parameter in model.parameters():
        detached = parameter.detach().float()
        finite = torch.isfinite(detached)
        nonfinite += int((~finite).sum().item())
        if bool(finite.any()):
            values = detached[finite]
            squared_norm += float(torch.sum(values * values))
            count += values.numel()
    return {
        "parameter_l2": math.sqrt(squared_norm),
        "parameter_nonfinite": nonfinite,
        "parameter_count": count,
    }


class TrainingTelemetry:
    def __init__(self, run_dir: Path, config: dict[str, Any], device: torch.device):
        self.enabled = bool(config.get("monitoring", {}).get("enabled", True))
        self.device = device
        self.config = config
        self.started = time.perf_counter()
        self.run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + f"_pid{os.getpid()}"
        self.directory = run_dir / "telemetry" / self.run_id
        self.events_path = self.directory / "events.jsonl"
        self._best: dict[str, float] = {}
        self._stale: dict[str, int] = {}
        self._closed = False
        self._resource_alerted = False
        self._previous_excepthook = sys.excepthook
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=False)
        package_root = Path(__file__).resolve().parent
        project_root = package_root.parent
        split_path = Path(config["paths"]["split_json"])
        baseline_path = Path(config["paths"].get("baseline_checkpoint", ""))
        baseline: dict[str, Any] = {"path": str(baseline_path), "exists": baseline_path.is_file()}
        if baseline_path.is_file():
            baseline["size_bytes"] = baseline_path.stat().st_size
            if bool(config.get("monitoring", {}).get("hash_baseline_checkpoint", True)):
                baseline["sha256"] = sha256_file(baseline_path)
        manifest = {
            "run_id": self.run_id,
            "started_at": _utc_now(),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "command": sys.argv,
            "working_directory": os.getcwd(),
            "python": sys.version,
            "platform": platform.platform(),
            "source_sha256": source_fingerprint(package_root),
            "git": _git_metadata(project_root),
            "device": _device_metadata(device),
            "split": _split_metadata(split_path),
            "baseline_checkpoint": baseline,
            "config": config,
        }
        (self.directory / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
        self.initial_resources = resource_snapshot(device)
        self.event("run_started", resources=self.initial_resources)
        sys.excepthook = self._exception_hook

    def _exception_hook(self, exception_type, exception, traceback) -> None:
        self.close(
            "crashed",
            exception_type=getattr(exception_type, "__name__", str(exception_type)),
            exception=str(exception),
        )
        self._previous_excepthook(exception_type, exception, traceback)

    def event(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record = {
            "run_id": self.run_id,
            "event": event,
            "timestamp": _utc_now(),
            "elapsed_seconds": time.perf_counter() - self.started,
            **payload,
        }
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")

    def reset_peak_memory(self) -> None:
        if self.enabled and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def epoch_finished(
        self,
        stage: str,
        epoch: int,
        metrics: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        duration_seconds: float,
    ) -> None:
        parameters = parameter_statistics(model)
        resources = resource_snapshot(self.device)
        self.event(
            "epoch_finished",
            stage=stage,
            epoch=epoch,
            duration_seconds=duration_seconds,
            next_learning_rates=[group["lr"] for group in optimizer.param_groups],
            metrics=metrics,
            parameters=parameters,
            resources=resources,
        )
        if float(metrics.get("gradient_nonfinite", 0.0)) > 0 or parameters["parameter_nonfinite"] > 0:
            self.event(
                "anomaly",
                severity="RED_FLAG",
                type="nonfinite_training_state",
                stage=stage,
                epoch=epoch,
                gradient_nonfinite=metrics.get("gradient_nonfinite", 0.0),
                parameter_nonfinite=parameters["parameter_nonfinite"],
            )
        if float(metrics.get("amp_skipped_steps", 0.0)) > 0:
            self.event(
                "anomaly",
                severity="ADVISORY",
                type="amp_steps_skipped",
                stage=stage,
                epoch=epoch,
                count=metrics["amp_skipped_steps"],
            )
        initial_rss = float(self.initial_resources.get("max_rss_mb", 0.0))
        multiplier = float(self.config.get("monitoring", {}).get("memory_multiplier_limit", 3.0))
        if (
            not self._resource_alerted
            and initial_rss > 0
            and float(resources.get("max_rss_mb", 0.0)) > multiplier * initial_rss
        ):
            self._resource_alerted = True
            self.event(
                "anomaly",
                severity="ADVISORY",
                type="resource_growth",
                stage=stage,
                epoch=epoch,
                initial_max_rss_mb=initial_rss,
                current_max_rss_mb=resources["max_rss_mb"],
                multiplier_limit=multiplier,
            )

    def validation_finished(self, stage: str, epoch: int, metrics: dict[str, Any]) -> None:
        self.event("validation_finished", stage=stage, epoch=epoch, metrics=metrics)
        primary_metric = "ece" if stage == "calibration" else "mean_dice"
        key = f"{stage}:{primary_metric}"
        value = float(metrics.get(primary_metric, float("nan")))
        if not math.isfinite(value):
            self.event(
                "anomaly",
                severity="RED_FLAG",
                type="nonfinite_metric",
                stage=stage,
                epoch=epoch,
                metric=primary_metric,
            )
            return
        previous_best = self._best.get(key, math.inf if stage == "calibration" else -math.inf)
        improved = value < previous_best if stage == "calibration" else value > previous_best
        if improved:
            self._best[key] = value
            self._stale[key] = 0
            return
        self._stale[key] = self._stale.get(key, 0) + 1
        patience = int(self.config.get("monitoring", {}).get("plateau_patience_validations", 10))
        if patience > 0 and self._stale[key] == patience:
            self.event(
                "anomaly",
                severity="ADVISORY",
                type="metric_plateau",
                stage=stage,
                epoch=epoch,
                primary_metric=primary_metric,
                best=self._best[key],
                validations_without_improvement=self._stale[key],
            )

    def close(self, status: str, **payload: Any) -> None:
        if self._closed:
            return
        self._closed = True
        self.event("run_finished", status=status, resources=resource_snapshot(self.device), **payload)
        if self.enabled and sys.excepthook == self._exception_hook:
            sys.excepthook = self._previous_excepthook
