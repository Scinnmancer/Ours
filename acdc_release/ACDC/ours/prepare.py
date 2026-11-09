from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import SimpleITK as sitk

from .config import load_config
from .data import preprocess_case, read_records, validate_datalist


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_split_builder() -> ModuleType:
    path = PROJECT_ROOT / "baseline" / "build_acdc_split.py"
    if not path.is_file():
        raise FileNotFoundError(f"ACDC baseline split builder is required: {path}")
    spec = importlib.util.spec_from_file_location("acdc_baseline_split_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline split builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_split(config: dict[str, Any]) -> dict[str, Any]:
    builder = _load_split_builder()
    args = argparse.Namespace(
        data_dir=config["paths"]["data_root"],
        metadata=str(PROJECT_ROOT / "baseline" / "acdc_patient001_100_scanner_metadata.csv"),
        output=config["paths"]["split_json"],
        seed=int(config["reproducibility"]["seed"]),
        val_fraction=0.2,
        check_labels=True,
    )
    payload = builder.build_split(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_dataset(config: dict[str, Any], limit: int = 0) -> dict[str, Any]:
    split_path = Path(config["paths"]["split_json"])
    with split_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    validate_datalist(payload)
    roi = tuple(int(value) for value in config["data"]["roi_size"])
    target_spacing = tuple(float(value) for value in config["data"]["target_spacing"])
    checked = 0
    problems: list[str] = []
    for split in ("train", "val", "test"):
        records = read_records(str(split_path), config["paths"]["data_root"], split)
        for record in records:
            if limit and checked >= limit:
                break
            try:
                image = sitk.ReadImage(record["image"])
                label = sitk.ReadImage(record["label"])
                if image.GetSize() != label.GetSize():
                    raise ValueError("image/label size mismatch")
                if image.GetSpacing() != label.GetSpacing():
                    raise ValueError("image/label spacing mismatch")
                prepared_image, prepared_label, spacing = preprocess_case(
                    record["image"], record["label"], roi, target_spacing
                )
                values = set(np.unique(prepared_label.numpy()).astype(int).tolist())
                if not values <= {0, 1, 2, 3}:
                    raise ValueError(f"invalid labels {sorted(values)}")
                if tuple(prepared_image.shape) != tuple(prepared_label.shape):
                    raise ValueError("preprocessed image/label shape mismatch")
                if not np.allclose(spacing, target_spacing[::-1], atol=1e-6):
                    raise ValueError(f"unexpected DHW spacing {spacing.tolist()}")
            except Exception as error:
                problems.append(f"{split}/{record.get('subject_id')}: {error}")
            checked += 1
        if limit and checked >= limit:
            break
    report = {
        "checked_records": checked,
        "patient_counts": payload["metadata"]["patient_counts"],
        "record_counts": payload["metadata"]["record_counts"],
        "source_domain": payload["metadata"]["source_domain"],
        "target_domain": payload["metadata"]["target_domain"],
        "problems": problems,
    }
    if problems:
        raise RuntimeError("Dataset validation failed:\n" + "\n".join(problems[:50]))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate the fixed ACDC 1.5T-to-3T split.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("configs") / "acdc.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit validation records; 0 checks all records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    if not args.no_generate:
        payload = generate_split(config)
        print(json.dumps(payload["metadata"]["record_counts"], indent=2))
    print(json.dumps(validate_dataset(config, args.limit), indent=2))


if __name__ == "__main__":
    main()
