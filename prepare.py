from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .config import load_config
from .data import generate_splits


def validate_dataset(config: dict[str, Any], limit: int = 0) -> dict[str, Any]:
    split_path = config["paths"]["split_json"]
    data_root = config["paths"]["data_root"]
    with open(split_path) as stream:
        payload = json.load(stream)
    splits = ["train", "val", *payload.get("metadata", {}).get("test_splits", [])]
    allowed = {0, 1, 2, 4}
    target_spacing = np.asarray(config["data"].get("resample", {}).get("spacing", [1.0, 1.0, 1.0]), dtype=float)
    tolerance = float(config["data"].get("spacing_tolerance", 0.05))
    resample_enabled = bool(config["data"].get("resample", {}).get("enabled", False))
    checked = 0
    problems: list[str] = []
    for split in splits:
        for record in payload.get(split, []):
            if limit and checked >= limit:
                break
            paths = [os.path.join(data_root, item) for item in record["image"]]
            label_path = os.path.join(data_root, record["label"])
            if len(paths) != 4:
                problems.append(f"{record.get('subject_id')}: expected four modalities")
                continue
            missing = [path for path in [*paths, label_path] if not os.path.isfile(path)]
            if missing:
                problems.extend(f"missing: {path}" for path in missing)
                continue
            images = [nib.load(path) for path in paths]
            label_image = nib.load(label_path)
            shapes = [image.shape for image in images] + [label_image.shape]
            if len(set(shapes)) != 1:
                problems.append(f"{record.get('subject_id')}: inconsistent shapes {shapes}")
            affines = [image.affine for image in images] + [label_image.affine]
            if not all(np.allclose(affines[0], affine, atol=1e-4) for affine in affines[1:]):
                problems.append(f"{record.get('subject_id')}: inconsistent affines")
            spacing = np.asarray(images[0].header.get_zooms()[:3], dtype=float)
            if not resample_enabled and not np.allclose(spacing, target_spacing, atol=tolerance):
                problems.append(f"{record.get('subject_id')}: spacing {spacing.tolist()} outside tolerance")
            labels = set(np.unique(np.asanyarray(label_image.dataobj)).astype(int).tolist())
            if not labels.issubset(allowed):
                problems.append(f"{record.get('subject_id')}: invalid labels {sorted(labels - allowed)}")
            checked += 1
        if limit and checked >= limit:
            break
    report = {"checked_cases": checked, "splits": splits, "problems": problems}
    if problems:
        raise RuntimeError("Dataset validation failed:\n" + "\n".join(problems[:50]))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and validate BraTS 2020 reliability splits.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("configs") / "brats2020.yaml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit validation cases; 0 checks all cases.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    if not args.no_generate:
        result = generate_splits(
            config["paths"]["data_root"],
            config["paths"]["split_json"],
            seed=int(config["reproducibility"]["seed"]),
            train_center=config["data"].get("train_center", "TCIA"),
            val_fraction=float(config["data"].get("val_fraction", 0.2)),
        )
        print(json.dumps(result["metadata"]["counts"], indent=2))
    report = validate_dataset(config, args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

