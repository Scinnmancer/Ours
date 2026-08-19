from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .probability import scalar_to_atomic, scalar_to_regions


MODALITIES = ("t1", "t1ce", "t2", "flair")
KNOWN_CENTERS = ("CBICA", "TCIA", "2013", "TMC")


def datafold_read(datalist: str, basedir: str, key: str) -> list[dict[str, Any]]:
    with open(datalist) as stream:
        payload = json.load(stream)
    if key not in payload:
        raise KeyError(f"Split '{key}' is absent from {datalist}")
    records = payload[key]
    result = []
    for record in records:
        item = dict(record)
        item["image"] = [os.path.join(basedir, path) for path in item["image"]]
        item["label"] = os.path.join(basedir, item["label"])
        result.append(item)
    return result


def training_root(data_root: str) -> str:
    return os.path.join(data_root, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")


def _normalize_center(subject_ids: list[str]) -> str | None:
    for subject_id in subject_ids:
        if not subject_id or subject_id == "NA":
            continue
        for part in subject_id.replace("-", "_").split("_"):
            if part.startswith("CBICA"):
                return "CBICA"
            if part.startswith("TCIA"):
                return "TCIA"
            if part in ("2013", "TMC"):
                return part
    return None


def _make_record(data_root: str, row: dict[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    case_id = row["BraTS_2020_subject_ID"]
    case_dir = os.path.join(training_root(data_root), case_id)
    images = [os.path.join(case_dir, f"{case_id}_{modality}.nii") for modality in MODALITIES]
    label = os.path.join(case_dir, f"{case_id}_seg.nii")
    if not all(os.path.exists(path) for path in images + [label]):
        return None, "missing_standard_files"
    center = _normalize_center(
        [
            row.get("BraTS_2017_subject_ID", ""),
            row.get("BraTS_2018_subject_ID", ""),
            row.get("TCGA_TCIA_subject_ID", ""),
            row.get("BraTS_2019_subject_ID", ""),
        ]
    )
    if center is None:
        return None, "unknown_center"
    return {
        "image": [os.path.relpath(path, data_root) for path in images],
        "label": os.path.relpath(label, data_root),
        "subject_id": case_id,
        "center": center,
        "grade": row.get("Grade", ""),
    }, None


def generate_splits(
    data_root: str,
    output_path: str,
    seed: int = 2026,
    train_center: str = "TCIA",
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    train_center = train_center.upper()
    if train_center not in KNOWN_CENTERS:
        raise ValueError(f"train_center must be one of {KNOWN_CENTERS}")
    mapping_path = os.path.join(training_root(data_root), "name_mapping.csv")
    cases: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    with open(mapping_path, newline="") as stream:
        for row in csv.DictReader(stream):
            record, reason = _make_record(data_root, row)
            if record is None:
                skipped.append({"subject_id": row.get("BraTS_2020_subject_ID", ""), "reason": str(reason)})
            else:
                cases.append(record)
    source = sorted((case for case in cases if case["center"] == train_center), key=lambda item: item["subject_id"])
    if len(source) < 2:
        raise ValueError(f"Not enough cases for source center {train_center}")
    random.Random(seed).shuffle(source)
    val_count = int(round(len(source) * val_fraction))
    if not 0 < val_count < len(source):
        raise ValueError("val_fraction produces an empty train or validation split")
    result: dict[str, Any] = {
        "train": sorted(source[val_count:], key=lambda item: item["subject_id"]),
        "val": sorted(source[:val_count], key=lambda item: item["subject_id"]),
    }
    test_splits = []
    test_cases = []
    for center in KNOWN_CENTERS:
        if center == train_center:
            continue
        key = f"test_{center}"
        result[key] = sorted((case for case in cases if case["center"] == center), key=lambda item: item["subject_id"])
        test_splits.append(key)
        test_cases.extend(result[key])
    result["test"] = sorted(test_cases, key=lambda item: item["subject_id"])
    result["center_ood"] = list(result["test"])
    result["id_test"] = []
    count_keys = ["train", "val", "id_test", "test", "center_ood", *test_splits]
    result["metadata"] = {
        "split_strategy": "single_center_train_separate_domain_tests",
        "seed": seed,
        "train_center": train_center,
        "test_centers": [key.removeprefix("test_") for key in test_splits],
        "test_splits": test_splits,
        "val_fraction": val_fraction,
        "modalities": list(MODALITIES),
        "counts": {key: len(result[key]) for key in count_keys},
        "center_counts": dict(Counter(case["center"] for case in cases)),
        "skipped": skipped,
    }
    result["metadata"]["counts"]["skipped"] = len(skipped)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(result, stream, indent=2)
    return result


class ConvertBraTSLabelsd:
    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        label = torch.as_tensor(result["label"])
        if label.ndim == 3:
            label = label.unsqueeze(0)
        if label.ndim != 4 or label.shape[0] != 1:
            raise ValueError(f"Expected scalar label [1,D,H,W], got {tuple(label.shape)}")
        batched = label.unsqueeze(0)
        result["label_scalar"] = label.long()
        result["label_atomic"] = scalar_to_atomic(batched).squeeze(0)
        result["label_regions"] = scalar_to_regions(batched).squeeze(0)
        return result


def _monai_transforms(config: dict[str, Any], mode: str):
    from monai import transforms

    roi = [int(value) for value in config["data"]["roi_size"]]
    common: list[Any] = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys="image", channel_dim=0),
        transforms.EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
    ]
    resample = config["data"].get("resample", {})
    if resample.get("enabled", False):
        common.extend(
            [
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=tuple(float(v) for v in resample.get("spacing", [1.0, 1.0, 1.0])),
                    mode=("bilinear", "nearest"),
                ),
            ]
        )
    if mode in ("train", "stats"):
        divisible = roi if mode == "train" else [32, 32, 32]
        common.append(
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=divisible, allow_smaller=True
            )
        )
    if mode == "train":
        common.append(transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False))
    if mode == "train":
        probability = float(config["data"].get("flip_probability", 0.2))
        for axis in range(3):
            common.append(transforms.RandFlipd(keys=["image", "label"], prob=probability, spatial_axis=axis))
    common.extend(
        [
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ConvertBraTSLabelsd(),
        ]
    )
    if mode == "train":
        common.extend(
            [
                transforms.CopyItemsd(keys="image", times=2, names=["image_view1", "image_view2"]),
                transforms.RandScaleIntensityd(
                    keys="image_view1", factors=0.1, prob=float(config["data"].get("scale_probability", 0.1))
                ),
                transforms.RandShiftIntensityd(
                    keys="image_view1", offsets=0.1, prob=float(config["data"].get("shift_probability", 0.1))
                ),
                transforms.RandScaleIntensityd(
                    keys="image_view2", factors=0.1, prob=float(config["data"].get("scale_probability", 0.1))
                ),
                transforms.RandShiftIntensityd(
                    keys="image_view2", offsets=0.1, prob=float(config["data"].get("shift_probability", 0.1))
                ),
            ]
        )
    keys = ["image", "label_scalar", "label_atomic", "label_regions"]
    if mode == "train":
        keys.extend(["image_view1", "image_view2"])
    common.append(transforms.ToTensord(keys=keys))
    return transforms.Compose(common)


def _loader_runtime_options(config: dict[str, Any], mode: str) -> dict[str, Any]:
    is_evaluation = mode == "eval"
    workers = int(
        config.get("evaluation", {}).get("workers", 0)
        if is_evaluation
        else config["data"].get("workers", 8)
    )
    return {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available() and not is_evaluation,
        "persistent_workers": workers > 0 and not is_evaluation,
    }


def build_loader(
    config: dict[str, Any],
    split: str,
    mode: str,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    from monai import data

    records = datafold_read(config["paths"]["split_json"], config["paths"]["data_root"], split)
    debug_cases = int(config["data"].get("debug_cases", 0))
    if debug_cases:
        records = records[:debug_cases]
    dataset = data.Dataset(data=records, transform=_monai_transforms(config, mode))
    sampler = None
    shuffle = mode == "train"
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    runtime_options = _loader_runtime_options(config, mode)
    return data.DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 1)) if mode == "train" else 1,
        shuffle=shuffle,
        sampler=sampler,
        # Evaluation batches contain several full-volume label representations.
        # Pinning/prefetching them can consume multiple GiB without accelerating
        # the single image transfer performed by the evaluator.
        **runtime_options,
    )


def brain_mask(image: torch.Tensor) -> torch.Tensor:
    return image.abs().sum(dim=1, keepdim=True) > 0
