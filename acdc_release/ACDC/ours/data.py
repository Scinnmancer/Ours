"""ACDC fixed-split loading using the current executable baseline recipe."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .augmentation import (
    additive_brightness,
    gaussian_noise,
    random_affine_3d,
    random_crop_3d,
    random_gamma,
)


EXPECTED_PATIENTS = {"train": 54, "val": 13, "test": 33}
EXPECTED_RECORDS = {"train": 108, "val": 26, "test": 66}
SOURCE_DOMAIN = "source_1.5T"
TARGET_DOMAIN = "target_3T"
CLASS_NAMES = ("BG", "RV", "MYO", "LV")


def validate_datalist(payload: dict[str, Any]) -> None:
    patient_sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        records = payload.get(split, [])
        patients = {row["patient_id"] for row in records}
        if len(records) != EXPECTED_RECORDS[split] or len(patients) != EXPECTED_PATIENTS[split]:
            raise ValueError(
                f"Unexpected {split} size: {len(patients)} patients/{len(records)} records; "
                f"expected {EXPECTED_PATIENTS[split]}/{EXPECTED_RECORDS[split]}"
            )
        subjects = [row["subject_id"] for row in records]
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"Duplicate subject_id in {split}")
        frames: dict[str, set[str]] = {}
        for row in records:
            frames.setdefault(row["patient_id"], set()).add(row["frame"])
        if any(value != {"ED", "ES"} for value in frames.values()):
            raise ValueError(f"Split {split} contains an incomplete ED/ES patient")
        patient_sets[split] = patients
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise ValueError(f"Patient leakage between {left} and {right}: {sorted(overlap)}")
    source = {row["domain"] for split in ("train", "val") for row in payload[split]}
    target = {row["domain"] for row in payload["test"]}
    if source != {SOURCE_DOMAIN} or target != {TARGET_DOMAIN}:
        raise ValueError("Expected source_1.5T train/val and target_3T test")


def read_records(json_list: str, data_dir: str, split: str, strict: bool = True) -> list[dict[str, Any]]:
    json_path = Path(json_list).expanduser().resolve()
    root = Path(data_dir).expanduser().resolve()
    with json_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if split not in ("train", "val", "test") or not isinstance(payload.get(split), list):
        raise ValueError(f"Missing ACDC split {split!r} in {json_path}")
    if strict:
        validate_datalist(payload)
    records = copy.deepcopy(payload[split])
    for record in records:
        for key in ("image", "label"):
            value = Path(record[key]).expanduser()
            resolved = value if value.is_absolute() else root / value
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            record[key] = str(resolved)
    return records


def _reference_image(source: sitk.Image, spacing: tuple[float, float, float]) -> sitk.Image:
    old_spacing = source.GetSpacing()
    old_size = source.GetSize()
    new_size = tuple(
        int(round(old_size[index] * old_spacing[index] / spacing[index]))
        for index in range(3)
    )
    result = sitk.Image(new_size, source.GetPixelIDValue())
    result.SetSpacing(spacing)
    result.SetOrigin(source.GetOrigin())
    result.SetDirection(source.GetDirection())
    return result


def _resample(
    image: sitk.Image,
    label: sitk.Image,
    target_spacing: tuple[float, float, float],
) -> tuple[sitk.Image, sitk.Image]:
    identity = sitk.Transform(3, sitk.sitkIdentity)
    spacing = image.GetSpacing()
    xy_spacing = (target_spacing[0], target_spacing[1], spacing[2])
    xy_reference = _reference_image(image, xy_spacing)
    image_xy = sitk.Resample(image, xy_reference, identity, sitk.sitkBSpline)
    label_xy = sitk.Resample(label, xy_reference, identity, sitk.sitkNearestNeighbor)
    xyz_reference = _reference_image(image_xy, target_spacing)
    image_xyz = sitk.Resample(image_xy, xyz_reference, identity, sitk.sitkNearestNeighbor)
    label_xyz = sitk.Resample(label_xy, xyz_reference, identity, sitk.sitkNearestNeighbor)
    return image_xyz, label_xyz


def _padding(shape: tuple[int, ...], minimum_shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    result = []
    for actual, minimum in zip(shape, minimum_shape):
        missing = max(minimum - actual, 0)
        before = missing // 2
        result.append((before, missing - before))
    return tuple(result)


def _pad(array: np.ndarray, minimum_shape: tuple[int, ...]) -> np.ndarray:
    return np.pad(array, _padding(array.shape, minimum_shape))


def _center_crop_xy(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[1] > 256:
        start = image.shape[1] // 2 - 128
        image, label = image[:, start : start + 256], label[:, start : start + 256]
    if image.shape[2] > 256:
        start = image.shape[2] // 2 - 128
        image, label = image[:, :, start : start + 256], label[:, :, start : start + 256]
    return image, label


def preprocess_case(
    image_path: str,
    label_path: str,
    roi_size: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)
    if image.GetSize() != label.GetSize() or image.GetSpacing() != label.GetSpacing():
        raise ValueError(f"Image/label geometry mismatch for {image_path}")
    image, label = _resample(image, label, target_spacing)
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    label_array = sitk.GetArrayFromImage(label).astype(np.uint8)
    values = set(np.unique(label_array).astype(int).tolist())
    if not values <= {0, 1, 2, 3}:
        raise ValueError(f"Unexpected ACDC labels in {label_path}: {sorted(values)}")
    percentile = float(np.percentile(image_array, 98))
    if not np.isfinite(percentile) or percentile <= 0:
        raise ValueError(f"Invalid intensity percentile for {image_path}")
    image_array = np.clip(image_array, 0, percentile) / percentile
    image_array = _pad(image_array, roi_size)
    label_array = _pad(label_array, roi_size)
    image_array, label_array = _center_crop_xy(image_array, label_array)
    spacing = np.asarray(label.GetSpacing(), dtype=np.float64)[::-1].copy()
    return (
        torch.from_numpy(image_array.astype(np.float32)),
        torch.from_numpy(label_array.astype(np.uint8)).long(),
        spacing,
    )


class ACDCDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        mode: str,
        roi_size: tuple[int, int, int],
        target_spacing: tuple[float, float, float],
    ):
        if mode not in ("train", "stats", "eval", "test"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.roi_size = tuple(roi_size)
        self.target_spacing = tuple(target_spacing)
        self.images: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.spacings: list[np.ndarray] = []
        self.records = records
        print(f"Loading ACDC {mode}: {len(records)} records")
        for index, record in enumerate(records, start=1):
            image, label, spacing = preprocess_case(
                record["image"], record["label"], self.roi_size, self.target_spacing
            )
            self.images.append(image)
            self.labels.append(label)
            self.spacings.append(spacing)
            if index == len(records) or index % 20 == 0:
                print(f"Loading ACDC {mode}: {index}/{len(records)}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _intensity_view(image: torch.Tensor) -> torch.Tensor:
        image = gaussian_noise(image, 0.02)
        image = additive_brightness(image, 0.7)
        return random_gamma(image, (0.5, 1.6), retain_stats=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base = self.images[index].unsqueeze(0).unsqueeze(0)
        label = self.labels[index].unsqueeze(0).unsqueeze(0)
        if self.mode == "train":
            view1 = self._intensity_view(base.clone())
            view2 = self._intensity_view(base.clone())
            paired = torch.cat((view1, view2), dim=1)
            paired, label = random_affine_3d(
                paired,
                label,
                scale=(0.1, 0.3, 0.3),
                rotate=(30, 0, 0),
                translate=(0, 0, 0),
            )
            paired, label = random_crop_3d(paired, label, self.roi_size)
            view1, view2 = paired[:, 0:1], paired[:, 1:2]
            image = view1
        else:
            image = base
            view1 = view2 = None
        result: dict[str, Any] = {
            "image": image.squeeze(0),
            "label": label.squeeze(0),
            "spacing": torch.as_tensor(np.ascontiguousarray(self.spacings[index]), dtype=torch.float64),
            "image_path": self.records[index]["image"],
            "label_path": self.records[index]["label"],
        }
        if view1 is not None and view2 is not None:
            result["image_view1"] = view1.squeeze(0)
            result["image_view2"] = view2.squeeze(0)
        result.update(
            {
                key: self.records[index].get(key, "")
                for key in ("subject_id", "patient_id", "frame", "domain", "group")
            }
        )
        return result


def build_loader(
    config: dict[str, Any],
    split: str,
    mode: str,
    distributed: bool = False,
) -> DataLoader:
    records = read_records(config["paths"]["split_json"], config["paths"]["data_root"], split)
    debug_cases = int(config["data"].get("debug_cases", 0))
    if debug_cases:
        records = records[:debug_cases]
    dataset = ACDCDataset(
        records,
        mode,
        tuple(int(value) for value in config["data"]["roi_size"]),
        tuple(float(value) for value in config["data"]["target_spacing"]),
    )
    sampler = None
    paper_shuffle = mode == "train" and bool(config["training"].get("paper_aligned", False))
    if mode == "train" and not paper_shuffle:
        sample_count = int(config["training"].get("iterations_per_epoch", 200)) * int(
            config["training"].get("batch_size", 1)
        )
        sampler = RandomSampler(dataset, replacement=True, num_samples=sample_count)
    if distributed:
        if sampler is not None:
            raise ValueError("Distributed ACDC training with replacement sampling is not supported")
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=paper_shuffle)
    is_evaluation = mode in ("eval", "test")
    workers = int(
        config.get("evaluation", {}).get("workers", 0)
        if is_evaluation
        else config["data"].get("workers", 2)
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 1)) if mode == "train" else 1,
        sampler=sampler,
        shuffle=paper_shuffle and not distributed,
        num_workers=workers,
        pin_memory=torch.cuda.is_available() and not is_evaluation,
        persistent_workers=workers > 0 and not is_evaluation,
    )


def brain_mask(image: torch.Tensor) -> torch.Tensor:
    return image.abs().sum(dim=1, keepdim=True) > 0


def restore_prediction_to_original(
    prediction: np.ndarray,
    image_path: str,
    label_path: str,
    roi_size: tuple[int, int, int],
    target_spacing: tuple[float, float, float],
) -> sitk.Image:
    """Undo center crop, padding and resampling for an ACDC class prediction."""
    original = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)
    resampled, _ = _resample(original, label, target_spacing)
    resampled_shape = tuple(reversed(resampled.GetSize()))
    padding = _padding(resampled_shape, roi_size)
    padded_shape = tuple(size + before + after for size, (before, after) in zip(resampled_shape, padding))
    y_start = padded_shape[1] // 2 - 128 if padded_shape[1] > 256 else 0
    x_start = padded_shape[2] // 2 - 128 if padded_shape[2] > 256 else 0
    y_size = min(padded_shape[1], 256)
    x_size = min(padded_shape[2], 256)
    expected = (padded_shape[0], y_size, x_size)
    if tuple(prediction.shape) != expected:
        raise ValueError(f"Prediction shape {tuple(prediction.shape)} does not match preprocessed shape {expected}")
    padded = np.zeros(padded_shape, dtype=np.uint8)
    padded[:, y_start : y_start + y_size, x_start : x_start + x_size] = prediction.astype(np.uint8)
    slices = tuple(slice(before, before + size) for size, (before, _after) in zip(resampled_shape, padding))
    resampled_array = padded[slices]
    prediction_image = sitk.GetImageFromArray(resampled_array)
    prediction_image.CopyInformation(resampled)
    restored = sitk.Resample(
        prediction_image,
        original,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    geometry_matches = (
        restored.GetSize() == original.GetSize()
        and np.allclose(restored.GetSpacing(), original.GetSpacing(), atol=1e-7)
        and np.allclose(restored.GetOrigin(), original.GetOrigin(), atol=1e-7)
        and np.allclose(restored.GetDirection(), original.GetDirection(), atol=1e-7)
    )
    if not geometry_matches:
        raise RuntimeError("Restored prediction geometry does not match the original ACDC image")
    return restored
