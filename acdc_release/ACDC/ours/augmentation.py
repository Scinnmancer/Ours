"""ACDC augmentations kept numerically aligned with the executable baseline."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_noise(image: torch.Tensor, std: float) -> torch.Tensor:
    return image + torch.randn_like(image) * std


def additive_brightness(image: torch.Tensor, std: float) -> torch.Tensor:
    offset = torch.normal(0.0, std, size=(1, 1, 1, 1, 1), device=image.device)
    return image + offset


def random_gamma(
    image: torch.Tensor,
    gamma_range: tuple[float, float],
    retain_stats: bool = True,
) -> torch.Tensor:
    shape = image.shape
    flattened = image.reshape(1, -1)
    minimum = flattened.min(dim=1, keepdim=True).values
    maximum = flattened.max(dim=1, keepdim=True).values
    value_range = (maximum - minimum).clamp_min(1e-7)
    old_mean = flattened.mean(dim=1, keepdim=True)
    old_std = flattened.std(dim=1, keepdim=True).clamp_min(1e-7)
    gamma = torch.empty(1, 1, device=image.device).uniform_(*gamma_range)
    flattened = torch.pow((flattened - minimum) / value_range, gamma) * value_range + minimum
    if retain_stats:
        flattened = flattened - flattened.mean(dim=1, keepdim=True)
        flattened = flattened / flattened.std(dim=1, keepdim=True).clamp_min(1e-7)
        flattened = flattened * old_std + old_mean
    return flattened.reshape(shape)


def random_affine_3d(
    image: torch.Tensor,
    label: torch.Tensor,
    scale,
    rotate,
    translate,
    shear: float | tuple[float, float, float] = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = [scale] * 3 if isinstance(scale, (int, float)) else scale
    rotate = [rotate] * 3 if isinstance(rotate, (int, float)) else rotate
    translate = [translate] * 3 if isinstance(translate, (int, float)) else translate
    shear = [shear] * 3 if isinstance(shear, (int, float)) else shear
    scales = [np.random.uniform(1 - value, 1 / (1 - value)) for value in scale]
    shears = [np.random.uniform(-value, value, size=2) for value in shear]
    shifts = [np.random.uniform(-value, value) for value in translate]
    affine = torch.tensor(
        [
            [scales[0], shears[0][0], shears[0][1], shifts[0]],
            [shears[1][0], scales[1], shears[1][1], shifts[1]],
            [shears[2][0], shears[2][1], scales[2], shifts[2]],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    angles = [math.radians(float(np.random.randint(-value, max(value, 1)))) for value in rotate]
    ax, ay, az = angles
    rotate_x = torch.tensor(
        [[1, 0, 0, 0], [0, math.cos(ax), -math.sin(ax), 0], [0, math.sin(ax), math.cos(ax), 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    rotate_y = torch.tensor(
        [[math.cos(ay), 0, -math.sin(ay), 0], [0, 1, 0, 0], [math.sin(ay), 0, math.cos(ay), 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    rotate_z = torch.tensor(
        [[math.cos(az), -math.sin(az), 0, 0], [math.sin(az), math.cos(az), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    theta = (rotate_x @ rotate_y @ rotate_z @ affine)[:3].unsqueeze(0).to(image.device)
    grid = F.affine_grid(theta, image.size(), align_corners=True)
    image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    label = F.grid_sample(label.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True).long()
    return image, label


def random_crop_3d(
    image: torch.Tensor,
    label: torch.Tensor,
    crop_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, depth, height, width = image.shape
    crop_d, crop_h, crop_w = crop_size
    starts = [
        np.random.randint(0, max(depth - crop_d, 1)),
        np.random.randint(0, max(height - crop_h, 1)),
        np.random.randint(0, max(width - crop_w, 1)),
    ]
    z, y, x = starts
    image = image[:, :, z : z + crop_d, y : y + crop_h, x : x + crop_w]
    label = label[:, :, z : z + crop_d, y : y + crop_h, x : x + crop_w]
    return image.contiguous(), label.contiguous()
