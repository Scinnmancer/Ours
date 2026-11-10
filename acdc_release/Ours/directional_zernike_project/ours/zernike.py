from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Order = tuple[int, int]


def _spherical_harmonic(m: int, l: int, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import sph_harm_y

        return sph_harm_y(l, m, theta, phi)
    except ImportError:
        from scipy.special import sph_harm

        return sph_harm(m, l, phi, theta)


def zernike_kernel(size: int, n: int, l: int, m: int) -> np.ndarray:
    """Create a sampled complex 3D Zernike basis on the unit ball."""
    if size < 3 or size % 2 == 0:
        raise ValueError("Zernike kernel size must be an odd integer >= 3")
    if n < l or (n - l) % 2 != 0 or abs(m) > l:
        raise ValueError(f"Invalid 3D Zernike indices n={n}, l={l}, m={m}")
    from scipy.special import eval_jacobi

    axis = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.sqrt(x * x + y * y + z * z)
    mask = radius <= 1.0 + 1e-12
    theta = np.zeros_like(radius)
    theta[mask] = np.arccos(np.clip(z[mask] / np.maximum(radius[mask], 1e-12), -1.0, 1.0))
    phi = np.arctan2(y, x)
    k = (n - l) // 2
    radial = math.sqrt(2 * n + 3) * np.power(radius, l) * eval_jacobi(k, 0.0, l + 0.5, 2.0 * radius * radius - 1.0)
    harmonic = _spherical_harmonic(m, l, phi, theta)
    basis = (3.0 / (4.0 * math.pi)) * radial * np.conjugate(harmonic)
    basis[~mask] = 0.0
    basis /= max(int(mask.sum()), 1)
    return basis.astype(np.complex64)


class MultiScaleZernike(nn.Module):
    def __init__(self, windows: Sequence[int], orders: Sequence[Order], chunk_depth: int = 0):
        super().__init__()
        self.windows = tuple(int(item) for item in windows)
        self.orders = tuple((int(n), int(l)) for n, l in orders)
        # Real-valued inputs satisfy A(n,l,-m)=(-1)^m conjugate(A(n,l,m)),
        # so m >= 0 contains every independent directional coefficient.  We
        # keep the real m=0 coefficient and both real/imaginary parts for m>0
        # instead of collapsing all m values into a rotation invariant norm.
        self.components = tuple(
            (order_index, m, part)
            for order_index, (_, l) in enumerate(self.orders)
            for m in range(l + 1)
            for part in (("real",) if m == 0 else ("real", "imag"))
        )
        self.chunk_depth = int(chunk_depth)
        for scale_index, size in enumerate(self.windows):
            for order_index, (n, l) in enumerate(self.orders):
                for m in range(l + 1):
                    kernel = zernike_kernel(size, n, l, m)
                    self.register_buffer(
                        f"kr_{scale_index}_{order_index}_{m}",
                        torch.from_numpy(kernel.real)[None, None],
                    )
                    if m > 0:
                        self.register_buffer(
                            f"ki_{scale_index}_{order_index}_{m}",
                            torch.from_numpy(kernel.imag)[None, None],
                        )

    @property
    def component_count(self) -> int:
        return len(self.components)

    def _describe_scale(self, probability: torch.Tensor, scale_index: int) -> torch.Tensor:
        channels = probability.shape[1]
        size = self.windows[scale_index]
        directional_components = []
        for order_index, (_, l) in enumerate(self.orders):
            for m in range(l + 1):
                real = getattr(self, f"kr_{scale_index}_{order_index}_{m}").to(dtype=probability.dtype)
                real = real.expand(channels, 1, size, size, size)
                real_moment = F.conv3d(probability, real, padding=size // 2, groups=channels)
                directional_components.append(real_moment)
                if m > 0:
                    imag = getattr(self, f"ki_{scale_index}_{order_index}_{m}").to(dtype=probability.dtype)
                    imag = imag.expand(channels, 1, size, size, size)
                    imag_moment = F.conv3d(probability, imag, padding=size // 2, groups=channels)
                    directional_components.append(imag_moment)
        return torch.stack(directional_components, dim=2)

    def _describe_full(self, probability: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._describe_scale(probability, idx) for idx in range(len(self.windows))], dim=1)

    def iter_descriptors(self, probability: torch.Tensor):
        """Yield (start, stop, descriptor) blocks with convolution halos removed."""
        if self.chunk_depth <= 0 or probability.shape[2] <= self.chunk_depth:
            yield 0, probability.shape[2], self._describe_full(probability)
            return
        halo = max(self.windows) // 2
        depth = probability.shape[2]
        for start in range(0, depth, self.chunk_depth):
            stop = min(depth, start + self.chunk_depth)
            input_start = max(0, start - halo)
            input_stop = min(depth, stop + halo)
            described = self._describe_full(probability[:, :, input_start:input_stop])
            crop_start = start - input_start
            crop_stop = crop_start + (stop - start)
            yield start, stop, described[:, :, :, :, crop_start:crop_stop]

    def forward(self, probability: torch.Tensor) -> torch.Tensor:
        """Return [B,scale,class,directional-component,D,H,W] descriptors."""
        return torch.cat([descriptor for _, _, descriptor in self.iter_descriptors(probability)], dim=4)

    def disagreement(
        self,
        probability1: torch.Tensor,
        probability2: torch.Tensor,
        statistics: "ZernikeStatistics",
    ) -> torch.Tensor:
        chunks = []
        iterator1 = self.iter_descriptors(probability1)
        iterator2 = self.iter_descriptors(probability2)
        for (start1, stop1, descriptor1), (start2, stop2, descriptor2) in zip(iterator1, iterator2):
            if start1 != start2 or stop1 != stop2:
                raise RuntimeError("Zernike descriptor chunk boundaries do not match")
            chunks.append(zernike_disagreement(descriptor1, descriptor2, statistics))
        return torch.cat(chunks, dim=2)

    def scale_disagreement(
        self,
        probability1: torch.Tensor,
        probability2: torch.Tensor,
        statistics: "ZernikeStatistics",
    ) -> torch.Tensor:
        """Return [B,scale,D,H,W], retaining the original directional moments."""
        chunks = []
        for first, second in zip(
            self.iter_descriptors(probability1), self.iter_descriptors(probability2)
        ):
            start1, stop1, descriptor1 = first
            start2, stop2, descriptor2 = second
            if (start1, stop1) != (start2, stop2):
                raise RuntimeError("Zernike descriptor chunk boundaries do not match")
            difference = (statistics(descriptor1) - statistics(descriptor2)).abs()
            chunks.append(difference.mean(dim=(2, 3)))
        return torch.cat(chunks, dim=2)


class ZernikeStatistics(nn.Module):
    def __init__(self, scales: int, classes: int, components: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(scales, classes, components))
        self.register_buffer("std", torch.ones(scales, classes, components))
        self.register_buffer("count", torch.zeros(scales, classes, components, dtype=torch.long))

    @property
    def fitted(self) -> bool:
        return bool(torch.all(self.count > 1))

    def set_values(self, mean: torch.Tensor, std: torch.Tensor, count: torch.Tensor) -> None:
        if mean.shape != self.mean.shape or std.shape != self.std.shape or count.shape != self.count.shape:
            raise ValueError("Zernike statistic shapes do not match configuration")
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(self.eps))
        self.count.copy_(count.long())

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Zernike statistics have not been fitted")
        shape = (1, *self.mean.shape, 1, 1, 1)
        return (descriptor - self.mean.view(shape)) / self.std.view(shape).clamp_min(self.eps)


@dataclass
class WelfordAccumulator:
    shape: tuple[int, int, int]

    def __post_init__(self):
        self.count = torch.zeros(self.shape, dtype=torch.long)
        self.mean = torch.zeros(self.shape, dtype=torch.float64)
        self.m2 = torch.zeros(self.shape, dtype=torch.float64)

    def update(self, descriptor: torch.Tensor, mask: torch.Tensor) -> None:
        data = descriptor.detach().to(device="cpu", dtype=torch.float64)
        valid = mask.detach().to(device="cpu", dtype=torch.bool)
        if valid.ndim == 5:
            valid = valid[:, 0]
        for scale in range(data.shape[1]):
            for channel in range(data.shape[2]):
                for order in range(data.shape[3]):
                    values = data[:, scale, channel, order][valid]
                    if values.numel() == 0:
                        continue
                    batch_count = values.numel()
                    batch_mean = values.mean()
                    batch_m2 = ((values - batch_mean) ** 2).sum()
                    old_count = self.count[scale, channel, order].item()
                    new_count = old_count + batch_count
                    delta = batch_mean - self.mean[scale, channel, order]
                    self.mean[scale, channel, order] += delta * batch_count / new_count
                    self.m2[scale, channel, order] += batch_m2 + delta.square() * old_count * batch_count / new_count
                    self.count[scale, channel, order] = new_count

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denominator = (self.count - 1).clamp_min(1).to(torch.float64)
        std = torch.sqrt(self.m2 / denominator).clamp_min(1e-6)
        return self.mean.float(), std.float(), self.count


def zernike_disagreement(
    descriptor1: torch.Tensor,
    descriptor2: torch.Tensor,
    statistics: ZernikeStatistics,
) -> torch.Tensor:
    standardized1 = statistics(descriptor1)
    standardized2 = statistics(descriptor2)
    return (standardized1 - standardized2).abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
