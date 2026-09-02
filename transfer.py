from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel3d(radius: int, sigma: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, dtype=dtype)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square() + zz.square()) / (2.0 * sigma * sigma))
    kernel[radius, radius, radius] = 0.0
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


class UncertaintyGatedLabelTransfer(nn.Module):
    def __init__(
        self,
        radius: int = 2,
        sigma: float = 1.0,
        gamma: float = 2.0,
        alpha_max: float = 0.35,
        beta: float = 2.0,
        iterations: int = 3,
        neighbor_reliability_power: float = 1.0,
        z0: float = 0.5,
        eps: float = 1e-7,
    ):
        super().__init__()
        if not 0.0 <= alpha_max < 1.0:
            raise ValueError("alpha_max must be in [0, 1)")
        self.radius = int(radius)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.alpha_max = float(alpha_max)
        self.beta = float(beta)
        self.iterations = int(iterations)
        self.neighbor_reliability_power = float(neighbor_reliability_power)
        self.eps = float(eps)
        self.register_buffer("kernel", gaussian_kernel3d(self.radius, self.sigma)[None, None])
        self.register_buffer("z0", torch.tensor(float(z0)))

    def set_z0(self, value: float | torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.z0.dtype, device=self.z0.device)
        self.z0.copy_(value.clamp_min(self.eps))

    def set_neighborhood(self, radius: int, sigma: float) -> None:
        """Replace the derived Gaussian kernel after checkpoint loading."""
        radius = int(radius)
        sigma = float(sigma)
        if radius < 1:
            raise ValueError("radius must be at least 1")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and greater than 0")
        self.radius = radius
        self.sigma = sigma
        self.kernel = gaussian_kernel3d(
            radius,
            sigma,
            dtype=self.kernel.dtype,
        )[None, None].to(device=self.kernel.device)

    def set_neighbor_reliability_power(self, power: float) -> None:
        power = float(power)
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError("neighbor reliability power must be finite and greater than 0")
        self.neighbor_reliability_power = power

    def evidence(self, uncertainty: torch.Tensor) -> torch.Tensor:
        reliability = (1.0 - uncertainty.clamp(0.0, 1.0)).pow(self.gamma)
        kernel = self.kernel.to(dtype=reliability.dtype)
        return F.conv3d(reliability, kernel, padding=self.radius)

    def forward(self, base_probability: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        if base_probability.shape[1] != 4 or uncertainty.shape[1] != 1:
            raise ValueError("Expected atomic [B,4,...] probability and [B,1,...] uncertainty")
        p = base_probability.clamp_min(0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        reliability = (1.0 - uncertainty.clamp(0.0, 1.0)).pow(self.gamma)
        kernel = self.kernel.to(dtype=p.dtype)
        z = F.conv3d(reliability, kernel, padding=self.radius)
        support = (z / self.z0.to(dtype=z.dtype).clamp_min(self.eps)).clamp(max=1.0)
        alpha = self.alpha_max * uncertainty.clamp(0.0, 1.0).pow(self.beta) * support
        q = p
        neighbor_reliability = reliability.pow(self.neighbor_reliability_power)
        neighbor_z = F.conv3d(neighbor_reliability, kernel, padding=self.radius)
        class_kernel = kernel.expand(p.shape[1], 1, *kernel.shape[2:]).contiguous()
        for _ in range(self.iterations):
            weighted = q * neighbor_reliability
            neighbor = F.conv3d(weighted, class_kernel, padding=self.radius, groups=p.shape[1])
            neighbor = neighbor / neighbor_z.clamp_min(self.eps)
            alternative = (1.0 - p) * neighbor
            alternative_sum = alternative.sum(dim=1, keepdim=True)
            alternative = torch.where(
                alternative_sum > self.eps,
                alternative / alternative_sum.clamp_min(self.eps),
                p,
            )
            q = (1.0 - alpha) * p + alpha * alternative
            q = q / q.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return q
