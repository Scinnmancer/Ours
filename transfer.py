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
        uncertainty_gain: float = 1.0,
        iterations: int = 3,
        z0: float = 0.5,
        eps: float = 1e-7,
    ):
        super().__init__()
        if not 0.0 <= alpha_max < 1.0:
            raise ValueError("alpha_max must be in [0, 1)")
        if not math.isfinite(uncertainty_gain) or uncertainty_gain <= 0.0:
            raise ValueError("uncertainty_gain must be finite and greater than 0")
        self.radius = int(radius)
        self.gamma = float(gamma)
        self.alpha_max = float(alpha_max)
        self.beta = float(beta)
        self.uncertainty_gain = float(uncertainty_gain)
        self.iterations = int(iterations)
        self.eps = float(eps)
        self.register_buffer("kernel", gaussian_kernel3d(self.radius, sigma)[None, None])
        self.register_buffer("z0", torch.tensor(float(z0)))

    def set_z0(self, value: float | torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.z0.dtype, device=self.z0.device)
        self.z0.copy_(value.clamp_min(self.eps))

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
        # Refine-only coverage control. The geometric uncertainty itself is not
        # changed; gain only remaps it while computing the test-time gate.
        refine_uncertainty = (
            self.uncertainty_gain * uncertainty.clamp(0.0, 1.0)
        ).clamp(max=1.0)
        alpha = self.alpha_max * refine_uncertainty.pow(self.beta) * support
        q = p
        class_kernel = kernel.expand(p.shape[1], 1, *kernel.shape[2:]).contiguous()
        for _ in range(self.iterations):
            weighted = q * reliability
            neighbor = F.conv3d(weighted, class_kernel, padding=self.radius, groups=p.shape[1])
            neighbor = neighbor / z.clamp_min(self.eps)
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
