from __future__ import annotations

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
        z0: float = 0.5,
        direction_mode: str = "legacy_complement",
        eps: float = 1e-7,
    ):
        super().__init__()
        if not 0.0 <= alpha_max < 1.0:
            raise ValueError("alpha_max must be in [0, 1)")
        self.radius = int(radius)
        self.gamma = float(gamma)
        self.alpha_max = float(alpha_max)
        self.beta = float(beta)
        self.iterations = int(iterations)
        if direction_mode not in ("legacy_complement", "local_excess_confidence"):
            raise ValueError(
                "direction_mode must be 'legacy_complement' or "
                "'local_excess_confidence'"
            )
        self.direction_mode = str(direction_mode)
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

    def _directional_target(
        self,
        base_probability: torch.Tensor,
        neighbor_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the refine target and its local excess-confidence coefficient."""
        p = base_probability
        neighbor = neighbor_probability
        alternative = (1.0 - p) * neighbor
        alternative_sum = alternative.sum(dim=1, keepdim=True)
        alternative = torch.where(
            alternative_sum > self.eps,
            alternative / alternative_sum.clamp_min(self.eps),
            p,
        )
        if self.direction_mode == "legacy_complement":
            rho = torch.ones_like(p[:, :1])
            return alternative, rho

        base_class = p.argmax(dim=1, keepdim=True)
        center_confidence = p.gather(1, base_class)
        neighbor_support = neighbor.gather(1, base_class).clamp(0.0, 1.0)
        rho = (
            (center_confidence - neighbor_support)
            / (1.0 - neighbor_support + self.eps)
        ).clamp(0.0, 1.0)
        target = (1.0 - rho) * neighbor + rho * alternative
        target = target / target.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return target, rho

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
        class_kernel = kernel.expand(p.shape[1], 1, *kernel.shape[2:]).contiguous()
        for _ in range(self.iterations):
            weighted = q * reliability
            neighbor = F.conv3d(weighted, class_kernel, padding=self.radius, groups=p.shape[1])
            neighbor = neighbor / z.clamp_min(self.eps)
            neighbor = torch.where(z > self.eps, neighbor, p)
            target, _ = self._directional_target(p, neighbor)
            q = (1.0 - alpha) * p + alpha * target
            q = q / q.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return q
