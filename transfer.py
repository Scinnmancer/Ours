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
        uncertainty_top_fraction: float = 1.0,
        percentile_roi_dilation: int = 0,
        z0: float = 0.5,
        eps: float = 1e-7,
    ):
        super().__init__()
        if not 0.0 <= alpha_max < 1.0:
            raise ValueError("alpha_max must be in [0, 1)")
        if (
            not math.isfinite(uncertainty_top_fraction)
            or not 0.0 < uncertainty_top_fraction <= 1.0
        ):
            raise ValueError("uncertainty_top_fraction must be in (0, 1]")
        if (
            isinstance(percentile_roi_dilation, bool)
            or float(percentile_roi_dilation) != int(percentile_roi_dilation)
            or int(percentile_roi_dilation) < 0
        ):
            raise ValueError("percentile_roi_dilation must be a non-negative integer")
        self.radius = int(radius)
        self.gamma = float(gamma)
        self.alpha_max = float(alpha_max)
        self.beta = float(beta)
        self.iterations = int(iterations)
        self.uncertainty_top_fraction = float(uncertainty_top_fraction)
        self.percentile_roi_dilation = int(percentile_roi_dilation)
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

    def selection_mask(
        self, base_probability: torch.Tensor, uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Select the exact highest-uncertainty fraction inside predicted ROI."""
        predicted_tumor = base_probability.argmax(dim=1, keepdim=True) != 0
        dilation = self.percentile_roi_dilation
        if dilation > 0:
            size = 2 * dilation + 1
            predicted_tumor = F.max_pool3d(
                predicted_tumor.to(dtype=base_probability.dtype),
                kernel_size=size,
                stride=1,
                padding=dilation,
            ) > 0
        selected = torch.zeros_like(predicted_tumor, dtype=torch.bool)
        detached_uncertainty = uncertainty.detach()
        for batch_index in range(base_probability.shape[0]):
            roi_flat = predicted_tumor[batch_index, 0].flatten()
            roi_indices = roi_flat.nonzero(as_tuple=False).flatten()
            if roi_indices.numel() == 0:
                continue
            count = max(
                1,
                int(math.ceil(roi_indices.numel() * self.uncertainty_top_fraction)),
            )
            values = detached_uncertainty[batch_index, 0].flatten()[roi_indices]
            chosen = roi_indices[torch.topk(values, k=count, sorted=False).indices]
            selected[batch_index, 0].view(-1)[chosen] = True
        return selected

    def forward(self, base_probability: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        if base_probability.shape[1] != 4 or uncertainty.shape[1] != 1:
            raise ValueError("Expected atomic [B,4,...] probability and [B,1,...] uncertainty")
        p = base_probability.clamp_min(0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        reliability = (1.0 - uncertainty.clamp(0.0, 1.0)).pow(self.gamma)
        kernel = self.kernel.to(dtype=p.dtype)
        z = F.conv3d(reliability, kernel, padding=self.radius)
        support = (z / self.z0.to(dtype=z.dtype).clamp_min(self.eps)).clamp(max=1.0)
        percentile_mask = self.selection_mask(p, uncertainty).to(dtype=p.dtype)
        alpha = (
            self.alpha_max
            * uncertainty.clamp(0.0, 1.0).pow(self.beta)
            * support
            * percentile_mask
        )
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
