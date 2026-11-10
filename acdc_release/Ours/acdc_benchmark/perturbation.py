"""Training-only perturbations for the final features of each ACDC decoder."""
import torch
from torch import nn


VARIANTS = ("dropout", "feature_noise", "cct_feature_dropout", "feature_cutout")
SUPPORTED_VARIANTS = (*VARIANTS, "feature_noise_strong")


class FeaturePerturbation(nn.Module):
    def __init__(self, variant, strength):
        super().__init__()
        self.variant = variant
        self.strength = strength

    def forward(self, x):
        if not self.training:
            return x
        if self.variant == "feature_noise":
            if self.strength == 0:
                return x
            noise = x.new_empty((*x.shape[:2], 1, 1, 1)).uniform_(-self.strength, self.strength)
            return x * (1 + noise)
        if self.variant == "cct_feature_dropout":
            attention = x.detach().float().mean(dim=1, keepdim=True)
            peak = attention.amax(dim=(2, 3, 4), keepdim=True)
            threshold = peak * peak.new_empty(peak.shape).uniform_(0.7, 0.9)
            keep = attention < threshold
            keep = keep | (peak <= 0) | ~keep.any(dim=(2, 3, 4), keepdim=True)
            return x * keep
        if self.variant != "feature_cutout":
            raise ValueError(f"Unknown feature perturbation {self.variant}")
        inside = []
        for axis, size in enumerate(x.shape[2:]):
            length = max(1, round(size * self.strength ** (1 / 3)))
            start = torch.randint(size - length + 1, (x.shape[0], 1, 1, 1, 1), device=x.device)
            shape = [1] * 5
            shape[axis + 2] = size
            position = torch.arange(size, device=x.device).view(shape)
            inside.append((start <= position) & (position < start + length))
        return x.masked_fill(inside[0] & inside[1] & inside[2], 0)


def configure_heads(model, variant):
    if variant == "dropout":
        return
    strengths = {
        "feature_noise": (0.10, 0.20),
        "feature_noise_strong": (0.20, 0.40),
        "cct_feature_dropout": (0.0, 0.0),
        "feature_cutout": (0.05, 0.10),
    }
    if variant not in strengths:
        raise ValueError(f"Unknown head perturbation {variant}")
    for head, strength in zip((model.head1, model.head2), strengths[variant]):
        operator = "feature_noise" if variant == "feature_noise_strong" else variant
        head.dropout = FeaturePerturbation(operator, strength)
