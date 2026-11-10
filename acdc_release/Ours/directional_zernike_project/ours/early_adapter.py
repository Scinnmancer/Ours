"""Shared residual at Swin patch embedding, with a fixed source reference."""
from contextlib import contextmanager

import torch
from torch import nn


class EarlyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv3d(48, 8, 1, bias=False)
        self.up = nn.Conv3d(8, 48, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        assert sum(p.numel() for p in self.parameters()) == 768

    def forward(self, feature):
        return feature + self.up(torch.relu(self.down(feature)))


class MultiScale3DAdapter(nn.Module):
    """PEOA-inspired spatial residual; no memory and no extra insertion sites."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(48)
        self.norm_scale = nn.Parameter(torch.full((48,), 1e-6))
        self.identity_scale = nn.Parameter(torch.ones(48))
        self.down = nn.Conv3d(48, 8, 1)
        self.branches = nn.ModuleList([
            nn.Conv3d(8, 8, 3, padding=d, dilation=d, groups=8) for d in (1, 2, 3)])
        self.pointwise = nn.Conv3d(8, 8, 1)
        self.up = nn.Conv3d(8, 48, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        assert sum(p.numel() for p in self.parameters()) == 1760

    def forward(self, feature):
        normalized = self.norm(feature.movedim(1, -1)).movedim(-1, 1)
        shape = (1, 48, 1, 1, 1)
        reduced = self.down(normalized * self.norm_scale.view(shape)
                            + feature * self.identity_scale.view(shape))
        mixed = self.branches[0](reduced)
        for branch in self.branches[1:]:
            mixed = mixed + branch(reduced)
        mixed = mixed / 3
        contextual = mixed + mixed.mean((2, 3, 4), keepdim=True)
        return feature + self.up(torch.relu(contextual + self.pointwise(contextual)))


def make_early_adapter(kind='pointwise'):
    if kind == 'pointwise':
        return EarlyAdapter()
    if kind == 'multiscale3d':
        return MultiScale3DAdapter()
    raise ValueError('Unknown early_adapter.kind: ' + str(kind))


@contextmanager
def reference_adapter(model, reference):
    """Select an immutable tiny reference without copying the frozen backbone."""
    previous = model._early_reference
    model._early_reference = reference
    try:
        yield
    finally:
        model._early_reference = previous


def patch_residual(model, module, inputs, output):
    reference = model._early_reference
    if reference is None:
        return model.early_adapter(output)
    # Works for both adapter types without copying the backbone or mutating state.
    return torch.func.functional_call(model.early_adapter, reference, (output,), strict=True)
