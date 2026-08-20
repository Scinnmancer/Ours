from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ours.checkpoint import load_checkpoint, save_checkpoint
from ours.model import DualHeadOutput, DualHeadSwinUNETR
from ours.probability import independent_logits_to_atomic, independent_logits_to_regions
from ours.train import (
    _make_scaler,
    _save_uncertainty_batch,
    configure_calibration_trainability,
    segmentation_state_sha256,
    train_epoch,
)
from ours.uncertainty import UncertaintyFusion


def _config() -> dict:
    return {
        "paths": {},
        "data": {"roi_size": [8, 8, 8], "workers": 0},
        "model": {
            "in_channels": 1,
            "region_channels": 3,
            "feature_size": 12,
            "head_dropout_rates": [0.1, 0.1],
            "output_mode": "independent_sigmoid",
        },
        "zernike": {"windows": [3], "orders": [[0, 0]], "chunk_depth": 0},
        "uncertainty": {
            "eta_init": 1.0,
            "xi_init": 1.0,
            "bias_init": -2.0,
            "max_balanced_samples": 128,
        },
        "label_transfer": {"radius": 1, "sigma": 1.0, "alpha_max": 0.2},
        "training": {
            "warmup_epochs": 1,
            "calibration_epochs": 1,
            "validation_every": 1,
            "freeze_segmentation_during_calibration": True,
            "amp": False,
        },
        "evaluation": {"workers": 0, "max_metric_voxels": 128},
        "monitoring": {"gradient_interval": 1, "memory_multiplier_limit": 3.0},
    }


class _TinyTemplate(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinViT = nn.Conv3d(1, 1, 1)
        self.encoder1 = nn.Conv3d(1, 1, 1)
        self.encoder2 = nn.Conv3d(1, 1, 1)
        self.encoder3 = nn.Conv3d(1, 1, 1)
        self.encoder4 = nn.Conv3d(1, 1, 1)
        self.encoder10 = nn.Conv3d(1, 1, 1)
        self.decoder5 = nn.Conv3d(1, 1, 1)
        self.decoder4 = nn.Conv3d(1, 1, 1)
        self.decoder3 = nn.Conv3d(1, 1, 1)
        self.decoder2 = nn.Conv3d(1, 1, 1)
        self.decoder1 = nn.Conv3d(1, 1, 1)
        self.out = nn.Conv3d(1, 3, 1)
        self.normalize = True


def _tiny_dual_head(monkeypatch) -> DualHeadSwinUNETR:
    monkeypatch.setattr("ours.model._make_swin_unetr", lambda config: _TinyTemplate())
    return DualHeadSwinUNETR(_config())


def test_geometric_only_fusion_uses_xi_and_ignores_legacy_eta():
    fusion = UncertaintyFusion(eta=1.0, xi=1.0, bias=-2.0)
    disagreement = torch.tensor([[[[[0.2]]]]])
    first = fusion(disagreement)
    with torch.no_grad():
        fusion.raw_eta.add_(20.0)
    second = fusion(disagreement)
    expected = torch.sigmoid(fusion.bias + fusion.xi * disagreement)

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert bool(((first >= 0.0) & (first <= 1.0)).all())


def test_model_uncertainty_path_uses_zernike_and_does_not_compute_js(monkeypatch):
    model = _tiny_dual_head(monkeypatch)
    expected_z = torch.full((1, 1, 4, 4, 4), 0.25)
    calls = []

    def geometric_disagreement(atomic1, atomic2, statistics):
        calls.append((atomic1, atomic2, statistics))
        return expected_z

    monkeypatch.setattr(model.zernike, "disagreement", geometric_disagreement)
    logits1 = torch.randn(1, 3, 4, 4, 4)
    logits2 = torch.randn(1, 3, 4, 4, 4)
    result = model.output_from_logits(logits1, logits2, compute_uncertainty=True)

    assert len(calls) == 1
    assert result.probability_disagreement is None
    torch.testing.assert_close(result.zernike_disagreement, expected_z)
    torch.testing.assert_close(result.uncertainty, model.fusion(expected_z))


def test_existing_warmup_state_dict_still_loads_strictly(monkeypatch, tmp_path):
    legacy = _tiny_dual_head(monkeypatch)
    path = tmp_path / "legacy_warmup.pt"
    save_checkpoint(path, legacy, "warmup", 4, _config(), metrics={"mean_dice": 0.7})

    current = _tiny_dual_head(monkeypatch)
    payload = load_checkpoint(path, current, strict=True)

    assert payload["stage"] == "warmup"
    assert current.state_dict().keys() == legacy.state_dict().keys()


class _RiskOnlyToyModel(nn.Module):
    ENCODER_PREFIXES = ("segmenter.",)

    def __init__(self):
        super().__init__()
        self.segmenter = nn.Conv3d(1, 3, 1)
        self.fusion = UncertaintyFusion()
        self.observed_training = None
        self.observed_second_view = None

    def forward(self, image, second_view=None, compute_uncertainty=False):
        self.observed_training = self.training
        self.observed_second_view = second_view
        logits1 = self.segmenter(image)
        logits2 = self.segmenter(image if second_view is None else second_view)
        regions1 = independent_logits_to_regions(logits1)
        regions2 = independent_logits_to_regions(logits2)
        atomic1 = independent_logits_to_atomic(logits1)
        atomic2 = independent_logits_to_atomic(logits2)
        base = 0.5 * (atomic1 + atomic2)
        result = DualHeadOutput(
            head_logits=(logits1, logits2),
            head_region_probabilities=(regions1, regions2),
            head_atomic_probabilities=(atomic1, atomic2),
            base_atomic_probability=base,
        )
        if compute_uncertainty:
            disagreement = (atomic1 - atomic2).abs().mean(dim=1, keepdim=True) + 0.1
            result.zernike_disagreement = disagreement
            result.uncertainty = self.fusion(disagreement)
        return result


class _ZeroSegmentationLoss(nn.Module):
    def forward(self, logits, target):
        return (logits * 0.0).sum()


def _toy_batch():
    image = torch.zeros(1, 1, 2, 2, 2)
    return {
        "image": image,
        "image_view1": image + 1.0,
        "image_view2": image + 2.0,
        "label_regions": torch.zeros(1, 3, 2, 2, 2),
        "label_atomic": torch.ones(1, 2, 2, 2, dtype=torch.uint8),
    }


def test_frozen_calibration_uses_same_image_and_only_trains_xi_and_bias():
    model = _RiskOnlyToyModel()
    parameters = configure_calibration_trainability(model, True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    before = segmentation_state_sha256(model)
    train_epoch(
        model,
        [_toy_batch()],
        optimizer,
        _make_scaler(False),
        torch.device("cpu"),
        _ZeroSegmentationLoss(),
        _config(),
        lambda_u=1.0,
        stage="calibration",
        epoch=1,
        freeze_segmentation=True,
    )

    assert model.observed_training is False
    assert model.observed_second_view is None
    assert segmentation_state_sha256(model) == before
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
        "fusion.raw_xi",
        "fusion.bias",
    }


def test_joint_calibration_keeps_geometry_and_disables_eta():
    model = _RiskOnlyToyModel()
    configure_calibration_trainability(model, False)
    assert model.segmenter.weight.requires_grad
    assert model.fusion.raw_xi.requires_grad
    assert model.fusion.bias.requires_grad
    assert not model.fusion.raw_eta.requires_grad


def test_uncertainty_snapshot_contains_zernike_component_only(tmp_path):
    probability = torch.full((1, 4, 2, 2, 2), 0.25)
    result = DualHeadOutput(
        head_logits=(torch.empty(0), torch.empty(0)),
        head_region_probabilities=(torch.empty(0), torch.empty(0)),
        head_atomic_probabilities=(probability, probability),
        base_atomic_probability=probability,
        zernike_disagreement=torch.full((1, 1, 2, 2, 2), 0.1),
        uncertainty=torch.full((1, 1, 2, 2, 2), 0.2),
    )
    batch = {"image": torch.zeros(1, 1, 2, 2, 2)}
    paths = _save_uncertainty_batch(
        result,
        batch,
        torch.zeros(1, 2, 2, 2, dtype=torch.uint8),
        tmp_path,
        batch_index=0,
        epoch=1,
        dtype="float32",
        save_components=True,
    )
    with np.load(paths[0]) as payload:
        assert "zernike_disagreement" in payload.files
        assert "probability_disagreement" not in payload.files
