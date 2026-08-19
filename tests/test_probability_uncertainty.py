from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from ours.checkpoint import load_checkpoint, save_checkpoint
from ours.config import validate_config
from ours.model import DualHeadOutput, DualHeadSwinUNETR
from ours.probability import independent_logits_to_atomic, independent_logits_to_regions, jensen_shannon_divergence
from ours.train import (
    _make_scaler,
    _save_uncertainty_batch,
    configure_calibration_trainability,
    run,
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
            "probability_disagreement_scale": 15.0,
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


def test_probability_only_fusion_uses_configured_scale_and_ignores_legacy_xi():
    fusion = UncertaintyFusion(eta=1.0, xi=1.0, bias=-2.0)
    disagreement = torch.tensor([[[[[0.02]]]]])
    first = fusion(disagreement, torch.zeros_like(disagreement), 15.0)
    with torch.no_grad():
        fusion.raw_xi.add_(20.0)
    second = fusion(disagreement, torch.full_like(disagreement, 100.0), 15.0)
    expected = torch.sigmoid(fusion.bias + fusion.eta * disagreement * 15.0)

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert bool(((first >= 0.0) & (first <= 1.0)).all())


def test_model_uncertainty_path_never_calls_zernike(monkeypatch):
    model = _tiny_dual_head(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Zernike disagreement must remain inactive")

    monkeypatch.setattr(model.zernike, "disagreement", fail_if_called)
    logits1 = torch.randn(1, 3, 4, 4, 4)
    logits2 = torch.randn(1, 3, 4, 4, 4)
    result = model.output_from_logits(logits1, logits2, compute_uncertainty=True)

    expected_js = jensen_shannon_divergence(
        independent_logits_to_atomic(logits1),
        independent_logits_to_atomic(logits2),
    )
    expected_u = model.fusion(
        expected_js,
        probability_disagreement_scale=model.probability_disagreement_scale,
    )
    assert result.zernike_disagreement is None
    torch.testing.assert_close(result.probability_disagreement, expected_js)
    torch.testing.assert_close(result.uncertainty, expected_u)


def test_legacy_warmup_state_dict_loads_strictly_without_overwriting_scale(monkeypatch, tmp_path):
    legacy = _tiny_dual_head(monkeypatch)
    path = tmp_path / "legacy_warmup.pt"
    save_checkpoint(path, legacy, "warmup", 4, _config(), metrics={"mean_dice": 0.7})

    current = _tiny_dual_head(monkeypatch)
    payload = load_checkpoint(path, current, strict=True)

    assert payload["stage"] == "warmup"
    assert current.probability_disagreement_scale == 15.0
    assert current.state_dict().keys() == legacy.state_dict().keys()


class _RiskOnlyToyModel(nn.Module):
    ENCODER_PREFIXES = ("segmenter.",)

    def __init__(self):
        super().__init__()
        self.segmenter = nn.Conv3d(1, 3, 1)
        self.fusion = UncertaintyFusion()
        self.probability_disagreement_scale = 15.0
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
            disagreement = jensen_shannon_divergence(atomic1, atomic2)
            result.probability_disagreement = disagreement
            result.uncertainty = self.fusion(
                disagreement,
                probability_disagreement_scale=self.probability_disagreement_scale,
            )
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


@pytest.mark.parametrize(
    ("freeze", "expected_training", "expects_second_view"),
    [(True, False, False), (False, True, True)],
)
def test_calibration_view_and_dropout_mode(freeze, expected_training, expects_second_view):
    model = _RiskOnlyToyModel()
    parameters = configure_calibration_trainability(model, freeze)
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
        lambda_u=1.0 if freeze else 0.1,
        stage="calibration",
        epoch=1,
        freeze_segmentation=freeze,
    )

    assert model.observed_training is expected_training
    assert (model.observed_second_view is not None) is expects_second_view
    if freeze:
        assert segmentation_state_sha256(model) == before
        assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
            "fusion.raw_eta",
            "fusion.bias",
        }


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_probability_disagreement_scale_must_be_positive_and_finite(invalid):
    config = copy.deepcopy(_config())
    config["uncertainty"]["probability_disagreement_scale"] = invalid
    with pytest.raises(ValueError, match="probability_disagreement_scale"):
        validate_config(config)


def test_stats_stage_is_explicitly_disabled():
    with pytest.raises(ValueError, match="stats stage is disabled"):
        run({}, "stats")


def test_uncertainty_snapshot_contains_probability_component_only(tmp_path):
    probability = torch.full((1, 4, 2, 2, 2), 0.25)
    result = DualHeadOutput(
        head_logits=(torch.empty(0), torch.empty(0)),
        head_region_probabilities=(torch.empty(0), torch.empty(0)),
        head_atomic_probabilities=(probability, probability),
        base_atomic_probability=probability,
        probability_disagreement=torch.zeros(1, 1, 2, 2, 2),
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
        assert "probability_disagreement" in payload.files
        assert "zernike_disagreement" not in payload.files
