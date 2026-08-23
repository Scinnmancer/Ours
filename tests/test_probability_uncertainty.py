from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ours.checkpoint import load_checkpoint, save_checkpoint
from ours.config import validate_config
from ours.losses import risk_brier_loss
from ours.model import DualHeadOutput, DualHeadSwinUNETR
from ours.probability import independent_logits_to_atomic, independent_logits_to_regions
from ours.train import (
    _make_scaler,
    _save_uncertainty_batch,
    calibration_checkpoint_key,
    encoder_state_sha256,
    frozen_calibration_state_sha256,
    configure_calibration_trainability,
    run,
    train_epoch,
)
from ours.uncertainty import UncertaintyFusion, uncertainty_weighted_margin_loss


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
            "lambda_u": 1.0,
            "margin_gradient": {
                "enabled": True,
                "weight": 0.01,
                "uncertainty_power": 2.0,
                "margin": 1.0,
            },
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
    assert not current.zernike_stats.fitted


class _RiskOnlyToyModel(nn.Module):
    ENCODER_PREFIXES = ("encoder.",)

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv3d(1, 1, 1)
        self.head1 = nn.Conv3d(1, 3, 1)
        self.head2 = nn.Conv3d(1, 3, 1)
        self.fusion = UncertaintyFusion()
        with torch.no_grad():
            self.encoder.weight.fill_(1.0)
            self.encoder.bias.zero_()
            self.head1.weight.zero_()
            self.head2.weight.zero_()
            self.head1.bias.fill_(-4.0)
            self.head2.bias.fill_(-4.0)
        self.observed_training = None
        self.observed_second_view = None

    def forward(self, image, second_view=None, compute_uncertainty=False):
        self.observed_training = self.training
        self.observed_second_view = second_view
        logits1 = self.head1(self.encoder(image))
        logits2 = self.head2(self.encoder(image if second_view is None else second_view))
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
    image = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2) / 8.0
    atomic = torch.zeros(1, 2, 2, 2, dtype=torch.uint8)
    atomic[..., 1:] = 1
    return {
        "image": image,
        "image_view1": image + 1.0,
        "image_view2": image + 2.0,
        "label_regions": torch.zeros(1, 3, 2, 2, 2),
        "label_atomic": atomic,
    }


def test_risk_brier_loss_fits_uncertainty_to_error_probability():
    uncertainty = torch.tensor([0.1, 0.8], requires_grad=True)
    error = torch.tensor([0.0, 1.0])

    loss = risk_brier_loss(uncertainty, error)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.025))
    assert float(uncertainty.grad[0]) > 0.0
    assert float(uncertainty.grad[1]) < 0.0


def test_risk_brier_loss_requires_aligned_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        risk_brier_loss(torch.zeros(2), torch.zeros(1, 2))


def test_margin_gradient_is_stronger_for_high_geometric_uncertainty_and_keeps_top_class():
    values = torch.tensor([0.90, 0.05, 0.03, 0.02]).reshape(1, 4, 1, 1, 1)

    high_atomic = values.clone().requires_grad_(True)
    high_loss = uncertainty_weighted_margin_loss(
        high_atomic,
        torch.full((1, 1, 1, 1, 1), 0.9),
        uncertainty_power=2.0,
        margin=1.0,
    )
    high_gradient = torch.autograd.grad(high_loss, high_atomic)[0]

    low_atomic = values.clone().requires_grad_(True)
    low_loss = uncertainty_weighted_margin_loss(
        low_atomic,
        torch.full((1, 1, 1, 1, 1), 0.1),
        uncertainty_power=2.0,
        margin=1.0,
    )
    low_gradient = torch.autograd.grad(low_loss, low_atomic)[0]
    updated = high_atomic.detach() - 1e-4 * high_gradient

    assert float(high_gradient.norm()) > float(low_gradient.norm())
    assert int(updated.argmax(dim=1)) == int(values.argmax(dim=1))
    assert not bool(high_loss.isnan())


def test_margin_loss_is_zero_below_margin_and_for_empty_mask():
    atomic = torch.tensor([0.30, 0.27, 0.23, 0.20]).reshape(1, 4, 1, 1, 1).requires_grad_(True)
    uncertainty = torch.ones(1, 1, 1, 1, 1)
    below_margin = uncertainty_weighted_margin_loss(
        atomic,
        uncertainty,
        uncertainty_power=2.0,
        margin=1.0,
    )
    empty = uncertainty_weighted_margin_loss(
        atomic,
        uncertainty,
        mask=torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        uncertainty_power=2.0,
        margin=1.0,
    )

    torch.testing.assert_close(below_margin, torch.zeros_like(below_margin))
    torch.testing.assert_close(empty, torch.zeros_like(empty))
    assert bool(torch.isfinite(empty))


def test_margin_loss_detaches_geometric_uncertainty():
    atomic = torch.tensor([0.90, 0.05, 0.03, 0.02]).reshape(1, 4, 1, 1, 1).requires_grad_(True)
    uncertainty = torch.full((1, 1, 1, 1, 1), 0.8, requires_grad=True)
    loss = uncertainty_weighted_margin_loss(atomic, uncertainty, margin=1.0)

    loss.backward()

    assert atomic.grad is not None
    assert uncertainty.grad is None


def test_margin_gradient_config_rejects_negative_weight():
    config = _config()
    config["uncertainty"]["margin_gradient"]["weight"] = -0.01

    with pytest.raises(ValueError, match="margin_gradient.weight"):
        validate_config(config)


def test_legacy_config_without_margin_gradient_remains_valid():
    config = _config()
    del config["uncertainty"]["margin_gradient"]

    validate_config(config)


def test_legacy_frozen_flag_maps_to_head_only_margin_calibration():
    model = _RiskOnlyToyModel()
    parameters = configure_calibration_trainability(model, True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    encoder_before = encoder_state_sha256(model)
    head_before = model.head1.weight.detach().clone()
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
    torch.testing.assert_close(model.observed_second_view, _toy_batch()["image_view2"])
    assert encoder_state_sha256(model) == encoder_before
    assert not torch.equal(model.head1.weight.detach(), head_before)
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
        "head1.weight",
        "head1.bias",
        "head2.weight",
        "head2.bias",
    }


def test_full_calibration_scope_is_rejected():
    model = _RiskOnlyToyModel()
    with pytest.raises(ValueError, match="only calibration_trainable_scope=heads"):
        configure_calibration_trainability(model, "full")


def test_head_only_calibration_trains_decoder_heads_and_keeps_encoder_frozen(monkeypatch):
    model = _tiny_dual_head(monkeypatch)
    before = encoder_state_sha256(model)
    configure_calibration_trainability(model, "heads")

    assert any(parameter.requires_grad for parameter in model.head1.parameters())
    assert any(parameter.requires_grad for parameter in model.head2.parameters())
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(model.ENCODER_PREFIXES)
    )
    assert not model.fusion.raw_xi.requires_grad
    assert not model.fusion.bias.requires_grad
    assert not model.fusion.raw_eta.requires_grad
    assert encoder_state_sha256(model) == before


def test_head_only_calibration_updates_heads_but_not_encoder():
    model = _RiskOnlyToyModel()
    parameters = configure_calibration_trainability(model, "heads")
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    encoder_before = encoder_state_sha256(model)
    frozen_before = frozen_calibration_state_sha256(model)
    head_before = model.head1.weight.detach().clone()
    fusion_bias_before = model.fusion.bias.detach().clone()
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
        calibration_scope="heads",
    )

    assert model.observed_training is False
    assert model.head1.training is True
    assert model.head2.training is True
    torch.testing.assert_close(model.observed_second_view, _toy_batch()["image_view2"])
    assert encoder_state_sha256(model) == encoder_before
    assert frozen_calibration_state_sha256(model) == frozen_before
    assert not torch.equal(model.head1.weight.detach(), head_before)
    torch.testing.assert_close(model.fusion.bias.detach(), fusion_bias_before)


def test_risk_brier_is_diagnostic_only_during_calibration():
    model = _RiskOnlyToyModel()
    config = _config()
    config["uncertainty"]["margin_gradient"]["enabled"] = False
    parameters = configure_calibration_trainability(model, "heads")
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    head_before = model.head1.weight.detach().clone()

    metrics = train_epoch(
        model,
        [_toy_batch()],
        optimizer,
        _make_scaler(False),
        torch.device("cpu"),
        _ZeroSegmentationLoss(),
        config,
        lambda_u=1.0,
        stage="calibration",
        epoch=1,
        calibration_scope="heads",
    )

    torch.testing.assert_close(model.head1.weight.detach(), head_before)
    assert metrics["calibration_brier"] > 0.0
    assert metrics["weighted_calibration_brier"] == 0.0
    assert metrics["risk_brier_backprop"] is False


def test_margin_gradient_updates_only_during_calibration():
    calibration_model = _RiskOnlyToyModel()
    calibration_parameters = configure_calibration_trainability(calibration_model, "heads")
    calibration_optimizer = torch.optim.SGD(calibration_parameters, lr=0.1)
    calibration_head_before = calibration_model.head1.weight.detach().clone()
    calibration_encoder_before = encoder_state_sha256(calibration_model)
    train_epoch(
        calibration_model,
        [_toy_batch()],
        calibration_optimizer,
        _make_scaler(False),
        torch.device("cpu"),
        _ZeroSegmentationLoss(),
        _config(),
        lambda_u=0.0,
        stage="calibration",
        epoch=1,
        calibration_scope="heads",
    )

    warmup_model = _RiskOnlyToyModel()
    warmup_optimizer = torch.optim.SGD(warmup_model.parameters(), lr=0.1)
    warmup_head_before = warmup_model.head1.weight.detach().clone()
    train_epoch(
        warmup_model,
        [_toy_batch()],
        warmup_optimizer,
        _make_scaler(False),
        torch.device("cpu"),
        _ZeroSegmentationLoss(),
        _config(),
        lambda_u=0.0,
        stage="warmup",
        epoch=1,
    )

    assert not torch.equal(calibration_model.head1.weight.detach(), calibration_head_before)
    assert encoder_state_sha256(calibration_model) == calibration_encoder_before
    torch.testing.assert_close(warmup_model.head1.weight.detach(), warmup_head_before)


def test_calibration_checkpoint_key_uses_only_basic_ece():
    better_ece = {"ece": 0.04, "risk_ece": 0.30, "risk_brier": 0.30, "mean_dice": 0.75}
    worse_ece = {"ece": 0.05, "risk_ece": 0.01, "risk_brier": 0.01, "mean_dice": 0.90}
    same_ece_better_dice = {"ece": 0.04, "risk_ece": 0.50, "risk_brier": 0.50, "mean_dice": 0.80}

    assert calibration_checkpoint_key(better_ece) < calibration_checkpoint_key(worse_ece)
    assert calibration_checkpoint_key(same_ece_better_dice) == calibration_checkpoint_key(better_ece)


class _TelemetryStub:
    instances = []

    def __init__(self, *args, **kwargs):
        self.run_id = "test-run"
        self.events = []
        self.closed = None
        self.__class__.instances.append(self)

    def event(self, event, **payload):
        self.events.append((event, payload))

    def epoch_finished(self, *args, **kwargs):
        return None

    def validation_finished(self, *args, **kwargs):
        return None

    def close(self, status, **payload):
        self.closed = (status, payload)


def _run_config(tmp_path):
    config = _config()
    split = tmp_path / "split.json"
    split.write_text("{}")
    config.update({"experiment": "direct-calibration", "reproducibility": {"seed": 1, "deterministic": True}})
    config["paths"] = {
        "run_root": str(tmp_path / "runs"),
        "split_json": str(split),
        "data_root": str(tmp_path),
    }
    config["training"].update(
        {
            "calibration_learning_rate": 1e-3,
            "weight_decay": 0.0,
            "dice_tolerance": 0.01,
            "calibration_trainable_scope": "heads",
        }
    )
    return config


def _patch_direct_calibration_runtime(monkeypatch, model, validation_metrics):
    monkeypatch.setattr("ours.train.DualHeadSwinUNETR", lambda config: model)
    monkeypatch.setattr("ours.train.RegionDiceLoss", _ZeroSegmentationLoss)
    monkeypatch.setattr("ours.train.build_loader", lambda *args, **kwargs: [_toy_batch()])
    monkeypatch.setattr("ours.train.save_run_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr("ours.train.TrainingTelemetry", _TelemetryStub)
    monkeypatch.setattr(
        "ours.train.train_epoch",
        lambda *args, **kwargs: {"duration_seconds": 0.0},
    )
    monkeypatch.setattr("ours.train.validate", lambda *args, **kwargs: dict(validation_metrics))
    monkeypatch.setattr("ours.train.fit_z0", lambda *args, **kwargs: 0.1)

    def fit_statistics(current, *args, **kwargs):
        current.zernike_stats.count.fill_(3)
        return {"fitted": True}

    monkeypatch.setattr("ours.train.fit_zernike_statistics", fit_statistics)


def test_direct_calibration_from_warmup_fits_stats_without_running_warmup(monkeypatch, tmp_path):
    config = _run_config(tmp_path)
    warmup = _tiny_dual_head(monkeypatch)
    checkpoint = tmp_path / "warmup_without_metrics.pt"
    save_checkpoint(checkpoint, warmup, "warmup", 4, config, metrics={})
    current = _tiny_dual_head(monkeypatch)
    _TelemetryStub.instances.clear()
    _patch_direct_calibration_runtime(
        monkeypatch,
        current,
        {"mean_dice": 0.8, "ece": 0.05, "risk_ece": 0.4, "risk_brier": 0.3},
    )

    final_path = run(config, "calibration", str(checkpoint))
    events = _TelemetryStub.instances[-1].events

    assert final_path.name == "final.pt"
    assert (final_path.parent / "stats_fitted.pt").is_file()
    assert any(event == "reference_dice_fitted" for event, _ in events)
    assert not any(event == "stage_started" and payload.get("stage") == "warmup" for event, payload in events)


def test_no_dice_eligible_checkpoint_keeps_only_diagnostic_last(monkeypatch, tmp_path):
    config = _run_config(tmp_path)
    warmup = _tiny_dual_head(monkeypatch)
    checkpoint = tmp_path / "warmup.pt"
    save_checkpoint(checkpoint, warmup, "warmup", 4, config, metrics={"mean_dice": 0.9})
    current = _tiny_dual_head(monkeypatch)
    _TelemetryStub.instances.clear()
    _patch_direct_calibration_runtime(
        monkeypatch,
        current,
        {"mean_dice": 0.5, "ece": 0.01, "risk_ece": 0.01, "risk_brier": 0.01},
    )

    with pytest.raises(RuntimeError, match="No calibration checkpoint satisfied"):
        run(config, "calibration", str(checkpoint))

    run_dir = tmp_path / "runs" / "direct-calibration"
    assert (run_dir / "last_calibration.pt").is_file()
    assert not (run_dir / "best_calibrated.pt").exists()
    assert not (run_dir / "final.pt").exists()


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
