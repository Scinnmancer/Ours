from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ours.config import load_config
from ours.test import _configure_test_time_refinement
from ours.transfer import UncertaintyGatedLabelTransfer


def test_test_time_refine_strength_doubles_alpha_max_without_state_change():
    transfer = UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    model = SimpleNamespace(label_transfer=transfer)
    state_before = {key: value.clone() for key, value in transfer.state_dict().items()}

    metadata = _configure_test_time_refinement(
        model,
        {"evaluation": {"refine_strength_scale": 2.0}},
    )

    assert transfer.alpha_max == pytest.approx(0.70)
    assert metadata == pytest.approx(
        {
            "refine_base_alpha_max": 0.35,
            "refine_strength_scale": 2.0,
            "refine_effective_alpha_max": 0.70,
            "refine_uncertainty_gain": 1.0,
        }
    )
    for key, value in transfer.state_dict().items():
        torch.testing.assert_close(value, state_before[key])


def test_larger_test_time_scale_produces_a_stronger_refine_update():
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = 1.0
    base[:, 0, 2, 2, 2] = 0.1
    base[:, 1, 2, 2, 2] = 0.9
    base[:, 0, 1:4, 1:4, 1:4] = 0.1
    base[:, 2, 1:4, 1:4, 1:4] = 0.9
    base[:, 2, 2, 2, 2] = 0.0
    uncertainty = torch.full((1, 1, 5, 5, 5), 0.8)

    weak = UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35, beta=2.0, iterations=3, z0=0.01)
    strong = UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35, beta=2.0, iterations=3, z0=0.01)
    _configure_test_time_refinement(
        SimpleNamespace(label_transfer=strong),
        {"evaluation": {"refine_strength_scale": 2.0}},
    )

    weak_update = (weak(base, uncertainty) - base).abs().mean()
    strong_update = (strong(base, uncertainty) - base).abs().mean()
    assert strong_update > weak_update


def test_refine_uncertainty_gain_expands_coverage_without_state_change():
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = 1.0
    base[:, 0, 2, 2, 2] = 0.1
    base[:, 1, 2, 2, 2] = 0.9
    base[:, 0, 1:4, 1:4, 1:4] = 0.1
    base[:, 2, 1:4, 1:4, 1:4] = 0.9
    base[:, 2, 2, 2, 2] = 0.0
    uncertainty = torch.full((1, 1, 5, 5, 5), 0.35)

    baseline = UncertaintyGatedLabelTransfer(
        radius=1, alpha_max=0.35, beta=0.75, uncertainty_gain=1.0,
        iterations=4, z0=0.01,
    )
    expanded = UncertaintyGatedLabelTransfer(
        radius=1, alpha_max=0.35, beta=0.75, uncertainty_gain=2.0,
        iterations=4, z0=0.01,
    )
    state_before = {key: value.clone() for key, value in baseline.state_dict().items()}
    expanded.load_state_dict(state_before, strict=True)

    baseline_update = (baseline(base, uncertainty) - base).abs().mean()
    expanded_update = (expanded(base, uncertainty) - base).abs().mean()

    assert expanded_update > baseline_update
    assert set(expanded.state_dict()) == set(state_before)


@pytest.mark.parametrize("gain", [0.0, -1.0, float("inf")])
def test_invalid_refine_uncertainty_gain_is_rejected(gain):
    with pytest.raises(ValueError, match="uncertainty_gain"):
        UncertaintyGatedLabelTransfer(uncertainty_gain=gain)


def test_default_config_uses_selected_refine_parameters():
    config = load_config(
        Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    )
    transfer = UncertaintyGatedLabelTransfer(
        alpha_max=config["label_transfer"]["alpha_max"],
        beta=config["label_transfer"]["beta"],
        iterations=config["label_transfer"]["iterations"],
    )
    metadata = _configure_test_time_refinement(
        SimpleNamespace(label_transfer=transfer), config
    )

    assert transfer.beta == pytest.approx(1.75)
    assert transfer.uncertainty_gain == pytest.approx(1.0)
    assert transfer.iterations == 4
    assert metadata["refine_strength_scale"] == pytest.approx(2.8)
    assert metadata["refine_effective_alpha_max"] == pytest.approx(0.98)
    assert metadata["refine_uncertainty_gain"] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, 3.0])
def test_invalid_effective_test_time_refine_strength_is_rejected(scale):
    model = SimpleNamespace(
        label_transfer=UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    )
    with pytest.raises(ValueError, match="Effective test-time refine alpha_max"):
        _configure_test_time_refinement(model, {"evaluation": {"refine_strength_scale": scale}})
