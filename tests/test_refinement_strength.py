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
            "refine_uncertainty_top_fraction": 1.0,
            "refine_percentile_roi_dilation": 0,
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


def test_percentile_gate_selects_exact_top_twenty_percent_inside_predicted_roi():
    base = torch.zeros(1, 4, 1, 1, 12)
    base[:, 0] = 1.0
    base[:, 0, :, :, :10] = 0.1
    base[:, 1, :, :, :10] = 0.9
    uncertainty = torch.arange(12, dtype=torch.float32).view(1, 1, 1, 1, 12)
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        uncertainty_top_fraction=0.20,
        percentile_roi_dilation=0,
    )

    selected = transfer.selection_mask(base, uncertainty)

    assert int(selected.sum()) == 2
    assert bool(selected[0, 0, 0, 0, 8])
    assert bool(selected[0, 0, 0, 0, 9])
    assert not bool(selected[0, 0, 0, 0, 10])
    assert not bool(selected[0, 0, 0, 0, 11])


def test_percentile_gate_uses_predicted_tumor_dilation_without_ground_truth():
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = 1.0
    base[:, 0, 2, 2, 2] = 0.1
    base[:, 1, 2, 2, 2] = 0.9
    uncertainty = torch.zeros(1, 1, 5, 5, 5)
    uncertainty[:, :, 2, 2, 3] = 1.0
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        uncertainty_top_fraction=0.04,
        percentile_roi_dilation=1,
    )

    selected = transfer.selection_mask(base, uncertainty)

    assert int(selected.sum()) == 2
    assert bool(selected[0, 0, 2, 2, 3])


def test_empty_predicted_tumor_roi_disables_refinement():
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = 1.0
    uncertainty = torch.rand(1, 1, 5, 5, 5)
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.9,
        beta=0.75,
        iterations=4,
        uncertainty_top_fraction=0.20,
        percentile_roi_dilation=1,
    )

    refined = transfer(base, uncertainty)

    torch.testing.assert_close(refined, base)
    assert not bool(transfer.selection_mask(base, uncertainty).any())


def test_default_config_uses_selected_refine_parameters():
    config = load_config(
        Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    )
    transfer = UncertaintyGatedLabelTransfer(
        alpha_max=config["label_transfer"]["alpha_max"],
        beta=config["label_transfer"]["beta"],
        iterations=config["label_transfer"]["iterations"],
        uncertainty_top_fraction=config["label_transfer"][
            "uncertainty_top_fraction"
        ],
        percentile_roi_dilation=config["label_transfer"][
            "percentile_roi_dilation"
        ],
    )
    metadata = _configure_test_time_refinement(
        SimpleNamespace(label_transfer=transfer), config
    )

    assert transfer.beta == pytest.approx(0.75)
    assert transfer.iterations == 4
    assert transfer.uncertainty_top_fraction == pytest.approx(0.20)
    assert transfer.percentile_roi_dilation == 1
    assert metadata["refine_strength_scale"] == pytest.approx(2.8)
    assert metadata["refine_effective_alpha_max"] == pytest.approx(0.98)
    assert metadata["refine_uncertainty_top_fraction"] == pytest.approx(0.20)
    assert metadata["refine_percentile_roi_dilation"] == 1


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("inf")])
def test_invalid_uncertainty_top_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="uncertainty_top_fraction"):
        UncertaintyGatedLabelTransfer(uncertainty_top_fraction=fraction)


@pytest.mark.parametrize("scale", [0.0, -1.0, 3.0])
def test_invalid_effective_test_time_refine_strength_is_rejected(scale):
    model = SimpleNamespace(
        label_transfer=UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    )
    with pytest.raises(ValueError, match="Effective test-time refine alpha_max"):
        _configure_test_time_refinement(model, {"evaluation": {"refine_strength_scale": scale}})
