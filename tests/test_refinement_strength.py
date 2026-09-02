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
            "refine_neighborhood_radius": 1,
            "refine_neighborhood_sigma": 1.0,
            "refine_neighbor_reliability_power": 1.0,
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


def test_all_voxel_refine_updates_a_low_uncertainty_boundary_voxel():
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = 1.0
    base[:, 0, 1:4, 1:4, 1:4] = 0.1
    base[:, 2, 1:4, 1:4, 1:4] = 0.9
    uncertainty = torch.full((1, 1, 5, 5, 5), 0.9)
    uncertainty[:, :, 2, 2, 0] = 0.2
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.994,
        beta=0.75,
        iterations=4,
        z0=0.01,
    )

    refined = transfer(base, uncertainty)

    assert refined[0, 0, 2, 2, 0] < base[0, 0, 2, 2, 0]
    assert refined[0, 2, 2, 2, 0] > base[0, 2, 2, 2, 0]


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

    assert transfer.beta == pytest.approx(0.75)
    assert transfer.iterations == 4
    assert "uncertainty_top_fraction" not in config["label_transfer"]
    assert "percentile_roi_dilation" not in config["label_transfer"]
    assert not hasattr(transfer, "selection_mask")
    assert metadata["refine_strength_scale"] == pytest.approx(2.84)
    assert metadata["refine_effective_alpha_max"] == pytest.approx(0.994)
    assert transfer.radius == 2
    assert transfer.sigma == pytest.approx(1.0)
    assert tuple(transfer.kernel.shape) == (1, 1, 5, 5, 5)
    assert transfer.neighbor_reliability_power == pytest.approx(2.0)
    assert metadata["refine_neighborhood_radius"] == 2
    assert metadata["refine_neighborhood_sigma"] == pytest.approx(1.0)
    assert metadata["refine_neighbor_reliability_power"] == pytest.approx(2.0)


def test_reliability_sharpening_increases_trusted_neighbor_influence():
    base = torch.zeros(1, 4, 3, 3, 3)
    base[:, 0] = 1.0
    base[:, 0, 1, 1, 0] = 0.0
    base[:, 1, 1, 1, 0] = 1.0
    base[:, 0, 1, 1, 2] = 0.0
    base[:, 2, 1, 1, 2] = 1.0
    uncertainty = torch.ones(1, 1, 3, 3, 3)
    uncertainty[:, :, 1, 1, 1] = 0.9
    uncertainty[:, :, 1, 1, 0] = 0.1
    uncertainty[:, :, 1, 1, 2] = 0.6
    plain = UncertaintyGatedLabelTransfer(
        radius=1, alpha_max=0.9, beta=0.0, iterations=1, z0=0.01
    )
    sharpened = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.9,
        beta=0.0,
        iterations=1,
        neighbor_reliability_power=2.0,
        z0=0.01,
    )

    plain_result = plain(base, uncertainty)
    sharpened_result = sharpened(base, uncertainty)

    assert sharpened_result[0, 1, 1, 1, 1] > plain_result[0, 1, 1, 1, 1]
    assert sharpened_result[0, 2, 1, 1, 1] < plain_result[0, 2, 1, 1, 1]


@pytest.mark.parametrize(
    ("radius", "sigma"),
    [(0, 1.0), (1, 0.0), (1, -0.5), (1, float("nan"))],
)
def test_invalid_test_time_neighborhood_is_rejected(radius, sigma):
    model = SimpleNamespace(
        label_transfer=UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    )
    with pytest.raises(ValueError, match="radius|sigma"):
        _configure_test_time_refinement(
            model,
            {
                "evaluation": {
                    "refine_strength_scale": 1.0,
                    "refine_neighborhood_radius": radius,
                    "refine_neighborhood_sigma": sigma,
                }
            },
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, 3.0])
def test_invalid_effective_test_time_refine_strength_is_rejected(scale):
    model = SimpleNamespace(
        label_transfer=UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    )
    with pytest.raises(ValueError, match="Effective test-time refine alpha_max"):
        _configure_test_time_refinement(model, {"evaluation": {"refine_strength_scale": scale}})
