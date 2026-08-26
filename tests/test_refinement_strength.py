from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ours.test import _configure_test_time_refinement
from ours.transfer import UncertaintyGatedLabelTransfer


def test_default_test_time_refine_strength_keeps_alpha_max_without_state_change():
    transfer = UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    model = SimpleNamespace(label_transfer=transfer)
    state_before = {key: value.clone() for key, value in transfer.state_dict().items()}

    metadata = _configure_test_time_refinement(
        model,
        {"evaluation": {"refine_strength_scale": 1.0}},
    )

    assert transfer.alpha_max == pytest.approx(0.35)
    assert metadata == pytest.approx(
        {
            "refine_base_alpha_max": 0.35,
            "refine_strength_scale": 1.0,
            "refine_effective_alpha_max": 0.35,
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


def _consensus_case(neighbor_probability: tuple[float, float], center_probability=(0.5005, 0.4995)):
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = neighbor_probability[0]
    base[:, 1] = neighbor_probability[1]
    base[:, 0, 2, 2, 2] = center_probability[0]
    base[:, 1, 2, 2, 2] = center_probability[1]
    uncertainty = torch.zeros(1, 1, 5, 5, 5)
    uncertainty[:, :, 2, 2, 2] = 1.0
    return base, uncertainty


def test_weak_neighbor_consensus_leaves_probability_unchanged():
    base, uncertainty = _consensus_case((0.55, 0.45), center_probability=(0.9, 0.1))
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.35,
        iterations=3,
        consensus_margin=0.25,
        class_change_margin=0.30,
        z0=0.01,
    )

    refined = transfer(base, uncertainty)

    torch.testing.assert_close(refined, base)


def test_zero_support_and_uniform_probability_are_stable():
    base = torch.full((1, 4, 3, 3, 3), 0.25)
    uncertainty = torch.ones(1, 1, 3, 3, 3)
    transfer = UncertaintyGatedLabelTransfer(radius=1, iterations=3, z0=0.01)

    refined = transfer(base, uncertainty)

    torch.testing.assert_close(refined, base)


def test_strong_neighbor_consensus_updates_and_can_safely_change_top_class():
    base, uncertainty = _consensus_case((0.05, 0.95))
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.35,
        iterations=3,
        consensus_margin=0.25,
        class_change_margin=0.30,
        z0=0.01,
    )

    refined = transfer(base, uncertainty)

    assert refined[0, 1, 2, 2, 2] > base[0, 1, 2, 2, 2]
    assert refined[0, :, 2, 2, 2].argmax() == 1
    torch.testing.assert_close(
        refined.sum(dim=1),
        torch.ones_like(refined[:, 0]),
    )
    assert torch.all(refined >= 0.0)


def test_class_change_below_safety_margin_is_rejected():
    base, uncertainty = _consensus_case((0.36, 0.64))
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.35,
        iterations=3,
        consensus_margin=0.25,
        class_change_margin=0.30,
        z0=0.01,
    )

    refined = transfer(base, uncertainty)

    torch.testing.assert_close(refined[:, :, 2, 2, 2], base[:, :, 2, 2, 2])
    assert refined[0, :, 2, 2, 2].argmax() == 0


def test_refinement_recomputes_neighborhood_for_all_three_iterations():
    base, uncertainty = _consensus_case((0.05, 0.95))
    transfer = UncertaintyGatedLabelTransfer(radius=1, iterations=3, z0=0.01)

    with patch("ours.transfer.F.conv3d", wraps=torch.nn.functional.conv3d) as conv3d:
        transfer(base, uncertainty)

    assert conv3d.call_count == 4  # One support convolution plus one per iteration.


def test_consensus_thresholds_do_not_change_checkpoint_state():
    legacy = UncertaintyGatedLabelTransfer(radius=1)
    configured = UncertaintyGatedLabelTransfer(
        radius=1,
        consensus_margin=0.4,
        class_change_margin=0.6,
    )

    assert configured.load_state_dict(legacy.state_dict(), strict=True) is not None
    assert configured.state_dict().keys() == legacy.state_dict().keys()


@pytest.mark.parametrize("scale", [0.0, -1.0, 3.0])
def test_invalid_effective_test_time_refine_strength_is_rejected(scale):
    model = SimpleNamespace(
        label_transfer=UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    )
    with pytest.raises(ValueError, match="Effective test-time refine alpha_max"):
        _configure_test_time_refinement(model, {"evaluation": {"refine_strength_scale": scale}})
