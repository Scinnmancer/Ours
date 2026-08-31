from types import SimpleNamespace

import pytest
import torch

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


@pytest.mark.parametrize("scale", [0.0, -1.0, 3.0])
def test_invalid_effective_test_time_refine_strength_is_rejected(scale):
    model = SimpleNamespace(
        label_transfer=UncertaintyGatedLabelTransfer(radius=1, alpha_max=0.35)
    )
    with pytest.raises(ValueError, match="Effective test-time refine alpha_max"):
        _configure_test_time_refinement(model, {"evaluation": {"refine_strength_scale": scale}})
