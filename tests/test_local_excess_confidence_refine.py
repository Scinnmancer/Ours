import pytest
import torch

from ours.config import validate_config
from ours.transfer import UncertaintyGatedLabelTransfer


def _distribution(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 4, 1, 1, 1)


def test_local_excess_confidence_is_continuous_and_parameter_free():
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        direction_mode="local_excess_confidence",
    )
    base = _distribution([0.9, 0.05, 0.03, 0.02])
    neighbor = _distribution([0.1, 0.7, 0.1, 0.1])

    target, rho = transfer._directional_target(base, neighbor)

    assert float(rho) == pytest.approx((0.9 - 0.1) / (1.0 - 0.1), rel=1e-6)
    torch.testing.assert_close(target.sum(dim=1), torch.ones_like(target[:, 0]))
    assert bool((target >= 0.0).all())


def test_supported_center_uses_neighbor_direction_without_hard_gate():
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        direction_mode="local_excess_confidence",
    )
    base = _distribution([0.6, 0.2, 0.1, 0.1])
    neighbor = _distribution([0.7, 0.1, 0.1, 0.1])

    target, rho = transfer._directional_target(base, neighbor)

    assert float(rho) == 0.0
    torch.testing.assert_close(target, neighbor)


def test_homogeneous_prediction_is_not_forced_away_from_its_class():
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = 0.9
    base[:, 1] = 0.1
    uncertainty = torch.full((1, 1, 5, 5, 5), 0.9)
    local = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.7,
        beta=1.0,
        iterations=3,
        z0=0.01,
        direction_mode="local_excess_confidence",
    )
    legacy = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.7,
        beta=1.0,
        iterations=3,
        z0=0.01,
        direction_mode="legacy_complement",
    )

    local_output = local(base, uncertainty)
    legacy_output = legacy(base, uncertainty)

    torch.testing.assert_close(local_output, base, atol=1e-6, rtol=1e-6)
    assert float((legacy_output - base).abs().mean()) > 0.05


def test_isolated_overconfident_class_is_corrected_toward_reliable_neighbors():
    base = torch.zeros(1, 4, 5, 5, 5)
    base[:, 0] = 0.95
    base[:, 1] = 0.05
    base[:, 0, 2, 2, 2] = 0.05
    base[:, 1, 2, 2, 2] = 0.95
    uncertainty = torch.full((1, 1, 5, 5, 5), 0.2)
    uncertainty[:, :, 2, 2, 2] = 0.95
    transfer = UncertaintyGatedLabelTransfer(
        radius=1,
        alpha_max=0.7,
        beta=1.0,
        iterations=3,
        z0=0.01,
        direction_mode="local_excess_confidence",
    )

    refined = transfer(base, uncertainty)

    assert refined[0, 1, 2, 2, 2] < base[0, 1, 2, 2, 2]
    assert refined[0, 0, 2, 2, 2] > base[0, 0, 2, 2, 2]
    torch.testing.assert_close(
        refined.sum(dim=1),
        torch.ones_like(refined[:, 0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_direction_mode_does_not_change_checkpoint_state_layout():
    legacy = UncertaintyGatedLabelTransfer(radius=1, direction_mode="legacy_complement")
    local = UncertaintyGatedLabelTransfer(
        radius=1,
        direction_mode="local_excess_confidence",
    )

    assert legacy.state_dict().keys() == local.state_dict().keys()
    local.load_state_dict(legacy.state_dict(), strict=True)


def test_invalid_direction_mode_is_rejected_by_config():
    config = {
        "paths": {},
        "data": {"roi_size": [8, 8, 8], "workers": 0},
        "model": {"head_dropout_rates": [0.2, 0.3]},
        "zernike": {"windows": [3]},
        "uncertainty": {},
        "label_transfer": {"alpha_max": 0.35, "direction_mode": "hard_gate"},
        "training": {
            "warmup_epochs": 1,
            "calibration_epochs": 1,
            "validation_every": 1,
        },
        "evaluation": {"workers": 0, "refine_strength_scale": 1.0},
    }
    with pytest.raises(ValueError, match="direction_mode"):
        validate_config(config)
