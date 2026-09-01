from pathlib import Path

import pytest

from ours.config import load_config
from ours.tune_refine_participation import (
    build_candidates,
    select_candidate,
    select_with_guardrails,
)


def _candidate(**updates):
    value = {
        "name": "candidate",
        "base_mean_dice": 0.80,
        "mean_dice": 0.801,
        "base_dice_ET": 0.75,
        "dice_ET": 0.7501,
        "basic_ece": 0.12,
        "atomic_change_rate": 0.02,
        "atomic_correction_precision": 0.60,
        "atomic_corrected_voxels": 60,
        "atomic_corrupted_voxels": 30,
    }
    value.update(updates)
    return value


def test_default_participation_grid_has_36_candidates():
    config = load_config(
        Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    )
    candidates = build_candidates(config)

    assert len(candidates) == 36
    assert {item["iterations"] for item in candidates} == {4, 5, 6}
    assert {item["beta"] for item in candidates} == {0.5, 0.75, 1.0}
    assert {item["uncertainty_gain"] for item in candidates} == {
        1.5,
        1.75,
        2.0,
        2.25,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"atomic_change_rate": 0.0149},
        {"atomic_change_rate": 0.0251},
        {"atomic_correction_precision": 0.549},
        {"atomic_corrected_voxels": 30, "atomic_corrupted_voxels": 30},
        {"mean_dice": 0.7999},
        {"dice_ET": 0.7494},
    ],
)
def test_selection_rejects_candidates_that_break_a_guardrail(updates):
    assert select_candidate([_candidate(**updates)]) is None


def test_selection_prefers_dice_then_ece_then_target_distance():
    lower_ece = _candidate(name="lower_ece", mean_dice=0.801, basic_ece=0.11)
    higher_dice = _candidate(name="higher_dice", mean_dice=0.802, basic_ece=0.13)
    farther = _candidate(name="farther", mean_dice=0.802, basic_ece=0.13, atomic_change_rate=0.024)

    assert select_candidate([lower_ece, farther, higher_dice]) is higher_dice


def test_fallback_allows_only_configured_tiny_dice_drop():
    config = load_config(
        Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    )
    tuning = config["evaluation"]["refine_participation_tuning"]
    candidate = _candidate(mean_dice=0.79995)

    best, tier = select_with_guardrails([candidate], tuning)

    assert best is candidate
    assert tier == "fallback"


def test_nearest_safe_tier_handles_a_grid_that_misses_target_band():
    config = load_config(
        Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    )
    tuning = config["evaluation"]["refine_participation_tuning"]
    far = _candidate(name="far", atomic_change_rate=0.006)
    near = _candidate(name="near", atomic_change_rate=0.012, basic_ece=0.13)

    best, tier = select_with_guardrails([far, near], tuning)

    assert best is near
    assert tier == "nearest_safe"


def test_no_safe_candidate_returns_diagnostics_instead_of_selecting():
    config = load_config(
        Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    )
    tuning = config["evaluation"]["refine_participation_tuning"]
    unsafe = _candidate(
        atomic_change_rate=0.03,
        atomic_corrected_voxels=10,
        atomic_corrupted_voxels=20,
    )

    best, tier = select_with_guardrails([unsafe], tuning)

    assert best is None
    assert tier is None
