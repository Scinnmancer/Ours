import json
from pathlib import Path

import pytest
import torch

from ours.config import load_config
from ours.metrics import evaluate_case
from ours.test import _atomic_prediction, _region_prediction_from_atomic
from ours.tune_refine import (
    _evaluate_selection_case,
    build_candidates,
    run_tuning,
    select_candidate,
    select_with_guardrails,
)


def _candidate(name: str, ece: float, dice: float, base_dice: float = 0.83):
    return {
        "name": name,
        "basic_ece": ece,
        "mean_dice": dice,
        "base_mean_dice": base_dice,
    }


def test_default_refine_grid_contains_recommended_high_uncertainty_setting():
    config_path = Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    config = load_config(config_path)
    candidates = build_candidates(config)

    assert len(candidates) == 75
    assert any(
        candidate["iterations"] == 3
        and candidate["beta"] == 2.0
        and candidate["refine_strength_scale"] == 2.6
        and candidate["effective_alpha_max"] == pytest.approx(0.91)
        for candidate in candidates
    )
    assert {candidate["iterations"] for candidate in candidates} == {
        1,
        2,
        3,
        4,
        5,
    }
    assert all(float(candidate["effective_alpha_max"]) < 1.0 for candidate in candidates)


def test_selection_minimizes_ece_after_dice_guardrail():
    candidates = [
        _candidate("ineligible", 0.05, 0.8290),
        _candidate("winner", 0.08, 0.8299),
        _candidate("higher-ece", 0.09, 0.84),
    ]
    best = select_candidate(candidates, dice_tolerance=0.0002)

    assert best is candidates[1]
    assert candidates[0]["eligible"] is False
    assert candidates[1]["eligible"] is True


def test_selection_uses_higher_dice_only_within_ece_tolerance():
    candidates = [
        _candidate("earlier", 0.08, 0.83),
        _candidate("higher-dice", 0.0801, 0.84),
    ]
    assert (
        select_candidate(
            candidates,
            dice_tolerance=0.0002,
            ece_tie_tolerance=0.0002,
        )
        is candidates[1]
    )


def test_selection_keeps_materially_lower_ece_despite_higher_dice():
    candidates = [
        _candidate("lower-ece", 0.08, 0.83),
        _candidate("higher-dice", 0.0803, 0.84),
    ]
    assert (
        select_candidate(
            candidates,
            dice_tolerance=0.0002,
            ece_tie_tolerance=0.0002,
        )
        is candidates[0]
    )


def test_selection_uses_small_fallback_only_when_strict_guardrail_is_empty():
    candidates = [
        _candidate("fallback", 0.08, 0.82995),
        _candidate("ineligible", 0.07, 0.8298),
    ]
    best, tier = select_with_guardrails(
        candidates,
        dice_tolerance=0.0,
        fallback_dice_tolerance=0.0001,
        ece_tie_tolerance=0.0002,
    )

    assert best is candidates[0]
    assert tier == "fallback"


def test_fast_source_selection_matches_full_dice_and_ece():
    probability = torch.full((1, 4, 5, 5, 5), 0.025)
    probability[:, 0] = 0.925
    probability[:, 0, 1:4, 1:4, 1:4] = 0.05
    probability[:, 1, 1:4, 1:4, 1:4] = 0.85
    probability[:, 2, 1:4, 1:4, 1:4] = 0.075
    target = torch.zeros(1, 1, 5, 5, 5, dtype=torch.uint8)
    target[:, :, 1:4, 1:4, 1:4] = 1
    atomic_prediction = _atomic_prediction(probability)
    region_prediction = _region_prediction_from_atomic(probability)

    fast = _evaluate_selection_case(
        probability,
        target,
        atomic_prediction,
        region_prediction,
        bins=15,
        max_voxels=200000,
    )
    full = evaluate_case(
        probability,
        torch.zeros(1, 1, 5, 5, 5),
        target,
        atomic_prediction=atomic_prediction,
        region_prediction=region_prediction,
        risk_reference_prediction=atomic_prediction,
        bins=15,
        max_voxels=200000,
    )

    assert fast["mean_dice"] == pytest.approx(full["mean_dice"])
    assert fast["basic_ece"] == pytest.approx(full["basic_ece"])


def test_one_click_tuning_runs_source_sweep_then_final_once(tmp_path, monkeypatch):
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "refine_tuning"
    final_calls = []

    def fake_source_sweep(_config, _checkpoint, sweep_output, candidates):
        for candidate in candidates:
            candidate_output = sweep_output / "source_val" / str(candidate["name"])
            candidate_output.mkdir(parents=True, exist_ok=True)
            is_winner = (
                candidate["iterations"] == 4
                and candidate["beta"] == 2.0
                and candidate["refine_strength_scale"] == 2.7
            )
            rows = [
                {
                    "domain": "source_val",
                    "prediction": "base",
                    "mean_dice": 0.83,
                    "basic_ece": 0.12,
                },
                {
                    "domain": "source_val",
                    "prediction": "refined",
                    "mean_dice": 0.8301,
                    "basic_ece": 0.08 if is_winner else 0.10,
                },
            ]
            (candidate_output / "summary_metrics.json").write_text(json.dumps(rows))

    def fake_final(config, _checkpoint, output_dir):
        final_calls.append(
            (
                config["label_transfer"]["beta"],
                config["label_transfer"]["iterations"],
                config["evaluation"]["refine_strength_scale"],
            )
        )
        result = Path(output_dir)
        result.mkdir(parents=True, exist_ok=True)
        (result / "summary_metrics.json").write_text("[]")
        return result

    monkeypatch.setattr("ours.tune_refine.run_source_sweep", fake_source_sweep)
    monkeypatch.setattr("ours.tune_refine.run_evaluation", fake_final)
    monkeypatch.setattr("ours.tune_refine.save_run_metadata", lambda *_args: None)
    config_path = Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"

    run_tuning(str(config_path), str(checkpoint), str(output))

    selection = json.loads((output / "selection.json").read_text())
    assert selection["refine_formula"] == "complement_neighbor_v1"
    assert selection["selection_tier"] == "strict"
    assert selection["best"]["iterations"] == 4
    assert selection["best"]["beta"] == 2.0
    assert selection["best"]["refine_strength_scale"] == 2.7
    assert final_calls == [(2.0, 4, 2.7)]
    assert (output / "candidates.csv").is_file()
