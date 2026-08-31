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
)


def _candidate(name: str, ece: float, dice: float, base_dice: float = 0.83):
    return {
        "name": name,
        "basic_ece": ece,
        "mean_dice": dice,
        "base_mean_dice": base_dice,
    }


def test_default_refine_grid_contains_central_smoothing_setting():
    config_path = Path(__import__("ours").__path__[0]) / "configs" / "brats2020.yaml"
    config = load_config(config_path)
    candidates = build_candidates(config)

    assert len(candidates) == 12
    assert any(
        candidate["beta"] == 1.5
        and candidate["refine_strength_scale"] == 2.0
        and candidate["effective_alpha_max"] == pytest.approx(0.70)
        for candidate in candidates
    )
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


def test_selection_uses_higher_dice_only_for_equal_ece():
    candidates = [
        _candidate("earlier", 0.08, 0.83),
        _candidate("higher-dice", 0.08, 0.84),
    ]
    assert select_candidate(candidates, dice_tolerance=0.0002) is candidates[1]


def test_selection_uses_higher_dice_inside_configured_ece_tie_band():
    candidates = [
        _candidate("slightly-lower-ece", 0.0800, 0.8300),
        _candidate("higher-dice", 0.0808, 0.8400),
    ]
    assert (
        select_candidate(
            candidates, dice_tolerance=0.0, ece_tie_tolerance=0.001
        )
        is candidates[1]
    )


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
    assert fast["mean_top_confidence"] == pytest.approx(0.85)
    assert fast["mean_entropy"] > 0.0
    assert fast["top1_flip_rate"] == 0.0


def test_confidence_diagnostics_measure_smoothing_and_top1_changes():
    probability = torch.zeros(1, 4, 2, 2, 2)
    probability[:, 0] = 0.7
    probability[:, 1] = 0.3
    target = torch.zeros(1, 1, 2, 2, 2, dtype=torch.uint8)
    probability[:, 0, 0, 0, 0] = 0.3
    probability[:, 1, 0, 0, 0] = 0.7
    target[:, :, 0, 0, 0] = 1
    prediction = _atomic_prediction(probability)
    reference = 1 - prediction

    metrics = _evaluate_selection_case(
        probability,
        target,
        prediction,
        _region_prediction_from_atomic(probability),
        bins=15,
        max_voxels=200000,
        reference_prediction=reference,
    )

    assert metrics["mean_top_confidence"] == pytest.approx(0.7)
    assert metrics["p95_top_confidence"] == pytest.approx(0.7)
    assert metrics["top1_flip_rate"] == pytest.approx(1.0)


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
                candidate["beta"] == 1.5
                and candidate["refine_strength_scale"] == 2.0
            )
            diagnostics = {
                "mean_top_confidence": 0.8,
                "p95_top_confidence": 0.95,
                "mean_error_top_confidence": 0.9,
                "mean_entropy": 0.4,
                "high_confidence_error_rate": 0.03,
                "top1_flip_rate": 0.0,
            }
            rows = [
                {
                    "domain": "source_val",
                    "prediction": "base",
                    "mean_dice": 0.83,
                    "basic_ece": 0.12,
                    **diagnostics,
                },
                {
                    "domain": "source_val",
                    "prediction": "refined",
                    "mean_dice": 0.8301,
                    "basic_ece": 0.08 if is_winner else 0.10,
                    **diagnostics,
                },
            ]
            (candidate_output / "summary_metrics.json").write_text(json.dumps(rows))

    def fake_final(config, _checkpoint, output_dir):
        final_calls.append(
            (
                config["label_transfer"]["beta"],
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
    assert selection["best"]["beta"] == 1.5
    assert selection["best"]["refine_strength_scale"] == 2.0
    assert selection["selection_tier"] == "strict"
    assert final_calls == [(1.5, 2.0)]
    assert (output / "candidates.csv").is_file()
