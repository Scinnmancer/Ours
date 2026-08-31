import json
from pathlib import Path

from ours.tune_refine import (
    _best_by_direction,
    _can_reuse_result,
    _candidate_name,
    _run_signature,
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


def test_select_candidate_uses_ece_after_dice_guardrail():
    candidates = [
        _candidate("lowest_but_ineligible", 0.08, 0.80),
        _candidate("eligible", 0.10, 0.825),
        _candidate("higher_ece", 0.12, 0.84),
    ]
    best = select_candidate(candidates, dice_tolerance=0.01)
    assert best is candidates[1]
    assert candidates[0]["eligible"] is False
    assert candidates[1]["eligible"] is True


def test_select_candidate_keeps_earlier_candidate_on_equal_ece():
    candidates = [
        _candidate("earlier", 0.10, 0.83),
        _candidate("later", 0.10, 0.84),
    ]
    assert select_candidate(candidates, dice_tolerance=0.01) is candidates[0]


def test_select_candidate_returns_none_when_all_fail_dice_guardrail():
    candidates = [_candidate("failed", 0.05, 0.80)]
    assert select_candidate(candidates, dice_tolerance=0.01) is None


def test_best_by_direction_keeps_a_fair_final_ablation():
    candidates = [
        {
            **_candidate("legacy", 0.11, 0.83),
            "direction_mode": "legacy_complement",
        },
        {
            **_candidate("new", 0.10, 0.8295),
            "direction_mode": "local_excess_confidence",
        },
    ]
    finalists = _best_by_direction(
        candidates,
        ["legacy_complement", "local_excess_confidence"],
        dice_tolerance=0.001,
    )
    assert [item["name"] for item in finalists] == ["legacy", "new"]


def test_tuning_reuses_completed_source_validation_results(tmp_path, monkeypatch):
    output = tmp_path / "tuning"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary = [
        {
            "domain": "source_val",
            "prediction": "base",
            "mean_dice": 0.83,
            "basic_ece": 0.12,
        },
        {
            "domain": "source_val",
            "prediction": "refined",
            "mean_dice": 0.8295,
            "basic_ece": 0.10,
        },
    ]
    def fail_if_evaluated(*_args, **_kwargs):
        raise AssertionError("completed candidates should be reused")

    monkeypatch.setattr("ours.tune_refine._can_reuse_result", lambda *_args: True)
    monkeypatch.setattr("ours.tune_refine.run_evaluation", fail_if_evaluated)
    config = __import__("ours").__path__[0] + "/configs/brats2020.yaml"
    for mode in ("legacy_complement", "local_excess_confidence"):
        candidate = output / "source_val" / _candidate_name(2.0, 2.0, mode)
        candidate.mkdir(parents=True)
        (candidate / "summary_metrics.json").write_text(json.dumps(summary))
    run_tuning(config, str(checkpoint), str(output), run_final=False)

    selection = json.loads((output / "selection.json").read_text())
    assert selection["selection_split"] == "source_val"
    assert selection["best"]["basic_ece"] == 0.10
    assert selection["best"]["direction_mode"] == "legacy_complement"
    assert (output / "candidates.csv").is_file()


def test_reuse_signature_changes_when_checkpoint_is_replaced(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"old")
    config = {"label_transfer": {"beta": 2.0}}
    old_signature = _run_signature(config, str(checkpoint))
    result = tmp_path / "result"
    result.mkdir()
    (result / "summary_metrics.json").write_text("[]")
    (result / "tuning_input.json").write_text(json.dumps(old_signature))
    assert _can_reuse_result(result, old_signature, force=False)

    checkpoint.write_bytes(b"new-checkpoint")
    new_signature = _run_signature(config, str(checkpoint))
    assert new_signature != old_signature
    assert not _can_reuse_result(result, new_signature, force=False)


def test_one_click_flow_reports_both_final_directions_without_test_selection(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "tuning"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_evaluation(config, _checkpoint, output_dir):
        mode = config["label_transfer"]["direction_mode"]
        splits = list(config["evaluation"]["splits"])
        calls.append((mode, splits))
        result = Path(output_dir)
        result.mkdir(parents=True, exist_ok=True)
        domains = ["source_val"] if splits == ["val"] else ["source_val", "CBICA"]
        rows = []
        for domain in domains:
            rows.extend(
                [
                    {
                        "domain": domain,
                        "prediction": "base",
                        "mean_dice": 0.83,
                        "basic_ece": 0.12,
                    },
                    {
                        "domain": domain,
                        "prediction": "refined",
                        "mean_dice": 0.8295,
                        "basic_ece": (
                            0.10 if mode == "local_excess_confidence" else 0.11
                        ),
                    },
                ]
            )
        (result / "summary_metrics.json").write_text(json.dumps(rows))
        return result

    monkeypatch.setattr("ours.tune_refine.run_evaluation", fake_evaluation)
    config = __import__("ours").__path__[0] + "/configs/brats2020.yaml"
    run_tuning(config, str(checkpoint), str(output), run_final=True)

    selection = json.loads((output / "selection.json").read_text())
    assert selection["best"]["direction_mode"] == "local_excess_confidence"
    assert set(selection["final_outputs"]) == {
        "legacy_complement",
        "local_excess_confidence",
    }
    assert (output / "final_comparison.csv").is_file()
    assert (output / "final_comparison.json").is_file()
    assert calls[:2] == [
        ("legacy_complement", ["val"]),
        ("local_excess_confidence", ["val"]),
    ]
    assert all(len(splits) > 1 for _, splits in calls[2:])
