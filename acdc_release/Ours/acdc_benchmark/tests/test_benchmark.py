import csv
import json

import numpy as np
import pytest

from acdc_benchmark.evaluate import METHODS, PROTOCOL, SPLIT_SHA256, score_frame, write_csv, write_json
from acdc_benchmark.metrics import (ECE_SOURCE_SHA256, FixedECEAccumulator, error_auroc, risk_coverage_area,
                                    score, uncertainty_error_overlap)
from acdc_benchmark.table import build_table


def fixture_volume():
    labels = np.array([[[0, 1, 2, 3, 0]]], dtype=np.uint8)
    valid = np.array([[[True, True, True, True, False]]])
    reference = np.array([[[True, False, True, False, True]]])
    p = np.array([[[[.95, .1, .95, .1, .1]]],
                  [[[.02, .7, .02, .1, .7]]],
                  [[[.02, .1, .02, .1, .1]]],
                  [[[.01, .1, .01, .7, .1]]]], dtype=np.float32)
    return p, labels, valid, reference


def test_today_ece_on_fixed_support():
    p, labels, valid, reference = fixture_volume()
    expected = FixedECEAccumulator()
    expected.add(p, labels, np.array([[[True, True, True, True, False]]]))
    result = score(p, labels, valid, reference)
    for key in ("ECE", "fixed_foreground_NLL", "fixed_foreground_Brier",
                "high_confidence_error_fraction", "ece_voxels"):
        assert result[key] == expected.result()[key]
    assert result["dice_RV"] == 1.0


def test_aurc_ties_use_one_coverage_threshold():
    probabilities = np.array([[.9, .8, .8, .99], [.1, .2, .2, .01],
                              [0., 0., 0., 0.], [0., 0., 0., 0.]])
    labels = np.array([0, 0, 1, 1])
    support = np.array([True, True, True, False])
    expected = (2 / 3) * (1 / 3)
    assert risk_coverage_area(probabilities, labels, support) == pytest.approx(expected)
    permutation = [0, 2, 1, 3]
    assert risk_coverage_area(probabilities[:, permutation], labels[permutation],
                              support[permutation]) == pytest.approx(expected)


def test_ueo_scores_only_fixed_support_and_includes_threshold_ties():
    probabilities = np.array([[.875, .75, .75, .5], [.125, .25, .25, .5],
                              [0., 0., 0., 0.], [0., 0., 0., 0.]])
    labels = np.array([0, 0, 1, 1])
    support = np.array([True, True, True, False])
    thresholds = np.array([0., .25, .5])
    actual = uncertainty_error_overlap(probabilities, labels, support, thresholds)
    assert actual == pytest.approx([.5, 2 / 3, 0])


def test_error_auroc_counts_ties_and_excludes_degenerate_frames():
    probabilities = np.array([[.875, .75, .75, .5], [.125, .25, .25, .5],
                              [0., 0., 0., 0.], [0., 0., 0., 0.]])
    labels = np.array([0, 0, 1, 1])
    support = np.array([True, True, True, False])
    assert error_auroc(probabilities, labels, support) == pytest.approx(.75)
    assert error_auroc(probabilities, np.zeros_like(labels), support) is None
    assert error_auroc(probabilities, np.ones_like(labels), support) is None


def test_complete_table_rejects_reference_mismatch(tmp_path):
    p, labels, valid, _ = fixture_volume()
    reference = tmp_path / "baseline"
    (reference / "support").mkdir(parents=True)
    protocol = {"protocol": PROTOCOL, "split": "test", "ece_source_sha256": ECE_SOURCE_SHA256,
                "split_sha256": SPLIT_SHA256}
    baseline = {"method": "baseline", "comparison": protocol, "checkpoint_sha256": "reference",
                "reference_checkpoint_sha256": "reference", "frames": {}}
    records = [{"subject_id": f"p{i}_{phase}", "patient_id": f"p{i}", "frame": phase,
                "domain": "target_3T"} for i in range(33) for phase in ("ED", "ES")]
    for method in METHODS:
        directory = tmp_path / method
        directory.mkdir(exist_ok=True)
        rows = []
        for record in records:
            row, identity = score_frame(p, labels, valid, record, directory, reference,
                                        baseline if method != "baseline" else None)
            rows.append(row)
            if method == "baseline":
                baseline["frames"][row["case"]] = identity
        write_csv(directory / "frame_metrics.csv", rows)
        write_json(directory / "protocol.json", {**baseline, "method": method})
    assert len(build_table(tmp_path)) == 8
    assert build_table(tmp_path)[-1]["n_patients"] == 33

    scores = tmp_path / "mbls/frame_metrics.csv"
    original = scores.read_text()
    with scores.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["ECE"] = str(100 * float(rows[0]["ECE"]))
    write_csv(scores, rows)
    with pytest.raises(ValueError, match="expected matching raw fixed ECE"):
        build_table(tmp_path)
    scores.write_text(original)

    method_protocol = tmp_path / "mbls/protocol.json"
    data = json.loads(method_protocol.read_text())
    data["frames"]["p0_ED"]["ece_support_sha256"] = "mismatch"
    write_json(method_protocol, data)
    with pytest.raises(ValueError, match="frame identities differ"):
        build_table(tmp_path)
