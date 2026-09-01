from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from ours.refine_audit import (
    audit_refinement,
    probability_snapshot,
    save_audit_nifti,
    summarize_audits,
)


def _probability(labels: list[int], confidence: list[float]) -> torch.Tensor:
    result = torch.empty(1, 4, 1, 1, len(labels), dtype=torch.float32)
    for index, (label, maximum) in enumerate(zip(labels, confidence)):
        result[0, :, 0, 0, index] = (1.0 - maximum) / 3.0
        result[0, label, 0, 0, index] = maximum
    return result


def _audit():
    base = probability_snapshot(
        _probability([1, 1, 2, 1, 2], [0.9, 0.9, 0.9, 0.9, 0.9])
    )
    refined = probability_snapshot(
        _probability([1, 0, 3, 2, 2], [0.8, 0.8, 0.95, 0.8, 0.9])
    )
    target = torch.tensor([[[[[1, 0, 2, 4, 1]]]]], dtype=torch.uint8)
    return audit_refinement(base, refined, target)


def test_refine_audit_classifies_atomic_changes_and_rates():
    audit = _audit()
    metrics = audit.metrics

    assert audit.atomic_change_map.flatten().tolist() == [1, 2, 3, 4, 1]
    assert metrics["roi_voxels"] == 5
    assert metrics["atomic_changed_voxels"] == 3
    assert metrics["atomic_corrected_voxels"] == 1
    assert metrics["atomic_corrupted_voxels"] == 1
    assert metrics["atomic_wrong_to_wrong_voxels"] == 1
    assert metrics["atomic_net_corrected_voxels"] == 0
    assert metrics["atomic_top1_preservation_rate"] == pytest.approx(2 / 5)
    assert metrics["atomic_correction_precision"] == pytest.approx(1 / 3)
    assert metrics["atomic_error_repair_rate"] == pytest.approx(1 / 3)
    assert metrics["atomic_error_change_coverage"] == pytest.approx(2 / 3)
    assert metrics["atomic_error_introduction_rate"] == pytest.approx(1 / 2)
    assert metrics["base_error_confidence_decrease_rate"] == pytest.approx(2 / 3)
    assert metrics["base_correct_confidence_decrease_rate"] == pytest.approx(1 / 2)
    assert metrics["base_correct_confidence_increase_rate"] == pytest.approx(1 / 2)


def test_refine_audit_records_atomic_and_region_transitions():
    audit = _audit()

    assert {
        (row["target_class"], row["base_class"], row["refined_class"], row["count"])
        for row in audit.atomic_transitions
    } == {
        (0, 1, 0, 1),
        (1, 1, 1, 1),
        (1, 2, 2, 1),
        (2, 2, 3, 1),
        (3, 1, 2, 1),
    }
    assert set(audit.region_change_maps) == {"TC", "WT", "ET"}
    assert any(row["region"] == "ET" and row["count"] > 0 for row in audit.region_transitions)
    for region in ("TC", "WT", "ET"):
        assert f"{region}_base_tp" in audit.metrics
        assert f"{region}_refined_fp" in audit.metrics
        assert f"{region}_net_corrected_voxels" in audit.metrics


def test_refine_audit_uses_null_for_undefined_ratios():
    probability = _probability([1], [0.9])
    target = torch.tensor([[[[[1]]]]], dtype=torch.uint8)
    audit = audit_refinement(
        probability_snapshot(probability), probability_snapshot(probability), target
    )

    assert audit.metrics["atomic_changed_voxels"] == 0
    assert audit.metrics["atomic_correction_precision"] is None
    assert audit.metrics["atomic_error_repair_rate"] is None
    assert audit.metrics["atomic_error_introduction_rate"] == pytest.approx(0.0)


def test_refine_audit_summary_sums_counts_and_averages_defined_rates():
    metrics = _audit().metrics
    rows = [
        {"domain": "A", "case": "one", **metrics},
        {"domain": "A", "case": "two", **metrics},
    ]
    summary = {row["domain"]: row for row in summarize_audits(rows)}

    assert summary["A"]["n_cases"] == 2
    assert summary["A"]["roi_voxels"] == 10
    assert summary["A"]["atomic_corrected_voxels"] == 2
    assert summary["A"]["pooled_atomic_correction_precision"] == pytest.approx(1 / 3)
    assert summary["A"]["case_macro_atomic_error_repair_rate"] == pytest.approx(1 / 3)
    assert summary["overall"]["n_cases"] == 2


def test_refine_audit_saves_spatial_nifti_maps(tmp_path: Path):
    audit = _audit()
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    save_audit_nifti(audit, affine, tmp_path, "case")

    expected = [
        "case__atomic_change_type.nii.gz",
        "case__confidence_delta.nii.gz",
        "case__TC_change_type.nii.gz",
        "case__WT_change_type.nii.gz",
        "case__ET_change_type.nii.gz",
    ]
    for name in expected:
        assert (tmp_path / name).is_file()
    change = nib.load(tmp_path / expected[0])
    confidence = nib.load(tmp_path / expected[1])
    np.testing.assert_allclose(change.affine, affine)
    np.testing.assert_array_equal(np.asarray(change.dataobj).reshape(-1), [1, 2, 3, 4, 1])
    np.testing.assert_allclose(
        np.asarray(confidence.dataobj).reshape(-1), [-0.1, -0.1, 0.05, -0.1, 0.0], atol=1e-6
    )
