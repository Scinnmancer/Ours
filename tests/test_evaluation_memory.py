import numpy as np
import pytest
import torch

from ours.metrics import (
    _sample_mask_indices,
    evaluate_case,
    expected_calibration_error,
    hd95_per_region,
)
from ours.probability import atomic_to_regions


def test_hd95_matches_simple_exact_distances():
    prediction = np.zeros((3, 9, 9, 9), dtype=np.uint8)
    target = np.zeros_like(prediction)
    for channel in range(3):
        prediction[channel, 4, 4, 5] = 1
        target[channel, 4, 4, 3] = 1

    assert hd95_per_region(prediction, target) == pytest.approx([2.0, 2.0, 2.0])
    assert hd95_per_region(target, target) == pytest.approx([0.0, 0.0, 0.0])

    prediction[:] = 0
    target[:] = 0
    prediction[:, 1:6, 1:6, 1:6] = 1
    target[:, 3:8, 2:7, 2:7] = 1
    assert hd95_per_region(prediction, target) == pytest.approx([np.sqrt(5.0)] * 3)


def test_hd95_crops_distance_transform_to_region_union(monkeypatch):
    from scipy import ndimage

    prediction = np.zeros((3, 64, 64, 64), dtype=np.uint8)
    target = np.zeros_like(prediction)
    prediction[:, 30:33, 30:33, 30:33] = 1
    target[:, 31:34, 30:33, 30:33] = 1
    shapes = []
    original = ndimage.distance_transform_edt

    def recording_distance_transform(value):
        shapes.append(value.shape)
        return original(value)

    monkeypatch.setattr(ndimage, "distance_transform_edt", recording_distance_transform)
    hd95_per_region(prediction, target)

    assert shapes == [(4, 3, 3)] * 6


def test_hd95_is_numerically_equivalent_to_monai():
    from monai.metrics.hausdorff_distance import compute_hausdorff_distance

    generator = np.random.default_rng(12)
    prediction = (generator.random((3, 20, 21, 22)) > 0.96).astype(np.uint8)
    target = (generator.random((3, 20, 21, 22)) > 0.96).astype(np.uint8)
    expected = compute_hausdorff_distance(
        torch.as_tensor(prediction[None]),
        torch.as_tensor(target[None]),
        include_background=True,
        percentile=95,
    )[0].numpy()

    np.testing.assert_allclose(hd95_per_region(prediction, target), expected, rtol=0, atol=1e-6)


def test_metric_index_sampling_is_bounded_and_deterministic():
    mask = torch.zeros(2_500_000, dtype=torch.bool)
    mask[::2] = True
    first = _sample_mask_indices(mask, 2_000)
    second = _sample_mask_indices(mask, 2_000)

    assert first.numel() == 2_000
    assert torch.equal(first, second)
    assert bool(mask[first].all())
    assert int(first[0]) == 0
    assert int(first[-1]) == 2_499_998


def test_precomputed_compact_predictions_preserve_metrics():
    generator = torch.Generator().manual_seed(7)
    probability = torch.rand((1, 4, 8, 8, 8), generator=generator)
    probability /= probability.sum(dim=1, keepdim=True)
    uncertainty = torch.rand((1, 1, 8, 8, 8), generator=generator)
    atomic_target = torch.randint(0, 4, (1, 1, 8, 8, 8), generator=generator)
    scalar_target = atomic_target.clone()
    scalar_target[scalar_target == 3] = 4
    region_probability = atomic_to_regions(probability)

    legacy = evaluate_case(
        probability,
        uncertainty,
        scalar_target,
        region_probability=region_probability,
        risk_reference_probability=probability,
        max_voxels=10_000,
    )
    compact = evaluate_case(
        probability,
        uncertainty,
        scalar_target,
        atomic_prediction=probability.argmax(dim=1)[0].to(torch.uint8),
        region_prediction=(region_probability[0] >= 0.5).to(torch.uint8),
        risk_reference_prediction=probability.argmax(dim=1)[0].to(torch.uint8),
        max_voxels=10_000,
    )

    assert compact.keys() == legacy.keys()
    for key in compact:
        assert compact[key] == pytest.approx(legacy[key], nan_ok=True)


def test_exact_aggregate_metrics_are_not_limited_by_ranking_sample_size():
    probability = torch.tensor(
        [
            [
                [[[0.90, 0.10, 0.10, 0.60]]],
                [[[0.05, 0.70, 0.10, 0.10]]],
                [[[0.03, 0.10, 0.70, 0.10]]],
                [[[0.02, 0.10, 0.10, 0.20]]],
            ]
        ],
        dtype=torch.float32,
    )
    uncertainty = torch.tensor([[[[[0.05, 0.20, 0.80, 0.90]]]]])
    scalar_target = torch.tensor([[[[[0, 1, 2, 4]]]]], dtype=torch.uint8)
    result = evaluate_case(probability, uncertainty, scalar_target, max_voxels=1)

    target = torch.tensor([0, 1, 2, 3])
    one_hot = torch.nn.functional.one_hot(target, num_classes=4).movedim(-1, 0).float()
    prediction = probability.argmax(dim=1).reshape(-1)
    mask = (target > 0) | (prediction > 0)
    expected_brier = ((probability.reshape(4, -1) - one_hot) ** 2).sum(dim=0)[mask].mean()
    expected_risk_brier = (
        (uncertainty.reshape(-1) - (prediction != target).float()) ** 2
    )[mask].mean()

    assert result["brier"] == pytest.approx(float(expected_brier))
    assert result["risk_brier"] == pytest.approx(float(expected_risk_brier))


def test_evaluate_case_reports_basic_ece_separately_from_risk_ece():
    probability = torch.tensor(
        [
            [
                [[[0.80, 0.10, 0.20, 0.10]]],
                [[[0.10, 0.70, 0.10, 0.10]]],
                [[[0.05, 0.10, 0.60, 0.20]]],
                [[[0.05, 0.10, 0.10, 0.60]]],
            ]
        ],
        dtype=torch.float32,
    )
    uncertainty = torch.tensor([[[[[0.10, 0.20, 0.30, 0.40]]]]])
    scalar_target = torch.tensor([[[[[0, 1, 2, 4]]]]], dtype=torch.uint8)

    result = evaluate_case(probability, uncertainty, scalar_target, max_voxels=100)
    confidence = np.asarray([0.70, 0.60, 0.60], dtype=np.float32)
    correct = np.ones(3, dtype=np.float32)
    expected = expected_calibration_error(confidence, correct, bins=15)

    assert result["basic_ece"] == pytest.approx(expected)
    assert result["ece"] == pytest.approx(result["basic_ece"])
    assert "risk_ece" in result
    assert result["basic_ece"] != pytest.approx(result["risk_ece"])
