import numpy as np
import pytest

from ours.metrics import MetricSampleReservoir


def _fill(reservoir: MetricSampleReservoir) -> None:
    for batch_index in range(25):
        voxel = np.arange(10_000, dtype=np.float32) + batch_index * 10_000
        reservoir.update(voxel=voxel, paired=voxel + 1)


def test_metric_sample_reservoir_stays_bounded_aligned_and_deterministic():
    first = MetricSampleReservoir(1_000, seed=7)
    second = MetricSampleReservoir(1_000, seed=7)
    _fill(first)
    _fill(second)

    assert len(first) == 1_000
    np.testing.assert_array_equal(first.values("paired"), first.values("voxel") + 1)
    np.testing.assert_array_equal(first.values("voxel"), second.values("voxel"))


def test_metric_sample_reservoir_rejects_misaligned_fields():
    reservoir = MetricSampleReservoir(10)
    with pytest.raises(ValueError, match="same size"):
        reservoir.update(first=np.arange(3), second=np.arange(4))

    reservoir.update(first=np.arange(3), second=np.arange(3))
    with pytest.raises(ValueError, match="remain unchanged"):
        reservoir.update(first=np.arange(3))
