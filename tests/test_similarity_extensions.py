"""第六阶段扩展算法（DTW、余弦相似度）的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import Algorithm, CalculationStatus, SortDirection
from core.exceptions import AlgorithmError
from modules.similarity.cosine import CosineAlgorithm
from modules.similarity.dtw import MAX_CELLS, DtwAlgorithm, dtw_distance
from modules.similarity.pearson import PearsonAlgorithm


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------
def test_dtw_identical_curves_distance_is_zero() -> None:
    result = DtwAlgorithm().compare([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.is_valid
    assert result.raw_value == pytest.approx(0.0, abs=1e-12)
    assert result.algorithm is Algorithm.DTW
    assert result.direction is SortDirection.ASC


def test_dtw_supports_length_mismatch() -> None:
    result = DtwAlgorithm().compare([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert result.is_valid
    assert result.sample_size == 5
    assert "长度" not in result.message


def test_dtw_normalized_scaled_curve_is_near_zero() -> None:
    base = np.array([10.0, 11.0, 10.5, 12.0, 13.0, 12.5])
    scaled = base * 3.0
    result = DtwAlgorithm().compare(base, scaled, normalize=True)
    assert result.is_valid
    assert result.raw_value == pytest.approx(0.0, abs=1e-9)
    assert result.normalized is True


def test_dtw_drops_invalid_points_per_series() -> None:
    result = DtwAlgorithm().compare(
        [1.0, 2.0, np.nan, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], min_valid_points=4
    )
    assert result.is_valid
    assert result.sample_size == 4


def test_dtw_reports_too_large_problem_as_failed() -> None:
    with pytest.raises(AlgorithmError):
        dtw_distance(np.zeros(2100), np.zeros(2100))

    result = DtwAlgorithm().compare(np.zeros(2100), np.zeros(2100))
    assert result.status is CalculationStatus.FAILED
    assert "计算量过大" in result.message
    assert np.sqrt(MAX_CELLS) == pytest.approx(2000.0)


def test_pearson_still_rejects_length_mismatch() -> None:
    result = PearsonAlgorithm().compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert result.status is CalculationStatus.LENGTH_MISMATCH


# ---------------------------------------------------------------------------
# 余弦相似度
# ---------------------------------------------------------------------------
def test_cosine_identical_and_scaled_curves() -> None:
    algorithm = CosineAlgorithm()
    base = [1.0, 3.0, 2.0, 5.0, 4.0]
    assert algorithm.compare(base, base).raw_value == pytest.approx(1.0)
    assert algorithm.compare(base, [3.0, 9.0, 6.0, 15.0, 12.0]).raw_value == pytest.approx(1.0)


def test_cosine_orthogonal_and_reverse() -> None:
    algorithm = CosineAlgorithm()
    orthogonal = algorithm.compare([1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0])
    assert orthogonal.raw_value == pytest.approx(0.0, abs=1e-12)

    reverse = algorithm.compare([1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0])
    assert reverse.raw_value == pytest.approx(-1.0)


def test_cosine_constant_curve_is_invalid() -> None:
    result = CosineAlgorithm().compare([2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.status is CalculationStatus.CONSTANT_SERIES


def test_cosine_direction_is_desc() -> None:
    assert CosineAlgorithm().direction is SortDirection.DESC
