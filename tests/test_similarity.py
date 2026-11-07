"""相似度模块与相似度/检索界面测试（设计文档 3.4、3.3.4）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QDate

from core.constants import CalculationStatus, SortDirection
from core.data_models import MarketDataset
from core.exceptions import UnknownAlgorithmError
from modules.similarity.service import SimilarityService
from modules.trend_analysis.trend_service import TrendService
from ui.similarity_page import SimilarityPage
from ui.trend_page import TrendPage


def _series(values: list[float]) -> pd.Series:
    return pd.Series(np.array(values, dtype=float))


@pytest.fixture()
def service() -> SimilarityService:
    return SimilarityService()


# ---------------------------------------------------------------------------
# 算法
# ---------------------------------------------------------------------------
def test_pearson_perfect_and_inverse(service: SimilarityService) -> None:
    up = _series([1, 2, 3, 4, 5])
    result = service.compare(up, up * 3, "pearson")
    assert result.is_valid
    assert result.raw_value == pytest.approx(1.0)
    assert result.direction is SortDirection.DESC
    assert result.sample_size == 5

    inverse = service.compare(up, up * -2 + 10, "pearson")
    assert inverse.raw_value == pytest.approx(-1.0)


def test_pearson_constant_series_is_invalid(service: SimilarityService) -> None:
    result = service.compare(_series([5, 5, 5, 5, 5]), _series([1, 2, 3, 4, 5]), "pearson")
    assert not result.is_valid
    assert result.status is CalculationStatus.CONSTANT_SERIES
    assert "常数" in result.message


def test_length_mismatch_is_reported(service: SimilarityService) -> None:
    result = service.compare(_series([1, 2, 3, 4]), _series([1, 2, 3, 4, 5]), "pearson")
    assert not result.is_valid
    assert result.status is CalculationStatus.LENGTH_MISMATCH
    assert "长度不一致" in result.message


def test_insufficient_valid_points(service: SimilarityService) -> None:
    left = _series([1, np.nan, np.nan, 4, np.nan])
    right = _series([1, 2, 3, 4, 5])
    result = service.compare(left, right, "pearson")
    assert not result.is_valid
    assert result.status is CalculationStatus.INSUFFICIENT_DATA
    assert result.sample_size == 2
    assert "有效数据点不足" in result.message


def test_nan_values_are_masked_pairwise(service: SimilarityService) -> None:
    left = _series([1, 2, np.nan, 4, 5, 6])
    right = _series([2, 4, 6, 8, 10, 12])
    result = service.compare(left, right, "pearson")
    assert result.is_valid
    assert result.sample_size == 5
    assert result.raw_value == pytest.approx(1.0)


def test_rmse_and_mae_values(service: SimilarityService) -> None:
    left = _series([1, 2, 3, 4, 5])
    right = _series([2, 4, 6, 8, 10])
    rmse = service.compare(left, right, "rmse", normalize=False)
    mae = service.compare(left, right, "mae", normalize=False)
    assert rmse.raw_value == pytest.approx(np.sqrt((1 + 4 + 9 + 16 + 25) / 5))
    assert mae.raw_value == pytest.approx(3.0)
    assert rmse.direction is SortDirection.ASC
    assert not rmse.normalized


def test_error_metrics_normalize_before_compare(service: SimilarityService) -> None:
    base = _series([100, 110, 90, 120, 105])
    scaled = base * 2
    normalized = service.compare(base, scaled, "rmse")
    raw = service.compare(base, scaled, "rmse", normalize=False)
    assert normalized.normalized is True
    assert normalized.raw_value == pytest.approx(0.0, abs=1e-9)
    assert raw.raw_value is not None and raw.raw_value > 50


def test_pearson_ignores_normalize_flag(service: SimilarityService) -> None:
    result = service.compare(_series([1, 2, 3, 4, 5]), _series([5, 4, 3, 2, 1]), "pearson", normalize=True)
    assert result.is_valid
    assert result.normalized is False


def test_compare_all_returns_every_algorithm(service: SimilarityService) -> None:
    results = service.compare_all(_series([1, 2, 3, 4, 5]), _series([2, 4, 6, 8, 10]))
    assert [item.algorithm.value for item in results] == ["pearson", "rmse", "mae", "dtw", "cosine"]


def test_unknown_algorithm_raises(service: SimilarityService) -> None:
    with pytest.raises(UnknownAlgorithmError):
        service.compare(_series([1, 2, 3]), _series([1, 2, 3]), "euclidean")


def test_available_algorithms_and_labels(service: SimilarityService) -> None:
    assert service.available_algorithms() == ("pearson", "rmse", "mae", "dtw", "cosine")
    assert service.default_algorithm == "pearson"
    assert service.label_of("rmse") == "RMSE 均方根误差"
    assert service.direction_of("mae") is SortDirection.ASC
    assert service.direction_of("dtw") is SortDirection.ASC
    assert service.direction_of("cosine") is SortDirection.DESC


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
def _dataset(periods: int = 150) -> MarketDataset:
    rng = np.random.default_rng(20240101)
    index = pd.bdate_range("2023-01-02", periods=periods)
    close = 100 + np.cumsum(rng.normal(0.2, 1.2, periods))
    frame = pd.DataFrame({"open": close - 0.3, "close": close}, index=index)
    return MarketDataset(name="测试指数", frame=frame)


def test_similarity_page_compares_two_curves(qapp) -> None:
    page = SimilarityPage(SimilarityService(), lambda: (_dataset(),), None)
    page.refresh()
    page.compare_current()

    assert len(page.results) == 5
    assert all(item.is_valid for item in page.results)
    assert page.results[0].raw_value == pytest.approx(1.0)
    assert page.table_results.rowCount() == 5
    assert len(page.figure.axes[0].get_lines()) == 2


def test_similarity_page_truncates_and_reports(qapp) -> None:
    dataset = _dataset()
    page = SimilarityPage(SimilarityService(), lambda: (dataset,), None)
    page.refresh()
    page.box_b.date_start.setDate(QDate(2023, 2, 1))
    page.compare_current()

    assert "截断" in page.label_status.text()
    assert all(item.is_valid for item in page.results)

    page.check_truncate.setChecked(False)
    page.compare_current()
    assert page.results[0].status is CalculationStatus.LENGTH_MISMATCH
    assert "未做任何截断" in page.label_status.text()


def test_trend_page_search_and_overlay(qapp) -> None:
    dataset = _dataset()
    similarity = SimilarityService()
    trend = TrendService(None, lambda: similarity)
    page = TrendPage(trend, lambda: (dataset,), None, lambda: similarity)
    page.date_start.setDate(QDate(2023, 2, 1))
    page.date_end.setDate(QDate(2023, 3, 1))
    page.search_current()

    assert page.table_results.rowCount() == 3
    assert len(page.search_results) == 3
    assert all(item.overlap_ratio <= 0.5 for item in page.search_results)
    assert page.search_results[0].rank == 1
    assert len(page.figure.axes[0].get_lines()) == 2
