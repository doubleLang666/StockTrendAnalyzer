"""第四阶段：趋势分析模块测试（绘图 / 多周期比较 / 历史检索 / 页面）。"""

from __future__ import annotations

from datetime import date

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from core.constants import Alignment, Normalization
from core.data_models import (
    MarketDataset,
    ProcessedDataset,
    SimilarityConfig,
    SimilarityResult,
)
from core.exceptions import (
    InsufficientDataError,
    ProcessingError,
    ServiceNotRegisteredError,
)
from modules.data_processing.processor import DataProcessor
from modules.trend_analysis import (
    CurveData,
    PlotRequest,
    Segment,
    TrendService,
    build_figure,
    compare_periods,
    find_similar_periods,
    series_from_source,
)


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
def make_series(length: int = 120, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=length)
    values = 20 + np.cumsum(rng.normal(0, 0.3, length))
    return pd.Series(values, index=index, name="close")


def pearson_metric(left: pd.Series, right: pd.Series) -> SimilarityResult:
    value = float(np.corrcoef(left.to_numpy(), right.to_numpy())[0, 1])
    return SimilarityResult(
        algorithm="pearson",
        raw_value=value,
        display_value=value,
        sample_size=len(left),
    )


class StubSimilarityService:
    """只在测试中使用的最小相似度实现。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def available_algorithms(self) -> tuple[str, ...]:
        return ("pearson",)

    def compare(self, series_a, series_b, algorithm: str = "pearson", **params):
        self.calls += 1
        return pearson_metric(series_a, series_b)


def make_dataset(series: pd.Series, name: str = "测试指数") -> MarketDataset:
    frame = pd.DataFrame(
        {
            "open": series.to_numpy() - 0.1,
            "high": series.to_numpy() + 0.2,
            "low": series.to_numpy() - 0.3,
            "close": series.to_numpy(),
            "volume": np.linspace(1000, 2000, len(series)),
        },
        index=series.index,
    )
    return MarketDataset(name=name, frame=frame)


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def test_build_figure_draws_curves_and_labels():
    first = make_series(30)
    second = make_series(30, seed=9)
    request = PlotRequest(
        curves=(CurveData("甲", first), CurveData("乙", second)),
        title="示例图",
        x_label="交易日期",
        y_label="收盘价",
    )
    figure = build_figure(request)
    axes = figure.axes[0]
    assert len(axes.get_lines()) == 2
    assert axes.get_title() == "示例图"
    assert axes.get_xlabel() == "交易日期"
    assert axes.get_ylabel() == "收盘价"
    assert axes.get_legend() is not None


def test_build_figure_reuses_given_figure():
    figure = build_figure(PlotRequest(curves=(CurveData("甲", make_series(10)),)))
    again = build_figure(PlotRequest(curves=(CurveData("乙", make_series(10)),)), figure=figure)
    assert again is figure
    assert len(again.axes) == 1
    assert [line.get_label() for line in again.axes[0].get_lines()] == ["乙"]


def test_build_figure_requires_curves():
    with pytest.raises(ProcessingError):
        build_figure(PlotRequest(curves=()))


# ----------------------------------------------------------------------
# 多周期比较
# ----------------------------------------------------------------------
def test_compare_periods_truncates_with_explicit_note():
    series = make_series(60)
    frame = series.to_frame("close")
    segments = [
        Segment(date(2023, 1, 2), date(2023, 3, 1)),   # 长
        Segment(date(2023, 3, 2), date(2023, 3, 20)),  # 短
    ]
    comparison = compare_periods(
        frame,
        "close",
        segments,
        normalization=Normalization.NONE,
        alignment=Alignment.REAL_DATE,
        truncate=True,
    )
    lengths = {item.trading_days for item in comparison.segments}
    assert len(lengths) == 1
    note = " ".join(comparison.notes)
    assert "最短" in note and "截断" in note
    assert "已统一按最短的" in note


def test_compare_periods_keeps_lengths_when_truncate_disabled():
    series = make_series(60)
    frame = series.to_frame("close")
    segments = [
        Segment(date(2023, 1, 2), date(2023, 3, 1)),
        Segment(date(2023, 3, 2), date(2023, 3, 20)),
    ]
    comparison = compare_periods(
        frame,
        "close",
        segments,
        alignment=Alignment.REAL_DATE,
        truncate=False,
    )
    assert len({item.trading_days for item in comparison.segments}) == 2
    assert any("按原长度绘制" in note for note in comparison.notes)


def test_compare_periods_relative_alignment_and_start_normalization():
    series = make_series(40)
    frame = series.to_frame("close")
    comparison = compare_periods(
        frame,
        "close",
        [Segment(date(2023, 1, 2), date(2023, 2, 10))],
        normalization=Normalization.START,
        alignment=Alignment.RELATIVE_DAY,
    )
    values = comparison.segments[0].values
    assert list(values.index[:3]) == [0, 1, 2]
    assert values.iloc[0] == pytest.approx(100.0)
    assert comparison.alignment is Alignment.RELATIVE_DAY


def test_compare_periods_min_max_and_skipped_segment():
    series = make_series(40)
    frame = series.to_frame("close")
    comparison = compare_periods(
        frame,
        "close",
        [
            Segment(date(2023, 1, 2), date(2023, 1, 20)),
            Segment(date(2019, 1, 1), date(2019, 2, 1)),
        ],
        normalization=Normalization.MIN_MAX,
        alignment=Alignment.REAL_DATE,
    )
    assert len(comparison.segments) == 1
    values = comparison.segments[0].values.dropna()
    assert values.min() >= -1e-9 and values.max() <= 1 + 1e-9
    assert any("没有数据，已跳过" in note for note in comparison.notes)


def test_compare_periods_errors():
    frame = make_series(20).to_frame("close")
    with pytest.raises(ProcessingError):
        compare_periods(frame, "open", [Segment(date(2023, 1, 2), date(2023, 1, 10))])
    with pytest.raises(InsufficientDataError):
        compare_periods(frame, "close", [Segment(date(2015, 1, 1), date(2015, 2, 1))])
    with pytest.raises(ProcessingError):
        compare_periods(frame, "close", [])


# ----------------------------------------------------------------------
# 历史检索
# ----------------------------------------------------------------------
def test_find_similar_periods_self_excluded_and_ranked():
    series = make_series(150)
    length = 20
    target_start = 10
    series.iloc[100:120] = series.iloc[target_start : target_start + length].to_numpy()
    results = find_similar_periods(
        series,
        series.index[target_start].date(),
        series.index[target_start + length - 1].date(),
        metric=pearson_metric,
        field="close",
        top_n=3,
    )
    assert len(results) == 3
    assert [item.rank for item in results] == [1, 2, 3]
    assert all(item.overlap_ratio <= 0.5 + 1e-9 for item in results)
    best = results[0]
    assert best.start_date == series.index[100].date()
    assert best.end_date == series.index[119].date()
    assert best.trading_days == length
    assert best.result.raw_value == pytest.approx(1.0)


def test_find_similar_periods_skips_sparse_windows():
    index = pd.bdate_range("2023-01-02", periods=80)
    values = np.full(80, np.nan)
    values[40:60] = make_series(20, seed=3).to_numpy()
    values[0:40:2] = 100.0  # 目标窗口隔一天才有值，有效点必然不足
    series = pd.Series(values, index=index)
    results = find_similar_periods(
        series,
        index[0].date(),
        index[19].date(),
        metric=pearson_metric,
        field="close",
        min_valid_points=15,
    )
    assert results == ()


def test_find_similar_periods_requires_datetime_index():
    series = pd.Series(np.arange(30, dtype="float64"))
    with pytest.raises(ProcessingError):
        find_similar_periods(series, date(2023, 1, 2), date(2023, 1, 10), metric=pearson_metric, field="close")


# ----------------------------------------------------------------------
# TrendService
# ----------------------------------------------------------------------
def test_trend_service_plot_from_datasets():
    dataset = make_dataset(make_series(40))
    service = TrendService()
    figure = service.plot([dataset])
    assert len(figure.axes[0].get_lines()) == 5


def test_series_from_source_applies_normalization_and_alignment():
    dataset = make_dataset(make_series(30))
    series, notes = series_from_source(
        dataset,
        "close",
        normalization=Normalization.START,
        alignment=Alignment.RELATIVE_DAY,
    )
    assert series.iloc[0] == pytest.approx(100.0)
    assert list(series.index[:2]) == [0, 1]


def test_series_from_source_processed_dataset_keeps_result():
    dataset = make_dataset(make_series(30))
    processed = DataProcessor().process(dataset, _processing_config())
    series, _notes = series_from_source(processed, "close", normalization=Normalization.START)
    assert series.equals(processed.frame["close"])


def test_trend_service_search_uses_similarity_service():
    stub = StubSimilarityService()
    service = TrendService(similarity_resolver=lambda: stub)
    dataset = make_dataset(make_series(120))
    target = Segment(dataset.frame.index[10].date(), dataset.frame.index[29].date())
    config = SimilarityConfig(algorithm="pearson", top_n=2, params={"field": "close"})
    results = service.search_similar_periods(target, dataset, config)
    assert len(results) == 2
    assert stub.calls > 0
    assert all(item.field == "close" for item in results)


def test_trend_service_search_without_similarity_raises():
    service = TrendService()
    dataset = make_dataset(make_series(60))
    target = Segment(dataset.frame.index[0].date(), dataset.frame.index[9].date())
    with pytest.raises(ServiceNotRegisteredError):
        service.search_similar_periods(target, dataset, SimilarityConfig(algorithm="pearson"))


def _processing_config():
    from core.data_models import ProcessingConfig

    dataset = make_dataset(make_series(30))
    return ProcessingConfig(
        fields=("close",),
        start_date=dataset.frame.index[0].date(),
        end_date=dataset.frame.index[-1].date(),
        normalization=Normalization.START,
    )


# ----------------------------------------------------------------------
# 页面
# ----------------------------------------------------------------------
def test_trend_page_draws_curve(qapp, sample_frame):
    from PySide6.QtCore import Qt

    from ui.trend_page import TrendPage

    dataset = MarketDataset(name="页面测试", frame=sample_frame)
    page = TrendPage(TrendService(), lambda: (dataset,), lambda: ())
    page.refresh()
    assert page.list_fields.count() == len(dataset.frame.columns)

    for row in range(1, page.list_fields.count()):
        page.list_fields.item(row).setCheckState(Qt.CheckState.Unchecked)
    page.plot_current()

    assert len(page.figure.axes) == 1
    assert len(page.figure.axes[0].get_lines()) == 1
    assert "已绘制 1 条曲线" in page.label_status.text()


def test_trend_page_comparison_adds_ranges(qapp, sample_frame):
    from ui.trend_page import TrendPage

    dataset = MarketDataset(name="页面测试", frame=sample_frame)
    page = TrendPage(TrendService(), lambda: (dataset,), lambda: ())
    page.refresh()
    page.add_current_range()
    assert len(page.segments) == 1
    assert page.table_ranges.rowCount() == 1
    page.plot_comparison()
    assert len(page.figure.axes[0].get_lines()) == 1
    assert "交易日" in page.label_status.text()
