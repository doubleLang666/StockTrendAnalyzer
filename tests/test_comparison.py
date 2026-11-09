"""阶段四：历史区间对比（模式 A / B）与相似区间定位高亮。"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.data_models import MarketDataset
from core.exceptions import InsufficientDataError
from modules.charting import (
    CandleChartController,
    ChartConfig,
    build_candlestick_figure,
    build_dual_candle_figure,
    build_normalized_figure,
    normalize_to_100,
    prepare_candle_data,
)
from modules.trend_analysis.segments import Segment


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def make_frame(length: int = 120, *, with_volume: bool = True) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=length, name="date")
    steps = np.arange(length, dtype="float64")
    close = 10.0 + np.sin(steps / 5.0) * 2.0 + steps * 0.05
    data = {
        "open": close - 0.15,
        "high": close + 0.4,
        "low": close - 0.5,
        "close": close,
    }
    if with_volume:
        data["volume"] = 1000.0 + steps * 13.0
    return pd.DataFrame(data, index=index)


def make_candle(length: int = 120, name: str = "样本"):
    return prepare_candle_data(make_frame(length), name)


# ----------------------------------------------------------------------
# 归一化
# ----------------------------------------------------------------------
def test_normalize_to_100_uses_first_finite_value():
    values = np.array([50.0, 75.0, 25.0])
    assert normalize_to_100(values).tolist() == [100.0, 150.0, 50.0]


def test_normalize_to_100_skips_leading_nan():
    values = np.array([np.nan, 20.0, 10.0])
    result = normalize_to_100(values)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(100.0)
    assert result[2] == pytest.approx(50.0)


def test_normalize_to_100_handles_zero_base_and_all_nan():
    assert np.isnan(normalize_to_100(np.array([0.0, 5.0]))).all()
    assert np.isnan(normalize_to_100(np.array([np.nan, np.nan]))).all()


# ----------------------------------------------------------------------
# 模式 A：双图并排
# ----------------------------------------------------------------------
def test_dual_figure_builds_two_columns_with_volume():
    left = make_candle(60, "区间一")
    right = make_candle(80, "区间二")
    result = build_dual_candle_figure(left, right, ChartConfig(ma_windows=(5,)))

    assert result.bars == 60  # 按较短区间截断
    assert len(result.candles[0]) == len(result.candles[1]) == 60
    assert len(result.figure.axes) == 4  # 两列各一张价格图 + 成交量
    assert result.volumes and all(ax is not None for ax in result.volumes)
    # 两列横轴范围一致（等宽）
    assert result.prices[0].get_xlim() == result.prices[1].get_xlim()
    # 图例里能看到各自的名称
    assert result.prices[0].get_title() == "区间一"
    assert "区间二" in result.prices[1].get_title()


def test_dual_figure_hides_volume_when_missing():
    left = prepare_candle_data(make_frame(30, with_volume=False), "无量一")
    right = prepare_candle_data(make_frame(30, with_volume=False), "无量二")
    result = build_dual_candle_figure(left, right, ChartConfig(ma_windows=(5,)))

    assert len(result.figure.axes) == 2
    assert result.volumes == ()
    assert result.volume_artists == (None, None)
    # 没有成交量时，价格图自己显示日期刻度
    labels = [tick.get_text() for tick in result.prices[0].get_xticklabels()]
    assert any(label for label in labels)


def test_dual_figure_requires_two_bars():
    left = make_candle(1, "一根")
    right = make_candle(1, "一根")
    with pytest.raises(InsufficientDataError):
        build_dual_candle_figure(left, right, ChartConfig())


# ----------------------------------------------------------------------
# 模式 B：归一化对比
# ----------------------------------------------------------------------
def test_normalized_figure_one_row_per_field():
    frame = make_frame(80)
    panels = [
        ("收盘价", [("区间一", frame["close"].iloc[:40]), ("区间二", frame["close"].iloc[40:])]),
        ("成交量", [("区间一", frame["volume"].iloc[:40]), ("区间二", frame["volume"].iloc[40:])]),
    ]
    figure = build_normalized_figure(panels, ChartConfig())
    assert len(figure.axes) == 2
    for ax, (label, curves) in zip(figure.axes, panels):
        lines = ax.get_lines()
        assert len(lines) == 3  # 两条曲线 + 100 基准虚线
        assert ax.get_legend() is not None
        for line, (curve_label, values) in zip(lines[:2], curves):
            data = line.get_ydata()
            first = next(value for value in values if np.isfinite(value))
            assert data[0] == pytest.approx(100.0)
            assert data[-1] == pytest.approx(values.iloc[-1] / first * 100.0)


def test_normalized_figure_prints_real_price_note():
    frame = make_frame(40)
    figure = build_normalized_figure(
        [("收盘价", [("A", frame["close"].iloc[:20]), ("B", frame["close"].iloc[20:])])],
        ChartConfig(),
    )
    texts = [text.get_text() for text in figure.texts]
    assert any("不代表真实价格" in text for text in texts)


# ----------------------------------------------------------------------
# 控制器：定位与高亮
# ----------------------------------------------------------------------
def make_controller(length: int = 120) -> CandleChartController:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4), dpi=100)
    candle = make_candle(length)
    result = build_candlestick_figure(candle, ChartConfig(initial_bars=40), figure=figure)
    return CandleChartController(result, figure.canvas)


def test_locate_finds_indices_and_expands_window():
    controller = make_controller()
    candle = controller.candle
    first, last = controller.locate(candle.dates[30], candle.dates[60])
    assert candle.dates[first] >= pd.Timestamp(candle.dates[30])
    assert candle.dates[last - 1] <= pd.Timestamp(candle.dates[60])
    start, stop = controller.window
    assert start <= first and stop >= last
    assert controller._highlight_range == (first, last)


def test_locate_draws_span_on_every_panel():
    controller = make_controller()
    controller.locate(controller.candle.dates[10], controller.candle.dates[20])
    axes = controller.result.axes
    # 每个面板一个高亮块，再加一句文字
    assert len(controller._highlight) == len(axes) + 1
    for ax in axes:
        assert any(getattr(patch, "get_xy", None) for patch in ax.patches)


def test_locate_returns_none_outside_data():
    controller = make_controller()
    assert controller.locate(date(2030, 1, 1), date(2030, 2, 1)) is None
    assert controller._highlight_range is None


def test_clear_highlight_removes_artists():
    controller = make_controller()
    controller.locate(controller.candle.dates[10], controller.candle.dates[20])
    controller.clear_highlight()
    assert controller._highlight == []
    assert controller._highlight_range is None


def test_close_removes_highlight():
    controller = make_controller()
    controller.locate(controller.candle.dates[10], controller.candle.dates[20])
    controller.close()
    assert controller._highlight == []


# ----------------------------------------------------------------------
# 页面接线
# ----------------------------------------------------------------------
def make_page(frame: pd.DataFrame):
    from app.config import DEFAULT_CONFIG, AppConfig
    from modules.trend_analysis.trend_service import TrendService
    from ui.trend_page import TrendPage

    dataset = MarketDataset(name="对比样本", frame=frame)
    page = TrendPage(
        TrendService(),
        lambda: (dataset,),
        lambda: (),
        config=AppConfig(DEFAULT_CONFIG),
    )
    page.refresh()
    return page


def add_segments(page, count: int = 2) -> None:
    starts = (date(2024, 1, 2), date(2024, 3, 4))
    for offset, start in enumerate(starts[:count]):
        end = date(2024, 2, 28) if offset == 0 else date(2024, 4, 30)
        page._segments.append(Segment(start, end))


def test_compare_mode_selector_has_three_modes(qapp, sample_frame):
    page = make_page(make_frame(120))
    modes = [page.combo_compare_mode.itemData(i) for i in range(page.combo_compare_mode.count())]
    assert modes == ["line", "candle_side", "normalized"]


def test_plot_comparison_candle_side_draws_two_columns(qapp):
    page = make_page(make_frame(160))
    add_segments(page, 2)
    page.combo_compare_mode.setCurrentIndex(1)
    page.plot_comparison()

    assert len(page.figure.axes) == 4
    assert page._controller is None
    assert page.toolbar.isVisible() or not page.chart_controls.isVisible()
    assert "模式 A" in page.label_chart_info.text()
    assert "K线并排" in page.label_status.text()


def test_plot_comparison_normalized_draws_panels(qapp):
    page = make_page(make_frame(160))
    add_segments(page, 2)
    page.combo_compare_mode.setCurrentIndex(2)
    page.plot_comparison()

    # 默认只勾选了收盘价 → 一行
    assert len(page.figure.axes) >= 1
    assert "模式 B" in page.label_chart_info.text()
    assert "不代表真实价格" in page.label_status.text()


def test_locate_result_highlights_in_candle_mode(qapp):
    page = make_page(make_frame(160))
    page.combo_chart_type.setCurrentIndex(1)
    page.plot_current()
    assert page._controller is not None

    result = SimpleNamespace(
        rank=1,
        start_date=date(2024, 2, 1),
        end_date=date(2024, 3, 1),
        trading_days=20,
        field="close",
        score=0.93,
        result=SimpleNamespace(label="皮尔逊相关系数"),
    )
    page._locate_result(result, "第 1 名：2024-02-01 ~ 2024-03-01，皮尔逊相关系数 = 0.9300")

    assert page._controller._highlight_range is not None
    assert "已定位" in page.label_status.text()
    assert "0.9300" in page.label_status.text()
