"""图表升级模块（modules.charting）的单元测试。

覆盖设计文档验收表的前三项：K 线 OHLC 与红涨绿跌、均线数值、成交量与
K 线日期一一对应；以及异常数据（乱序、重复、缺失、非法价格）的处理约定。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from core.data_models import MarketDataset
from core.exceptions import InsufficientDataError, ProcessingError
from modules.charting import (
    CandleData,
    ChartConfig,
    build_candlestick_figure,
    build_moving_averages,
    prepare_candle_data,
    sma,
)
from modules.charting.indicators import require_minimum_bars


def make_frame(length: int = 30) -> pd.DataFrame:
    """构造一段确定性的日线数据，便于精确断言。"""
    index = pd.bdate_range("2024-01-02", periods=length, name="date")
    close = np.linspace(10.0, 12.0, length)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1000.0, 2000.0, length),
        },
        index=index,
    )


# ----------------------------------------------------------------------
# 均线
# ----------------------------------------------------------------------
def test_sma_matches_manual_average():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(values, 3)
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:], [2.0, 3.0, 4.0])


def test_sma_propagates_missing_values():
    values = np.array([1.0, np.nan, 3.0, 4.0])
    result = sma(values, 2)
    assert np.isnan(result[1]) and np.isnan(result[2])
    np.testing.assert_allclose(result[3], 3.5)


def test_sma_rejects_non_positive_window():
    with pytest.raises(ValueError):
        sma(np.array([1.0, 2.0]), 0)


def test_moving_averages_use_configured_windows(sample_frame):
    candle = prepare_candle_data(sample_frame, "样本")
    result = build_moving_averages(candle, (5, 10))
    assert set(result) == {5, 10}
    assert np.isnan(result[5][:4]).all()
    np.testing.assert_allclose(result[5][4], sample_frame["close"].iloc[:5].mean())


def test_require_minimum_bars():
    candle = prepare_candle_data(make_frame(3), "短数据")
    require_minimum_bars(candle, 3)
    with pytest.raises(ProcessingError):
        require_minimum_bars(candle, 4)


# ----------------------------------------------------------------------
# 行情整理
# ----------------------------------------------------------------------
def test_prepare_candle_data_preserves_dates_and_volume(sample_frame):
    candle = prepare_candle_data(sample_frame, "样本")
    assert isinstance(candle, CandleData)
    assert len(candle) == len(sample_frame)
    assert candle.dates.equals(pd.DatetimeIndex(sample_frame.index))
    assert candle.has_volume
    # 最后一根的成交量在原始数据里缺失，必须保持缺失而不是补零
    assert np.isnan(candle.volume[-1])
    assert candle.price_span == pytest.approx(
        float(sample_frame["high"].max() - sample_frame["low"].min())
    )


def test_prepare_candle_data_computes_pct_change_from_previous_close(sample_frame):
    candle = prepare_candle_data(sample_frame, "样本")
    assert np.isnan(candle.pct_change[0])
    expected = (sample_frame["close"].iloc[1] - sample_frame["close"].iloc[0]) / sample_frame[
        "close"
    ].iloc[0] * 100
    assert candle.pct_change[1] == pytest.approx(expected)


def test_prepare_candle_data_sorts_unsorted_index():
    frame = make_frame(5).iloc[::-1]
    candle = prepare_candle_data(frame, "乱序")
    assert candle.dates.is_monotonic_increasing
    assert any("升序" in note for note in candle.notes)


def test_prepare_candle_data_keeps_last_duplicate_date():
    frame = make_frame(4)
    duplicated = pd.concat([frame, frame.iloc[[2]].assign(close=99.0)])
    candle = prepare_candle_data(duplicated, "重复")
    assert len(candle) == 4
    assert candle.close[2] == pytest.approx(99.0)
    assert any("重复" in note for note in candle.notes)


def test_prepare_candle_data_drops_invalid_prices():
    frame = make_frame(6)
    frame.iloc[1, frame.columns.get_loc("close")] = 0.0
    frame.iloc[3, frame.columns.get_loc("high")] = 1.0  # 最高价低于最低价
    candle = prepare_candle_data(frame, "异常")
    assert len(candle) == 4
    assert any("剔除异常记录" in note for note in candle.notes)


def test_prepare_candle_data_hides_missing_volume():
    frame = make_frame(6).drop(columns=["volume"])
    candle = prepare_candle_data(frame, "无成交量")
    assert not candle.has_volume
    assert candle.volume is None
    assert any("成交量" in note for note in candle.notes)


def test_prepare_candle_data_hides_invalid_volume():
    frame = make_frame(6)
    frame["volume"] = np.nan
    candle = prepare_candle_data(frame, "成交量缺失")
    assert not candle.has_volume
    assert candle.volume is None


def test_prepare_candle_data_requires_datetime_index():
    frame = make_frame(5).reset_index(drop=True)
    with pytest.raises(ProcessingError) as info:
        prepare_candle_data(frame, "相对数据")
    assert "交易日期" in info.value.user_message()


def test_prepare_candle_data_requires_ohlc():
    frame = make_frame(5).drop(columns=["high"])
    with pytest.raises(ProcessingError) as info:
        prepare_candle_data(frame, "缺字段")
    assert "最高价" in info.value.user_message()


def test_prepare_candle_data_rejects_frame_without_valid_rows():
    frame = make_frame(4)
    frame["close"] = 0.0
    with pytest.raises(InsufficientDataError):
        prepare_candle_data(frame, "全异常")


# ----------------------------------------------------------------------
# 组装图表
# ----------------------------------------------------------------------
def test_build_candlestick_figure_draws_candles_and_volume(sample_frame):
    candle = prepare_candle_data(sample_frame, "样本")
    result = build_candlestick_figure(candle, ChartConfig(show_macd=False))

    assert len(result.axes) == 2
    assert len(result.artists.bodies.get_paths()) == len(candle)
    assert len(result.artists.wicks.get_segments()) == len(candle)
    assert result.volume_artists is not None
    assert result.volume_artists.count == len(candle)
    assert "样本" in result.header
    assert f"共 {len(candle)} 个交易日" in result.header


def test_build_candlestick_figure_colors_follow_close_vs_open():
    frame = make_frame(6)
    frame.iloc[0, frame.columns.get_loc("close")] = frame["open"].iloc[0] + 0.5  # 涨
    frame.iloc[1, frame.columns.get_loc("close")] = frame["open"].iloc[1] - 0.5  # 跌
    candle = prepare_candle_data(frame, "涨跌")
    config = ChartConfig()
    result = build_candlestick_figure(candle, config)
    colors = result.artists.bodies.get_facecolor()

    np.testing.assert_allclose(colors[0], to_rgba(config.up_color))
    np.testing.assert_allclose(colors[1], to_rgba(config.down_color))


def test_build_candlestick_figure_shows_latest_bars_by_default():
    frame = make_frame(30)
    candle = prepare_candle_data(frame, "样本")
    result = build_candlestick_figure(candle, ChartConfig(initial_bars=10))

    start, stop = result.visible_range
    assert (start, stop) == (len(candle) - 10, len(candle))
    assert result.price_ax.get_xlim() == pytest.approx((start - 0.8, stop - 0.2))
    # 顶部信息栏只描述整段数据；当前窗口由界面标签实时显示
    assert "共 30 个交易日" in result.header
    assert "当前显示" not in result.header

    low = frame["low"].iloc[start:stop].min()
    high = frame["high"].iloc[start:stop].max()
    bottom, top = result.price_ax.get_ylim()
    assert bottom < float(low)
    assert top > float(high)
    # 纵轴跟随可视区，而不是被全历史高低点撑开
    assert top - bottom < (high - low) * 3


def test_build_candlestick_figure_hides_volume_without_data():
    frame = make_frame(20).drop(columns=["volume"])
    candle = prepare_candle_data(frame, "无成交量")
    result = build_candlestick_figure(candle, ChartConfig(show_macd=False))

    assert len(result.axes) == 1
    assert result.volume_ax is None
    assert result.volume_artists is None


def test_build_candlestick_figure_respects_show_volume_flag(sample_frame):
    candle = prepare_candle_data(sample_frame, "样本")
    result = build_candlestick_figure(candle, ChartConfig(show_volume=False, show_macd=False))
    assert len(result.axes) == 1
    assert result.volume_ax is None


def test_build_candlestick_figure_reuses_given_figure(sample_frame):
    from matplotlib.figure import Figure

    candle = prepare_candle_data(sample_frame, "样本")
    figure = Figure(figsize=(6.0, 4.0), dpi=100)
    result = build_candlestick_figure(candle, ChartConfig(show_macd=False), figure=figure)

    assert result.figure is figure
    assert len(figure.axes) == 2
    # 复用同一块画布重绘时不应叠加坐标轴
    build_candlestick_figure(candle, ChartConfig(show_macd=False), figure=figure)
    assert len(figure.axes) == 2


def test_build_candlestick_figure_draws_configured_moving_averages(sample_frame):
    candle = prepare_candle_data(sample_frame, "样本")
    result = build_candlestick_figure(candle, ChartConfig(ma_windows=(3,)))

    assert set(result.artists.ma_lines) == {3}
    assert len(result.price_ax.get_lines()) == 1


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
def test_chart_config_cleans_windows():
    config = ChartConfig(ma_windows=(0, 5, 5, 10))
    assert config.ma_windows == (5, 10)
    # 全部取消勾选时允许没有均线，是否回落到默认值由界面决定
    assert ChartConfig(ma_windows=()).ma_windows == ()


def test_chart_config_clamps_initial_bars():
    assert ChartConfig(initial_bars=3).initial_bars == 10
    assert ChartConfig(initial_bars=500).initial_bars == 500


def test_chart_config_from_app_config():
    from app.config import DEFAULT_CONFIG, AppConfig

    config = ChartConfig.from_app_config(AppConfig(DEFAULT_CONFIG))
    assert isinstance(config, ChartConfig)
    assert config.ma_windows == (5, 10, 20)
    assert config.initial_bars == 240

    custom = AppConfig({"chart": {"ma_windows": [5, 30], "initial_bars": 120}})
    overridden = ChartConfig.from_app_config(custom)
    assert overridden.ma_windows == (5, 30)
    assert overridden.initial_bars == 120


# ----------------------------------------------------------------------
# 页面接线
# ----------------------------------------------------------------------
def test_trend_page_candle_mode_draws_panels(qapp, sample_frame):
    from app.config import DEFAULT_CONFIG, AppConfig
    from modules.trend_analysis.trend_service import TrendService
    from ui.trend_page import TrendPage

    dataset = MarketDataset(name="页面K线", frame=sample_frame)
    page = TrendPage(
        TrendService(),
        lambda: (dataset,),
        lambda: (),
        config=AppConfig(DEFAULT_CONFIG),
    )
    page.refresh()
    assert page.combo_chart_type.currentData() == "line"

    page.combo_chart_type.setCurrentIndex(1)
    assert page.btn_draw.text() == "绘制K线图"
    page.plot_current()

    # 默认：主图 + 成交量 + MACD（RSI / BOLL 默认关闭，避免面板过多）
    assert len(page.figure.axes) == 3
    result = page._chart_result
    assert result.macd_ax is not None and result.rsi_ax is None
    assert page.check_macd.isChecked() and not page.check_rsi.isChecked()
    header = page.figure._suptitle.get_text()
    assert "页面K线" in header
    assert "共 12 个交易日" in header
    # 界面信息栏显示当前可视窗口，并可交互缩放
    assert not page.label_chart_info.isHidden()
    assert "显示" in page.label_chart_info.text()
    assert "共 12 个" in page.label_chart_info.text()
    assert page._controller is not None
    assert page.btn_zoom_in.isEnabled()
    assert page.toolbar.isHidden()
    assert not page.chart_controls.isHidden()

    page.combo_chart_type.setCurrentIndex(0)
    page.plot_current()
    assert len(page.figure.axes) == 1
    # 折线图回到原有工具栏，K线控制器必须断开
    assert page._controller is None
    assert not page.btn_zoom_in.isEnabled()
    assert page.chart_controls.isHidden()


def test_trend_page_rejects_normalized_data_for_candles(qapp, sample_frame, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from app.config import DEFAULT_CONFIG, AppConfig
    from core.constants import Normalization
    from core.data_models import ProcessedDataset, ProcessingConfig
    from modules.trend_analysis.trend_service import TrendService
    from ui.trend_page import TrendPage

    config = ProcessingConfig(fields=("close",), normalization=Normalization.START)
    processed = ProcessedDataset("归一化结果", sample_frame, config)
    page = TrendPage(
        TrendService(),
        lambda: (),
        lambda: (processed,),
        config=AppConfig(DEFAULT_CONFIG),
    )
    page.refresh()
    page.combo_chart_type.setCurrentIndex(1)

    calls: list[tuple] = []

    def fake_warning(*args, **kwargs):
        calls.append(args)
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(fake_warning))
    page.plot_current()

    assert calls, "归一化数据必须给出提示而不是画图"
    assert "K线" in str(calls[0][1])
    assert len(page.figure.axes) == 0


def test_trend_page_shows_data_quality_notice(qapp, sample_frame):
    from app.config import DEFAULT_CONFIG, AppConfig
    from modules.trend_analysis.trend_service import TrendService
    from ui.trend_page import TrendPage

    frame = sample_frame.drop(columns=["volume"])
    dataset = MarketDataset(name="无成交量", frame=frame)
    page = TrendPage(
        TrendService(),
        lambda: (dataset,),
        lambda: (),
        config=AppConfig(DEFAULT_CONFIG),
    )
    page.refresh()
    page.combo_chart_type.setCurrentIndex(1)
    page.plot_current()

    # 没有成交量就只剩主图 + MACD 两个面板
    assert len(page.figure.axes) == 2
    assert page._chart_result.volume_ax is None
    # 数据质量说明放在状态栏；信息栏留给光标行情与可视窗口
    assert "成交量" in page.label_status.text()
    assert "显示" in page.label_chart_info.text()
    assert not page.check_volume.isEnabled()
