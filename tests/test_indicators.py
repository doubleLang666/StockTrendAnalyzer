"""技术指标（MACD / RSI / BOLL）的数值对拍与图表接线测试。

数值口径按设计文档 3.3 节：MACD(12,26,9)、RSI(6,12,24)、BOLL(20,2)，
全部由本地收盘价计算，与导入文件里的同名指标列无关。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from core.data_models import MarketDataset
from modules.charting import (
    ChartConfig,
    build_candlestick_figure,
    build_macd,
    build_rsi,
    prepare_candle_data,
)
from modules.charting.chart_interaction import CandleChartController
from modules.charting.indicators import (
    DEFAULT_BOLL_K,
    DEFAULT_BOLL_WINDOW,
    DEFAULT_MACD_PARAMS,
    boll,
    ema,
    macd,
    rsi,
    wilder_average,
)


def make_frame(length: int = 120, *, trend: float = 0.05) -> pd.DataFrame:
    """确定性日线数据：收盘价带轻微波动，避免 MACD / RSI 出现全零极值。"""
    index = pd.bdate_range("2024-01-02", periods=length, name="date")
    offsets = np.arange(length, dtype="float64")
    close = 20.0 + trend * offsets + np.sin(offsets / 3.0) * 0.6
    return pd.DataFrame(
        {
            "open": close - 0.15,
            "high": close + 0.3,
            "low": close - 0.35,
            "close": close,
            "volume": np.linspace(1000.0, 3000.0, length),
        },
        index=index,
    )


# ----------------------------------------------------------------------
# 平滑与 MACD
# ----------------------------------------------------------------------
def test_ema_matches_pandas_ewm():
    values = np.array([10.0, 11.5, 10.8, 12.4, 13.0, 12.2, 12.9])
    expected = pd.Series(values).ewm(span=5, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(ema(values, 5), expected)


def test_ema_skips_missing_values_without_changing_state():
    values = np.array([10.0, np.nan, 12.0, np.nan, 14.0])
    result = ema(values, 3)
    assert np.isnan(result[[1, 3]]).all()
    # NaN 位置不推进状态：后面两个有效值等价于直接对 [10, 12, 14] 递推
    reference = ema(np.array([10.0, 12.0, 14.0]), 3)
    np.testing.assert_allclose(result[[0, 2, 4]], reference)


def test_ema_rejects_invalid_span():
    with pytest.raises(ValueError):
        ema([1.0, 2.0], 0)


def test_wilder_average_uses_one_over_window():
    values = np.array([10.0, 20.0, 20.0])
    result = wilder_average(values, 5)
    # avg = avg + (x - avg) / N
    np.testing.assert_allclose(result, [10.0, 12.0, 13.6])


def test_macd_matches_manual_formula():
    values = np.linspace(10.0, 20.0, 80) + np.sin(np.arange(80))
    data = macd(values)
    dif = ema(values, 12) - ema(values, 26)
    dea = ema(dif, 9)
    np.testing.assert_allclose(data.dif, dif, equal_nan=True)
    np.testing.assert_allclose(data.dea, dea, equal_nan=True)
    np.testing.assert_allclose(data.hist, (dif - dea) * 2.0, equal_nan=True)


def test_macd_rejects_inverted_periods():
    with pytest.raises(ValueError):
        macd([1.0, 2.0, 3.0], fast=26, slow=12)


def test_build_macd_uses_close_prices():
    candle = prepare_candle_data(make_frame(60), "样本")
    data = build_macd(candle, DEFAULT_MACD_PARAMS)
    expected = macd(candle.close)
    np.testing.assert_allclose(data.hist, expected.hist, equal_nan=True)


# ----------------------------------------------------------------------
# RSI
# ----------------------------------------------------------------------
def test_rsi_is_100_for_monotonic_rise_and_0_for_monotonic_fall():
    rising = rsi(np.arange(1.0, 40.0), 6)
    falling = rsi(np.arange(40.0, 1.0, -1.0), 6)
    assert np.isnan(rising[0]) and np.isnan(falling[0])
    np.testing.assert_allclose(rising[1:], 100.0)
    np.testing.assert_allclose(falling[1:], 0.0)


def test_rsi_is_neutral_for_flat_prices():
    flat = rsi(np.full(20, 8.0), 6)
    np.testing.assert_allclose(flat[1:], 50.0)


def test_rsi_stays_within_bounds_on_real_looking_data():
    values = make_frame(200)["close"].to_numpy()
    for window in (6, 12, 24):
        result = rsi(values, window)
        finite = result[np.isfinite(result)]
        assert finite.size
        assert finite.min() >= 0.0 and finite.max() <= 100.0
        assert np.isnan(result[0])


def test_build_rsi_returns_requested_windows():
    candle = prepare_candle_data(make_frame(60), "样本")
    data = build_rsi(candle, (6, 12))
    assert set(data) == {6, 12}
    np.testing.assert_allclose(data[6], rsi(candle.close, 6), equal_nan=True)


# ----------------------------------------------------------------------
# BOLL
# ----------------------------------------------------------------------
def test_boll_bands_are_two_standard_deviations_around_sma():
    values = make_frame(80)["close"].to_numpy()
    data = boll(values, DEFAULT_BOLL_WINDOW, DEFAULT_BOLL_K)
    window = DEFAULT_BOLL_WINDOW
    manual = pd.Series(values).rolling(window).mean().to_numpy()
    deviation = pd.Series(values).rolling(window).std(ddof=0).to_numpy()
    np.testing.assert_allclose(data.middle[window - 1 :], manual[window - 1 :])
    np.testing.assert_allclose(
        data.upper[window - 1 :],
        (manual + DEFAULT_BOLL_K * deviation)[window - 1 :],
    )
    np.testing.assert_allclose(
        data.lower[window - 1 :],
        (manual - DEFAULT_BOLL_K * deviation)[window - 1 :],
    )
    assert np.isnan(data.middle[: window - 1]).all()


def test_boll_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        boll([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        boll([1.0, 2.0], 2, 0)


# ----------------------------------------------------------------------
# 组装图表
# ----------------------------------------------------------------------
def test_figure_adds_macd_and_rsi_panels_in_order():
    candle = prepare_candle_data(make_frame(60), "样本")
    config = ChartConfig(show_macd=True, show_rsi=True, show_boll=True)
    result = build_candlestick_figure(candle, config)

    assert [ax is result.price_ax for ax in result.axes] == [True, False, False, False]
    assert result.axes[1] is result.volume_ax
    assert result.axes[2] is result.macd_ax
    assert result.axes[3] is result.rsi_ax
    assert result.has_indicators

    # 时间轴共享：缩放主图，副图时间范围必须一致
    result.price_ax.set_xlim(10.0, 40.0)
    for ax in result.axes[1:]:
        assert ax.get_xlim() == pytest.approx((10.0, 40.0))


def test_macd_panel_contains_zero_axis_and_bars():
    candle = prepare_candle_data(make_frame(60), "样本")
    result = build_candlestick_figure(candle, ChartConfig(show_macd=True, show_rsi=False))

    low, high = result.macd_ax.get_ylim()
    assert low < 0.0 < high
    assert result.macd_artists is not None
    start, stop = result.visible_range
    assert len(result.macd_artists.bars.get_paths()) == stop - start
    assert len(result.macd_artists.dif_line.get_xdata()) == stop - start


def test_rsi_panel_is_fixed_to_0_100():
    candle = prepare_candle_data(make_frame(60), "样本")
    result = build_candlestick_figure(candle, ChartConfig(show_macd=False, show_rsi=True))

    assert result.rsi_ax.get_ylim() == pytest.approx((0.0, 100.0))
    assert set(result.rsi_artists.lines) == {6, 12, 24}


def test_boll_overlay_draws_three_lines_under_candles():
    candle = prepare_candle_data(make_frame(60), "样本")
    result = build_candlestick_figure(
        candle, ChartConfig(show_macd=False, show_boll=True, ma_windows=())
    )

    assert result.boll_artists is not None
    assert result.boll_data is not None
    assert len(result.price_ax.get_lines()) == 3
    for line in result.price_ax.get_lines():
        assert line.get_zorder() < result.artists.bodies.get_zorder()


def test_indicators_can_be_switched_off_completely():
    candle = prepare_candle_data(make_frame(60), "样本")
    result = build_candlestick_figure(
        candle,
        ChartConfig(show_volume=False, show_macd=False, show_rsi=False, show_boll=False),
    )
    assert len(result.axes) == 1
    assert not result.has_indicators
    assert result.boll_artists is None


# ----------------------------------------------------------------------
# 控制器：缩放平移时指标同步刷新
# ----------------------------------------------------------------------
def make_controller(**config_kwargs) -> tuple[object, CandleChartController]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    candle = prepare_candle_data(make_frame(120), "样本")
    figure = Figure(figsize=(6.0, 4.5), dpi=100)
    result = build_candlestick_figure(
        candle, ChartConfig(initial_bars=60, **config_kwargs), figure=figure
    )
    canvas = FigureCanvasAgg(figure)
    controller = CandleChartController(result, canvas)
    return result, controller


def test_window_change_reslices_indicator_artists():
    result, controller = make_controller(show_macd=True, show_rsi=True)
    controller.zoom_in()
    start, stop = controller.window

    assert len(result.macd_artists.bars.get_paths()) == stop - start
    macd_line = result.macd_artists.dif_line
    np.testing.assert_allclose(macd_line.get_xdata(), np.arange(start, stop))
    rsi_line = result.rsi_artists.lines[12]
    np.testing.assert_allclose(rsi_line.get_xdata(), np.arange(start, stop))
    low, high = result.macd_ax.get_ylim()
    assert low <= 0.0 <= high
    controller.close()


def test_window_change_keeps_boll_in_sync():
    result, controller = make_controller(show_boll=True, show_macd=False)
    controller.show_all()
    assert len(result.boll_artists.middle.get_xdata()) == controller.visible_bars
    controller.close()


# ----------------------------------------------------------------------
# 配置与页面
# ----------------------------------------------------------------------
def test_chart_config_indicator_defaults_and_overrides():
    config = ChartConfig()
    assert config.show_macd is True
    assert config.show_rsi is False
    assert config.show_boll is False
    assert config.rsi_windows == (6, 12, 24)
    assert config.macd_params == (12, 26, 9)

    toggled = config.with_indicators(rsi=True, boll=True)
    assert toggled.show_rsi and toggled.show_boll and toggled.show_macd


def test_chart_config_from_app_config_reads_indicator_keys():
    from app.config import AppConfig

    config = ChartConfig.from_app_config(
        AppConfig(
            {
                "chart": {
                    "show_macd": False,
                    "show_rsi": True,
                    "show_boll": True,
                    "rsi_windows": [6, 24],
                    "boll_window": 26,
                    "boll_k": 2.5,
                }
            }
        )
    )
    assert config.show_macd is False
    assert config.show_rsi is True
    assert config.rsi_windows == (6, 24)
    assert config.boll_window == 26
    assert config.boll_k == pytest.approx(2.5)


def test_chart_config_rejects_invalid_macd_params():
    config = ChartConfig(macd_params=(26, 12, 9))
    assert config.macd_params == DEFAULT_MACD_PARAMS


def test_trend_page_indicator_toggles_redraw(qapp, sample_frame):
    from app.config import DEFAULT_CONFIG, AppConfig
    from modules.trend_analysis.trend_service import TrendService
    from ui.trend_page import TrendPage

    dataset = MarketDataset(name="指标页", frame=sample_frame)
    page = TrendPage(
        TrendService(),
        lambda: (dataset,),
        lambda: (),
        config=AppConfig(DEFAULT_CONFIG),
    )
    page.refresh()
    page.combo_chart_type.setCurrentIndex(1)
    page.plot_current()

    assert len(page.figure.axes) == 3  # 主图 + 成交量 + MACD

    page.check_macd.setChecked(False)
    assert len(page.figure.axes) == 2  # 主图 + 成交量
    assert page._chart_result.macd_ax is None

    page.check_rsi.setChecked(True)
    page.check_boll.setChecked(True)
    assert len(page.figure.axes) == 3  # 主图 + 成交量 + RSI
    assert page._chart_result.rsi_ax is not None
    assert page._chart_result.boll_artists is not None
