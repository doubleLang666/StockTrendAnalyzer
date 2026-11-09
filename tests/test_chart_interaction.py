"""阶段二交互控制器测试：缩放、平移、十字光标与数据提示。

控制器不依赖 Qt，因此这里用 ``FigureCanvasAgg`` 离屏驱动，直接调用控制器方法或
伪造 matplotlib 事件，断言"窗口怎么变、蜡烛只画了多少根、光标显示什么"。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from modules.charting import (
    CandleChartController,
    ChartConfig,
    CursorInfo,
    build_candlestick_figure,
    prepare_candle_data,
)
from modules.charting.chart_interaction import ZOOM_FACTOR

COUNT = 120


def make_frame(count: int = COUNT) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=count)
    base = 100.0 + np.linspace(0.0, 20.0, count)
    wave = np.sin(np.arange(count) / 5.0) * 2.0
    close = base + wave
    open_ = close - 0.5 + np.cos(np.arange(count) / 7.0) * 0.8
    high = np.maximum(open_, close) + 1.2
    low = np.minimum(open_, close) - 1.1
    volume = 1e6 + np.arange(count) * 1e4
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def make_chart(initial_bars: int = 40, frame: pd.DataFrame | None = None):
    data = frame if frame is not None else make_frame()
    candle = prepare_candle_data(data, "样本")
    result = build_candlestick_figure(candle, ChartConfig(initial_bars=initial_bars))
    canvas = FigureCanvasAgg(result.figure)
    return result, canvas


def make_controller(initial_bars: int = 40, frame: pd.DataFrame | None = None):
    result, canvas = make_chart(initial_bars, frame)
    cursors: list[CursorInfo | None] = []
    windows: list[object] = []
    controller = CandleChartController(
        result,
        canvas,
        on_cursor=cursors.append,
        on_window=windows.append,
    )
    return result, canvas, controller, cursors, windows


# ----------------------------------------------------------------------
# 初始窗口
# ----------------------------------------------------------------------
def test_controller_starts_with_latest_initial_bars():
    result, _canvas, controller, _cursors, windows = make_controller(initial_bars=40)

    assert controller.window == (len(result.candle) - 40, len(result.candle))
    assert controller.visible_bars == 40
    info = controller.window_info()
    assert info.start_date == result.candle.dates[-40]
    assert info.end_date == result.candle.dates[-1]
    assert "显示" in info.text() and "40 个交易日" in info.text()
    assert windows and "40 个交易日" in windows[-1].text()
    controller.close()


def test_controller_slices_artists_to_visible_window():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)

    # 只把可视区的 40 根塞进 artist，而不是把 120 根全画出来
    assert len(result.artists.bodies.get_paths()) == 40
    assert len(result.artists.wicks.get_segments()) == 40
    assert len(result.volume_artists.bars.get_paths()) == 40
    controller.close()


# ----------------------------------------------------------------------
# 缩放
# ----------------------------------------------------------------------
def test_zoom_in_shrinks_window_around_center():
    _result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=60)
    start, stop = controller.window
    center = (start + stop) / 2.0

    controller.zoom_in()

    new_start, new_stop = controller.window
    assert new_stop - new_start == pytest.approx(round(60 / ZOOM_FACTOR), abs=1)
    assert new_start <= center <= new_stop
    controller.close()


def test_zoom_out_grows_window_until_all_data():
    _result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=20)

    controller.zoom_out()
    assert controller.visible_bars > 20

    for _ in range(20):
        controller.zoom_out()
    assert controller.window == (0, COUNT)
    controller.close()


def test_zoom_respects_minimum_bars():
    _result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)

    for _ in range(30):
        controller.zoom_in()
    assert controller.visible_bars == 10  # MIN_VISIBLE_BARS
    controller.close()


def test_zoom_keeps_anchor_inside_window():
    _result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=60)
    start, stop = controller.window
    anchor = start + 5

    controller.zoom_by(0.5, anchor)

    new_start, new_stop = controller.window
    assert new_start <= anchor <= new_stop
    controller.close()


def test_scroll_event_zooms_in_and_out():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=60)
    before = controller.window
    event = SimpleNamespace(inaxes=result.price_ax, xdata=before[1] - 10, button="up")

    controller._on_scroll(event)
    after_in = controller.window
    assert after_in[1] - after_in[0] < before[1] - before[0]

    controller._on_scroll(SimpleNamespace(inaxes=result.price_ax, xdata=after_in[0] + 5, button="down"))
    assert controller.visible_bars >= after_in[1] - after_in[0]
    controller.close()


def test_scroll_outside_axes_is_ignored():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    before = controller.window
    controller._on_scroll(SimpleNamespace(inaxes=None, xdata=None, button="up"))
    assert controller.window == before
    controller.close()


# ----------------------------------------------------------------------
# 平移
# ----------------------------------------------------------------------
def test_pan_by_moves_window_without_touching_data():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    close_before = result.candle.close.copy()
    start, stop = controller.window

    controller.pan_by(-10)

    assert controller.window == (start - 10, stop - 10)
    np.testing.assert_allclose(result.candle.close, close_before)
    # 平移后仍然只渲染窗口内的蜡烛
    assert len(result.artists.bodies.get_paths()) == 40
    controller.close()


def test_pan_clamps_at_data_edges():
    _result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)

    controller.pan_by(10_000)
    assert controller.window == (COUNT - 40, COUNT)

    controller.pan_by(-10_000)
    assert controller.window == (0, 40)
    controller.close()


def test_drag_pans_the_window():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    start, stop = controller.window
    axes = result.price_ax

    # 向右拖动 = 往回看历史：窗口整体左移 10 根
    controller._on_press(SimpleNamespace(button=1, inaxes=axes, xdata=float(start + 1)))
    controller._on_motion(SimpleNamespace(inaxes=axes, xdata=float(start + 11), ydata=None))
    controller._on_release(SimpleNamespace())

    assert controller.window == (start - 10, stop - 10)
    controller.close()


def test_show_all_and_reset():
    _result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=30)

    controller.show_all()
    assert controller.window == (0, COUNT)

    controller.reset()
    assert controller.visible_bars == 30
    assert controller.window == (COUNT - 30, COUNT)
    controller.close()


# ----------------------------------------------------------------------
# 十字光标与数据提示
# ----------------------------------------------------------------------
def test_cursor_info_text_contains_full_quote():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    index = len(result.candle) - 5

    info = controller.cursor_info(index)

    text = info.text()
    assert result.candle.dates[index].strftime("%Y-%m-%d") in text
    assert "开" in text and "高" in text and "低" in text and "收" in text
    assert "涨跌幅" in text
    assert "成交量" in text
    controller.close()


def test_cursor_info_hides_missing_fields():
    frame = make_frame()
    frame.loc[frame.index[0], "volume"] = np.nan
    _result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40, frame=frame)

    # 第一根没有成交量，也没有涨跌幅（没有前收）
    info = controller.cursor_info(0)
    text = info.text()
    assert "成交量" not in text
    assert "涨跌幅" not in text
    controller.close()


def test_set_cursor_positions_lines_in_both_panels():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    start, stop = controller.window
    index = start + 10

    controller.set_cursor(index, price=123.45)

    # 主图、成交量与指标副图各有一条竖线，位置一致（主副图同步定位）
    assert len(controller._verticals) == len(result.axes)
    for vertical in controller._verticals:
        assert vertical.get_xdata()[0] == index
        assert vertical.get_visible()
    assert controller._price_h.get_ydata()[0] == pytest.approx(123.45)
    controller.close()


def test_set_cursor_outside_window_is_clamped():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    start, stop = controller.window

    controller.set_cursor(-5)
    assert controller._price_v.get_xdata()[0] == start

    controller.set_cursor(COUNT + 100)
    assert controller._price_v.get_xdata()[0] == stop - 1
    controller.close()


def test_clear_cursor_hides_lines_and_notifies():
    result, _canvas, controller, cursors, _windows = make_controller(initial_bars=40)
    controller.set_cursor(len(result.candle) - 1)

    controller.set_cursor(None)

    assert not controller._price_h.get_visible()
    for vertical in controller._verticals:
        assert not vertical.get_visible()
    assert cursors[-1] is None
    controller.close()


def test_motion_outside_axes_clears_cursor():
    result, _canvas, controller, cursors, _windows = make_controller(initial_bars=40)
    controller._on_motion(SimpleNamespace(inaxes=result.price_ax, xdata=float(COUNT - 1), ydata=100.0))
    assert cursors[-1] is not None

    controller._on_motion(SimpleNamespace(inaxes=None, xdata=None, ydata=None))
    assert cursors[-1] is None
    controller.close()


def test_leave_clears_drag_state():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    controller._on_press(SimpleNamespace(button=1, inaxes=result.price_ax, xdata=float(COUNT - 1)))
    assert controller._dragging

    controller._on_leave(SimpleNamespace())
    assert not controller._dragging
    controller.close()


# ----------------------------------------------------------------------
# 生命周期
# ----------------------------------------------------------------------
def test_close_disconnects_and_removes_crosshair():
    result, _canvas, controller, _cursors, _windows = make_controller(initial_bars=40)
    price_lines_before = len(result.price_ax.get_lines())
    volume_lines_before = len(result.volume_ax.get_lines())

    controller.close()

    assert controller._cids == []
    # 十字光标（主图一条竖线 + 一条横线，副图各一条竖线）被移除，其余 artist 保持原样
    assert len(result.price_ax.get_lines()) == price_lines_before - 2
    assert len(result.volume_ax.get_lines()) == volume_lines_before - 1
    assert controller._price_v is None
    assert controller._verticals == []


def test_window_change_reports_window_callback():
    _result, _canvas, controller, _cursors, windows = make_controller(initial_bars=40)
    windows.clear()

    controller.show_all()

    assert windows and "120 个交易日" in windows[-1].text()
    controller.close()
