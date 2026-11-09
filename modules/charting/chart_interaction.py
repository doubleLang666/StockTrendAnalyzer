"""图表交互控制器（设计文档 4 节）：十字光标、滚轮缩放、拖动平移、数据提示。

设计取舍：

* 控制器只依赖 matplotlib 的 canvas（``mpl_connect``），不 import Qt，
  因此测试可以用 ``FigureCanvasAgg`` 离屏驱动，界面层只负责把回调接到标签上；
* 缩放/平移**只改可视窗口**，不触碰 ``CandleData`` 里的原始数据（文档要求
  "平移或缩放不得改变原始数据"）；
* 横轴固定为交易日序号（全局索引），窗口一变就用
  :func:`update_candlesticks` / :func:`update_volume` 只把可视区的那几根数据
  塞进已有的 artist，控件里的几千根蜡烛不会每帧重建，缩放平移才能跟手；
* 十字光标用 ``animated=True`` 的三条线 + ``copy_from_bbox`` / ``blit`` 局部刷新，
  鼠标移动不会触发整图重绘；窗口变化时才做一次完整重绘并重新抓背景。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from modules.charting.candle_data import CandleData
from modules.charting.candlestick_chart import update_candlesticks
from modules.charting.chart_config import ChartConfig
from modules.charting.chart_figure import (
    CandleFigure,
    date_tick_label,
    visible_price_bounds,
)
from modules.charting.indicator_chart import (
    macd_bounds,
    update_boll,
    update_macd,
    update_rsi,
)
from modules.charting.volume_chart import update_volume, volume_axis_formatter

__all__ = [
    "CandleChartController",
    "CursorInfo",
    "WindowInfo",
    "MIN_VISIBLE_BARS",
    "ZOOM_FACTOR",
    "format_price",
]

MIN_VISIBLE_BARS = 10
ZOOM_FACTOR = 1.4
_PRICE_DECIMALS = 2


def format_price(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value):,.{_PRICE_DECIMALS}f}"


def _format_volume(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return volume_axis_formatter(float(value))


@dataclass(frozen=True, slots=True)
class CursorInfo:
    """光标所在交易日的一行行情；缺失字段（NaN）不展示，不做任何补零。"""

    index: int
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    pct_change: float
    volume: float | None

    @property
    def direction(self) -> str:
        """``up`` / ``down`` / ``flat``，界面据此把文字染成红/绿/灰。"""
        if not np.isfinite(self.pct_change) or self.pct_change == 0:
            return "flat"
        return "up" if self.pct_change > 0 else "down"

    def text(self) -> str:
        parts = [
            self.date.strftime("%Y-%m-%d"),
            f"开 {format_price(self.open)}",
            f"高 {format_price(self.high)}",
            f"低 {format_price(self.low)}",
            f"收 {format_price(self.close)}",
        ]
        if np.isfinite(self.pct_change):
            parts.append(f"涨跌幅 {float(self.pct_change):+.2f}%")
        if self.volume is not None and np.isfinite(self.volume):
            parts.append(f"成交量 {_format_volume(self.volume)}")
        return "    ".join(parts)


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """当前可视窗口，用于界面信息栏的"显示 A ~ B（N 个交易日）"。"""

    start: int
    stop: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total: int

    @property
    def count(self) -> int:
        return self.stop - self.start

    def text(self) -> str:
        return (
            f"显示 {self.start_date.strftime('%Y-%m-%d')} ~ "
            f"{self.end_date.strftime('%Y-%m-%d')}"
            f"（{self.count} 个交易日，共 {self.total} 个）"
        )


class CandleChartController:
    """把一张 :class:`CandleFigure` 变成可缩放、可平移、带十字光标的行情图。"""

    def __init__(
        self,
        result: CandleFigure,
        canvas: FigureCanvasBase,
        *,
        on_cursor: Callable[[CursorInfo | None], None] | None = None,
        on_window: Callable[[WindowInfo], None] | None = None,
        min_bars: int = MIN_VISIBLE_BARS,
    ) -> None:
        self.result = result
        self.canvas = canvas
        self.figure: Figure = result.figure
        self.candle: CandleData = result.candle
        self.config: ChartConfig = result.config
        self._count = len(result.candle)
        self._min_bars = max(2, min(int(min_bars), self._count))
        self._on_cursor = on_cursor
        self._on_window = on_window
        self._axes: tuple[Axes, ...] = result.axes
        self._price_ax = result.price_ax
        self._volume_ax = result.volume_ax

        self._initial_window = self._normalize(*result.visible_range)
        self._window = self._initial_window
        self._rendered: tuple[int, int] | None = None
        self._background: object | None = None
        self._use_blit = True
        self._dragging = False
        self._drag_x = 0.0
        self._drag_window = self._initial_window
        self._cursor_index: int | None = None
        self._cids: list[int] = []

        self._crosshair: list[Line2D] = []
        self._verticals: list[Line2D] = []
        self._price_v: Line2D | None = None
        self._price_h: Line2D | None = None
        self._highlight: list[object] = []
        self._highlight_range: tuple[int, int] | None = None
        self._build_crosshair()
        self._connect()

        self._apply_window(self._window, redraw=True, notify=True)

    # ------------------------------------------------------------------
    # 对外属性与方法
    # ------------------------------------------------------------------
    @property
    def window(self) -> tuple[int, int]:
        return self._window

    @property
    def visible_bars(self) -> int:
        return self._window[1] - self._window[0]

    @property
    def total_bars(self) -> int:
        return self._count

    def zoom_in(self) -> None:
        self.zoom_by(1.0 / ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self.zoom_by(ZOOM_FACTOR)

    def zoom_by(self, factor: float, anchor: float | None = None) -> None:
        """以 ``anchor``（交易日索引，缺省为窗口中心）为锚点缩放 ``factor`` 倍。

        ``factor < 1`` 放大（看得更少更细），``factor > 1`` 缩小（看得更多）。
        """
        if factor <= 0:
            return
        start, stop = self._window
        size = stop - start
        new_size = int(round(size * factor))
        new_size = max(self._min_bars, min(new_size, self._count))
        if new_size == size:
            return
        pivot = (start + stop) / 2.0 if anchor is None else float(anchor)
        pivot = min(max(pivot, float(start)), float(stop - 1))
        ratio = (pivot - start) / size
        new_start = int(round(pivot - ratio * new_size))
        self._apply_window(self._normalize(new_start, new_start + new_size))

    def pan_by(self, bars: float) -> None:
        """把窗口向"更晚"的方向平移 ``bars`` 个交易日（负数表示向更早）。"""
        shift = int(round(bars))
        if shift == 0:
            return
        start, stop = self._window
        self._apply_window(self._normalize(start + shift, stop + shift))

    def reset(self) -> None:
        """恢复默认窗口（最近 ``initial_bars`` 根）。"""
        self._apply_window(self._initial_window)

    def show_all(self) -> None:
        """显示全部历史数据。"""
        self._apply_window((0, self._count))

    def set_cursor(self, index: int | None, *, price: float | None = None) -> None:
        """定位十字光标；``index=None`` 时隐藏。

        每个面板各有一条竖线，主图再加一条横线，因此"主副图同步定位"
        在视觉上是一根贯穿全图的时间线。
        """
        if index is None:
            self._cursor_index = None
            for vertical in self._verticals:
                vertical.set_visible(False)
            if self._price_h is not None:
                self._price_h.set_visible(False)
            self._push_cursor(None)
            self._clear_crosshair()
            return

        start, stop = self._window
        index = max(start, min(int(index), stop - 1))
        self._cursor_index = index
        for vertical in self._verticals:
            vertical.set_xdata([index, index])
            vertical.set_visible(True)
        if self._price_h is not None:
            anchor = price if price is not None and np.isfinite(price) else self.candle.close[index]
            self._price_h.set_ydata([anchor, anchor])
            self._price_h.set_visible(np.isfinite(anchor))
        self._push_cursor(self._cursor_info(index))
        self._refresh_crosshair()

    def window_info(self) -> WindowInfo:
        start, stop = self._window
        return WindowInfo(
            start=start,
            stop=stop,
            start_date=self.candle.dates[start],
            end_date=self.candle.dates[stop - 1],
            total=self._count,
        )

    def cursor_info(self, index: int) -> CursorInfo:
        return self._cursor_info(index)

    def locate(self, start_date, stop_date, *, label: str | None = None, margin: float = 0.08):
        """定位到一段真实日期区间并高亮。

        返回命中的交易日索引区间 ``(start, stop)``（``stop`` 为区间后一根），
        区间与数据没有交集时返回 ``None``。
        """
        dates = self.candle.dates
        first = int(dates.searchsorted(pd.Timestamp(start_date), side="left"))
        last = int(dates.searchsorted(pd.Timestamp(stop_date), side="right"))
        if last <= first or first >= self._count:
            return None
        last = min(last, self._count)
        span = last - first
        pad = max(2, int(round(span * margin)))
        window = self._normalize(first - pad, last + pad)
        self._apply_window(window)
        self._draw_highlight(first, last, label)
        return first, last

    def clear_highlight(self) -> None:
        """移除相似区间高亮。"""
        for artist in self._highlight:
            try:
                artist.remove()
            except Exception:  # pragma: no cover
                pass
        self._highlight.clear()
        self._highlight_range = None

    def close(self) -> None:
        """断开所有 matplotlib 事件连接并移除十字光标，允许安全丢弃本控制器。"""
        for cid in self._cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:  # pragma: no cover - 画布已销毁时忽略
                pass
        self._cids.clear()
        for artist in self._crosshair:
            try:
                artist.remove()
            except Exception:  # pragma: no cover
                pass
        self._crosshair.clear()
        self._verticals.clear()
        self._price_v = self._price_h = None
        self.clear_highlight()
        self._background = None

    # ------------------------------------------------------------------
    # 内部：光标与窗口
    # ------------------------------------------------------------------
    def _draw_highlight(self, first: int, last: int, label: str | None) -> None:
        """在区间上画一层半透明底色，并在主图顶部写一句说明。"""
        self.clear_highlight()
        left = first - 0.5
        right = last - 0.5
        for index, ax in enumerate(self._axes):
            span = ax.axvspan(
                left,
                right,
                color=self.config.highlight_color,
                alpha=self.config.highlight_alpha,
                linewidth=0,
                zorder=0.5,
            )
            self._highlight.append(span)
            if index == 0:
                text = label or (
                    f"{self.candle.dates[first].strftime('%Y-%m-%d')} ~ "
                    f"{self.candle.dates[last - 1].strftime('%Y-%m-%d')}"
                )
                self._highlight.append(
                    ax.text(
                        left + 1.0,
                        0.88,
                        text,
                        transform=ax.get_xaxis_transform(),
                        color=self.config.highlight_color,
                        fontsize=7.5,
                        ha="left",
                        va="top",
                        zorder=6,
                        clip_on=True,
                        bbox={
                            "facecolor": self.config.axes_background,
                            "edgecolor": self.config.highlight_color,
                            "linewidth": 0.6,
                            "alpha": 0.88,
                            "boxstyle": "round,pad=0.25",
                        },
                    )
                )
        self._highlight_range = (first, last)

    def _cursor_info(self, index: int) -> CursorInfo:
        candle = self.candle
        volume = None
        if candle.volume is not None:
            value = float(candle.volume[index])
            volume = value if np.isfinite(value) else None
        return CursorInfo(
            index=index,
            date=candle.dates[index],
            open=float(candle.open[index]),
            high=float(candle.high[index]),
            low=float(candle.low[index]),
            close=float(candle.close[index]),
            pct_change=float(candle.pct_change[index]),
            volume=volume,
        )

    def _push_cursor(self, info: CursorInfo | None) -> None:
        if self._on_cursor is not None:
            self._on_cursor(info)

    def _normalize(self, start: int, stop: int) -> tuple[int, int]:
        count = self._count
        size = int(stop) - int(start)
        size = max(self._min_bars, min(size, count))
        start = int(start)
        start = max(0, min(start, count - size))
        return start, start + size

    def _apply_window(
        self, window: tuple[int, int], *, redraw: bool = True, notify: bool = True
    ) -> None:
        start, stop = self._normalize(*window)
        changed = (start, stop) != self._rendered
        self._window = (start, stop)
        if changed:
            self._slice_artists(start, stop)
            self._rendered = (start, stop)
            self._hide_crosshair_for_draw()
        if notify:
            self._push_window()
        if redraw and changed:
            self.canvas.draw_idle()

    def _slice_artists(self, start: int, stop: int) -> None:
        update_candlesticks(
            self.result.artists,
            self.candle,
            self.config,
            start,
            stop,
            self.result.moving_averages,
        )
        self._price_ax.set_xlim(start - 0.8, stop - 0.2)
        self._price_ax.set_ylim(
            *visible_price_bounds(
                self.candle,
                start,
                stop,
                self.result.moving_averages,
                boll=self.result.boll_data,
            )
        )
        if self.result.boll_artists is not None and self.result.boll_data is not None:
            update_boll(self.result.boll_artists, self.result.boll_data, start, stop)

        volume_ax = self._volume_ax
        volume_artists = self.result.volume_artists
        if volume_ax is not None and volume_artists is not None and self.candle.volume is not None:
            update_volume(
                volume_artists,
                self.candle,
                self.config,
                start,
                stop,
                self.result.volume_moving_averages,
            )
            top = float(np.nanmax(self.candle.volume[start:stop]))
            volume_ax.set_ylim(0.0, top * 1.2 if np.isfinite(top) and top > 0 else 1.0)

        macd_ax = self.result.macd_ax
        macd_data = self.result.macd_data
        if macd_ax is not None and macd_data is not None and self.result.macd_artists is not None:
            update_macd(self.result.macd_artists, macd_data, self.config, start, stop)
            macd_ax.set_ylim(*macd_bounds(macd_data, start, stop))

        rsi_ax = self.result.rsi_ax
        if rsi_ax is not None and self.result.rsi_artists is not None and self.result.rsi_data:
            update_rsi(self.result.rsi_artists, self.result.rsi_data, start, stop)
            rsi_ax.set_ylim(0.0, 100.0)

    def _push_window(self) -> None:
        if self._on_window is not None:
            self._on_window(self.window_info())

    def _hide_crosshair_for_draw(self) -> None:
        for artist in self._crosshair:
            artist.set_visible(False)

    # ------------------------------------------------------------------
    # 内部：十字光标绘制（blit）
    # ------------------------------------------------------------------
    def _build_crosshair(self) -> None:
        config = self.config
        style = {
            "color": config.foreground,
            "linewidth": 0.7,
            "linestyle": (0, (4, 3)),
            "alpha": 0.75,
            "zorder": 9,
        }
        for panel in self._axes:
            vertical = panel.axvline(0.0, animated=True, visible=False, **style)
            self._verticals.append(vertical)
            self._crosshair.append(vertical)
        self._price_v = self._verticals[0] if self._verticals else None
        self._price_h = self._price_ax.axhline(0.0, animated=True, visible=False, **style)
        self._crosshair.append(self._price_h)

    def _refresh_crosshair(self) -> None:
        if not any(artist.get_visible() for artist in self._crosshair):
            return
        if self._use_blit and self._background is not None:
            if self._blit_crosshair():
                return
        elif self._use_blit and self._background is None:
            # 还没抓到背景：先整图绘制一次，draw_event 里会补抓背景
            self.canvas.draw_idle()
            return
        self.canvas.draw_idle()

    def _clear_crosshair(self) -> None:
        """隐藏光标后清掉屏幕上残留的线：blit 用背景覆盖，否则整图重绘。"""
        if not self._use_blit:
            self.canvas.draw_idle()
            return
        if self._background is not None:
            try:
                self.canvas.restore_region(self._background)  # type: ignore[arg-type]
                self.canvas.blit(self.figure.bbox)  # type: ignore[attr-defined]
                return
            except Exception:  # pragma: no cover
                self._disable_blit()
        self.canvas.draw_idle()

    def _blit_crosshair(self) -> bool:
        try:
            self.canvas.restore_region(self._background)  # type: ignore[arg-type]
            for artist in self._crosshair:
                if artist.get_visible():
                    artist.axes.draw_artist(artist)
            self.canvas.blit(self.figure.bbox)  # type: ignore[attr-defined]
            return True
        except Exception:  # pragma: no cover - 后端不支持 blit 时退回整图重绘
            self._disable_blit()
            return False

    def _disable_blit(self) -> None:
        """后端不支持局部刷新时，把十字光标改成普通 artist，随整图重绘一起画。"""
        self._use_blit = False
        self._background = None
        for artist in self._crosshair:
            artist.set_animated(False)

    def _on_draw(self, _event: Event) -> None:
        try:
            self._background = self.canvas.copy_from_bbox(self.figure.bbox)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            self._disable_blit()

    # ------------------------------------------------------------------
    # 内部：事件
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        self._cids = [
            self.canvas.mpl_connect("scroll_event", self._on_scroll),
            self.canvas.mpl_connect("button_press_event", self._on_press),
            self.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.canvas.mpl_connect("button_release_event", self._on_release),
            self.canvas.mpl_connect("figure_leave_event", self._on_leave),
            self.canvas.mpl_connect("draw_event", self._on_draw),
        ]

    def _on_scroll(self, event: Event) -> None:
        if getattr(event, "inaxes", None) not in self._axes:
            return
        button = getattr(event, "button", None)
        factor = 1.0 / ZOOM_FACTOR if button == "up" else ZOOM_FACTOR
        anchor = getattr(event, "xdata", None)
        self.zoom_by(factor, anchor)

    def _on_press(self, event: Event) -> None:
        if getattr(event, "button", None) != 1:
            return
        if getattr(event, "inaxes", None) not in self._axes:
            return
        self._dragging = True
        self._drag_x = float(getattr(event, "xdata", 0.0) or 0.0)
        self._drag_window = self._window

    def _on_motion(self, event: Event) -> None:
        if self._dragging:
            if getattr(event, "xdata", None) is None:
                return
            delta = float(event.xdata) - self._drag_x
            start, stop = self._drag_window
            self._apply_window(self._normalize(start - int(round(delta)), stop - int(round(delta))))
            return
        in_axes = getattr(event, "inaxes", None)
        if in_axes not in self._axes or getattr(event, "xdata", None) is None:
            self.set_cursor(None)
            return
        index = int(round(float(event.xdata)))
        price = float(event.ydata) if in_axes is self._price_ax and getattr(event, "ydata", None) else None
        self.set_cursor(index, price=price)

    def _on_release(self, _event: Event) -> None:
        self._dragging = False

    def _on_leave(self, _event: Event) -> None:
        self._dragging = False
        self.set_cursor(None)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def tick_label(self, value: float) -> str:
        return date_tick_label(self.candle.dates, value)
