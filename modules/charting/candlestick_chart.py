"""K 线蜡烛图绘制（设计文档 3.1）。

只负责在给定的 ``Axes`` 上画东西，不创建 Figure、不处理日期刻度、不涉及 Qt，
因此可以在离屏环境里直接断言"画了几根蜡烛、什么颜色"。

实现要点：

* **影线**用 ``vlines`` 一次性生成 ``LineCollection``，**实体**用 ``PolyCollection``
  一次性提交，避免逐根 ``Rectangle`` 带来的几千个 artist；
* 红涨绿跌：``close >= open`` 用红色，``close < open`` 用绿色，完全相等时按平盘
  色画一根细实体（十字星），不会因为零高度而"消失"；
* 均线在同一坐标系上叠加，颜色来自 :class:`ChartConfig`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from modules.charting.candle_data import CandleData
from modules.charting.chart_config import ChartConfig

__all__ = ["CandleArtists", "draw_candlesticks", "price_axis_formatter", "update_candlesticks"]


@dataclass(frozen=True, slots=True)
class CandleArtists:
    """绘制结果中的艺术家引用，供阶段二的十字光标/区间高亮复用。"""

    wicks: LineCollection
    bodies: PolyCollection
    ma_lines: dict[int, Line2D] = field(default_factory=dict)


def price_axis_formatter(value: float, _position: int | None = None) -> str:
    """价格轴刻度：千分位 + 两位小数（指数点位与股价都适用）。"""
    if not np.isfinite(value):
        return ""
    return f"{value:,.2f}"


def body_colors(candle: CandleData, config: ChartConfig) -> np.ndarray:
    """按涨跌给出每根蜡烛的颜色：红涨、绿跌、平盘中性色。"""
    open_ = candle.open
    close = candle.close
    colors = np.where(close >= open_, config.up_color, config.down_color).astype(object)
    flat = close == open_
    if bool(np.any(flat)):
        colors[flat] = config.flat_color
    return colors


def _slice_geometry(
    candle: CandleData,
    config: ChartConfig,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 ``[start, stop)`` 区间的实体顶点、影线线段与颜色。

    局部横坐标从 0 开始，因此同一份几何既可用于整段绘制，也可用于缩放平移后的
    可视区重绘（见 :func:`update_candlesticks`）。
    """
    count = max(0, stop - start)
    half = 0.5 * config.body_width_ratio
    x = np.arange(start, stop, dtype="float64")

    open_ = candle.open[start:stop]
    close = candle.close[start:stop]
    low = candle.low[start:stop]
    high = candle.high[start:stop]

    low_edge = np.minimum(open_, close)
    high_edge = np.maximum(open_, close)
    flat = close == open_
    if bool(np.any(flat)):
        span = np.maximum(high - low, candle.price_span * 1e-4)
        minimum = np.maximum(span * config.flat_body_ratio, candle.price_span * 1e-3)
        center = (open_ + close) * 0.5
        low_edge = np.where(flat, center - minimum * 0.5, low_edge)
        high_edge = np.where(flat, center + minimum * 0.5, high_edge)

    vertices = np.empty((count, 4, 2), dtype="float64")
    vertices[:, 0, 0] = x - half
    vertices[:, 1, 0] = x - half
    vertices[:, 2, 0] = x + half
    vertices[:, 3, 0] = x + half
    vertices[:, 0, 1] = low_edge
    vertices[:, 1, 1] = high_edge
    vertices[:, 2, 1] = high_edge
    vertices[:, 3, 1] = low_edge

    segments = np.stack(
        (np.column_stack((x, low)), np.column_stack((x, high))),
        axis=1,
    )
    colors = body_colors(candle, config)[start:stop]
    return vertices, segments, colors


def draw_candlesticks(
    ax: Axes,
    candle: CandleData,
    config: ChartConfig,
    moving_averages: dict[int, np.ndarray] | None = None,
) -> CandleArtists:
    """在主坐标系上绘制蜡烛与均线。"""
    count = len(candle)
    if count == 0:
        raise ValueError("没有可绘制的K线数据")
    vertices, segments, colors = _slice_geometry(candle, config, 0, count)

    wicks = LineCollection(segments, colors=colors, linewidths=config.wick_linewidth, zorder=2)
    ax.add_collection(wicks)
    bodies = PolyCollection(
        vertices,
        facecolors=colors,
        edgecolors=colors,
        linewidths=0.55,
        antialiased=False,
        closed=True,
        zorder=3,
    )
    ax.add_collection(bodies)

    ma_lines: dict[int, Line2D] = {}
    x = np.arange(count, dtype="float64")
    for window, values in (moving_averages or {}).items():
        if values.size != count:
            continue
        (line,) = ax.plot(
            x,
            values,
            color=config.color_for_window(int(window)),
            linewidth=config.ma_linewidth,
            label=f"MA{int(window)}",
            zorder=4,
        )
        ma_lines[int(window)] = line

    padding = candle.price_span * 0.04
    ax.set_ylim(float(np.nanmin(candle.low)) - padding, float(np.nanmax(candle.high)) + padding)
    ax.set_xlim(-1.0, float(count) + 0.0)

    ax.yaxis.set_major_formatter(FuncFormatter(price_axis_formatter))
    if config.price_axis_on_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.tick_params(axis="both", colors=config.muted, labelsize=8, length=3)
    ax.grid(True, axis="y", color=config.grid_color, alpha=config.grid_alpha, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.set_facecolor(config.axes_background)
    for spine in ax.spines.values():
        spine.set_color(config.grid_color)

    if ma_lines:
        legend = ax.legend(
            loc="upper left",
            fontsize=8,
            frameon=False,
            ncols=min(len(ma_lines), 5),
            handlelength=1.2,
            columnspacing=1.2,
        )
        for text in legend.get_texts():
            text.set_color(config.foreground)

    return CandleArtists(wicks=wicks, bodies=bodies, ma_lines=ma_lines)


def update_candlesticks(
    artists: CandleArtists,
    candle: CandleData,
    config: ChartConfig,
    start: int,
    stop: int,
    moving_averages: dict[int, np.ndarray] | None = None,
) -> None:
    """把已有艺术家改成只包含 ``[start, stop)`` 区间（阶段二缩放/平移用）。

    只替换顶点与线段数据，不新建 artist、不清空坐标轴，因此几千根数据的一帧
    重绘可以控制在几十毫秒内；越界区间会被钳制到合法范围。
    """
    start = max(0, min(int(start), len(candle)))
    stop = max(start, min(int(stop), len(candle)))
    vertices, segments, colors = _slice_geometry(candle, config, start, stop)
    artists.bodies.set_verts(vertices)
    artists.bodies.set_facecolors(colors)
    artists.bodies.set_edgecolors(colors)
    artists.wicks.set_segments(segments)
    artists.wicks.set_color(colors)

    x = np.arange(start, stop, dtype="float64")
    for window, line in artists.ma_lines.items():
        values = (moving_averages or {}).get(window)
        if values is not None and values.size >= stop:
            line.set_data(x, values[start:stop])
        else:
            line.set_data(x, np.full(stop - start, np.nan))
