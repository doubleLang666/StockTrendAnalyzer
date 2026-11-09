"""成交量副图（设计文档 3.2）。

与 K 线共用横坐标（交易日序号），每根柱子对应一个交易日，颜色跟随当日涨跌。
成交量缺失时本模块不参与绘制（由 :mod:`modules.charting.chart_figure` 决定不建副图），
绝不填充或估算成交量。

柱子不用 ``Axes.bar``：那会为每个交易日创建一个 ``Rectangle``，几千根时重绘需要
数秒。这里与蜡烛体一致，用一个 ``PolyCollection`` 一次性提交，重绘保持在百毫秒级，
为阶段二的缩放、平移留出性能余量。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from modules.charting.candle_data import CandleData
from modules.charting.candlestick_chart import body_colors
from modules.charting.chart_config import ChartConfig

__all__ = ["VolumeArtists", "draw_volume", "update_volume", "volume_axis_formatter"]


@dataclass(frozen=True, slots=True)
class VolumeArtists:
    """成交量柱子（单个 PolyCollection）与 VOL_MA 线的引用。"""

    bars: PolyCollection
    ma_lines: dict[int, Line2D] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.bars.get_paths())


def volume_axis_formatter(value: float, _position: int | None = None) -> str:
    """成交量刻度：万 / 亿，与国内行情软件一致。"""
    if not np.isfinite(value):
        return ""
    magnitude = abs(value)
    if magnitude >= 1e8:
        return f"{value / 1e8:.1f}亿"
    if magnitude >= 1e4:
        return f"{value / 1e4:.0f}万"
    return f"{value:,.0f}"


def _bar_vertices(volume: np.ndarray, width_ratio: float, start: int = 0) -> np.ndarray:
    """每根柱子一个四顶点矩形，底边固定在 0；横坐标从 ``start`` 开始。"""
    count = volume.size
    half = 0.5 * width_ratio
    x = np.arange(start, start + count, dtype="float64")
    vertices = np.empty((count, 4, 2), dtype="float64")
    vertices[:, 0, 0] = x - half
    vertices[:, 1, 0] = x - half
    vertices[:, 2, 0] = x + half
    vertices[:, 3, 0] = x + half
    vertices[:, 0, 1] = 0.0
    vertices[:, 1, 1] = volume
    vertices[:, 2, 1] = volume
    vertices[:, 3, 1] = 0.0
    return vertices


def draw_volume(
    ax: Axes,
    candle: CandleData,
    config: ChartConfig,
    moving_averages: dict[int, np.ndarray] | None = None,
) -> VolumeArtists:
    """在副图坐标系上绘制成交量柱与成交量均线。"""
    if not candle.has_volume:
        raise ValueError("该数据没有可用成交量，不应创建成交量副图")
    assert candle.volume is not None
    count = len(candle)
    x = np.arange(count, dtype="float64")
    colors = body_colors(candle, config)
    volume = np.nan_to_num(candle.volume, nan=0.0, posinf=0.0, neginf=0.0)

    bars = PolyCollection(
        _bar_vertices(volume, config.body_width_ratio),
        facecolors=colors,
        edgecolors="none",
        linewidths=0,
        alpha=config.volume_alpha,
        antialiased=False,
        closed=True,
        zorder=2,
    )
    ax.add_collection(bars)
    ax.set_xlim(-1.0, float(count))

    ma_lines: dict[int, Line2D] = {}
    for window, values in (moving_averages or {}).items():
        if values.size != count:
            continue
        (line,) = ax.plot(
            x,
            values,
            color=config.color_for_window(int(window)),
            linewidth=1.0,
            label=f"VOL_MA{int(window)}",
            zorder=3,
        )
        ma_lines[int(window)] = line

    top = float(np.nanmax(volume)) if count else 0.0
    ax.set_ylim(0.0, top * 1.2 if top > 0 else 1.0)

    ax.yaxis.set_major_formatter(FuncFormatter(volume_axis_formatter))
    if config.price_axis_on_right:
        ax.yaxis.tick_right()
    ax.tick_params(axis="both", colors=config.muted, labelsize=8, length=3)
    ax.grid(True, axis="y", color=config.grid_color, alpha=config.grid_alpha, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(config.axes_background)
    ax.set_ylabel("成交量", color=config.muted, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(config.grid_color)

    if ma_lines:
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, 0.90),
            fontsize=6.5,
            frameon=False,
            ncols=min(len(ma_lines), 3),
            handlelength=0.9,
            columnspacing=0.9,
            handletextpad=0.4,
            borderpad=0.1,
            borderaxespad=0.1,
        )
        for text in legend.get_texts():
            text.set_color(config.muted)

    return VolumeArtists(bars=bars, ma_lines=ma_lines)


def update_volume(
    artists: VolumeArtists,
    candle: CandleData,
    config: ChartConfig,
    start: int,
    stop: int,
    moving_averages: dict[int, np.ndarray] | None = None,
) -> None:
    """把成交量柱与 VOL_MA 改成只包含 ``[start, stop)`` 区间（阶段二用）。"""
    if not candle.has_volume:
        raise ValueError("该数据没有可用成交量，不应更新成交量副图")
    assert candle.volume is not None
    start = max(0, min(int(start), len(candle)))
    stop = max(start, min(int(stop), len(candle)))
    volume = np.nan_to_num(candle.volume[start:stop], nan=0.0, posinf=0.0, neginf=0.0)
    artists.bars.set_verts(_bar_vertices(volume, config.body_width_ratio, start))
    artists.bars.set_facecolors(body_colors(candle, config)[start:stop])

    x = np.arange(start, stop, dtype="float64")
    for window, line in artists.ma_lines.items():
        values = (moving_averages or {}).get(window)
        if values is not None and values.size >= stop:
            line.set_data(x, values[start:stop])
        else:
            line.set_data(x, np.full(stop - start, np.nan))
