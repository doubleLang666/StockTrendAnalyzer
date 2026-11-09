"""历史区间对比（设计文档 3.4）的图表构建。

三种对比形式：

* :func:`build_dual_candle_figure` —— 模式 A：双图并排，两侧各自画蜡烛、
  成交量与均线，横轴一律按"第 N 根"对齐、等宽显示，便于形态对照；
* :func:`build_normalized_figure` —— 模式 B：归一化走势对比，每组曲线以
  首个有效值为 100，图上明确标注"非真实价格"；
* :func:`normalize_to_100` —— 上面两种模式共用的归一化函数。

本模块只负责画图，不改变原始数据，也不修改相似度算法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from modules.charting.candle_data import CandleData
from modules.charting.candlestick_chart import CandleArtists, draw_candlesticks
from modules.charting.chart_config import ChartConfig
from modules.charting.chart_figure import DATE_FORMAT, date_tick_label
from modules.charting.indicators import build_moving_averages, build_volume_moving_averages
from modules.charting.volume_chart import VolumeArtists, draw_volume

__all__ = [
    "DualCandleFigure",
    "NORMALIZED_NOTE",
    "build_dual_candle_figure",
    "build_normalized_figure",
    "normalize_to_100",
]

NORMALIZED_NOTE = "曲线按区间首个有效值归一化为 100，仅用于形态对比，不代表真实价格。"
_LINE_COLORS = ("#5aa9ff", "#f2c14e", "#d977d9", "#7bd389", "#e5484d", "#39c5bb")


def normalize_to_100(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """把序列缩放到"首个有效值 = 100"，缺失值保持 NaN。"""
    array = np.asarray(values, dtype="float64").ravel()
    result = np.full(array.shape, np.nan, dtype="float64")
    finite = np.isfinite(array)
    if not finite.any():
        return result
    base = float(array[finite][0])
    if base == 0:
        return result
    result[finite] = array[finite] / base * 100.0
    return result


@dataclass(frozen=True, slots=True)
class DualCandleFigure:
    """双图并排的结果，便于测试断言与界面取用。"""

    figure: Figure
    candles: tuple[CandleData, ...]
    prices: tuple[Axes, ...]
    volumes: tuple[Axes, ...] = ()
    artists: tuple[CandleArtists, ...] = ()
    volume_artists: tuple[VolumeArtists | None, ...] = ()
    titles: tuple[str, ...] = field(default_factory=tuple)

    @property
    def bars(self) -> int:
        return min((len(candle) for candle in self.candles), default=0)


def _style_price_axis(ax: Axes, config: ChartConfig) -> None:
    ax.set_facecolor(config.axes_background)
    if config.price_axis_on_right:
        ax.yaxis.tick_right()
    ax.tick_params(axis="both", colors=config.muted, labelsize=8, length=3)
    ax.grid(True, axis="y", color=config.grid_color, alpha=config.grid_alpha, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(config.grid_color)


def build_dual_candle_figure(
    left: CandleData,
    right: CandleData,
    config: ChartConfig | None = None,
    *,
    figure: Figure | None = None,
    labels: tuple[str, str] | None = None,
) -> DualCandleFigure:
    """模式 A：两张 K 线图并排。

    两侧横轴都换算成"第 N 根"，并统一成较短一方的长度，保证同跨度可比；
    日期刻度按各自数据的真实日期显示。
    """
    chart_config = config or ChartConfig()
    fig = figure if figure is not None else Figure(figsize=(11.0, 5.8), dpi=100)
    fig.clear()
    fig.patch.set_facecolor(chart_config.background)

    bars = min(len(left), len(right))
    if bars < 2:
        from core.exceptions import InsufficientDataError

        raise InsufficientDataError("对比区间太短", detail="两侧区间至少各需要 2 个交易日才能并排对比")

    left = _truncate(left, bars)
    right = _truncate(right, bars)
    titles = labels or (left.name, right.name)

    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[3.2, 0.9],
        left=0.080,
        right=0.930,
        top=0.865,
        bottom=0.11,
        wspace=0.16,
        hspace=0.10,
    )

    prices: list[Axes] = []
    volumes: list[Axes] = []
    artists: list[CandleArtists] = []
    volume_artists: list[VolumeArtists | None] = []
    show_volume = chart_config.show_volume

    for column, (candle, title) in enumerate(((left, titles[0]), (right, titles[1]))):
        price_ax = fig.add_subplot(grid[0, column])
        _style_price_axis(price_ax, chart_config)
        ma = build_moving_averages(candle, chart_config.ma_windows)
        artists.append(draw_candlesticks(price_ax, candle, chart_config, ma))
        if column == 0:
            # 左图把纵轴放到左侧，避免两列的价格刻度在中间挤在一起
            price_ax.yaxis.tick_left()
            price_ax.yaxis.set_label_position("left")
        price_ax.set_title(title, color=chart_config.foreground, fontsize=9, pad=4)
        price_ax.set_ylim(*_bounds(candle, ma, 0, bars))
        price_ax.set_xlim(-0.8, bars - 0.2)
        prices.append(price_ax)

        if show_volume and candle.has_volume:
            volume_ax = fig.add_subplot(grid[1, column])
            _style_price_axis(volume_ax, chart_config)
            vma = build_volume_moving_averages(candle, chart_config.volume_ma_windows)
            volume_artists.append(draw_volume(volume_ax, candle, chart_config, vma))
            if column == 0:
                volume_ax.yaxis.tick_left()
                volume_ax.yaxis.set_label_position("left")
            top = float(np.nanmax(candle.volume)) if candle.volume is not None else 0.0
            volume_ax.set_ylim(0.0, top * 1.2 if top > 0 else 1.0)
            volume_ax.set_xlim(-0.8, bars - 0.2)
            volume_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            volume_ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _pos, data=candle: date_tick_label(data.dates, value)
                )
            )
            prices[-1].tick_params(axis="x", labelbottom=False)
            volumes.append(volume_ax)
        else:
            price_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            price_ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _pos, data=candle: date_tick_label(data.dates, value)
                )
            )
            volume_artists.append(None)

    fig.suptitle(
        "历史区间对比（模式 A：双图并排，横轴等宽）",
        color=chart_config.foreground,
        fontsize=10.5,
        x=0.045,
        y=0.955,
        ha="left",
        va="top",
    )
    fig.text(
        0.045,
        0.915,
        "两侧都按区间内第 N 根 K 线对齐；纵轴按各自区间缩放，成交量缺失的一侧自动隐藏副图。",
        color=chart_config.muted,
        fontsize=8,
        ha="left",
        va="top",
    )
    return DualCandleFigure(
        figure=fig,
        candles=(left, right),
        prices=tuple(prices),
        volumes=tuple(volumes),
        artists=tuple(artists),
        volume_artists=tuple(volume_artists),
        titles=titles,
    )


def _truncate(candle: CandleData, bars: int) -> CandleData:
    """截取每个区间的前 ``bars`` 根，保证两张图等宽。"""
    if len(candle) == bars:
        return candle
    from dataclasses import replace

    sliced = candle.dates[:bars]
    return replace(
        candle,
        dates=sliced,
        open=candle.open[:bars],
        high=candle.high[:bars],
        low=candle.low[:bars],
        close=candle.close[:bars],
        volume=None if candle.volume is None else candle.volume[:bars],
        pct_change=candle.pct_change[:bars],
    )


def _bounds(candle: CandleData, moving_averages: dict[int, np.ndarray], start: int, stop: int):
    high = float(np.nanmax(candle.high[start:stop]))
    low = float(np.nanmin(candle.low[start:stop]))
    for values in moving_averages.values():
        segment = values[start:stop]
        if segment.size and np.isfinite(segment).any():
            high = max(high, float(np.nanmax(segment)))
            low = min(low, float(np.nanmin(segment)))
    padding = (high - low) * 0.06 or max(abs(high) * 0.01, 1.0)
    return low - padding, high + padding


def build_normalized_figure(
    panels: Sequence[tuple[str, Sequence[tuple[str, Sequence[float]]]]],
    config: ChartConfig | None = None,
    *,
    figure: Figure | None = None,
    title: str = "归一化走势对比（模式 B）",
    note: str = NORMALIZED_NOTE,
) -> Figure:
    """模式 B：每个字段一行，把两段区间的走势都归一化成"首个有效值 = 100"。

    ``panels`` 形如 ``[("收盘价", [("区间一", values), ("区间二", values)]), ...]``。
    """
    chart_config = config or ChartConfig()
    fig = figure if figure is not None else Figure(figsize=(9.0, 5.6), dpi=100)
    fig.clear()
    fig.patch.set_facecolor(chart_config.background)

    rows = max(1, len(panels))
    grid = fig.add_gridspec(
        rows,
        1,
        left=0.075,
        right=0.955,
        top=0.855 if rows > 1 else 0.845,
        bottom=0.16 if rows > 1 else 0.20,
        hspace=0.42 if rows > 1 else 0.0,
    )

    for row, (field_label, curves) in enumerate(panels):
        ax = fig.add_subplot(grid[row])
        _style_price_axis(ax, chart_config)
        for position, (label, values) in enumerate(curves):
            normalized = normalize_to_100(values)
            x = np.arange(normalized.size, dtype="float64")
            ax.plot(
                x,
                normalized,
                color=_LINE_COLORS[position % len(_LINE_COLORS)],
                linewidth=1.5,
                label=label,
            )
        ax.set_title(field_label, color=chart_config.foreground, fontsize=9.5, pad=4, loc="left")
        ax.grid(True, axis="y", color=chart_config.grid_color, alpha=chart_config.grid_alpha)
        ax.tick_params(axis="both", colors=chart_config.muted, labelsize=8, length=3)
        legend = ax.legend(
            loc="upper left",
            fontsize=7.5,
            frameon=False,
            ncols=min(len(curves), 2),
        )
        for text in legend.get_texts():
            text.set_color(chart_config.muted)
        ax.set_ylabel("归一化数值", color=chart_config.muted, fontsize=8)
        if row == rows - 1:
            ax.set_xlabel("相对交易日（第 N 日）", color=chart_config.muted, fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}"))
        ax.axhline(100.0, color=chart_config.muted, linewidth=0.7, linestyle=(0, (4, 3)), alpha=0.5)

    fig.suptitle(
        title,
        color=chart_config.foreground,
        fontsize=10.5,
        x=0.045,
        y=0.965,
        ha="left",
        va="top",
    )
    fig.text(0.045, 0.075, note, color=chart_config.muted, fontsize=8, ha="left", va="bottom")
    return fig
