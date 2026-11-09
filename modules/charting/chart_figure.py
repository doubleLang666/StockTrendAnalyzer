"""组装一张完整的行情 K 线图（主图 + 成交量 + MACD / RSI 副图）。

设计文档 2 节的布局：顶部基本信息、中部主 K 线、下部成交量与指标副图、
底部日期轴。是否出现某个面板完全由 :class:`ChartConfig` 的开关决定，
指标计算在本地完成，不使用导入文件里可能存在的同名列。

横轴使用**交易日序号**而不是真实日期：

* 周末与长假不会留下空白（与券商软件一致）；
* 缩放、平移、区间高亮都退化为整数索引运算；
* 主副图通过 ``sharex`` 共享，天然满足“缩放平移时主副图时间轴同步”。

本模块不依赖 Qt，测试可离屏断言坐标轴数量、蜡烛数量与可视区范围。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from modules.charting.candle_data import CandleData
from modules.charting.candlestick_chart import CandleArtists, draw_candlesticks
from modules.charting.chart_config import ChartConfig
from modules.charting.indicator_chart import (
    BollArtists,
    MacdArtists,
    RsiArtists,
    boll_bounds,
    draw_boll,
    draw_macd,
    draw_rsi,
    macd_bounds,
)
from modules.charting.indicators import (
    BollData,
    MacdData,
    build_boll,
    build_macd,
    build_moving_averages,
    build_rsi,
    build_volume_moving_averages,
)
from modules.charting.volume_chart import VolumeArtists, draw_volume

__all__ = [
    "CandleFigure",
    "build_candlestick_figure",
    "build_header",
    "date_tick_label",
    "visible_price_bounds",
    "DATE_FORMAT",
]

DATE_FORMAT = "%Y-%m-%d"
_PERIOD_LABEL = "日线"


@dataclass(frozen=True, slots=True)
class CandleFigure:
    """一次绘制的结果，阶段二的交互直接复用这些引用，不需要重建图。"""

    figure: Figure
    candle: CandleData
    config: ChartConfig
    price_ax: Axes
    volume_ax: Axes | None
    artists: CandleArtists
    volume_artists: VolumeArtists | None
    header: str
    dates: pd.DatetimeIndex
    visible_range: tuple[int, int]
    moving_averages: dict[int, np.ndarray] = field(default_factory=dict)
    volume_moving_averages: dict[int, np.ndarray] = field(default_factory=dict)
    macd_ax: Axes | None = None
    rsi_ax: Axes | None = None
    macd_data: MacdData | None = None
    rsi_data: dict[int, np.ndarray] = field(default_factory=dict)
    boll_data: BollData | None = None
    macd_artists: MacdArtists | None = None
    rsi_artists: RsiArtists | None = None
    boll_artists: BollArtists | None = None

    @property
    def axes(self) -> tuple[Axes, ...]:
        """按从上到下的顺序返回所有面板。"""
        panels: list[Axes] = [self.price_ax]
        for ax in (self.volume_ax, self.macd_ax, self.rsi_ax):
            if ax is not None:
                panels.append(ax)
        return tuple(panels)

    @property
    def has_indicators(self) -> bool:
        return self.macd_ax is not None or self.rsi_ax is not None


def date_tick_label(dates: pd.DatetimeIndex, value: float) -> str:
    """把横轴上的序号换算成交易日标签；越界返回空串。"""
    index = int(round(float(value)))
    if 0 <= index < len(dates):
        return dates[index].strftime(DATE_FORMAT)
    return ""


def visible_price_bounds(
    candle: CandleData,
    start: int,
    stop: int,
    moving_averages: dict[int, np.ndarray],
    *,
    boll: BollData | None = None,
) -> tuple[float, float]:
    """按可视区间的最高/最低价（含均线与布林轨）给出纵轴范围。

    只按全量历史取范围会让当前窗口挤在图的顶部，因此纵轴必须跟随可视区。
    """
    high = float(np.nanmax(candle.high[start:stop]))
    low = float(np.nanmin(candle.low[start:stop]))
    for values in moving_averages.values():
        segment = values[start:stop]
        if segment.size and np.isfinite(segment).any():
            high = max(high, float(np.nanmax(segment)))
            low = min(low, float(np.nanmin(segment)))
    if boll is not None:
        band = boll_bounds(boll, start, stop)
        if band is not None:
            high = max(high, band[1])
            low = min(low, band[0])
    padding = (high - low) * 0.06
    if padding <= 0:
        padding = max(abs(high) * 0.01, 1.0)
    return low - padding, high + padding


def build_header(candle: CandleData) -> str:
    """顶部基本信息：股票名、周期、数据全区间与交易日总数。

    当前可视区间的起止日期由界面信息栏显示，随缩放平移实时更新，
    避免每次缩放都去改图内的文字。
    """
    return (
        f"{candle.name} · {_PERIOD_LABEL} · "
        f"{candle.start.strftime(DATE_FORMAT)} ~ {candle.end.strftime(DATE_FORMAT)} · "
        f"共 {len(candle)} 个交易日"
    )


def _layout_rows(show_volume: bool, show_macd: bool, show_rsi: bool) -> tuple[int, list[float]]:
    ratios: list[float] = [3.1]
    if show_volume:
        ratios.append(0.85)
    if show_macd:
        ratios.append(1.1)
    if show_rsi:
        ratios.append(1.1)
    return len(ratios), ratios


def build_candlestick_figure(
    candle: CandleData,
    config: ChartConfig | None = None,
    *,
    figure: Figure | None = None,
    moving_averages: dict[int, np.ndarray] | None = None,
    volume_moving_averages: dict[int, np.ndarray] | None = None,
    macd_data: MacdData | None = None,
    rsi_data: dict[int, np.ndarray] | None = None,
    boll_data: BollData | None = None,
) -> CandleFigure:
    """在 ``figure`` 上绘制 K 线图并返回全部绘制结果。

    ``figure`` 传入时会被清空后复用（与 ``plotting.build_figure`` 的约定一致），
    便于趋势页只用一块画布在折线与 K 线之间切换。指标数据可以预先算好传入，
    这样切换显隐或重建图时不必重复计算历史序列。
    """
    chart_config = config or ChartConfig()
    fig = figure if figure is not None else Figure(figsize=(8.8, 5.6), dpi=100)
    fig.clear()
    fig.patch.set_facecolor(chart_config.background)

    show_volume = chart_config.show_volume and candle.has_volume
    rows, height_ratios = _layout_rows(show_volume, chart_config.show_macd, chart_config.show_rsi)
    margins = dict(
        left=0.055,
        right=0.935 if chart_config.price_axis_on_right else 0.975,
        top=0.885,
        bottom=0.115,
        hspace=0.16 if rows > 2 else 0.08,
    )
    grid = fig.add_gridspec(rows, 1, height_ratios=height_ratios, **margins)

    ma = (
        moving_averages
        if moving_averages is not None
        else build_moving_averages(candle, chart_config.ma_windows)
    )
    boll = boll_data
    if chart_config.show_boll and boll is None:
        boll = build_boll(candle, chart_config.boll_window, chart_config.boll_k)

    price_ax = fig.add_subplot(grid[0])
    price_ax.set_facecolor(chart_config.axes_background)
    boll_artists = draw_boll(price_ax, boll, chart_config) if boll is not None else None
    artists = draw_candlesticks(price_ax, candle, chart_config, ma)

    row = 1
    volume_ax: Axes | None = None
    volume_artists: VolumeArtists | None = None
    if show_volume:
        volume_ax = fig.add_subplot(grid[row], sharex=price_ax)
        row += 1
        volume_ax.set_facecolor(chart_config.axes_background)
        volume_ma = (
            volume_moving_averages
            if volume_moving_averages is not None
            else build_volume_moving_averages(candle, chart_config.volume_ma_windows)
        )
        volume_artists = draw_volume(volume_ax, candle, chart_config, volume_ma)

    macd_ax: Axes | None = None
    macd_artists: MacdArtists | None = None
    macd_series = macd_data
    if chart_config.show_macd:
        macd_ax = fig.add_subplot(grid[row], sharex=price_ax)
        row += 1
        if macd_series is None:
            macd_series = build_macd(candle, chart_config.macd_params)
        macd_artists = draw_macd(macd_ax, macd_series, chart_config)

    rsi_ax: Axes | None = None
    rsi_artists: RsiArtists | None = None
    rsi_series: dict[int, np.ndarray] = dict(rsi_data or {})
    if chart_config.show_rsi:
        rsi_ax = fig.add_subplot(grid[row], sharex=price_ax)
        row += 1
        if not rsi_series:
            rsi_series = build_rsi(candle, chart_config.rsi_windows)
        rsi_artists = draw_rsi(rsi_ax, rsi_series, chart_config)

    count = len(candle)
    visible = max(1, min(int(chart_config.initial_bars), count))
    start = count - visible
    stop = count
    price_ax.set_xlim(start - 0.8, stop - 0.2)
    price_ax.set_ylim(*visible_price_bounds(candle, start, stop, ma, boll=boll))
    if volume_ax is not None and candle.volume is not None:
        visible_top = float(np.nanmax(candle.volume[start:stop]))
        volume_ax.set_ylim(0.0, visible_top * 1.2 if visible_top > 0 else 1.0)
    if macd_ax is not None and macd_series is not None:
        macd_ax.set_ylim(*macd_bounds(macd_series, start, stop))

    panels = [ax for ax in (price_ax, volume_ax, macd_ax, rsi_ax) if ax is not None]
    for panel in panels[:-1]:
        panel.tick_params(axis="x", labelbottom=False)
    tick_ax = panels[-1]
    tick_ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    tick_ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: date_tick_label(candle.dates, value))
    )
    tick_ax.tick_params(axis="x", colors=chart_config.muted, labelsize=8, length=3)
    if len(panels) > 1:
        tick_ax.set_xlabel("交易日期", color=chart_config.muted, fontsize=8, labelpad=2)

    header = build_header(candle)
    fig.suptitle(
        header,
        color=chart_config.foreground,
        fontsize=11,
        x=margins["left"],
        y=0.965,
        ha="left",
        va="top",
    )

    return CandleFigure(
        figure=fig,
        candle=candle,
        config=chart_config,
        price_ax=price_ax,
        volume_ax=volume_ax,
        artists=artists,
        volume_artists=volume_artists,
        header=header,
        dates=candle.dates,
        visible_range=(start, stop),
        moving_averages=ma,
        volume_moving_averages=volume_ma if show_volume else {},
        macd_ax=macd_ax,
        rsi_ax=rsi_ax,
        macd_data=macd_series,
        rsi_data=rsi_series if chart_config.show_rsi else {},
        boll_data=boll if chart_config.show_boll else None,
        macd_artists=macd_artists,
        rsi_artists=rsi_artists,
        boll_artists=boll_artists,
    )
