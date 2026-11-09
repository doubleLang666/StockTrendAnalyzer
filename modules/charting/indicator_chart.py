"""MACD / RSI 副图与 BOLL 主图叠加（设计文档 3.3）。

与蜡烛、成交量一致，所有柱体都用单个 ``PolyCollection`` 提交，
线的数据也只在窗口变化时用 ``set_data`` 更新，保证缩放平移不重算指标、
不重建艺术家。

口径见 :mod:`modules.charting.indicators`：

* MACD 柱状值为 ``2 * (DIF - DEA)``，红柱在 0 轴上方、绿柱在下方；
* RSI 纵轴固定 0~100，并画出 30 / 70 两条参考线；
* BOLL 叠加在主图上（压在所有蜡烛之下），只画中轨与上下轨。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

from modules.charting.chart_config import ChartConfig, RSI_COLORS
from modules.charting.indicators import BollData, MacdData

__all__ = [
    "BollArtists",
    "MacdArtists",
    "RsiArtists",
    "boll_bounds",
    "draw_boll",
    "draw_macd",
    "draw_rsi",
    "macd_bounds",
    "update_boll",
    "update_macd",
    "update_rsi",
]

_DIF_COLOR = "#c9d1d9"
_DEA_COLOR = "#f2c14e"
_BOLL_MID_COLOR = "#b6bfcb"
_BOLL_BAND_COLOR = "#39c5bb"
_RSI_GUIDE_LEVELS = (30.0, 70.0)


@dataclass(frozen=True, slots=True)
class MacdArtists:
    bars: PolyCollection
    dif_line: Line2D
    dea_line: Line2D


@dataclass(frozen=True, slots=True)
class RsiArtists:
    lines: dict[int, Line2D] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BollArtists:
    middle: Line2D
    upper: Line2D
    lower: Line2D


# ----------------------------------------------------------------------
# 公共样式
# ----------------------------------------------------------------------
def _style_indicator_axis(ax: Axes, config: ChartConfig) -> None:
    """副图统一外观：深色底、右侧刻度、低对比横向网格、弱化边框。"""
    ax.set_facecolor(config.axes_background)
    if config.price_axis_on_right:
        ax.yaxis.tick_right()
    ax.tick_params(axis="both", colors=config.muted, labelsize=8, length=3)
    ax.grid(True, axis="y", color=config.grid_color, alpha=config.grid_alpha, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(config.grid_color)


def _legend(ax: Axes, handles: list[object], labels: list[str], config: ChartConfig):
    """左上角横排图例。

    不使用 ``title=``：matplotlib 在多列图例下会把标题摆到与条目同一行，
    参数说明因此单独画在右上角（:func:`_param_label`）。
    """
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.90),
        fontsize=6.5,
        frameon=False,
        ncols=len(labels),
        handlelength=0.9,
        columnspacing=0.9,
        handletextpad=0.4,
        borderpad=0.1,
        borderaxespad=0.1,
    )
    for text in legend.get_texts():
        text.set_color(config.muted)
    return legend


def _param_label(ax: Axes, text: str, config: ChartConfig) -> None:
    """把指标参数写在面板右上角，与图例分开，避免挤压图例。"""
    ax.text(
        0.997,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=config.muted,
        zorder=6,
    )


def _hist_vertices(hist: np.ndarray, width_ratio: float, start: int = 0) -> np.ndarray:
    """MACD 柱：每根一个四顶点矩形，从 0 轴画到柱值（正负都可）。"""
    values = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    count = values.size
    half = 0.5 * width_ratio
    x = np.arange(start, start + count, dtype="float64")
    vertices = np.empty((count, 4, 2), dtype="float64")
    vertices[:, 0, 0] = x - half
    vertices[:, 1, 0] = x - half
    vertices[:, 2, 0] = x + half
    vertices[:, 3, 0] = x + half
    vertices[:, 0, 1] = 0.0
    vertices[:, 1, 1] = values
    vertices[:, 2, 1] = values
    vertices[:, 3, 1] = 0.0
    return vertices


def _hist_colors(hist: np.ndarray, config: ChartConfig) -> np.ndarray:
    positive = np.asarray(hist, dtype="float64") >= 0
    return np.where(positive, config.up_color, config.down_color)


# ----------------------------------------------------------------------
# MACD
# ----------------------------------------------------------------------
def macd_bounds(macd: MacdData, start: int, stop: int) -> tuple[float, float]:
    """可视区内 DIF / DEA / 柱的纵轴范围，0 轴始终包含在内。"""
    parts = [
        macd.dif[start:stop],
        macd.dea[start:stop],
        macd.hist[start:stop],
    ]
    values = np.concatenate(parts) if parts else np.array([], dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    padding = (high - low) * 0.12 or 0.1
    return low - padding, high + padding


def draw_macd(ax: Axes, macd: MacdData, config: ChartConfig) -> MacdArtists:
    """在副图上画 MACD 柱、DIF 与 DEA。"""
    count = macd.hist.size
    hist = np.nan_to_num(macd.hist, nan=0.0, posinf=0.0, neginf=0.0)
    bars = PolyCollection(
        _hist_vertices(hist, config.body_width_ratio * 0.8),
        facecolors=_hist_colors(macd.hist, config),
        edgecolors="none",
        linewidths=0,
        antialiased=False,
        closed=True,
        alpha=0.9,
        zorder=2,
    )
    ax.add_collection(bars)
    x = np.arange(count, dtype="float64")
    (dif_line,) = ax.plot(x, macd.dif, color=_DIF_COLOR, linewidth=1.0, label="DIF", zorder=3)
    (dea_line,) = ax.plot(x, macd.dea, color=_DEA_COLOR, linewidth=1.0, label="DEA", zorder=3)
    ax.axhline(0.0, color=config.foreground, linewidth=0.8, alpha=0.35, zorder=1)
    ax.set_xlim(-1.0, float(count))

    _style_indicator_axis(ax, config)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _p: f"{value:.2f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=False))
    fast, slow, signal = config.macd_params
    _legend(
        ax,
        [Patch(facecolor=config.up_color, alpha=0.9), dif_line, dea_line],
        ["MACD", "DIF", "DEA"],
        config,
    )
    _param_label(ax, f"MACD({fast},{slow},{signal})", config)
    ax.set_ylim(*macd_bounds(macd, 0, count) if count else (-1.0, 1.0))
    return MacdArtists(bars=bars, dif_line=dif_line, dea_line=dea_line)


def update_macd(
    artists: MacdArtists,
    macd: MacdData,
    config: ChartConfig,
    start: int,
    stop: int,
) -> None:
    """把 MACD 的三部分改成只包含 ``[start, stop)``。"""
    start = max(0, min(int(start), macd.hist.size))
    stop = max(start, min(int(stop), macd.hist.size))
    hist = np.nan_to_num(macd.hist[start:stop], nan=0.0, posinf=0.0, neginf=0.0)
    artists.bars.set_verts(_hist_vertices(hist, config.body_width_ratio * 0.8, start))
    artists.bars.set_facecolors(_hist_colors(macd.hist[start:stop], config))
    x = np.arange(start, stop, dtype="float64")
    artists.dif_line.set_data(x, macd.dif[start:stop])
    artists.dea_line.set_data(x, macd.dea[start:stop])


# ----------------------------------------------------------------------
# RSI
# ----------------------------------------------------------------------
def draw_rsi(ax: Axes, rsi_data: dict[int, np.ndarray], config: ChartConfig) -> RsiArtists:
    """在副图上画 RSI 多周期曲线，纵轴固定 0~100。"""
    count = next(iter(rsi_data.values())).size if rsi_data else 0
    x = np.arange(count, dtype="float64")
    lines: dict[int, Line2D] = {}
    for position, (window, values) in enumerate(rsi_data.items()):
        color = RSI_COLORS[position % len(RSI_COLORS)]
        (line,) = ax.plot(x, values, color=color, linewidth=1.0, label=f"RSI{window}", zorder=3)
        lines[int(window)] = line

    for level in _RSI_GUIDE_LEVELS:
        ax.axhline(
            level,
            color=config.muted,
            linewidth=0.7,
            linestyle=(0, (4, 3)),
            alpha=0.45,
            zorder=1,
        )
    ax.set_xlim(-1.0, float(count) if count else 1.0)
    ax.set_ylim(0.0, 100.0)
    ax.yaxis.set_major_locator(MultipleLocator(25.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _p: f"{value:.0f}"))

    _style_indicator_axis(ax, config)
    periods = ",".join(str(window) for window in rsi_data) or "6,12,24"
    _legend(
        ax,
        list(lines.values()),
        [line.get_label() for line in lines.values()],
        config,
    )
    _param_label(ax, f"RSI({periods})", config)
    return RsiArtists(lines=lines)


def update_rsi(
    artists: RsiArtists,
    rsi_data: dict[int, np.ndarray],
    start: int,
    stop: int,
) -> None:
    """把 RSI 曲线改成只包含 ``[start, stop)``。"""
    length = next(iter(rsi_data.values())).size if rsi_data else 0
    start = max(0, min(int(start), length))
    stop = max(start, min(int(stop), length))
    x = np.arange(start, stop, dtype="float64")
    for window, line in artists.lines.items():
        values = rsi_data.get(window)
        if values is None or values.size < stop:
            line.set_data(x, np.full(stop - start, np.nan))
        else:
            line.set_data(x, values[start:stop])


# ----------------------------------------------------------------------
# BOLL（叠加在主图）
# ----------------------------------------------------------------------
def boll_bounds(boll: BollData, start: int, stop: int) -> tuple[float, float] | None:
    """BOLL 上下轨在可视区间的取值范围；全为空时返回 ``None``。"""
    parts = [boll.upper[start:stop], boll.lower[start:stop]]
    values = np.concatenate(parts)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def draw_boll(ax: Axes, boll: BollData, config: ChartConfig) -> BollArtists:
    """在主图上叠加 BOLL 中轨与上下轨（zorder 低于蜡烛，不遮挡价格）。"""
    count = boll.middle.size
    x = np.arange(count, dtype="float64")
    (lower,) = ax.plot(
        x, boll.lower, color=_BOLL_BAND_COLOR, linewidth=0.9, alpha=0.85, zorder=1
    )
    (middle,) = ax.plot(
        x, boll.middle, color=_BOLL_MID_COLOR, linewidth=1.0, alpha=0.9, zorder=1
    )
    (upper,) = ax.plot(
        x, boll.upper, color=_BOLL_BAND_COLOR, linewidth=0.9, alpha=0.85, zorder=1
    )
    window, multiple = config.boll_window, config.boll_k
    ax.text(
        0.997,
        0.955,
        f"BOLL({window},{multiple:g})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=config.muted,
        zorder=6,
    )
    return BollArtists(middle=middle, upper=upper, lower=lower)


def update_boll(artists: BollArtists, boll: BollData, start: int, stop: int) -> None:
    """把 BOLL 三条轨道改成只包含 ``[start, stop)``。"""
    size = boll.middle.size
    start = max(0, min(int(start), size))
    stop = max(start, min(int(stop), size))
    x = np.arange(start, stop, dtype="float64")
    artists.middle.set_data(x, boll.middle[start:stop])
    artists.upper.set_data(x, boll.upper[start:stop])
    artists.lower.set_data(x, boll.lower[start:stop])
