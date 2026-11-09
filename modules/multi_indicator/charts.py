"""多指标可视化的绘图后端（纯 Matplotlib，不依赖 PySide6）。

三种视图对应设计文档「2.2 多指标可视化」与「2.5 结果展示 C/D」：

* :func:`build_indicator_panels` —— 模式 A：每个指标一行独立子图，各自纵轴，
  每行都同时画出目标区间与对比区间两段，便于逐指标看形态；
* :func:`build_overlay_figure` —— 模式 B：所有指标按各自的变换口径归到 100 附近
  后叠加在同一坐标系里，只看形态不看绝对数值；
* :func:`build_radar_figure` —— 指标相似度雷达图，一眼看出短板在哪。

这里只负责画图，不做任何相似度计算；单位变换由
:mod:`modules.multi_indicator.preparation` 完成，页面把结果传进来即可。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from core.exceptions import InsufficientDataError
from modules.multi_indicator.preparation import SeriesPair

__all__ = [
    "COMPARE_COLOR",
    "INDICATOR_COLORS",
    "TARGET_COLOR",
    "IndicatorPanels",
    "OverlayFigure",
    "RadarFigure",
    "build_indicator_panels",
    "build_overlay_figure",
    "build_radar_figure",
    "build_score_radar",
]

TARGET_COLOR = "#1f5fd0"
COMPARE_COLOR = "#d97706"
INDICATOR_COLORS = (
    "#1f5fd0",
    "#1a7f4b",
    "#7c3aed",
    "#c8102e",
    "#0891b2",
    "#b45309",
    "#0f766e",
)
_BACKGROUND = "#ffffff"
_AXES_BACKGROUND = "#fbfcfd"
_GRID_COLOR = "#e3e7ec"
_TEXT_COLOR = "#3d4852"
_MUTED = "#7b8794"


@dataclass(frozen=True, slots=True)
class IndicatorPanels:
    """模式 A 的产物：每个指标一个子图。"""

    figure: Figure
    axes: tuple
    pairs: tuple[SeriesPair, ...]

    def __len__(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True, slots=True)
class OverlayFigure:
    """模式 B 的产物：所有指标叠加在同一坐标系。"""

    figure: Figure
    ax: object
    pairs: tuple[SeriesPair, ...]
    lines: dict[str, tuple[Line2D, Line2D]]


@dataclass(frozen=True, slots=True)
class RadarFigure:
    """雷达图的产物。"""

    figure: Figure
    ax: object
    labels: tuple[str, ...]
    values: tuple[float, ...]


def _style(ax, *, show_x: bool = True, ylabel: str = "") -> None:
    ax.set_facecolor(_AXES_BACKGROUND)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.tick_params(colors=_MUTED, labelsize=8, length=3)
    if show_x:
        ax.set_xlabel("相对交易日（第 N 个交易日）", color=_MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=_MUTED, fontsize=8)


def _line_color(pair: SeriesPair, index: int) -> str:
    return INDICATOR_COLORS[index % len(INDICATOR_COLORS)]


def build_indicator_panels(
    pairs,
    *,
    figure: Figure | None = None,
    title: str = "多指标趋势对比（模式 A：每个指标独立子图）",
) -> IndicatorPanels:
    """为每个指标画一行子图，目标与对比两段各一条线。"""
    items = tuple(pairs)
    if not items:
        raise InsufficientDataError("没有可绘制的指标", detail="请至少勾选一个有效指标后再绘图。")

    fig = figure if figure is not None else Figure(figsize=(9.2, 2.1 * len(items) + 0.9), dpi=100)
    fig.clear()
    fig.set_facecolor(_BACKGROUND)
    rows = len(items)
    axes = []
    for row, pair in enumerate(items):
        ax = fig.add_subplot(rows, 1, row + 1)
        x = np.arange(pair.length)
        target = pair.target.to_numpy(dtype=float)
        compare = pair.compare.to_numpy(dtype=float)
        ax.plot(x, target, color=_line_color(pair, row), linewidth=1.5, label="目标区间")
        ax.plot(
            x,
            compare,
            color=COMPARE_COLOR,
            linewidth=1.3,
            linestyle=(0, (5, 2)),
            label="对比区间",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        _style(ax, show_x=row == rows - 1, ylabel=pair.spec.label)
        ax.set_title(
            f"{pair.spec.label}　{pair.transform_label}",
            loc="left",
            fontsize=9,
            color=_TEXT_COLOR,
        )
        ymin, ymax = _finite_bounds(target, compare)
        if ymin is not None:
            pad = (ymax - ymin) * 0.08 or 1.0
            ax.set_ylim(ymin - pad, ymax + pad)
        axes.append(ax)

    handles = [
        Line2D([], [], color=_TEXT_COLOR, linewidth=1.6),
        Line2D([], [], color=_TEXT_COLOR, linewidth=1.3, linestyle=(0, (5, 2))),
    ]
    legend = fig.legend(
        handles,
        ["目标区间", "对比区间"],
        loc="upper right",
        fontsize=8,
        frameon=False,
        ncols=2,
        handlelength=1.8,
        bbox_to_anchor=(0.985, 0.995),
    )
    for text in legend.get_texts():
        text.set_color(_MUTED)

    if not title:
        fig.suptitle("")
    else:
        fig.suptitle(title, fontsize=11, color=_TEXT_COLOR, x=0.01, ha="left")
    margins = dict(left=0.115, right=0.985, top=0.90 if title else 0.98, bottom=0.115)
    fig.subplots_adjust(hspace=0.42, **margins)
    return IndicatorPanels(figure=fig, axes=tuple(axes), pairs=items)


def build_overlay_figure(
    pairs,
    *,
    figure: Figure | None = None,
    title: str = "多指标归一化叠加（模式 B：统一 100 起始基准）",
) -> OverlayFigure:
    """把所有指标画在同一坐标系里，只看形态差异。"""
    items = tuple(pairs)
    if not items:
        raise InsufficientDataError("没有可绘制的指标", detail="请至少勾选一个有效指标后再绘图。")

    fig = figure if figure is not None else Figure(figsize=(9.2, 4.6), dpi=100)
    fig.clear()
    fig.set_facecolor(_BACKGROUND)
    ax = fig.add_subplot(111)
    lines: dict[str, tuple[Line2D, Line2D]] = {}
    for index, pair in enumerate(items):
        color = _line_color(pair, index)
        x = np.arange(pair.length)
        target_line, = ax.plot(
            x,
            pair.target.to_numpy(dtype=float),
            color=color,
            linewidth=1.5,
            label=pair.spec.label,
        )
        compare_line, = ax.plot(
            x,
            pair.compare.to_numpy(dtype=float),
            color=color,
            linewidth=1.2,
            linestyle=(0, (5, 2)),
            alpha=0.85,
        )
        lines[pair.spec.key] = (target_line, compare_line)

    ax.axhline(100.0, color=COMPARE_COLOR, linewidth=0.9, linestyle=(0, (2, 3)), alpha=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    _style(ax, ylabel="归一化数值")
    legend = ax.legend(
        loc="upper left",
        fontsize=8,
        frameon=False,
        ncols=min(len(items), 4),
        handlelength=1.8,
        columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_color(_TEXT_COLOR)

    handles = [
        Line2D([], [], color=_TEXT_COLOR, linewidth=1.4),
        Line2D([], [], color=_TEXT_COLOR, linewidth=1.2, linestyle=(0, (5, 2))),
    ]
    extra = ax.legend(
        handles,
        ["实线 = 目标区间", "虚线 = 对比区间"],
        loc="lower left",
        fontsize=7.5,
        frameon=False,
        handlelength=1.8,
    )
    ax.add_artist(legend)
    for text in extra.get_texts():
        text.set_color(_MUTED)

    if title:
        fig.suptitle(title, fontsize=11, color=_TEXT_COLOR, x=0.01, ha="left")
    fig.text(
        0.012,
        0.012,
        "各指标按自身的变换口径对齐到 100 附近（虚线为 100 基准线），仅用于比较形态，不代表真实价格或成交量。",
        fontsize=7.5,
        color=_MUTED,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90 if title else 0.97, bottom=0.215)
    return OverlayFigure(figure=fig, ax=ax, pairs=items, lines=lines)


def build_radar_figure(
    scores,
    *,
    figure: Figure | None = None,
    mode: str = "combined",
    title: str = "指标相似度雷达图",
) -> RadarFigure:
    """把每个有效指标的一个 0–100 分数画成多边形。"""
    items = [score for score in scores if score.is_valid and score.score_for(mode) is not None]
    if len(items) < 3:
        raise InsufficientDataError(
            "雷达图至少需要 3 个有效指标",
            detail="勾选更多指标，或调整区间后重试（少于 3 个指标无法构成多边形）。",
        )

    labels = tuple(score.label for score in items)
    values = tuple(float(score.score_for(mode)) for score in items)
    count = len(items)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    closed_values = list(values) + [values[0]]

    fig = figure if figure is not None else Figure(figsize=(4.6, 4.6), dpi=100)
    fig.clear()
    fig.set_facecolor(_BACKGROUND)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(_AXES_BACKGROUND)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color=_TEXT_COLOR, fontsize=9)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color=_MUTED, fontsize=7.5)
    ax.set_rlabel_position(205.0)
    ax.grid(color=_GRID_COLOR, linewidth=0.9)
    ax.spines["polar"].set_color(_GRID_COLOR)
    ax.plot(closed_angles, closed_values, color=TARGET_COLOR, linewidth=1.8)
    ax.fill(closed_angles, closed_values, color=TARGET_COLOR, alpha=0.16)
    for angle, value in zip(angles, values):
        ax.scatter([angle], [value], color=TARGET_COLOR, s=18, zorder=3)
    for angle, label, value in zip(angles, labels, values):
        ax.annotate(
            f"{value:.0f}",
            xy=(angle, value),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.5,
            color=TARGET_COLOR,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
        )
    if title:
        fig.suptitle(title, fontsize=11, color=_TEXT_COLOR, x=0.02, ha="left")
    fig.subplots_adjust(left=0.12, right=0.92, top=0.86, bottom=0.08)
    return RadarFigure(figure=fig, ax=ax, labels=labels, values=values)


def build_score_radar(
    result,
    *,
    figure: Figure | None = None,
) -> RadarFigure:
    """便捷入口：直接拿 :class:`MultiIndicatorResult` 画雷达图。"""
    return build_radar_figure(result.scores, figure=figure, mode=result.score_mode)


def _finite_bounds(*arrays) -> tuple[float | None, float | None]:
    values = []
    for array in arrays:
        arr = np.asarray(array, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return None, None
    merged = np.concatenate(values)
    return float(np.min(merged)), float(np.max(merged))
