"""曲线绘制（设计文档 3.3.2）。

本模块只负责把已经准备好的数值序列画成 ``matplotlib.figure.Figure``，
不接触任何 Qt 窗口对象；界面层拿到 Figure 后再包进画布。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd
from matplotlib.figure import Figure

from core.exceptions import ProcessingError

__all__ = ["CurveData", "PlotRequest", "build_figure"]


@dataclass(frozen=True, slots=True)
class CurveData:
    """一条待绘制的曲线。"""

    label: str
    values: pd.Series
    style: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlotRequest:
    """一张图表的完整描述。"""

    curves: tuple[CurveData, ...]
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    legend: bool = True


def build_figure(request: PlotRequest, *, figure: Figure | None = None) -> Figure:
    """按请求绘制曲线，返回（或复用）Figure。

    传入 ``figure`` 时会先清空，便于界面反复重绘而不累积子图。
    """
    if not request.curves:
        raise ProcessingError("没有可绘制的曲线")

    fig = figure or Figure(figsize=(7.2, 4.6))
    fig.clear()
    axes = fig.add_subplot(111)

    has_date_axis = False
    for index, curve in enumerate(request.curves):
        if not isinstance(curve.values, pd.Series):
            raise ProcessingError(f"曲线 {curve.label!r} 不是 pandas.Series")
        if curve.values.empty:
            continue
        has_date_axis = has_date_axis or isinstance(curve.values.index, pd.DatetimeIndex)
        style = dict(curve.style)
        style.setdefault("label", curve.label)
        style.setdefault("linewidth", 1.6)
        if index < 10:
            style.setdefault("color", f"C{index}")
        axes.plot(curve.values.index, curve.values.to_numpy(dtype="float64"), **style)

    if not axes.get_lines():
        raise ProcessingError("没有可绘制的有效数据")

    if request.title:
        axes.set_title(request.title)
    if request.x_label:
        axes.set_xlabel(request.x_label)
    if request.y_label:
        axes.set_ylabel(request.y_label)
    if has_date_axis:
        fig.autofmt_xdate(rotation=0, ha="center")
    if request.legend and len(axes.get_lines()) > 1:
        axes.legend(loc="best", fontsize=9)
    axes.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
