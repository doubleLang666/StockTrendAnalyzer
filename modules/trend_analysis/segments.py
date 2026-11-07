"""多周期曲线比较（设计文档 3.3.3）。

把同一数据的多个历史区间转换成可以放在同一张图里比较的曲线：

* 每个区间独立切片，缺失值保持缺失；
* ``real_date`` 保留真实日期，``relative_day`` 把每个区间的起始日编为第 0 日；
* 支持原始数值、起始值归一化、Min-Max 归一化三种显示模式；
* 区间长度不同时**不会静默截断**：默认按最短区间截断并给出明确说明，
  也可以选择保持原长度（此时长度差异同样会写入说明）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from core.constants import (
    DEFAULT_MIN_MAX_HIGH,
    DEFAULT_MIN_MAX_LOW,
    ALIGNMENT_LABELS,
    NORMALIZATION_LABELS,
    Alignment,
    Normalization,
)
from core.exceptions import InsufficientDataError, ProcessingError
from modules.data_processing.alignment import apply_alignment
from modules.data_processing.normalization import normalize_series
from modules.trend_analysis.plotting import CurveData

__all__ = ["Segment", "ComparisonSegment", "PeriodComparison", "compare_periods"]


@dataclass(frozen=True, slots=True)
class Segment:
    """一个待比较的历史区间。"""

    start: date
    end: date
    label: str = ""

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ProcessingError("区间起始日期不能晚于结束日期")

    @property
    def display_label(self) -> str:
        return self.label or f"{self.start.isoformat()} ~ {self.end.isoformat()}"


@dataclass(frozen=True, slots=True)
class ComparisonSegment:
    """比较结果中的一条曲线。"""

    label: str
    start: date | None
    end: date | None
    values: pd.Series
    trading_days: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class PeriodComparison:
    """多周期比较的完整结果。"""

    field: str
    segments: tuple[ComparisonSegment, ...]
    alignment: Alignment
    normalization: Normalization
    notes: tuple[str, ...] = field(default_factory=tuple)

    def curves(self) -> tuple[CurveData, ...]:
        return tuple(CurveData(label=item.label, values=item.values) for item in self.segments)

    @property
    def length_summary(self) -> str:
        lengths = sorted({item.trading_days for item in self.segments})
        if len(lengths) == 1:
            return f"各区间均为 {lengths[0]} 个交易日"
        return "各区间交易日数：" + "、".join(str(item.trading_days) for item in self.segments)


def compare_periods(
    frame: pd.DataFrame,
    field: str,
    segments: tuple[Segment, ...] | list[Segment],
    *,
    normalization: Normalization | str = Normalization.NONE,
    alignment: Alignment | str = Alignment.REAL_DATE,
    truncate: bool = True,
    low: float = DEFAULT_MIN_MAX_LOW,
    high: float = DEFAULT_MIN_MAX_HIGH,
) -> PeriodComparison:
    """把多个历史区间转换成同一张图上的比较曲线。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ProcessingError("多周期比较需要非空数据")
    if field not in frame.columns:
        raise ProcessingError(f"数据中不存在字段：{field}")
    if not segments:
        raise ProcessingError("请至少选择一个比较区间")

    try:
        resolved_alignment = Alignment(getattr(alignment, "value", alignment))
        resolved_normalization = Normalization(getattr(normalization, "value", normalization))
    except ValueError as exc:  # pragma: no cover - 参数错误属于编程错误
        raise ProcessingError("时间对齐或归一化方式不合法") from exc

    notes: list[str] = []
    prepared: list[ComparisonSegment] = []
    for segment in segments:
        window = frame.loc[
            (frame.index >= pd.Timestamp(segment.start)) & (frame.index <= pd.Timestamp(segment.end))
        ]
        if window.empty:
            notes.append(f"{segment.display_label}：该区间没有数据，已跳过")
            continue
        if len(window) < 2:
            notes.append(f"{segment.display_label}：不足 2 个交易日，无法形成曲线，已跳过")
            continue
        prepared.append(
            ComparisonSegment(
                label=segment.display_label,
                start=window.index[0].date() if isinstance(window.index, pd.DatetimeIndex) else None,
                end=window.index[-1].date() if isinstance(window.index, pd.DatetimeIndex) else None,
                values=window[field].astype("float64"),
                trading_days=int(len(window)),
            )
        )

    if not prepared:
        raise InsufficientDataError("所选区间内没有可比较的数据")

    lengths = {item.trading_days for item in prepared}
    if len(lengths) > 1:
        if truncate:
            shortest = min(lengths)
            prepared = [
                ComparisonSegment(
                    label=item.label,
                    start=item.start,
                    end=item.values.index[shortest - 1].date()
                    if isinstance(item.values.index, pd.DatetimeIndex)
                    else item.end,
                    values=item.values.iloc[:shortest],
                    trading_days=shortest,
                    truncated=item.trading_days > shortest,
                )
                for item in prepared
            ]
            notes.append(
                f"各区间交易日数不一致（{len(lengths)} 种），已统一按最短的 {shortest} 个交易日截断，"
                "截断行为在此明确说明，不做静默处理"
            )
        else:
            notes.append(
                "各区间交易日数不一致，已按原长度绘制（长度差异见各曲线说明）"
            )

    adjusted: list[ComparisonSegment] = []
    for item in prepared:
        series = item.values
        if resolved_normalization is not Normalization.NONE:
            outcome = normalize_series(series, resolved_normalization, low=low, high=high)
            series = outcome.series
            if outcome.note:
                notes.append(f"{item.label}：{outcome.note}")
        aligned_frame = apply_alignment(series.to_frame(name=field), resolved_alignment)
        adjusted.append(
            ComparisonSegment(
                label=item.label,
                start=item.start,
                end=item.end,
                values=aligned_frame[field],
                trading_days=item.trading_days,
                truncated=item.truncated,
            )
        )

    notes.insert(
        0,
        f"对齐方式：{ALIGNMENT_LABELS[resolved_alignment.value]}；"
        f"显示模式：{NORMALIZATION_LABELS[resolved_normalization.value]}",
    )
    return PeriodComparison(
        field=field,
        segments=tuple(adjusted),
        alignment=resolved_alignment,
        normalization=resolved_normalization,
        notes=tuple(notes),
    )
