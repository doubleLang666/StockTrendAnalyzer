"""数据归一化（设计文档 3.2.4）。

* 起始值归一化：区间内第一个有效数据作为基准，转换为 100；
* Min-Max 归一化：映射到配置区间（默认 0 ~ 1）。

两种方式都会返回 :class:`NormalizationOutcome`，其中记录了基准值、
实际使用的上下界等参数，便于写入处理结果的追溯信息。
原始数据不会被修改，函数只返回新序列。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.constants import DEFAULT_MIN_MAX_HIGH, DEFAULT_MIN_MAX_LOW, START_NORMALIZED_BASE, Normalization
from core.exceptions import ProcessingError

__all__ = ["NormalizationOutcome", "normalize_series"]


@dataclass(frozen=True, slots=True)
class NormalizationOutcome:
    """单列归一化结果及其参数记录。"""

    series: pd.Series
    method: Normalization
    base_value: float | None = None
    low: float | None = None
    high: float | None = None
    note: str | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def parameters(self) -> dict[str, float | str | None]:
        """可序列化的参数快照。"""
        return {
            "method": self.method.value,
            "base_value": self.base_value,
            "low": self.low,
            "high": self.high,
            "note": self.note,
            **self.extra,
        }


def _start_normalized(series: pd.Series, base: float) -> NormalizationOutcome:
    values = series.astype("float64")
    valid = values.dropna()
    if valid.empty:
        return NormalizationOutcome(series=values, method=Normalization.START)
    first = float(valid.iloc[0])
    if first == 0:
        raise ProcessingError("首个有效值为 0，无法进行起始值归一化", detail="请改用 Min-Max 归一化或排除该字段")
    return NormalizationOutcome(
        series=values / first * base,
        method=Normalization.START,
        base_value=first,
    )


def _min_max_normalized(
    series: pd.Series,
    low: float,
    high: float,
) -> NormalizationOutcome:
    values = series.astype("float64")
    valid = values.dropna()
    if valid.empty:
        return NormalizationOutcome(series=values, method=Normalization.MIN_MAX, low=low, high=high)
    data_low = float(valid.min())
    data_high = float(valid.max())
    span = data_high - data_low
    if span == 0:
        # 最大值与最小值相同：不得除零，统一映射到下界并记录说明。
        flattened = values.where(values.isna(), low)
        return NormalizationOutcome(
            series=flattened,
            method=Normalization.MIN_MAX,
            low=low,
            high=high,
            note="该字段在所选区间内取值恒定，已统一映射到下界",
            extra={"data_min": data_low, "data_max": data_high},
        )
    scaled = (values - data_low) / span * (high - low) + low
    return NormalizationOutcome(
        series=scaled,
        method=Normalization.MIN_MAX,
        low=low,
        high=high,
        extra={"data_min": data_low, "data_max": data_high},
    )


def normalize_series(
    series: pd.Series,
    method: Normalization | str,
    *,
    low: float = DEFAULT_MIN_MAX_LOW,
    high: float = DEFAULT_MIN_MAX_HIGH,
    base: float = START_NORMALIZED_BASE,
) -> NormalizationOutcome:
    """对单列数据做归一化，缺失值保持缺失。"""
    try:
        resolved = Normalization(method)
    except ValueError as exc:
        raise ProcessingError(f"未知的归一化方式：{method}") from exc
    if resolved is Normalization.NONE:
        return NormalizationOutcome(series=series.astype("float64"), method=Normalization.NONE)
    if resolved is Normalization.START:
        return _start_normalized(series, base)
    if high <= low:
        raise ProcessingError("Min-Max 归一化的上界必须大于下界")
    return _min_max_normalized(series, low, high)
