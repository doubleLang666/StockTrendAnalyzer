"""多指标联动分析：区间对齐与尺度变换（对应设计文档 2.2 / 4.1 / 4.2 / 4.3）。

职责：

* 把目标区间与对比区间各自切成指标序列；
* **按第 N 个交易日逐位对齐**（长度不同截断到较短一方，沿用现有系统的约定）；
* 按指标类型做尺度变换，使不同量纲的指标可以放在同一套评分体系里；
* 统计有效数据点比例，低于门槛的指标判为无效（不填 0、不插值）。

变换规则（同一指标的两段区间必须完全一致）：

============  ==========================================  ================
指标类型       变换                                         说明
============  ==========================================  ================
价格           ``x / 首个有效值 × 100``                     首个有效值不为正时退回原始数值
成交量/换手率   ``x / 首个有效值 × 100``                     首个有效值非正时改用 ``log1p`` 后再做基准变换
涨跌幅         ``100 + 涨跌幅(%)``                          100 表示当日不变，避免出现负值/零值基准
============  ==========================================  ================

原始数据一律不改，所有结果是新序列。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from core.exceptions import InsufficientDataError, ProcessingError
from modules.multi_indicator.specs import KIND_PRICE, KIND_RATE, IndicatorSpec

__all__ = [
    "DEFAULT_MIN_VALID_RATIO",
    "SeriesPair",
    "TRANSFORM_BASE100",
    "TRANSFORM_LOG_BASE100",
    "TRANSFORM_OFFSET100",
    "TRANSFORM_RAW",
    "align_pair",
    "prepare_pair",
    "transform_values",
]

#: 默认最低有效数据比例（文档 4.3 建议 80%）。
DEFAULT_MIN_VALID_RATIO = 0.8

TRANSFORM_BASE100 = "base100"
TRANSFORM_LOG_BASE100 = "log_base100"
TRANSFORM_OFFSET100 = "offset100"
TRANSFORM_RAW = "raw"

_TRANSFORM_LABELS = {
    TRANSFORM_BASE100: "首个有效值换算为 100",
    TRANSFORM_LOG_BASE100: "log1p 变换后首个有效值换算为 100",
    TRANSFORM_OFFSET100: "100 + 涨跌幅（100 表示当日不变）",
    TRANSFORM_RAW: "原始数值（无法做基准变换）",
}


def _first_finite(values: pd.Series) -> float | None:
    finite = values.to_numpy(dtype="float64")
    finite = finite[np.isfinite(finite)]
    return float(finite[0]) if finite.size else None


def transform_values(spec: IndicatorSpec, values: pd.Series) -> tuple[pd.Series, str, str]:
    """按指标类型做尺度变换，返回 ``(新序列, 变换标识, 说明)``。"""
    numeric = values.astype("float64")
    if spec.kind == KIND_RATE:
        return numeric + 100.0, TRANSFORM_OFFSET100, _TRANSFORM_LABELS[TRANSFORM_OFFSET100]

    base = _first_finite(numeric)
    if base is None:
        return numeric, TRANSFORM_RAW, "区间内没有有效数值"

    if spec.kind == KIND_PRICE:
        if base <= 0:
            return numeric, TRANSFORM_RAW, "首个有效价格不为正，未做基准变换"
        return numeric / base * 100.0, TRANSFORM_BASE100, _TRANSFORM_LABELS[TRANSFORM_BASE100]

    # 成交量 / 换手率：优先按首值做基准，首值非正时退回对数变换（文档 4.2）。
    if base > 0:
        return numeric / base * 100.0, TRANSFORM_BASE100, _TRANSFORM_LABELS[TRANSFORM_BASE100]
    positive = numeric[numeric > 0]
    if not positive.empty:
        # 首个正值做对数基准，零值保持 0（0 表示当日无成交，不会污染比例放大）。
        logged = np.log1p(numeric.where(numeric >= 0))
        logged_base = float(np.log1p(positive.iloc[0]))
        if logged_base > 0:
            return logged / logged_base * 100.0, TRANSFORM_LOG_BASE100, _TRANSFORM_LABELS[TRANSFORM_LOG_BASE100]
    return numeric, TRANSFORM_RAW, "首值非正且无法做对数变换，直接比较原始数值"


def _slice(series: pd.Series, start: Any, end: Any, label: str) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ProcessingError(
            "多指标分析需要真实交易日期索引",
            detail="相对交易日数据没有真实日期，无法按区间对齐",
        )
    frame_start = pd.Timestamp(start)
    frame_end = pd.Timestamp(end)
    if frame_start > frame_end:
        raise ProcessingError(f"{label}的起始日期晚于结束日期")
    sliced = series.loc[frame_start:frame_end]
    if sliced.empty:
        raise InsufficientDataError(
            f"{label}内没有可用数据",
            detail=f"{frame_start.date()} ~ {frame_end.date()}",
        )
    return sliced


def align_pair(target: pd.Series, compare: pd.Series) -> tuple[pd.Series, pd.Series]:
    """按第 N 个交易日逐位对齐，长度不同时截断到较短一方。"""
    length = min(len(target), len(compare))
    if length < 2:
        raise InsufficientDataError(
            "对比区间太短",
            detail="两段区间至少各需要 2 个交易日",
        )
    return (
        target.iloc[:length].reset_index(drop=True),
        compare.iloc[:length].reset_index(drop=True),
    )


@dataclass(frozen=True, slots=True)
class SeriesPair:
    """一个指标在两段区间上的对齐结果。"""

    spec: IndicatorSpec
    target: pd.Series
    compare: pd.Series
    target_raw: pd.Series
    compare_raw: pd.Series
    target_dates: pd.DatetimeIndex
    compare_dates: pd.DatetimeIndex
    transform: str
    note: str
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO

    @property
    def length(self) -> int:
        return int(len(self.target))

    @property
    def valid_mask(self) -> np.ndarray:
        left = self.target.to_numpy(dtype="float64")
        right = self.compare.to_numpy(dtype="float64")
        return np.isfinite(left) & np.isfinite(right)

    @property
    def valid_count(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def valid_ratio(self) -> float:
        return self.valid_count / self.length if self.length else 0.0

    @property
    def is_valid(self) -> bool:
        return self.length >= 2 and self.valid_ratio >= self.min_valid_ratio

    @property
    def reason(self) -> str:
        """无效原因（有效时为空字符串）。"""
        if self.length < 2:
            return "对齐后不足 2 个交易日"
        if self.valid_ratio < self.min_valid_ratio:
            return (
                f"有效数据点 {self.valid_count}/{self.length}（{self.valid_ratio:.0%}），"
                f"低于 {self.min_valid_ratio:.0%} 门槛"
            )
        return ""

    @property
    def transform_label(self) -> str:
        return _TRANSFORM_LABELS.get(self.transform, self.transform)

    def scored_values(self) -> tuple[np.ndarray, np.ndarray]:
        """返回只保留有效点的两段数值，供相似度算法使用。"""
        mask = self.valid_mask
        left = self.target.to_numpy(dtype="float64")[mask]
        right = self.compare.to_numpy(dtype="float64")[mask]
        return left, right

    def axis_label(self) -> str:
        if self.spec.kind == KIND_RATE:
            return f"{self.spec.label}（100 = 不变，%）"
        if self.transform == TRANSFORM_RAW:
            return f"{self.spec.label}（原始值）"
        return f"{self.spec.label}（首个有效值 = 100）"


def prepare_pair(
    frame: pd.DataFrame,
    spec: IndicatorSpec,
    target_start: date,
    target_end: date,
    compare_start: date,
    compare_end: date,
    *,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
) -> SeriesPair:
    """取出两段区间、对齐并做尺度变换，返回 :class:`SeriesPair`。

    结构性错误（没有该字段、区间为空）抛异常，由调用方按指标记录为无效行。
    """
    from modules.multi_indicator.specs import indicator_series

    if not 0.0 < min_valid_ratio <= 1.0:
        raise ProcessingError("最低有效数据比例必须位于 (0, 1] 区间")

    series = indicator_series(frame, spec)
    target_raw = _slice(series, target_start, target_end, "目标区间")
    compare_raw = _slice(series, compare_start, compare_end, "对比区间")

    target_transformed, transform, note = transform_values(spec, target_raw)
    compare_transformed, compare_kind, compare_note = transform_values(spec, compare_raw)
    if compare_kind != transform:  # 同一指标两段必须使用同一种变换
        transform = TRANSFORM_RAW
        note = "两段区间的首值性质不同，改用原始数值比较"
    if compare_note and compare_note != note:
        note = f"{note}；对比区间：{compare_note}"

    target_aligned, compare_aligned = align_pair(
        target_transformed.reset_index(drop=True),
        compare_transformed.reset_index(drop=True),
    )
    return SeriesPair(
        spec=spec,
        target=target_aligned,
        compare=compare_aligned,
        target_raw=target_raw.reset_index(drop=True),
        compare_raw=compare_raw.reset_index(drop=True),
        target_dates=pd.DatetimeIndex(target_raw.index[: len(target_aligned)]),
        compare_dates=pd.DatetimeIndex(compare_raw.index[: len(compare_aligned)]),
        transform=transform,
        note=note,
        min_valid_ratio=min_valid_ratio,
    )
