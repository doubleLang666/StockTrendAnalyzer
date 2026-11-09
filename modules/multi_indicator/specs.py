"""多指标联动分析：指标注册表（对应设计文档 2.1）。

指标分为两类：

* **原始指标**：直接取行情表里的列（开盘价、收盘价、最高价、最低价、成交量、换手率）；
* **派生指标**：由原始字段在本地计算（涨跌幅 = 收盘价的变化率）。

约定：

* 缺列的原始指标一律判为"不可用"，界面置灰并说明原因，绝不补 0 或插值；
* 派生指标只在源字段存在且区间内有效点足够时可用；
* 指标的计算只用本地历史数据，不涉及实时行情。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from core.constants import FIELD_LABELS, DerivedMethod
from core.exceptions import ProcessingError
from modules.data_processing.derived_data import compute_derived

__all__ = [
    "DEFAULT_INDICATOR_KEYS",
    "IndicatorSpec",
    "INDICATORS",
    "available_indicators",
    "get_indicator",
    "indicator_label",
    "indicator_map",
    "indicator_series",
    "select_default_indicators",
    "unavailable_reason",
]

#: 指标类型：决定尺度变换方式与坐标轴标签。
KIND_PRICE = "price"
KIND_VOLUME = "volume"
KIND_TURNOVER = "turnover"
KIND_RATE = "rate"


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    """一个可参与分析的行情指标。"""

    key: str
    label: str
    field: str
    kind: str
    description: str
    #: 派生指标的算法标识（``DerivedMethod`` 的值）；原始指标为 ``None``。
    derived: str | None = None

    @property
    def is_derived(self) -> bool:
        return self.derived is not None

    @property
    def unit_hint(self) -> str:
        """写入图注/结果的量纲说明。"""
        if self.kind == KIND_RATE:
            return "百分比（%）"
        if self.kind == KIND_VOLUME:
            return "股"
        if self.kind == KIND_TURNOVER:
            return "%"
        return "元"


INDICATORS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec("close", FIELD_LABELS["close"], "close", KIND_PRICE, "观察整体价格趋势"),
    IndicatorSpec("open", FIELD_LABELS["open"], "open", KIND_PRICE, "观察开盘价格变化"),
    IndicatorSpec("high", FIELD_LABELS["high"], "high", KIND_PRICE, "观察阶段价格高点"),
    IndicatorSpec("low", FIELD_LABELS["low"], "low", KIND_PRICE, "观察阶段价格低点"),
    IndicatorSpec("volume", FIELD_LABELS["volume"], "volume", KIND_VOLUME, "观察交易活跃程度"),
    IndicatorSpec("turnover", FIELD_LABELS["turnover"], "turnover", KIND_TURNOVER, "观察筹码交换活跃程度"),
    IndicatorSpec(
        "pct_change",
        "涨跌幅",
        "close",
        KIND_RATE,
        "观察每日价格变化（由收盘价计算）",
        derived=DerivedMethod.PCT_CHANGE.value,
    ),
)

#: 文档 2.1 的默认勾选顺序；实际只会勾选数据中真的存在的指标。
DEFAULT_INDICATOR_KEYS: tuple[str, ...] = ("close", "volume", "turnover", "pct_change")

_MAP: dict[str, IndicatorSpec] = {spec.key: spec for spec in INDICATORS}


def indicator_map() -> Mapping[str, IndicatorSpec]:
    """返回 ``key -> IndicatorSpec`` 的只读映射。"""
    return _MAP


def get_indicator(key: str) -> IndicatorSpec:
    """按 ``key`` 取指标定义；未知 key 抛 :class:`ProcessingError`。"""
    try:
        return _MAP[key]
    except KeyError as exc:
        raise ProcessingError(f"未知的行情指标：{key}") from exc


def indicator_label(key: str) -> str:
    spec = _MAP.get(key)
    return spec.label if spec is not None else FIELD_LABELS.get(key, key)


def unavailable_reason(spec: IndicatorSpec, columns: Iterable[str]) -> str | None:
    """返回该指标在给定列集合下不可用的原因；可用时返回 ``None``。"""
    available = {str(column).lower() for column in columns}
    if spec.is_derived:
        if spec.field not in available:
            return f"缺少源字段「{FIELD_LABELS.get(spec.field, spec.field)}」，无法计算{spec.label}"
        return None
    if spec.field not in available:
        return f"源数据没有「{spec.label}」字段"
    return None


def available_indicators(frame_or_columns: pd.DataFrame | Iterable[str]) -> tuple[IndicatorSpec, ...]:
    """返回当前数据中可参与分析的指标（保持注册顺序）。"""
    columns = frame_or_columns.columns if isinstance(frame_or_columns, pd.DataFrame) else frame_or_columns
    columns = [str(column) for column in columns]
    return tuple(spec for spec in INDICATORS if unavailable_reason(spec, columns) is None)


def select_default_indicators(specs: Iterable[IndicatorSpec]) -> tuple[str, ...]:
    """按文档 2.1 的默认列表挑出当前可用的指标。"""
    available = {spec.key for spec in specs}
    return tuple(key for key in DEFAULT_INDICATOR_KEYS if key in available)


def indicator_series(frame: pd.DataFrame, spec: IndicatorSpec) -> pd.Series:
    """取出指标对应的完整序列（派生指标在本地计算，返回与原索引对齐的新序列）。"""
    if not isinstance(frame, pd.DataFrame):
        raise ProcessingError("行情指标需要 pandas.DataFrame 输入")
    if spec.field not in frame.columns:
        raise ProcessingError(
            f"数据缺少「{FIELD_LABELS.get(spec.field, spec.field)}」字段，无法计算{spec.label}"
        )
    source = frame[spec.field].astype("float64")
    if not spec.is_derived:
        return source
    # 涨跌幅等派生指标必须先在完整序列上计算，切区间时首日才有前收基准。
    result = compute_derived(source, spec.derived)
    result.name = spec.key
    return result
