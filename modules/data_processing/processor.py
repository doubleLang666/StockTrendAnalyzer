"""数据处理服务（设计文档 3.2）。

按 :class:`~core.data_models.ProcessingConfig` 编排完整流程：

1. 数据选择：字段 + 日期范围；
2. 二级数据计算（可选）：变化率 / 差分 / 对数收益率；
3. 归一化（可选）：起始值归一化 / Min-Max；
4. 时间对齐：真实交易日期 / 相对交易日；
5. 生成 :class:`~core.data_models.ProcessedDataset`，并把每一步的参数写入追溯信息。

原始 :class:`~core.data_models.MarketDataset` 始终保持不变。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

import pandas as pd

from app.config import AppConfig
from core.constants import (
    DATA_KIND_LABELS,
    DEFAULT_MIN_MAX_HIGH,
    DEFAULT_MIN_MAX_LOW,
    DataKind,
    Normalization,
)
from core.data_models import (
    DerivedField,
    MarketDataset,
    ProcessedDataset,
    ProcessingConfig,
    field_label,
)
from core.exceptions import InsufficientDataError, ProcessingError
from modules.data_processing.alignment import apply_alignment, index_axis_label
from modules.data_processing.derived_data import compute_derived
from modules.data_processing.normalization import normalize_series
from modules.data_processing.selector import select_frame

__all__ = ["DataProcessor", "resolve_data_kind"]


def resolve_data_kind(config: ProcessingConfig) -> DataKind:
    """按配置推导数据形态：做了归一化或计算了二级数据即为相对数据。"""
    if config.normalization is not Normalization.NONE or config.derived:
        return DataKind.RELATIVE
    return config.data_kind


class DataProcessor:
    """``DataProcessor`` 协议实现：Dataset + Config -> ProcessedDataset。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # 协议实现
    # ------------------------------------------------------------------
    def process(self, dataset: MarketDataset, config: ProcessingConfig) -> ProcessedDataset:
        if not isinstance(dataset, MarketDataset):
            raise ProcessingError("数据处理需要 MarketDataset 输入")
        if not isinstance(config, ProcessingConfig):
            raise ProcessingError("数据处理需要 ProcessingConfig 输入")

        notes: list[str] = []
        frame, selection_notes = select_frame(
            dataset,
            config.fields,
            config.start_date,
            config.end_date,
        )
        notes.extend(selection_notes)
        if frame.empty:
            raise InsufficientDataError("所选范围内没有数据")

        source_fields = tuple(frame.columns)
        derived_frame, derived_notes = self._build_derived(frame, config.derived)
        notes.extend(derived_notes)

        normalized_frame, normalization_info, normalization_notes = self._normalize(frame, config)
        notes.extend(normalization_notes)

        value_frame = pd.concat([normalized_frame, derived_frame], axis=1)
        value_frame = value_frame.loc[:, [*source_fields, *derived_frame.columns]]

        aligned = apply_alignment(value_frame, config.alignment)
        resolved_config = replace(config, data_kind=resolve_data_kind(config))
        if config.data_kind is not resolved_config.data_kind:
            notes.append(
                f"配置声明为{DATA_KIND_LABELS[config.data_kind.value]}，但因已归一化或计算二级数据，"
                f"结果按{DATA_KIND_LABELS[resolved_config.data_kind.value]}标记"
            )

        provenance: dict[str, Any] = {
            "source_rows": len(dataset),
            "result_rows": len(aligned),
            "source_range": _range_text(dataset.start_date, dataset.end_date),
            "applied_range": _frame_range_text(aligned),
            "alignment_axis": index_axis_label(config.alignment),
            "fields": list(source_fields),
            "derived": [item.output_name for item in config.derived],
            "normalization": normalization_info,
            "notes": list(notes),
        }
        return ProcessedDataset(
            name=dataset.name,
            frame=aligned,
            config=resolved_config,
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # 步骤实现
    # ------------------------------------------------------------------
    @staticmethod
    def _build_derived(
        frame: pd.DataFrame,
        derived: tuple[DerivedField, ...],
    ) -> tuple[pd.DataFrame, list[str]]:
        notes: list[str] = []
        columns: dict[str, pd.Series] = {}
        for item in derived:
            if item.base_field not in frame.columns:
                notes.append(f"二级数据跳过：{field_label(item.base_field)} 不在所选字段中")
                continue
            columns[item.output_name] = compute_derived(frame[item.base_field], item.method)
        if not columns:
            return pd.DataFrame(index=frame.index), notes
        return pd.DataFrame(columns, index=frame.index), notes

    @staticmethod
    def _normalize(
        frame: pd.DataFrame,
        config: ProcessingConfig,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
        notes: list[str] = []
        if config.normalization is Normalization.NONE:
            return frame.copy(deep=True), {}, notes

        low = config.min_max_low if config.min_max_low is not None else DEFAULT_MIN_MAX_LOW
        high = config.min_max_high if config.min_max_high is not None else DEFAULT_MIN_MAX_HIGH
        columns: dict[str, pd.Series] = {}
        info: dict[str, dict[str, Any]] = {}
        for name in frame.columns:
            outcome = normalize_series(frame[name], config.normalization, low=low, high=high)
            columns[name] = outcome.series
            info[name] = outcome.parameters()
            if outcome.note:
                notes.append(f"{field_label(name)}：{outcome.note}")
        return pd.DataFrame(columns, index=frame.index), info, notes


def _range_text(start: date | None, end: date | None) -> str:
    if start is None or end is None:
        return "未知"
    return f"{start.isoformat()} ~ {end.isoformat()}"


def _frame_range_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "未知"
    if isinstance(frame.index, pd.DatetimeIndex):
        return f"{frame.index[0].date().isoformat()} ~ {frame.index[-1].date().isoformat()}"
    return f"第 {int(frame.index[0])} 日 ~ 第 {int(frame.index[-1])} 日（共 {len(frame)} 个交易日）"
