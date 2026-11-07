"""数据处理模块（第三阶段）。

对外提供 :class:`~modules.data_processing.processor.DataProcessor`：
按 :class:`~core.data_models.ProcessingConfig` 完成数据选择、二级数据计算、
归一化与时间对齐，产出 :class:`~core.data_models.ProcessedDataset`。
"""

from __future__ import annotations

from modules.data_processing.alignment import (
    alignment_label,
    apply_alignment,
    index_axis_label,
)
from modules.data_processing.derived_data import compute_derived, previous_valid
from modules.data_processing.normalization import NormalizationOutcome, normalize_series
from modules.data_processing.processor import DataProcessor, resolve_data_kind
from modules.data_processing.selector import numeric_fields, select_frame

__all__ = [
    "DataProcessor",
    "resolve_data_kind",
    "compute_derived",
    "previous_valid",
    "normalize_series",
    "NormalizationOutcome",
    "apply_alignment",
    "alignment_label",
    "index_axis_label",
    "numeric_fields",
    "select_frame",
]
