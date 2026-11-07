"""核心层：统一数据结构、接口与常量。

本层不依赖项目内任何其他模块，也不依赖 GUI 框架。
"""

from __future__ import annotations

from core.data_models import (
    DataIssue,
    DerivedField,
    ImportReport,
    IssueSeverity,
    MarketDataset,
    ProcessedDataset,
    ProcessingConfig,
    SearchResult,
    SimilarityConfig,
    SimilarityResult,
    field_label,
)

__all__ = [
    "MarketDataset",
    "ProcessedDataset",
    "ProcessingConfig",
    "DerivedField",
    "SimilarityConfig",
    "SimilarityResult",
    "SearchResult",
    "DataIssue",
    "ImportReport",
    "IssueSeverity",
    "field_label",
]
