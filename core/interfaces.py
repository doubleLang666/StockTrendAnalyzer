"""模块对外接口定义（Protocol）。

UI 层与应用协调层只依赖这些协议，不依赖具体实现类，
从而满足设计文档 2.3「UI 不直接操作底层解析器或算法实现」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

import pandas as pd

from core.data_models import (
    MarketDataset,
    ProcessedDataset,
    ProcessingConfig,
    SearchResult,
    SimilarityConfig,
    SimilarityResult,
)

__all__ = [
    "DataImporter",
    "DataProcessor",
    "TrendService",
    "SimilarityService",
    "ResultExporter",
    "Algorithm",
]


@runtime_checkable
class DataImporter(Protocol):
    """数据导入模块接口（设计文档 4.4）。"""

    def load_file(self, file_path: str | Path) -> MarketDataset:
        """解析单个 TXT 行情文件并返回 :class:`MarketDataset`。"""
        ...

    def load_files(self, file_paths: Iterable[str | Path]) -> list[MarketDataset]:
        """批量导入；单个文件失败不应中断其余文件。"""
        ...


@runtime_checkable
class DataProcessor(Protocol):
    """数据处理模块接口（设计文档 3.2）。"""

    def process(self, dataset: MarketDataset, config: ProcessingConfig) -> ProcessedDataset:
        """按配置返回处理后数据，原数据集保持不变。"""
        ...


@runtime_checkable
class TrendService(Protocol):
    """趋势分析模块接口（设计文档 3.3）。"""

    def plot(
        self,
        datasets: Sequence[MarketDataset | ProcessedDataset],
        config: Any = None,
    ) -> Any:
        """生成 ``matplotlib.figure.Figure``，不直接操作任何窗口对象。"""
        ...

    def search_similar_periods(
        self,
        target_period: Any,
        dataset: MarketDataset | ProcessedDataset,
        config: SimilarityConfig,
    ) -> list[SearchResult]:
        """以目标区间长度滑动检索历史相似区间。"""
        ...


@runtime_checkable
class SimilarityService(Protocol):
    """相似度分析模块接口（设计文档 3.4）。"""

    @property
    def available_algorithms(self) -> tuple[str, ...]:
        """已注册的算法标识。"""
        ...

    def compare(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        algorithm: str = "pearson",
        **params: Any,
    ) -> SimilarityResult:
        """比较两条数值序列。"""
        ...


@runtime_checkable
class ResultExporter(Protocol):
    """分析结果导出接口（设计文档第六阶段）。"""

    def export(self, payload: Any, target_path: str | Path) -> Path:
        ...
