"""分析结果导出（设计文档第六阶段"分析结果导出"）。

本模块只负责把已经算好的结果写盘，不参与任何计算：

* :meth:`CsvExporter.export_processed` —— 处理后的数据表；
* :meth:`CsvExporter.export_search_results` —— 历史区间检索结果；
* :meth:`CsvExporter.export_similarity_results` —— 相似度对比结果。

统一采用 ``utf-8-sig``（带 BOM）编码，Excel 直接双击即可正确显示中文；
参数与来源信息写在表格前的 ``#`` 注释行里，既保证 CSV 仍可解析，
又满足设计文档"必须记录参数"的要求。
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from core.constants import ALGORITHM_LABELS, Alignment, DataKind
from core.data_models import (
    ProcessedDataset,
    SearchResult,
    SimilarityResult,
    field_label,
)
from core.exceptions import ExportError

__all__ = ["CsvExporter", "ExportPayload"]

ExportPayload = Any

_ENCODING = "utf-8-sig"
_ALIGNMENT_INDEX_LABELS = {
    Alignment.REAL_DATE.value: "日期",
    Alignment.RELATIVE_DAY.value: "相对交易日",
}


class CsvExporter:
    """把处理数据、检索结果、相似度结果导出为 CSV。"""

    #: 导出文件的编码（带 BOM 的 UTF-8，Excel 友好）。
    encoding = _ENCODING

    # -- 对外统一入口 -----------------------------------------------------
    def export(self, payload: ExportPayload, target_path: str | Path, **options: Any) -> Path:
        """按数据类型自动分派到具体导出方法。"""
        path = Path(target_path)
        if isinstance(payload, ProcessedDataset):
            return self.export_processed(payload, path)
        if isinstance(payload, pd.DataFrame):
            return self.export_frame(payload, path, **options)
        if isinstance(payload, SimilarityResult):
            return self.export_similarity_results((payload,), path, **options)
        if isinstance(payload, SearchResult):
            return self.export_search_results((payload,), path, **options)
        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
            items = tuple(payload)
            if not items:
                raise ExportError("没有可导出的结果")
            if all(isinstance(item, SearchResult) for item in items):
                return self.export_search_results(items, path, **options)
            if all(isinstance(item, SimilarityResult) for item in items):
                return self.export_similarity_results(items, path, **options)
        raise ExportError(
            "不支持导出这种数据",
            detail="支持：处理后的数据、历史区间检索结果、相似度对比结果",
        )

    # -- 处理后的数据 -----------------------------------------------------
    def export_frame(
        self,
        frame: pd.DataFrame,
        target_path: str | Path,
        *,
        index_label: str | None = None,
        metadata: Sequence[str] = (),
    ) -> Path:
        """导出一张数据表（行 = 日期，列 = 字段）。"""
        if frame is None or len(frame.columns) == 0:
            raise ExportError("没有可导出的数据列")
        label = index_label or self._index_label(frame)
        rows = [[self._index_text(index, label), *[self._number(value) for value in row]]
                for index, row in zip(frame.index, frame.to_numpy(), strict=False)]
        header = [label, *[self._column_text(column) for column in frame.columns]]
        return self._write(target_path, metadata, header, rows)

    def export_processed(
        self,
        dataset: ProcessedDataset,
        target_path: str | Path,
        *,
        metadata: Sequence[str] | None = None,
    ) -> Path:
        """导出处理结果，并在表头前写明来源、对齐方式、归一化等参数。"""
        config = dataset.config
        info = [
            f"数据集：{dataset.name}",
            f"对齐方式：{self._alignment_text(config.alignment)}",
            f"数据形态：{self._data_kind_text(config.data_kind)}",
            f"归一化：{config.normalization.value}",
            f"二级数据：{'、'.join(item.output_name for item in config.derived) or '无'}",
        ]
        if metadata:
            info.extend(metadata)
        return self.export_frame(
            dataset.frame,
            target_path,
            metadata=info,
        )

    # -- 检索结果 ---------------------------------------------------------
    def export_search_results(
        self,
        results: Sequence[SearchResult],
        target_path: str | Path,
        *,
        field: str | None = None,
        target: Any | None = None,
        algorithm: str | None = None,
        metadata: Sequence[str] = (),
    ) -> Path:
        """导出 Top N 历史相似区间。"""
        if not results:
            raise ExportError("没有可导出的检索结果")
        field_name = field or results[0].field
        info = [f"检索字段：{field_label(field_name)}"]
        if target is not None:
            info.append(f"目标区间：{self._segment_text(target)}")
        algo = algorithm or results[0].algorithm.value
        info.append(f"算法：{ALGORITHM_LABELS.get(algo, algo)}")
        info.extend(metadata)
        header = ["排名", "起始日期", "结束日期", "交易日", "字段", "算法", "原始值", "状态", "说明"]
        rows = []
        for item in results:
            raw = "" if item.result.raw_value is None else f"{item.result.raw_value:.6f}"
            rows.append(
                [
                    str(item.rank),
                    item.start_date.isoformat(),
                    item.end_date.isoformat(),
                    str(item.trading_days),
                    field_label(item.field),
                    ALGORITHM_LABELS.get(item.algorithm.value, item.algorithm.value),
                    raw,
                    item.result.status.value,
                    item.result.message,
                ]
            )
        return self._write(target_path, info, header, rows)

    # -- 相似度结果 -------------------------------------------------------
    def export_similarity_results(
        self,
        results: Sequence[SimilarityResult],
        target_path: str | Path,
        *,
        curve_a: str = "",
        curve_b: str = "",
        metadata: Sequence[str] = (),
    ) -> Path:
        """导出同一对曲线的多算法对比结果。"""
        if not results:
            raise ExportError("没有可导出的相似度结果")
        info = []
        if curve_a:
            info.append(f"曲线 A：{curve_a}")
        if curve_b:
            info.append(f"曲线 B：{curve_b}")
        info.extend(metadata)
        header = ["算法", "原始值", "有效点", "先归一化", "状态", "说明"]
        rows = []
        for item in results:
            raw = "" if item.raw_value is None else f"{item.raw_value:.6f}"
            rows.append(
                [
                    ALGORITHM_LABELS.get(item.algorithm.value, item.algorithm.value),
                    raw,
                    str(item.sample_size),
                    "是" if item.normalized else "否",
                    item.status.value,
                    item.message,
                ]
            )
        return self._write(target_path, info, header, rows)

    # -- 内部工具 ---------------------------------------------------------
    def _write(
        self,
        target_path: str | Path,
        metadata: Sequence[str],
        header: Sequence[str],
        rows: Iterable[Sequence[str]],
    ) -> Path:
        path = Path(target_path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        parent = path.parent
        if parent and not parent.exists():
            raise ExportError("目标目录不存在", detail=str(parent))
        try:
            with path.open("w", encoding=self.encoding, newline="") as handle:
                writer = csv.writer(handle)
                for line in metadata:
                    writer.writerow([f"# {line}"])
                writer.writerow(list(header))
                writer.writerows(rows)
        except OSError as exc:
            raise ExportError("写入文件失败", detail=str(exc)) from exc
        return path

    @staticmethod
    def _index_label(frame: pd.DataFrame) -> str:
        if isinstance(frame.index, pd.DatetimeIndex):
            return "日期"
        return "相对交易日"

    @staticmethod
    def _index_text(index: Any, label: str) -> str:
        if isinstance(index, pd.Timestamp):
            return index.strftime("%Y-%m-%d")
        return str(index)

    @staticmethod
    def _number(value: Any) -> str:
        if value is None:
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if pd.isna(number):
            return ""
        return f"{number:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _column_text(column: Any) -> str:
        name = str(column)
        if "." in name:
            base, _, method = name.partition(".")
            return f"{field_label(base)}（{method}）"
        return field_label(name)

    @staticmethod
    def _alignment_text(alignment: Alignment) -> str:
        return _ALIGNMENT_INDEX_LABELS.get(alignment.value, alignment.value)

    @staticmethod
    def _data_kind_text(kind: DataKind) -> str:
        return "绝对数据" if kind is DataKind.ABSOLUTE else "相对数据"

    @staticmethod
    def _segment_text(segment: Any) -> str:
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        if start is not None and end is not None:
            return f"{start} ~ {end}"
        return str(segment)
