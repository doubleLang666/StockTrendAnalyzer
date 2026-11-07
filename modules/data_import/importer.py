"""TXT 行情导入服务。

对外提供两种粒度：

* ``load_file`` / ``load_files``：只返回 :class:`~core.data_models.MarketDataset`，
  满足 :class:`~core.interfaces.DataImporter` 协议，供后续模块使用；
* ``import_file`` / ``import_files``：额外返回 :class:`ImportResult`，其中包含
  编码、字段映射、行数与问题清单，供界面展示。

整个过程不修改源文件，也不填充缺失值（缺失值保持 ``NaN``）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.config import AppConfig
from core.constants import FIELD_DATE, STANDARD_FIELDS
from core.data_models import DataIssue, ImportReport, MarketDataset
from core.exceptions import DataImportError, StockAnalyzerError
from modules.data_import.dialect import delimiter_label
from modules.data_import.encoding import decode_bytes, detect_encoding
from modules.data_import.parser import parse_table
from modules.data_import.validation import validate_frame

__all__ = ["TxtImporter", "ImportResult"]


@dataclass(frozen=True, slots=True)
class ImportResult:
    """一次导入的成果：数据集 + 元信息报告。"""

    dataset: MarketDataset
    report: ImportReport

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def issues(self) -> tuple[DataIssue, ...]:
        return self.report.issues

    @property
    def has_errors(self) -> bool:
        return self.report.has_errors


class TxtImporter:
    """券商 TXT 行情文件导入器。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def _encodings(self) -> tuple[str, ...] | None:
        if self._config is None:
            return None
        values = self._config.get("import.encodings")
        return tuple(values) if values else None

    def _delimiters(self) -> tuple[str, ...] | None:
        if self._config is None:
            return None
        values = self._config.get("import.delimiters")
        return tuple(values) if values else None

    def _missing_tokens(self) -> tuple[str, ...] | None:
        if self._config is None:
            return None
        values = self._config.get("import.missing_tokens")
        return tuple(str(item) for item in values) if values else None

    # ------------------------------------------------------------------
    # DataImporter 协议
    # ------------------------------------------------------------------
    def load_file(self, path: str | Path) -> MarketDataset:
        return self.import_file(path).dataset

    def load_files(self, paths: Iterable[str | Path]) -> tuple[MarketDataset, ...]:
        return tuple(result.dataset for result in self.import_files(paths))

    # ------------------------------------------------------------------
    # 详细导入
    # ------------------------------------------------------------------
    def import_file(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
    ) -> ImportResult:
        started = time.perf_counter()
        source = Path(path)
        raw = self._read_bytes(source)

        if encoding:
            try:
                text = decode_bytes(raw, encoding)
            except StockAnalyzerError as exc:
                raise DataImportError(
                    f"无法按指定编码读取 {source.name}",
                    detail=exc.user_message(),
                ) from exc
        else:
            encoding, text = detect_encoding(raw, self._encodings())

        parsed = parse_table(
            text,
            delimiters=self._delimiters(),
            missing_tokens=self._missing_tokens(),
        )
        issues = list(parsed.issues) + list(validate_frame(parsed.frame))

        name = parsed.title_name or source.stem
        notes = self._build_notes(source, encoding, parsed, issues)
        dataset = MarketDataset(
            name=name,
            frame=parsed.frame,
            source_path=source,
            source_encoding=encoding,
            declared_fields=(FIELD_DATE, *[col for col in STANDARD_FIELDS if col in parsed.frame.columns]),
            notes=notes,
        )

        report = ImportReport(
            source_path=source,
            dataset_name=name,
            code=parsed.title_code,
            encoding=encoding,
            delimiter_label=delimiter_label(parsed.delimiter),
            header_line=parsed.header_line + 1,
            mapping=dict(parsed.mapping.columns),
            missing_fields=parsed.mapping.missing,
            ignored_columns=parsed.mapping.ignored,
            total_rows=parsed.total_rows,
            imported_rows=parsed.imported_rows,
            dropped_rows=parsed.dropped_rows,
            issues=tuple(issues),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return ImportResult(dataset=dataset, report=report)

    def import_files(
        self,
        paths: Iterable[str | Path],
        *,
        encoding: str | None = None,
    ) -> tuple[ImportResult, ...]:
        """逐个导入；某个文件失败不会中断其余文件，由调用方决定如何处理。"""
        results: list[ImportResult] = []
        for path in paths:
            results.append(self.import_file(path, encoding=encoding))
        return tuple(results)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _read_bytes(source: Path) -> bytes:
        if not source.exists():
            raise DataImportError(f"文件不存在：{source.name}", detail=str(source))
        if not source.is_file():
            raise DataImportError(f"不是文件：{source.name}", detail=str(source))
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise DataImportError(
                f"无法读取文件：{source.name}",
                detail=str(exc),
            ) from exc
        if not raw:
            raise DataImportError(f"文件为空：{source.name}", detail=str(source))
        return raw

    @staticmethod
    def _build_notes(
        source: Path,
        encoding: str,
        parsed,
        issues: Sequence[DataIssue],
    ) -> tuple[str, ...]:
        notes = [
            f"来源文件：{source.name}",
            f"编码：{encoding}；分隔符：{delimiter_label(parsed.delimiter)}",
            f"表头位于第 {parsed.header_line + 1} 行，共导入 {parsed.imported_rows} 条记录",
        ]
        if parsed.mapping.ignored:
            notes.append("已忽略的源文件列：" + "、".join(parsed.mapping.ignored))
        if parsed.mapping.missing:
            notes.append("源文件缺少的字段：" + "、".join(parsed.mapping.missing))
        errors = sum(1 for issue in issues if issue.severity.value == "error")
        if errors:
            notes.append(f"存在 {errors} 类数据异常，请在校验结果中查看")
        return tuple(notes)
