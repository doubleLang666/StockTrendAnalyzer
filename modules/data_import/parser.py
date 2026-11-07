"""TXT 行情文件解析。

解析流程严格对应设计文档 3.1.3：

1. 定位表头（能在同一行里映射出至少两个标准字段，且必须包含日期）；
2. 识别标题行（表头之前最后一个非空行，形如 ``创业板指 (399006)``）；
3. 逐行转成标准字段，日期非法则丢弃该行并记录，数值非法则记为 ``NaN``；
4. 按日期升序排序、检查重复日期，最后交给 :mod:`validation` 复核逻辑关系。

解析器只产出"数据 + 问题清单"，不擅自修正、删除或填充异常值。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from core.constants import DATE_FORMATS, DELIMITER_CANDIDATES, FIELD_DATE, MISSING_TOKENS, STANDARD_FIELDS
from core.data_models import DataIssue, IssueSeverity
from core.exceptions import ParseError, ValidationError
from modules.data_import.dialect import detect_delimiter, split_lines, split_row
from modules.data_import.field_mapping import FieldMapping, build_mapping, parse_title

__all__ = ["ParsedTable", "parse_table", "parse_date", "parse_number", "find_header"]

#: 表头搜索的最大行数（券商文件通常前几行就是标题与表头）。
_HEADER_SEARCH_LIMIT = 80

#: 单个问题最多保留的样例条数，避免报告过长。
_MAX_SAMPLES = 3


def parse_date(text: str) -> pd.Timestamp | None:
    """按 :data:`core.constants.DATE_FORMATS` 逐种格式解析日期，失败返回 ``None``。"""
    value = str(text).strip()
    if not value:
        return None
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(value, fmt))
        except ValueError:
            continue
    # 兜底：交给 pandas 处理 "2010-06-01T00:00:00" 之类的变体。
    try:
        parsed = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(parsed) else parsed


def parse_number(text: str) -> float | None:
    """把单元格文本转成浮点数；缺失标记与非法内容都返回 ``None``（后续记 NaN）。"""
    value = str(text).strip().replace(",", "").replace("，", "")
    if value.endswith("%"):
        value = value[:-1]
    if value.lower() in MISSING_TOKENS:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _HeaderCandidate:
    line_index: int
    delimiter: str
    headers: tuple[str, ...]
    mapping: FieldMapping


@dataclass(frozen=True, slots=True)
class ParsedTable:
    """解析结果：数据表 + 识别信息 + 问题清单。"""

    frame: pd.DataFrame
    mapping: FieldMapping
    headers: tuple[str, ...]
    delimiter: str
    header_line: int
    title_name: str
    title_code: str
    total_rows: int
    imported_rows: int
    dropped_rows: int
    issues: tuple[DataIssue, ...]
    elapsed_ms: float = 0.0


class _IssueCollector:
    """按 (编码, 字段) 聚合问题，避免逐行刷屏。"""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str | None], dict[str, object]] = {}
        self._order: list[tuple[str, str | None]] = []

    def add(
        self,
        code: str,
        message: str,
        severity: IssueSeverity = IssueSeverity.WARNING,
        field: str | None = None,
        sample: str | None = None,
    ) -> None:
        key = (code, field)
        item = self._items.get(key)
        if item is None:
            item = {
                "message": message,
                "severity": severity,
                "count": 0,
                "samples": [],
            }
            self._items[key] = item
            self._order.append(key)
        item["count"] = int(item["count"]) + 1  # type: ignore[arg-type]
        samples = item["samples"]
        if sample and len(samples) < _MAX_SAMPLES:  # type: ignore[arg-type]
            samples.append(sample)  # type: ignore[attr-defined]

    def issues(self) -> tuple[DataIssue, ...]:
        result: list[DataIssue] = []
        for key in self._order:
            code, field = key
            item = self._items[key]
            result.append(
                DataIssue(
                    code=code,
                    message=str(item["message"]),
                    severity=item["severity"],  # type: ignore[arg-type]
                    field=field,
                    count=int(item["count"]),  # type: ignore[arg-type]
                    samples=tuple(item["samples"]),  # type: ignore[arg-type]
                )
            )
        # 错误优先展示，其次警告、提示。
        order = {IssueSeverity.ERROR: 0, IssueSeverity.WARNING: 1, IssueSeverity.INFO: 2}
        result.sort(key=lambda issue: order.get(issue.severity, 9))
        return tuple(result)


def find_header(lines: Sequence[str], candidates: Sequence[str] | None = None) -> _HeaderCandidate:
    """在前若干行中找出最像表头的一行。"""
    delimiters = tuple(candidates or DELIMITER_CANDIDATES)
    best: _HeaderCandidate | None = None
    best_score = 0

    for index, line in enumerate(lines[:_HEADER_SEARCH_LIMIT]):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        delimiter = detect_delimiter(line, delimiters)
        headers = tuple(split_row(line, delimiter))
        if len(headers) < 2:
            continue
        mapping = build_mapping(headers)
        if not mapping.has_date:
            continue
        score = len(mapping.columns) * 10 + len(headers)
        if score > best_score:
            best = _HeaderCandidate(index, delimiter, headers, mapping)
            best_score = score

    if best is None:
        raise ValidationError(
            "未能识别表头",
            detail="前 %d 行中没有找到同时包含日期与行情字段的表头" % _HEADER_SEARCH_LIMIT,
        )
    return best


def _find_title(lines: Sequence[str], header_line: int) -> tuple[str, str]:
    """表头之前最后一个非空行视为标题行。"""
    for index in range(header_line - 1, -1, -1):
        text = lines[index].strip()
        if text:
            return parse_title(text)
    return "", ""


def _column_positions(headers: Sequence[str]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, header in enumerate(headers):
        positions.setdefault(header.strip(), index)
    return positions


def parse_table(
    text: str,
    *,
    delimiters: Sequence[str] | None = None,
    missing_tokens: Iterable[str] | None = None,
) -> ParsedTable:
    """解析整个文本内容。"""
    started = time.perf_counter()
    lines = split_lines(text)
    if not lines:
        raise ParseError("文件内容为空", detail="没有可解析的文本行")

    header = find_header(lines, delimiters)
    title_name, title_code = _find_title(lines, header.line_index)
    positions = _column_positions(header.headers)

    # 标准字段 -> 源行下标
    column_index: dict[str, int] = {}
    for field, source in header.mapping.columns.items():
        position = positions.get(source.strip())
        if position is not None:
            column_index[field] = position

    if FIELD_DATE not in column_index:
        raise ValidationError("表头中缺少日期列，无法解析行情数据")

    extra_missing = {str(item).strip().lower() for item in (missing_tokens or ())}
    collector = _IssueCollector()

    width = len(header.headers)
    dates: list[pd.Timestamp] = []
    records: dict[str, list[float]] = {field: [] for field in column_index if field != FIELD_DATE}
    total_rows = 0
    dropped_rows = 0

    for offset, line in enumerate(lines[header.line_index + 1 :], start=header.line_index + 2):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = split_row(line, header.delimiter)
        if not tokens:
            continue
        if len(tokens) == 1:
            # 尾部说明行（如 "数据来源:xxx"）直接跳过，不计入有效行。
            continue

        total_rows += 1
        if len(tokens) < width:
            tokens = tokens + [""] * (width - len(tokens))
        elif len(tokens) > width:
            collector.add(
                "row_too_wide",
                f"第 {offset} 行字段数（{len(tokens)}）多于表头（{width}），多余部分已忽略",
                IssueSeverity.WARNING,
                sample=f"第 {offset} 行",
            )
            tokens = tokens[:width]

        raw_date = tokens[column_index[FIELD_DATE]]
        stamp = parse_date(raw_date)
        if stamp is None:
            dropped_rows += 1
            collector.add(
                "invalid_date",
                "日期无法解析，该行已跳过",
                IssueSeverity.ERROR,
                field=FIELD_DATE,
                sample=f"第 {offset} 行：{raw_date!r}",
            )
            continue

        for field, position in column_index.items():
            if field == FIELD_DATE:
                continue
            raw_value = tokens[position]
            value = parse_number(raw_value)
            if value is None and raw_value.strip().lower() not in (MISSING_TOKENS | extra_missing):
                collector.add(
                    "invalid_number",
                    "数值无法解析，已按缺失值 NaN 处理",
                    IssueSeverity.WARNING,
                    field=field,
                    sample=f"第 {offset} 行：{raw_value!r}",
                )
            records[field].append(np.nan if value is None else value)

        dates.append(stamp)

    if not dates:
        raise ValidationError(
            "没有解析到任何有效数据行",
            detail=f"共扫描 {total_rows} 行，日期均无法识别",
        )

    index = pd.DatetimeIndex(dates, name=FIELD_DATE)
    frame = pd.DataFrame(records, index=index)
    frame = frame.loc[:, [name for name in STANDARD_FIELDS if name in frame.columns]]

    # 排序：设计文档要求统一按日期升序存放。
    if not frame.index.is_monotonic_increasing:
        collector.add(
            "unsorted",
            "原始数据未按日期升序排列，已重新排序",
            IssueSeverity.INFO,
        )
        frame = frame.sort_index(kind="mergesort")

    # 重复日期：保留首次出现，并如实反馈。
    if frame.index.has_duplicates:
        duplicated = frame.index.duplicated(keep="first")
        duplicate_count = int(duplicated.sum())
        sample_dates = ", ".join(str(item.date()) for item in frame.index[duplicated].unique()[:3])
        collector.add(
            "duplicate_date",
            "存在重复日期的记录，已保留首次出现的行",
            IssueSeverity.WARNING,
            field=FIELD_DATE,
            sample=sample_dates,
        )
        frame = frame.loc[~duplicated]
        dropped_rows += duplicate_count

    missing_fields = tuple(field for field in STANDARD_FIELDS if field not in frame.columns)
    if missing_fields:
        collector.add(
            "missing_fields",
            "源文件未提供以下字段：" + "、".join(
                {"open": "开盘价", "high": "最高价", "low": "最低价", "volume": "成交量",
                 "amount": "成交额", "turnover": "换手率"}.get(field, field)
                for field in missing_fields
            ),
            IssueSeverity.INFO,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ParsedTable(
        frame=frame,
        mapping=header.mapping,
        headers=header.headers,
        delimiter=header.delimiter,
        header_line=header.line_index,
        title_name=title_name,
        title_code=title_code,
        total_rows=total_rows,
        imported_rows=int(len(frame)),
        dropped_rows=dropped_rows,
        issues=collector.issues(),
        elapsed_ms=elapsed_ms,
    )
