"""分隔符与行的拆分。

券商导出的 TXT 主要有三种形态：

* 制表符分隔（通达信、同花顺导出常见，字段用空格补齐）；
* 逗号或分号分隔（CSV 风格）；
* 纯空格对齐（固定宽度表格，无显式分隔符）。

本模块只负责"把一行拆成若干字段"，字段含义的识别交给
:mod:`modules.data_import.field_mapping`。
"""

from __future__ import annotations

import re
from typing import Sequence

from core.constants import DELIMITER_CANDIDATES

__all__ = [
    "WHITESPACE",
    "DELIMITER_LABELS",
    "split_lines",
    "detect_delimiter",
    "split_row",
    "delimiter_label",
]

#: 无显式分隔符时的标记（空格对齐）。
WHITESPACE = "whitespace"

DELIMITER_LABELS: dict[str, str] = {
    "\t": "制表符",
    ",": "逗号",
    ";": "分号",
    "|": "竖线",
    WHITESPACE: "空格对齐",
}

_MULTI_SPACE = re.compile(r"[ \u3000]{2,}")
_ANY_SPACE = re.compile(r"\s+")


def delimiter_label(delimiter: str) -> str:
    return DELIMITER_LABELS.get(delimiter, repr(delimiter))


def split_lines(text: str) -> list[str]:
    """按通用换行符拆行，并去掉行尾多余空白（不删除行内空格）。

    文件末尾常见的空行一律丢弃，避免被当成数据行统计。
    """
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip("\n").rstrip() for line in text.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def detect_delimiter(line: str, candidates: Sequence[str] | None = None) -> str:
    """根据表头行判断分隔符；没有显式分隔符时返回 :data:`WHITESPACE`。"""
    best = ""
    best_count = 0
    for candidate in candidates or DELIMITER_CANDIDATES:
        count = line.count(candidate)
        if count > best_count:
            best, best_count = candidate, count
    if best_count == 0:
        return WHITESPACE
    return best


def split_row(line: str, delimiter: str) -> list[str]:
    """拆分单行数据，结果去掉首尾空白并丢弃尾部空字段。"""
    stripped = line.strip()
    if not stripped:
        return []

    if delimiter == WHITESPACE or (delimiter != "\t" and delimiter not in stripped):
        parts = _MULTI_SPACE.split(stripped.replace("\t", "  "))
        if len(parts) <= 1:
            parts = _ANY_SPACE.split(stripped)
    else:
        parts = stripped.split(delimiter)

    result = [part.strip() for part in parts]
    while result and result[-1] == "":
        result.pop()
    return result
