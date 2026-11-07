"""表头识别与字段映射。

设计文档 3.1.2 要求"设计字段映射机制，避免将数据解析逻辑绑定到某一家券商"。
这里通过 :data:`core.constants.FIELD_ALIASES` 把各家表头统一映射到标准字段，
无法识别的列（通达信导出的 MA、VOL-TDX、MACD 等指标）记录为"已忽略列"，
既不影响导入，也不丢弃信息。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from core.constants import FIELD_ALIASES, FIELD_DATE, FIELD_LABELS, REQUIRED_FIELDS, STANDARD_FIELDS

__all__ = [
    "FieldMapping",
    "normalize_token",
    "match_field",
    "build_mapping",
    "parse_title",
    "TITLE_PATTERN",
]

#: 标题行形如：``创业板指 (399006)``、``上证指数 000001``。
TITLE_PATTERN = re.compile(
    r"^(?P<name>.*?)[\s\u3000]*[（(]?[\s\u3000]*(?P<code>\d{6})[\s\u3000]*[)）]?\s*$"
)

#: 表头常见修饰后缀，例如 "收盘价(元)"、"成交量(手)"。
_SUFFIX_SPLIT = re.compile(r"[（(\[【:：]", re.UNICODE)


def normalize_token(text: str) -> str:
    """表头归一化：去 BOM、去首尾空白、压缩内部空白、转小写。"""
    value = str(text).replace("\ufeff", "").strip().lower()
    return re.sub(r"\s+", " ", value)


def match_field(token: str) -> str | None:
    """把一个表头单元格映射为标准字段，未识别返回 ``None``。"""
    key = normalize_token(token)
    if not key:
        return None
    if key in FIELD_ALIASES:
        return FIELD_ALIASES[key]

    # 去掉 "(元)"、"（手）" 之类的单位后缀再试一次。
    base = normalize_token(_SUFFIX_SPLIT.split(key, 1)[0])
    if base and base in FIELD_ALIASES:
        return FIELD_ALIASES[base]
    return None


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """表头到标准字段的映射结果。"""

    #: 标准字段 -> 源文件列名。
    columns: dict[str, str]
    #: 源文件中存在但未识别的列。
    ignored: tuple[str, ...] = ()
    #: 标准字段中源文件缺失的部分（如缺少成交额、换手率）。
    missing: tuple[str, ...] = ()

    def source_of(self, field: str) -> str | None:
        return self.columns.get(field)

    @property
    def has_date(self) -> bool:
        return FIELD_DATE in self.columns

    def label_of(self, field: str) -> str:
        return FIELD_LABELS.get(field, field)


def build_mapping(headers: Sequence[str]) -> FieldMapping:
    """根据表头顺序构建字段映射。

    同一个标准字段被多列命中时保留第一列（多数券商文件会重复导出同名指标）。
    """
    columns: dict[str, str] = {}
    ignored: list[str] = []
    for header in headers:
        field = match_field(header)
        if field is None:
            if normalize_token(header):
                ignored.append(header.strip())
            continue
        columns.setdefault(field, header.strip())

    missing = tuple(field for field in STANDARD_FIELDS if field not in columns)
    return FieldMapping(columns=dict(columns), ignored=tuple(ignored), missing=missing)


def parse_title(line: str) -> tuple[str, str]:
    """从标题行提取数据名称与代码，识别失败时原样作为名称返回。

    返回 ``(名称, 代码)``，代码可能为空字符串。
    """
    text = str(line).replace("\ufeff", "").strip()
    if not text:
        return "", ""
    match = TITLE_PATTERN.match(text)
    if match:
        name = match.group("name").strip()
        code = match.group("code").strip()
        return (name or text), code
    return text, ""


def required_fields_present(mapping: FieldMapping) -> bool:
    """是否满足最低字段要求（日期 + 收盘价）。"""
    return all(field in mapping.columns for field in REQUIRED_FIELDS)
