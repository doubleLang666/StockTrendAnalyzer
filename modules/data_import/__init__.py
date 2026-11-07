"""数据导入模块。

对外统一通过 :class:`~modules.data_import.importer.TxtImporter` 使用：

    >>> importer = TxtImporter(config)
    >>> result = importer.import_file("data/399006.txt")
    >>> result.dataset.name
    '创业板指'
"""

from __future__ import annotations

from modules.data_import.dialect import delimiter_label
from modules.data_import.encoding import decode_bytes, detect_encoding
from modules.data_import.field_mapping import FieldMapping, build_mapping
from modules.data_import.importer import ImportResult, TxtImporter
from modules.data_import.parser import ParsedTable, parse_table
from modules.data_import.validation import validate_frame

__all__ = [
    "TxtImporter",
    "ImportResult",
    "ParsedTable",
    "parse_table",
    "detect_encoding",
    "decode_bytes",
    "build_mapping",
    "FieldMapping",
    "validate_frame",
    "delimiter_label",
]
