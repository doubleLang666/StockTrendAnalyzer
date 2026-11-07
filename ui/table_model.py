"""预览表格的数据模型。

把 :class:`~core.data_models.MarketDataset` 的 DataFrame 适配成 Qt 表格模型：
下标（交易日）作为行号，标准字段作为列，缺失值（NaN）显示为空白，
符合设计文档 3.1.5 "空值留白，不填充 0" 的要求。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from core.constants import (
    FIELD_AMOUNT,
    FIELD_CLOSE,
    FIELD_HIGH,
    FIELD_LOW,
    FIELD_OPEN,
    FIELD_TURNOVER,
    FIELD_VOLUME,
    STANDARD_FIELDS,
)
from core.data_models import field_label

__all__ = ["MarketTableModel", "ProcessedTableModel", "format_value"]

_PRICE_FIELDS = (FIELD_OPEN, FIELD_HIGH, FIELD_LOW, FIELD_CLOSE)
_COUNT_FIELDS = (FIELD_VOLUME, FIELD_AMOUNT)


def format_value(field: str, value: Any) -> str:
    """按字段类型格式化单元格文本；缺失值一律返回空字符串。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return ""
    if field in _PRICE_FIELDS:
        return f"{number:.2f}"
    if field in _COUNT_FIELDS:
        return f"{number:,.0f}"
    if field == FIELD_TURNOVER:
        return f"{number:.2f}"
    return f"{number:g}"


class MarketTableModel(QAbstractTableModel):
    """只读表格模型：行 = 交易日，列 = 标准字段。"""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame: pd.DataFrame | None = None
        self._columns: list[str] = []

    # ------------------------------------------------------------------
    # 数据装载
    # ------------------------------------------------------------------
    def set_frame(self, frame: pd.DataFrame | None) -> None:
        self.beginResetModel()
        if frame is None or frame.empty:
            self._frame = None
            self._columns = []
        else:
            self._frame = frame
            self._columns = [col for col in STANDARD_FIELDS if col in frame.columns]
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame | None:
        return self._frame

    # ------------------------------------------------------------------
    # QAbstractTableModel 接口
    # ------------------------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid() or self._frame is None:
            return 0
        return int(len(self._frame))

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid() or self._frame is None:
            return 0
        return len(self._columns)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if self._frame is None:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole and 0 <= section < len(self._columns):
                return field_label(self._columns[section])
            return None
        # 纵向表头即日期列（设计文档 3.1.5：行 = 日期，列 = 字段）
        if role == Qt.ItemDataRole.DisplayRole and 0 <= section < len(self._frame):
            return self._frame.index[section].strftime("%Y-%m-%d")
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if self._frame is None or not index.isValid():
            return None
        row, column = index.row(), index.column()
        if not (0 <= row < len(self._frame) and 0 <= column < len(self._columns)):
            return None

        field = self._columns[column]
        value = self._frame.iloc[row][field]
        if role == Qt.ItemDataRole.DisplayRole:
            return format_value(field, value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            text = format_value(field, value)
            date_text = self._frame.index[row].strftime("%Y-%m-%d")
            return f"{date_text}　{field_label(field)}：{text or '缺失（NaN）'}"
        return None


class ProcessedTableModel(QAbstractTableModel):
    """处理结果的只读表格模型。

    列可以是原始字段，也可以是二级数据（如 ``close_pct_change``），
    行索引既可能是真实交易日期，也可能是相对交易日序号。
    """

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame: pd.DataFrame | None = None
        self._columns: list[str] = []
        self._labels: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 数据装载
    # ------------------------------------------------------------------
    def set_data(
        self,
        frame: pd.DataFrame | None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.beginResetModel()
        self._labels = dict(labels or {})
        if frame is None or frame.empty:
            self._frame = None
            self._columns = []
        else:
            self._frame = frame
            self._columns = [str(col) for col in frame.columns]
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame | None:
        return self._frame

    def is_relative(self) -> bool:
        return isinstance(self._frame, pd.DataFrame) and not isinstance(
            self._frame.index, pd.DatetimeIndex
        )

    # ------------------------------------------------------------------
    # QAbstractTableModel 接口
    # ------------------------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid() or self._frame is None:
            return 0
        return int(len(self._frame))

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid() or self._frame is None:
            return 0
        return len(self._columns)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if self._frame is None or role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                name = self._columns[section]
                return self._labels.get(name, field_label(name))
            return None
        if not (0 <= section < len(self._frame)):
            return None
        index = self._frame.index[section]
        if isinstance(index, pd.Timestamp):
            return index.strftime("%Y-%m-%d")
        return f"第 {int(index)} 日"

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if self._frame is None or not index.isValid():
            return None
        row, column = index.row(), index.column()
        if not (0 <= row < len(self._frame) and 0 <= column < len(self._columns)):
            return None

        name = self._columns[column]
        value = self._frame.iloc[row][name]
        if role == Qt.ItemDataRole.DisplayRole:
            return _format_column_value(name, value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            text = _format_column_value(name, value)
            label = self._labels.get(name, field_label(name))
            return f"{label}：{text or '缺失（NaN）'}"
        return None


def _format_column_value(name: str, value: Any) -> str:
    """处理结果列的统一格式化：已知字段沿用行情规则，其余保留 4 位小数。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if name.split("_")[0] in STANDARD_FIELDS and "_" not in name:
        return format_value(name, value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return ""
    if name in _PRICE_FIELDS:
        return f"{number:.2f}"
    if name in _COUNT_FIELDS:
        return f"{number:,.0f}"
    return f"{number:.4f}"
