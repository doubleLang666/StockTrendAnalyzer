"""数据选择（设计文档 3.2.2）。

负责把用户在界面上勾选的字段、日期范围落到实际数据上：

* 字段为空时使用该数据的全部数值字段；
* 日期范围为空时使用数据自身的起止日期；
* 超出数据范围时按实际可用范围裁剪，并返回一条提示。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from core.constants import FIELD_DATE
from core.data_models import MarketDataset
from core.exceptions import InsufficientDataError, ProcessingError

__all__ = ["numeric_fields", "select_frame"]


def numeric_fields(dataset: MarketDataset) -> tuple[str, ...]:
    """返回数据集中可供分析的数值字段（按标准字段顺序）。"""
    return tuple(name for name in dataset.fields if name != FIELD_DATE)


def select_frame(
    dataset: MarketDataset,
    fields: tuple[str, ...] | list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """按字段与日期范围裁剪数据，返回 ``(frame, 提示信息)``。"""
    if not isinstance(dataset, MarketDataset):
        raise ProcessingError("数据选择需要 MarketDataset 输入")

    notes: list[str] = []
    available = numeric_fields(dataset)
    if not available:
        raise InsufficientDataError("该数据没有可用于分析的数值字段")

    if fields:
        missing = [name for name in fields if name not in available]
        selected = tuple(name for name in fields if name in available)
        if not selected:
            raise ProcessingError("所选字段均不在该数据中", detail="、".join(map(str, fields)))
        if missing:
            notes.append("以下字段在该数据中不存在，已跳过：" + "、".join(missing))
    else:
        selected = available

    frame = dataset.frame
    if start_date is not None or end_date is not None:
        original = frame
        mask = pd.Series(True, index=frame.index)
        if start_date is not None:
            mask &= frame.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= frame.index <= pd.Timestamp(end_date)
        frame = frame.loc[mask]
        if frame.empty:
            raise InsufficientDataError(
                "所选日期范围内没有数据",
                detail=f"{start_date} ~ {end_date}",
            )
        if len(frame) != len(original):
            notes.append(f"已按日期范围裁剪，保留 {len(frame)} / {len(original)} 条记录")

    selected_frame = frame.loc[:, list(selected)].copy(deep=True)
    return selected_frame, tuple(notes)
