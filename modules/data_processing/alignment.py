"""时间对齐（设计文档 3.2.5）。

* ``real_date``：保留真实交易日期作为索引；
* ``relative_day``：把区间内第一个交易日编为第 0 日，之后依次递增。

两种模式返回的 DataFrame 结构一致，仅索引不同，方便后续绘图与相似度计算。
"""

from __future__ import annotations

import pandas as pd

from core.constants import ALIGNMENT_LABELS, Alignment
from core.exceptions import ProcessingError

__all__ = ["apply_alignment", "alignment_label", "index_axis_label"]


def alignment_label(alignment: Alignment | str) -> str:
    return ALIGNMENT_LABELS.get(getattr(alignment, "value", alignment), str(alignment))


def index_axis_label(alignment: Alignment | str) -> str:
    """返回适合作为坐标轴标题的索引名称。"""
    resolved = Alignment(getattr(alignment, "value", alignment))
    return "交易日期" if resolved is Alignment.REAL_DATE else "相对交易日"


def apply_alignment(frame: pd.DataFrame, alignment: Alignment | str) -> pd.DataFrame:
    """按对齐方式重设索引，返回新 DataFrame（不修改入参）。"""
    if not isinstance(frame, pd.DataFrame):
        raise ProcessingError("时间对齐需要 pandas.DataFrame 输入")
    try:
        resolved = Alignment(getattr(alignment, "value", alignment))
    except ValueError as exc:
        raise ProcessingError(f"未知的对齐方式：{alignment}") from exc

    result = frame.copy(deep=True)
    if resolved is Alignment.REAL_DATE:
        result.index = pd.DatetimeIndex(frame.index, name="date")
        return result

    result.index = pd.RangeIndex(0, len(frame), name="relative_day")
    return result
