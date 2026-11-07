"""二级数据计算（设计文档 3.2.3）。

支持三种可扩展的计算方法：

* ``pct_change``  变化率 =（当前值 - 前一有效值）/ 前一有效值 × 100%
* ``diff``        差分 = 当前值 - 前一有效值
* ``log_return``  对数收益率 = ln(当前值 / 前一有效值)

约定：

* "前一有效值" 指前一个非缺失值，中间缺失的记录不会中断计算链；
* 当前值或前一有效值缺失时结果保持 ``NaN``；
* 前一有效值为 0 时不参与计算（结果为 ``NaN``，绝不做错误除法）；
* 计算结果沿用原始日期索引，首条记录必然为空。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.constants import DerivedMethod
from core.exceptions import ProcessingError

__all__ = ["compute_derived", "previous_valid"]


def previous_valid(series: pd.Series) -> pd.Series:
    """返回「前一有效值」序列：先按有效值前向填充，再整体后移一格。"""
    return series.astype("float64").ffill().shift(1)


def _pct_change(series: pd.Series) -> pd.Series:
    previous = previous_valid(series)
    current = series.astype("float64")
    result = (current - previous) / previous * 100.0
    return result.where(previous.notna() & (previous != 0))


def _diff(series: pd.Series) -> pd.Series:
    previous = previous_valid(series)
    return (series.astype("float64") - previous).where(previous.notna())


def _log_return(series: pd.Series) -> pd.Series:
    previous = previous_valid(series)
    current = series.astype("float64")
    valid = previous.notna() & (previous > 0) & (current > 0)
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result[valid] = np.log(current[valid] / previous[valid])
    return result


_METHODS = {
    DerivedMethod.PCT_CHANGE: _pct_change,
    DerivedMethod.DIFF: _diff,
    DerivedMethod.LOG_RETURN: _log_return,
}


def compute_derived(series: pd.Series, method: DerivedMethod | str) -> pd.Series:
    """按 ``method`` 计算二级数据，结果与原序列同索引。"""
    try:
        resolved = DerivedMethod(method)
    except ValueError as exc:  # pragma: no cover - 参数错误属于编程错误
        raise ProcessingError(f"未知的二级数据计算方法：{method}") from exc
    if not isinstance(series, pd.Series):
        raise ProcessingError("二级数据计算需要 pandas.Series 输入")
    return _METHODS[resolved](series)
