"""导入数据的一致性校验。

对应设计文档 3.1.4 的校验项：能否解析、日期合法性、重复日期、非法字符、
缺失值、负价格、最高/最低价与开盘/收盘价的逻辑冲突。

**校验只负责报告问题，不修改数据**——异常记录保留在数据集中，
由用户在预览界面判断是否继续使用，这符合"异常数据不得直接删除"的要求。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.constants import (
    FIELD_CLOSE,
    FIELD_HIGH,
    FIELD_LOW,
    FIELD_OPEN,
    FIELD_VOLUME,
    PRICE_FIELDS,
    STANDARD_FIELDS,
)
from core.data_models import DataIssue, IssueSeverity, field_label

__all__ = ["validate_frame", "MISSING_VALUE_TOLERANCE"]

#: 浮点比较容差，避免因导出精度导致的假报警。
MISSING_VALUE_TOLERANCE = 1e-9


def _sample_dates(index: pd.DatetimeIndex, mask: np.ndarray, limit: int = 3) -> tuple[str, ...]:
    hits = [str(stamp.date()) for stamp, flag in zip(index, mask) if flag]
    return tuple(hits[:limit])


def validate_frame(frame: pd.DataFrame) -> tuple[DataIssue, ...]:
    """检查行情数据中的异常，返回按严重程度排序的问题清单。"""
    issues: list[DataIssue] = []
    index = frame.index

    # ---- 负价格 -------------------------------------------------------
    for col in PRICE_FIELDS:
        if col not in frame.columns:
            continue
        values = frame[col].to_numpy(dtype=float, na_value=np.nan)
        mask = np.isfinite(values) & (values < 0)
        if mask.any():
            issues.append(
                DataIssue(
                    code="negative_price",
                    message=f"{field_label(col)}出现负值",
                    severity=IssueSeverity.ERROR,
                    field=col,
                    count=int(mask.sum()),
                    samples=_sample_dates(index, mask),
                )
            )

    # ---- 最高价 / 最低价 逻辑冲突 -------------------------------------
    if FIELD_HIGH in frame.columns and FIELD_LOW in frame.columns:
        high = frame[FIELD_HIGH].to_numpy(dtype=float, na_value=np.nan)
        low = frame[FIELD_LOW].to_numpy(dtype=float, na_value=np.nan)
        both = np.isfinite(high) & np.isfinite(low)
        mask = both & (high < low - MISSING_VALUE_TOLERANCE)
        if mask.any():
            issues.append(
                DataIssue(
                    code="high_below_low",
                    message="最高价低于最低价",
                    severity=IssueSeverity.ERROR,
                    field=FIELD_HIGH,
                    count=int(mask.sum()),
                    samples=_sample_dates(index, mask),
                )
            )

    # ---- 最高价低于开/收盘、最低价高于开/收盘 -------------------------
    for base, base_label, below in (
        (FIELD_HIGH, "最高价", "低于"),
        (FIELD_LOW, "最低价", "高于"),
    ):
        if base not in frame.columns:
            continue
        limit = frame[base].to_numpy(dtype=float, na_value=np.nan)
        for other in (FIELD_OPEN, FIELD_CLOSE):
            if other not in frame.columns:
                continue
            values = frame[other].to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(limit) & np.isfinite(values)
            if base == FIELD_HIGH:
                mask = valid & (limit < values - MISSING_VALUE_TOLERANCE)
            else:
                mask = valid & (limit > values + MISSING_VALUE_TOLERANCE)
            if mask.any():
                issues.append(
                    DataIssue(
                        code=f"{base}_conflict_{other}",
                        message=f"{base_label}{below}{field_label(other)}",
                        severity=IssueSeverity.WARNING,
                        field=base,
                        count=int(mask.sum()),
                        samples=_sample_dates(index, mask),
                    )
                )

    # ---- 成交量负值 ---------------------------------------------------
    if FIELD_VOLUME in frame.columns:
        volume = frame[FIELD_VOLUME].to_numpy(dtype=float, na_value=np.nan)
        mask = np.isfinite(volume) & (volume < 0)
        if mask.any():
            issues.append(
                DataIssue(
                    code="negative_volume",
                    message="成交量出现负值",
                    severity=IssueSeverity.WARNING,
                    field=FIELD_VOLUME,
                    count=int(mask.sum()),
                    samples=_sample_dates(index, mask),
                )
            )

    # ---- 收盘价非正 ---------------------------------------------------
    if FIELD_CLOSE in frame.columns:
        close = frame[FIELD_CLOSE].to_numpy(dtype=float, na_value=np.nan)
        mask = np.isfinite(close) & (close <= 0)
        if mask.any():
            issues.append(
                DataIssue(
                    code="non_positive_close",
                    message="收盘价不是正数",
                    severity=IssueSeverity.WARNING,
                    field=FIELD_CLOSE,
                    count=int(mask.sum()),
                    samples=_sample_dates(index, mask),
                )
            )

    # ---- 缺失值统计（保持 NaN，不填充） -------------------------------
    for col in STANDARD_FIELDS:
        if col not in frame.columns:
            continue
        missing = frame[col].isna().to_numpy()
        if not missing.any():
            continue
        count = int(missing.sum())
        total = len(frame)
        severity = IssueSeverity.ERROR if count == total else IssueSeverity.WARNING
        issues.append(
            DataIssue(
                code="missing_values",
                message=f"{field_label(col)}存在缺失值（保留为空，未填充）",
                severity=severity,
                field=col,
                count=count,
                samples=_sample_dates(index, missing),
            )
        )

    return tuple(_sort_issues(issues))


def _sort_issues(issues: Iterable[DataIssue]) -> list[DataIssue]:
    order = {IssueSeverity.ERROR: 0, IssueSeverity.WARNING: 1, IssueSeverity.INFO: 2}
    return sorted(issues, key=lambda issue: (order.get(issue.severity, 9), issue.code))
