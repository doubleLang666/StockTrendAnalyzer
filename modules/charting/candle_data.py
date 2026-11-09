"""K 线绘制所需的整理后数据（设计文档 7.2「数据要求」）。

职责边界：

* 只做"整理"：排序、去重、异常价格剔除、成交量可用性判断；
* **不修改原始行情数据**：所有变换都发生在副本上；
* 不伪造：缺字段直接报错，缺成交量返回 ``None``，绝不插值补值；
* 每一项自动修复都写入 :attr:`CandleData.notes`，界面必须原样告知用户。

整理后的数组以 ``numpy`` 形式保存，绘制与交互阶段只读，避免每次重绘再做
pandas 切片。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.constants import FIELD_CLOSE, FIELD_HIGH, FIELD_LOW, FIELD_OPEN, FIELD_VOLUME
from core.data_models import field_label
from core.exceptions import InsufficientDataError, ProcessingError

__all__ = ["CandleData", "prepare_candle_data", "PRICE_COLUMNS"]

PRICE_COLUMNS: tuple[str, ...] = (FIELD_OPEN, FIELD_HIGH, FIELD_LOW, FIELD_CLOSE)


@dataclass(frozen=True, slots=True)
class CandleData:
    """一张 K 线图所需的全部数值，日期索引保持升序且唯一。"""

    name: str
    dates: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray | None
    pct_change: np.ndarray
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.dates.size)

    @property
    def start(self) -> pd.Timestamp:
        return self.dates[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.dates[-1]

    @property
    def has_volume(self) -> bool:
        return self.volume is not None and self.volume.size == len(self)

    @property
    def price_span(self) -> float:
        """价格区间宽度，用于给十字星实体一个可见的最小高度。"""
        if len(self) == 0:
            return 0.0
        span = float(np.nanmax(self.high) - np.nanmin(self.low))
        return span if span > 0 else max(abs(float(self.close[0])), 1.0)


def prepare_candle_data(
    frame: pd.DataFrame,
    name: str = "",
    *,
    require_ohlc: bool = True,
) -> CandleData:
    """把行情表整理成 :class:`CandleData`。

    参数
    ----
    frame:
        列名已标准化的行情表，索引必须是真实交易日期。
    name:
        股票/指数名称，用于图表标题。
    require_ohlc:
        为 ``True`` 时缺开高低收任一字段即报错（K 线不能靠收盘价伪造）。
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ProcessingError("K线图需要非空行情数据")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ProcessingError(
            "K线图需要真实交易日期索引",
            detail="相对交易日数据没有真实日期，无法确定每根K线对应的交易日",
        )

    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing and require_ohlc:
        raise ProcessingError(
            "数据缺少以下字段，无法绘制K线："
            + "、".join(field_label(column) for column in missing),
            detail="K线必须由开盘价、最高价、最低价、收盘价共同构成，不做推测绘制",
        )

    notes: list[str] = []
    columns = [column for column in (*PRICE_COLUMNS, FIELD_VOLUME) if column in frame.columns]
    work = frame.loc[:, columns].copy(deep=True)
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    if not work.index.is_monotonic_increasing:
        work = work.sort_index(kind="mergesort")
        notes.append("数据日期未按时间升序，已重新排序")

    duplicated = int(work.index.duplicated(keep="last").sum())
    if duplicated:
        work = work.loc[~work.index.duplicated(keep="last")]
        notes.append(f"发现 {duplicated} 个重复日期，已保留每个日期的最后一条记录")

    total = int(len(work))
    prices = work.loc[:, list(PRICE_COLUMNS)]
    invalid_missing = prices.isna().any(axis=1) | (prices <= 0).any(axis=1)
    invalid_logic = work[FIELD_HIGH] < work[FIELD_LOW]
    invalid = invalid_missing | invalid_logic
    dropped = int(invalid.sum())
    if dropped:
        parts: list[str] = []
        missing_count = int(invalid_missing.sum())
        logic_count = int(invalid_logic.sum())
        if missing_count:
            parts.append(f"价格缺失或非正 {missing_count} 条")
        if logic_count:
            parts.append(f"最高价低于最低价 {logic_count} 条")
        work = work.loc[~invalid]
        notes.append(
            f"已剔除异常记录 {dropped}/{total} 条（" + "、".join(parts) + "），不参与绘图"
        )
    if work.empty:
        raise InsufficientDataError("清洗后没有可用于绘制K线的数据")

    volume: np.ndarray | None = None
    if FIELD_VOLUME in work.columns:
        raw_volume = work[FIELD_VOLUME].astype("float64")
        raw_volume = raw_volume.mask(raw_volume < 0)
        if raw_volume.notna().any():
            volume = raw_volume.to_numpy(dtype="float64", copy=True)
        else:
            notes.append("该数据不含有效成交量，已隐藏成交量副图")
    else:
        notes.append("该数据不含成交量字段，已隐藏成交量副图")

    close = work[FIELD_CLOSE].to_numpy(dtype="float64", copy=True)
    previous = np.empty_like(close)
    previous[0] = np.nan
    previous[1:] = close[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = np.where(previous > 0, (close - previous) / previous * 100.0, np.nan)

    return CandleData(
        name=name or "行情数据",
        dates=pd.DatetimeIndex(work.index, name="date"),
        open=work[FIELD_OPEN].to_numpy(dtype="float64", copy=True),
        high=work[FIELD_HIGH].to_numpy(dtype="float64", copy=True),
        low=work[FIELD_LOW].to_numpy(dtype="float64", copy=True),
        close=close,
        volume=volume,
        pct_change=pct_change,
        notes=tuple(notes),
    )
