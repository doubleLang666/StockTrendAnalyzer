"""技术指标的本地计算（设计文档 3.3、7.3）。

所有指标都由本地历史数据计算，不使用行情文件里自带的 MA/MACD 列，
也不做任何插值。函数只接受并返回数值，不涉及绘图与 Qt，便于单独做数值对拍。

接口约定统一：输入序列 + 参数，输出与输入等长的 ``numpy`` 数组，前导位为 NaN；
数据不足时整列 NaN，由调用方决定是否提示，而不是抛错。

指标口径（与国内行情软件一致，代码里也写清楚，便于答辩时对拍）：

* ``EMA``：``alpha = 2 / (span + 1)``，以第一个有效值作为初值，递推计算；
* ``MACD``：``DIF = EMA(close, 12) - EMA(close, 26)``，``DEA = EMA(DIF, 9)``，
  柱状值取 ``2 * (DIF - DEA)``（国内软件的画法，不是 1 倍）；
* ``RSI``：涨跌幅分别做"通达信 SMA(X, N, 1)"平滑（等价于 alpha = 1/N 的递推均值），
  ``RSI = 100 * 平均涨幅 / (平均涨幅 + 平均跌幅)``；涨跌都为 0 时取中性值 50；
* ``BOLL``：中轨为 N 日简单均线，上下轨为 ``中轨 ± k * 总体标准差``（``ddof=0``）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.exceptions import ProcessingError
from modules.charting.candle_data import CandleData
from modules.charting.chart_config import clean_windows

__all__ = [
    "BollData",
    "MacdData",
    "boll",
    "build_boll",
    "build_macd",
    "build_moving_averages",
    "build_rsi",
    "build_volume_moving_averages",
    "ema",
    "macd",
    "require_minimum_bars",
    "rsi",
    "sma",
    "wilder_average",
]

DEFAULT_MACD_PARAMS = (12, 26, 9)
DEFAULT_RSI_WINDOWS = (6, 12, 24)
DEFAULT_BOLL_WINDOW = 20
DEFAULT_BOLL_K = 2.0


@dataclass(frozen=True, slots=True)
class MacdData:
    """MACD 三条线，长度均与 ``close`` 相同。"""

    dif: np.ndarray
    dea: np.ndarray
    hist: np.ndarray


@dataclass(frozen=True, slots=True)
class BollData:
    """BOLL 三条轨道，长度均与 ``close`` 相同。"""

    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray



def sma(values: np.ndarray | list[float], window: int) -> np.ndarray:
    """简单移动平均。

    * 前 ``window - 1`` 个位置为 NaN（数据不足，不做前向填充）；
    * 窗口内含缺失值则结果为 NaN，保证"缺数据就是缺数据"；
    * 数据长度不足窗口时整列 NaN，由调用方决定是否提示，而不是抛错。
    """
    if window <= 0:
        raise ValueError("均线窗口必须是正整数")
    array = np.asarray(values, dtype="float64").ravel()
    result = np.full(array.shape[0], np.nan, dtype="float64")
    if array.size < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(array, window)
    result[window - 1 :] = windows.mean(axis=1)
    return result


def build_moving_averages(
    candle: CandleData,
    windows: tuple[int, ...] | list[int],
) -> dict[int, np.ndarray]:
    """按收盘价一次算出全部均线，供绘制与交互复用（不在鼠标移动时重算）。"""
    cleaned = clean_windows(windows)
    return {window: sma(candle.close, window) for window in cleaned}


def build_volume_moving_averages(
    candle: CandleData,
    windows: tuple[int, ...] | list[int],
) -> dict[int, np.ndarray]:
    """成交量均线；没有成交量数据时返回空字典。"""
    if not candle.has_volume:
        return {}
    cleaned = clean_windows(windows)
    volume = candle.volume
    assert volume is not None  # for type checkers; has_volume 已保证
    return {window: sma(volume, window) for window in cleaned}


def require_minimum_bars(candle: CandleData, window: int) -> None:
    """绘制某条均线前的数据量检查（错误信息面向用户）。"""
    if len(candle) < window:
        raise ProcessingError(
            f"数据只有 {len(candle)} 个交易日，不足以计算 MA{window}",
            detail="请扩大日期范围，或关闭该条均线",
        )


def ema(values: np.ndarray | list[float], span: int) -> np.ndarray:
    """指数移动平均（``alpha = 2 / (span + 1)``，以第一个有效值为初值）。

    缺失值不参与计算：该位置输出 NaN，内部状态保持不变（不做插值填补）。
    """
    if span <= 0:
        raise ValueError("EMA 周期必须是正整数")
    array = np.asarray(values, dtype="float64").ravel()
    result = np.full(array.shape[0], np.nan, dtype="float64")
    alpha = 2.0 / (span + 1.0)
    state: float | None = None
    for position, value in enumerate(array):
        if not np.isfinite(value):
            continue
        state = float(value) if state is None else alpha * float(value) + (1.0 - alpha) * state
        result[position] = state
    return result


def wilder_average(values: np.ndarray | list[float], window: int) -> np.ndarray:
    """通达信 ``SMA(X, N, 1)`` 平滑：``avg = avg + (x - avg) / N``。

    与 :func:`ema` 的差别是平滑系数为 ``1/N``（而不是 ``2/(N+1)``），
    这正是国内软件的 RSI 口径；首值直接作为初值，缺失值所在位置输出 NaN。
    """
    if window <= 0:
        raise ValueError("平滑周期必须是正整数")
    array = np.asarray(values, dtype="float64").ravel()
    result = np.full(array.shape[0], np.nan, dtype="float64")
    state: float | None = None
    for position, value in enumerate(array):
        if not np.isfinite(value):
            continue
        state = float(value) if state is None else state + (float(value) - state) / window
        result[position] = state
    return result


def macd(
    values: np.ndarray | list[float],
    fast: int = DEFAULT_MACD_PARAMS[0],
    slow: int = DEFAULT_MACD_PARAMS[1],
    signal: int = DEFAULT_MACD_PARAMS[2],
) -> MacdData:
    """MACD(12,26,9)：DIF、DEA 与柱状值 ``2 * (DIF - DEA)``。"""
    if not 0 < fast < slow:
        raise ValueError("MACD 的快线周期必须为正整数且小于慢线周期")
    if signal <= 0:
        raise ValueError("MACD 的信号线周期必须是正整数")
    close = np.asarray(values, dtype="float64").ravel()
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return MacdData(dif=dif, dea=dea, hist=(dif - dea) * 2.0)


def rsi(values: np.ndarray | list[float], window: int) -> np.ndarray:
    """RSI：``100 * 平均涨幅 / (平均涨幅 + 平均跌幅)``，首日无前收故为 NaN。

    涨跌都为 0（价格连续不变）时取中性值 50；只有涨幅没有跌幅时为 100。
    """
    if window <= 0:
        raise ValueError("RSI 周期必须是正整数")
    array = np.asarray(values, dtype="float64").ravel()
    result = np.full(array.shape[0], np.nan, dtype="float64")
    if array.size < 2:
        return result
    delta = np.diff(array)
    gain = np.where(np.isfinite(delta) & (delta > 0), delta, np.where(np.isfinite(delta), 0.0, np.nan))
    loss = np.where(np.isfinite(delta) & (delta < 0), -delta, np.where(np.isfinite(delta), 0.0, np.nan))
    avg_gain = wilder_average(gain, window)
    avg_loss = wilder_average(loss, window)
    total = avg_gain + avg_loss
    ratio = np.full(total.shape, np.nan, dtype="float64")
    valid = np.isfinite(total)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio[valid] = np.where(
            total[valid] > 0,
            avg_gain[valid] / total[valid] * 100.0,
            50.0,
        )
    result[1:] = ratio
    return result


def boll(
    values: np.ndarray | list[float],
    window: int = DEFAULT_BOLL_WINDOW,
    k: float = DEFAULT_BOLL_K,
) -> BollData:
    """BOLL(20, 2)：中轨为简单均线，上下轨为 ``中轨 ± k * 总体标准差``。"""
    if window <= 0:
        raise ValueError("BOLL 周期必须是正整数")
    if k <= 0:
        raise ValueError("BOLL 的倍数必须是正数")
    array = np.asarray(values, dtype="float64").ravel()
    middle = sma(array, window)
    deviation = np.full(array.shape[0], np.nan, dtype="float64")
    if array.size >= window:
        windows = np.lib.stride_tricks.sliding_window_view(array, window)
        deviation[window - 1 :] = windows.std(axis=1, ddof=0)
    return BollData(
        middle=middle,
        upper=middle + k * deviation,
        lower=middle - k * deviation,
    )


def build_macd(candle: CandleData, params: tuple[int, int, int] = DEFAULT_MACD_PARAMS) -> MacdData:
    """按收盘价计算 MACD，参数形如 ``(12, 26, 9)``。"""
    if len(params) != 3:
        raise ValueError("MACD 参数必须是 (快线, 慢线, 信号线)")
    fast, slow, signal = (int(item) for item in params)
    return macd(candle.close, fast=fast, slow=slow, signal=signal)


def build_rsi(candle: CandleData, windows: tuple[int, ...] | list[int]) -> dict[int, np.ndarray]:
    """按收盘价一次算出全部 RSI 周期，键为周期。"""
    cleaned = clean_windows(windows)
    return {window: rsi(candle.close, window) for window in cleaned}


def build_boll(
    candle: CandleData,
    window: int = DEFAULT_BOLL_WINDOW,
    k: float = DEFAULT_BOLL_K,
) -> BollData:
    """按收盘价计算 BOLL 轨道。"""
    return boll(candle.close, window=window, k=k)
