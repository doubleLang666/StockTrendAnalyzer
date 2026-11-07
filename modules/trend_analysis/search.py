"""历史区间滑动检索（设计文档 3.3.4）。

以目标区间的交易日长度作为窗口长度，在历史数据上逐日滑动，
对每个窗口调用外部传入的相似度函数（本模块不实现任何相似度算法）。

明确跳过规则：

* 与目标区间完全重合的窗口（自身匹配）；
* 与目标区间重叠比例超过 ``max_overlap_ratio`` 的窗口；
* 有效数据点少于 ``min_valid_points`` 的窗口；
* 归一化基准为 0 或相似度函数返回无效结果的窗口。

只检索已导入的历史数据，不涉及目标区间结束日之后的任何外推。
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core.constants import DEFAULT_MAX_OVERLAP_RATIO, DEFAULT_MIN_VALID_POINTS, DEFAULT_TOP_N, Normalization
from core.data_models import SearchResult, SimilarityResult
from core.exceptions import InsufficientDataError, ProcessingError
from modules.data_processing.normalization import normalize_series

__all__ = ["SimilarityMetric", "find_similar_periods"]

#: 相似度函数：输入两条等长数值序列，返回 :class:`~core.data_models.SimilarityResult`。
SimilarityMetric = Callable[[pd.Series, pd.Series], SimilarityResult]

_EPS = 1e-9


def find_similar_periods(
    series: pd.Series,
    target_start: date,
    target_end: date,
    *,
    metric: SimilarityMetric,
    field: str,
    top_n: int = DEFAULT_TOP_N,
    min_valid_points: int = DEFAULT_MIN_VALID_POINTS,
    max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO,
    normalize: bool = True,
) -> tuple[SearchResult, ...]:
    """在 ``series`` 中检索与目标区间最相似的历史窗口。"""
    if not isinstance(series, pd.Series):
        raise ProcessingError("历史检索需要 pandas.Series 输入")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ProcessingError("历史检索要求以交易日期为索引")
    if not series.index.is_monotonic_increasing:
        raise ProcessingError("历史检索要求数据按日期升序排列")
    if top_n < 1:
        raise ProcessingError("Top N 必须大于 0")
    if not 0 <= max_overlap_ratio < 1:
        raise ProcessingError("重叠比例阈值必须位于 [0, 1) 区间")

    window_length = int(
        series.loc[pd.Timestamp(target_start): pd.Timestamp(target_end)].shape[0]
    )
    target = series.loc[pd.Timestamp(target_start): pd.Timestamp(target_end)]
    if target.empty:
        raise InsufficientDataError("目标区间内没有数据", detail=f"{target_start} ~ {target_end}")
    if window_length < 2:
        raise InsufficientDataError("目标区间至少需要 2 个交易日")
    if window_length > len(series):
        raise InsufficientDataError("目标区间长度超过历史数据长度")

    target_index = target.index
    target_raw = target.astype("float64").reset_index(drop=True)
    target_array = target_raw.to_numpy(dtype="float64")
    scored: list[tuple[float, SearchResult]] = []

    for position in range(len(series) - window_length + 1):
        window = series.iloc[position: position + window_length]
        overlap = len(target_index.intersection(window.index)) / window_length
        if overlap > max_overlap_ratio + _EPS:
            continue  # 自身匹配或高度重叠，不作为独立样本

        window_values = np.asarray(window.to_numpy(dtype="float64"), dtype="float64")
        valid_mask = np.isfinite(target_array) & np.isfinite(window_values)
        if int(valid_mask.sum()) < min_valid_points:
            continue

        left = target_raw
        right = pd.Series(window_values, name="close")
        if normalize:
            target_outcome = normalize_series(left, Normalization.START)
            window_outcome = normalize_series(right, Normalization.START)
            if target_outcome.base_value in (None, 0) or window_outcome.base_value in (None, 0):
                continue
            left = target_outcome.series
            right = window_outcome.series

        result = metric(left, right)
        if not isinstance(result, SimilarityResult) or not result.is_valid:
            continue

        start_date = window.index[0].date()
        end_date = window.index[-1].date()
        scored.append(
            (
                result.sort_key(),
                SearchResult(
                    rank=0,
                    start_date=start_date,
                    end_date=end_date,
                    trading_days=window_length,
                    field=field,
                    result=result,
                    series=window.copy(),
                    overlap_ratio=overlap,
                ),
            )
        )

    if not scored:
        return ()

    scored.sort(key=lambda item: item[0])
    ranked: list[SearchResult] = []
    for rank, (_, item) in enumerate(scored[:top_n], start=1):
        ranked.append(
            SearchResult(
                rank=rank,
                start_date=item.start_date,
                end_date=item.end_date,
                trading_days=item.trading_days,
                field=item.field,
                result=item.result,
                series=item.series,
                overlap_ratio=item.overlap_ratio,
            )
        )
    return tuple(ranked)


def describe_results(results: Sequence[SearchResult]) -> str:
    """把检索结果格式化成便于日志/界面展示的文本。"""
    if not results:
        return "没有找到符合条件的相似历史区间"
    lines = []
    for item in results:
        value = item.result.display_value if item.result.display_value is not None else item.result.raw_value
        lines.append(
            f"第 {item.rank} 名　{item.start_date} ~ {item.end_date}　"
            f"{item.trading_days} 个交易日　{item.result.label} = {value:.4f}"
        )
    return "\n".join(lines)
