"""趋势分析模块（第四阶段）。

对外提供 :class:`~modules.trend_analysis.trend_service.TrendService`：

* 单/多条曲线绘制（Figure，不依赖任何窗口对象）；
* 多周期历史区间比较（真实日期 / 相对交易日，原始数值 / 起始值归一化 / Min-Max）；
* 以目标区间长度进行的滑动窗口历史检索（相似度算法由外部注入）。
"""

from __future__ import annotations

from modules.trend_analysis.plotting import CurveData, PlotRequest, build_figure
from modules.trend_analysis.search import SimilarityMetric, describe_results, find_similar_periods
from modules.trend_analysis.segments import (
    ComparisonSegment,
    PeriodComparison,
    Segment,
    compare_periods,
)
from modules.trend_analysis.trend_service import TrendService, series_from_source, source_frame

__all__ = [
    "TrendService",
    "CurveData",
    "PlotRequest",
    "build_figure",
    "Segment",
    "ComparisonSegment",
    "PeriodComparison",
    "compare_periods",
    "find_similar_periods",
    "describe_results",
    "SimilarityMetric",
    "series_from_source",
    "source_frame",
]
