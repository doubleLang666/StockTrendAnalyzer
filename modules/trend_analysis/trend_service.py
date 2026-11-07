"""趋势分析服务（设计文档 3.3）。

* :meth:`TrendService.plot`        生成 Figure，供界面嵌入画布；
* :meth:`TrendService.compare`     多周期比较曲线；
* :meth:`TrendService.search_similar_periods`  滑动窗口检索。

相似度算法本身不在本模块实现：检索时通过注册表取得相似度服务并按
:class:`~core.data_models.SimilarityConfig` 调用，符合文档"趋势模块只负责可视化与检索"。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
from matplotlib.figure import Figure

from app.config import AppConfig
from core.constants import (
    DEFAULT_MAX_OVERLAP_RATIO,
    DEFAULT_MIN_VALID_POINTS,
    DEFAULT_TOP_N,
    Alignment,
    Normalization,
)
from core.data_models import (
    MarketDataset,
    ProcessedDataset,
    SearchResult,
)
from core.exceptions import InsufficientDataError, ProcessingError, ServiceNotRegisteredError
from modules.data_processing.alignment import apply_alignment, index_axis_label
from modules.data_processing.normalization import normalize_series
from modules.trend_analysis.plotting import CurveData, PlotRequest, build_figure
from modules.trend_analysis.search import find_similar_periods
from modules.trend_analysis.segments import PeriodComparison, Segment, compare_periods

__all__ = ["TrendService", "series_from_source", "source_frame"]


def source_frame(source: MarketDataset | ProcessedDataset) -> tuple[pd.DataFrame, str]:
    """取出数据源的可绘制表格与名称。"""
    if isinstance(source, ProcessedDataset):
        return source.frame, source.name
    if isinstance(source, MarketDataset):
        return source.frame, source.name
    raise ProcessingError("趋势分析需要 MarketDataset 或 ProcessedDataset 输入")


def series_from_source(
    source: MarketDataset | ProcessedDataset,
    field: str,
    *,
    start: date | None = None,
    end: date | None = None,
    normalization: Normalization | str = Normalization.NONE,
    alignment: Alignment | str = Alignment.REAL_DATE,
) -> tuple[pd.Series, tuple[str, ...]]:
    """从数据源取出一条可绘制曲线，并返回需要告知用户的说明。

    对 :class:`MarketDataset` 会按传入的归一化与对齐参数即时处理；
    对 :class:`ProcessedDataset` 直接使用其处理结果（已应用过设置），
    不再重复归一化，避免两次变换叠加。
    """
    frame, _name = source_frame(source)
    if field not in frame.columns:
        raise ProcessingError(f"数据中不存在字段：{field}")

    notes: list[str] = []
    series = frame[field]
    is_processed = isinstance(source, ProcessedDataset)

    if isinstance(frame.index, pd.DatetimeIndex):
        if start is not None or end is not None:
            mask = pd.Series(True, index=frame.index)
            if start is not None:
                mask &= frame.index >= pd.Timestamp(start)
            if end is not None:
                mask &= frame.index <= pd.Timestamp(end)
            series = series.loc[mask]
            if series.empty:
                raise InsufficientDataError("所选日期范围内没有数据")
    else:
        if start is not None or end is not None:
            notes.append("该数据是相对交易日数据，日期范围筛选不适用，已忽略")
        alignment = Alignment.RELATIVE_DAY

    if is_processed:
        return series.astype("float64"), tuple(notes)

    resolved_normalization = Normalization(getattr(normalization, "value", normalization))
    resolved_alignment = Alignment(getattr(alignment, "value", alignment))
    if resolved_normalization is not Normalization.NONE:
        series = normalize_series(series, resolved_normalization).series
        notes.append(f"已按{resolved_normalization.value}归一化显示")
    aligned = apply_alignment(series.to_frame(name=field), resolved_alignment)
    return aligned[field], tuple(notes)


class TrendService:
    """``TrendService`` 协议实现。"""

    def __init__(
        self,
        config: AppConfig | None = None,
        similarity_resolver: Callable[[], Any | None] | None = None,
    ) -> None:
        self._config = config
        self._similarity_resolver = similarity_resolver

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    def plot(
        self,
        datasets: Sequence[MarketDataset | ProcessedDataset] | PlotRequest,
        config: Any = None,
    ) -> Figure:
        """生成 Figure。

        * 传入 :class:`PlotRequest` 时直接按请求绘制；
        * 传入数据集序列时，把每个数据集的所有数据列各画一条曲线。
        """
        if isinstance(datasets, PlotRequest):
            return build_figure(datasets)
        if config is not None and not isinstance(config, PlotRequest):
            raise ProcessingError("绘图配置必须是 PlotRequest")

        curves: list[CurveData] = []
        for dataset in datasets:
            frame, name = source_frame(dataset)
            for column in frame.columns:
                label = (
                    dataset.column_label(column)
                    if isinstance(dataset, ProcessedDataset)
                    else str(column)
                )
                curves.append(CurveData(label=f"{name}·{label}", values=frame[column].astype("float64")))
        if not curves:
            raise ProcessingError("没有可绘制的数据")
        request = config or PlotRequest(
            curves=tuple(curves),
            title="行情曲线",
            x_label="交易日期",
            y_label="数值",
            legend=True,
        )
        return build_figure(request)

    # ------------------------------------------------------------------
    # 多周期比较
    # ------------------------------------------------------------------
    def compare(
        self,
        frame: pd.DataFrame,
        field: str,
        segments: Iterable[Segment],
        *,
        normalization: Normalization | str = Normalization.NONE,
        alignment: Alignment | str = Alignment.REAL_DATE,
        truncate: bool = True,
    ) -> PeriodComparison:
        return compare_periods(
            frame,
            field,
            tuple(segments),
            normalization=normalization,
            alignment=alignment,
            truncate=truncate,
        )

    # ------------------------------------------------------------------
    # 历史检索
    # ------------------------------------------------------------------
    def search_similar_periods(
        self,
        target_period: Segment | tuple[date, date],
        dataset: MarketDataset | ProcessedDataset,
        config: Any = None,
    ) -> list[SearchResult]:
        """以目标区间长度滑动检索历史相似区间。"""
        from core.data_models import SimilarityConfig

        if isinstance(target_period, Segment):
            start, end = target_period.start, target_period.end
        else:
            start, end = target_period

        similarity = self._similarity_resolver() if self._similarity_resolver else None
        if similarity is None:
            raise ServiceNotRegisteredError(
                "相似度模块尚未接入",
                detail="历史区间检索需要相似度算法，将在第五阶段接通",
            )

        similarity_config = (
            config
            if isinstance(config, SimilarityConfig)
            else SimilarityConfig(
                algorithm=getattr(config, "algorithm", "pearson"),
                top_n=getattr(config, "top_n", DEFAULT_TOP_N),
            )
        )
        params = dict(similarity_config.params)
        field = params.pop("field", None)
        frame, _name = source_frame(dataset)
        if field is None:
            candidates = [column for column in frame.columns if column != "date"]
            if len(candidates) != 1:
                raise ProcessingError("数据包含多个字段，请在检索参数中指定 field")
            field = candidates[0]
        if field not in frame.columns:
            raise ProcessingError(f"数据中不存在字段：{field}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ProcessingError("历史检索需要真实交易日期索引，请使用真实日期对齐的数据")

        algorithm = similarity_config.algorithm.value

        def metric(left: pd.Series, right: pd.Series):  # noqa: ANN202
            return similarity.compare(left, right, algorithm=algorithm)

        results = find_similar_periods(
            frame[field].astype("float64"),
            start,
            end,
            metric=metric,
            field=field,
            top_n=similarity_config.top_n,
            min_valid_points=similarity_config.min_valid_points
            if similarity_config.min_valid_points
            else DEFAULT_MIN_VALID_POINTS,
            max_overlap_ratio=similarity_config.max_overlap_ratio
            if similarity_config.max_overlap_ratio is not None
            else DEFAULT_MAX_OVERLAP_RATIO,
            normalize=similarity_config.normalize,
        )
        return list(results)

    # ------------------------------------------------------------------
    def axis_label(self, alignment: Alignment | str) -> str:
        return index_axis_label(alignment)
