"""多指标联动分析：编排（指标 → 对齐 → 单指标评分 → 加权综合分）。

一次分析只做纯计算，不碰任何界面：

* 逐指标准备 :class:`SeriesPair`（缺字段 / 区间为空 / 有效点不足都会记录原因）； 
* 逐指标算 Pearson / RMSE / MAE 与 0–100 评分；
* 只让**有效指标**参与加权，权重重新归一化并记录实际权重（文档 2.4）；
* 输出综合分、分级、参与指标、被剔除指标及原因。

原始行情数据在分析过程中只读不改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Sequence

import pandas as pd

from core.data_models import MarketDataset
from core.exceptions import ProcessingError, StockAnalyzerError
from modules.multi_indicator.preparation import DEFAULT_MIN_VALID_RATIO, SeriesPair, prepare_pair
from modules.multi_indicator.scoring import (
    IndicatorScore,
    compute_indicator_score,
    grade_label,
    normalize_weights,
    score_mode_label,
)
from modules.multi_indicator.specs import get_indicator

__all__ = [
    "IndicatorWeight",
    "MultiIndicatorRequest",
    "MultiIndicatorResult",
    "analyze_frame",
    "default_weights",
]

#: 默认权重（文档 2.4 示例）：收盘价 40% / 成交量 30% / 换手率 20% / 涨跌幅 10%。
_DEFAULT_WEIGHT_ORDER: tuple[tuple[str, float], ...] = (
    ("close", 40.0),
    ("volume", 30.0),
    ("turnover", 20.0),
    ("pct_change", 10.0),
)


@dataclass(frozen=True, slots=True)
class IndicatorWeight:
    key: str
    weight: float = 0.0


def default_weights(keys: Sequence[str]) -> dict[str, float]:
    """为给定指标生成默认权重；未在默认表中出现的指标平均分配剩余份额。"""
    defaults: dict[str, float] = {}
    known = {key: weight for key, weight in _DEFAULT_WEIGHT_ORDER if key in keys}
    if known:
        total = sum(known.values())
        for key, weight in known.items():
            defaults[key] = weight / total
        return defaults
    if not keys:
        return {}
    share = 1.0 / len(keys)
    return {key: share for key in keys}


@dataclass(frozen=True, slots=True)
class MultiIndicatorRequest:
    """一次多指标分析的全部输入。"""

    target_start: date
    target_end: date
    compare_start: date
    compare_end: date
    keys: tuple[str, ...]
    weights: Mapping[str, float] = field(default_factory=dict)
    score_mode: str = "combined"
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO


@dataclass(frozen=True, slots=True)
class MultiIndicatorResult:
    """一次多指标分析的输出。"""

    target_start: date
    target_end: date
    compare_start: date
    compare_end: date
    pairs: tuple[SeriesPair, ...] = ()
    scores: tuple[IndicatorScore, ...] = ()
    weights: Mapping[str, float] = field(default_factory=dict)
    input_weights: Mapping[str, float] = field(default_factory=dict)
    weight_changed: bool = False
    total_score: float | None = None
    score_mode: str = "combined"
    skipped: tuple[tuple[str, str], ...] = ()
    trading_days: int = 0

    @property
    def valid_scores(self) -> tuple[IndicatorScore, ...]:
        return tuple(score for score in self.scores if score.is_valid)

    @property
    def invalid_scores(self) -> tuple[IndicatorScore, ...]:
        return tuple(score for score in self.scores if not score.is_valid)

    @property
    def grade(self) -> str:
        if self.total_score is None:
            return "无法评定"
        return grade_label(self.total_score)

    @property
    def has_result(self) -> bool:
        return self.total_score is not None

    @property
    def mode_label(self) -> str:
        return score_mode_label(self.score_mode)

    def weight_of(self, key: str) -> float:
        return float(self.weights.get(key, 0.0))

    def pair_of(self, key: str) -> SeriesPair | None:
        for pair in self.pairs:
            if pair.spec.key == key:
                return pair
        return None

    def score_of(self, key: str) -> IndicatorScore | None:
        for score in self.scores:
            if score.spec_key == key:
                return score
        return None

    def weakest(self) -> IndicatorScore | None:
        """相似度最低的有效指标（用于差异分析，文档 2.5-E）。"""
        valid = self.valid_scores
        if not valid:
            return None
        return min(valid, key=lambda score: score.score_for(self.score_mode) or 0.0)

    def summary_text(self) -> str:
        """一行摘要，用于状态栏与结果区标题。"""
        if self.total_score is None:
            return "没有可参与计算的指标（请检查区间与指标有效点占比）"
        participant = "、".join(score.label for score in self.valid_scores)
        parts = [
            f"综合相似度 {self.total_score:.2f}（{self.grade}）",
            f"模式：{self.mode_label}",
            f"参与指标：{participant}",
            f"有效交易日 {self.trading_days}",
        ]
        if self.weight_changed:
            parts.append("部分指标无效，权重已按有效指标重新归一化")
        return "｜".join(parts)

    def weight_rows(self) -> tuple[tuple[str, float, float], ...]:
        """``(指标名, 输入权重, 实际权重)``，用于界面展示重新归一化结果。"""
        rows = []
        for score in self.scores:
            rows.append(
                (
                    score.label,
                    float(self.input_weights.get(score.spec_key, 0.0)),
                    self.weight_of(score.spec_key),
                )
            )
        return tuple(rows)


def analyze_frame(
    frame: pd.DataFrame,
    request: MultiIndicatorRequest,
    *,
    similarity_service: object,
) -> MultiIndicatorResult:
    """对一张行情表做多指标联动分析。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ProcessingError("没有可用于分析的行情数据")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ProcessingError(
            "多指标分析需要真实交易日期索引",
            detail="相对交易日数据没有真实日期，无法按区间对齐",
        )

    keys = tuple(dict.fromkeys(request.keys)) or ()
    if not keys:
        raise ProcessingError("请至少选择一个指标")

    input_weights = dict(request.weights or default_weights(keys))
    pairs: list[SeriesPair] = []
    scores: list[IndicatorScore] = []
    skipped: list[tuple[str, str]] = []

    for key in keys:
        spec = get_indicator(key)
        try:
            pair = prepare_pair(
                frame,
                spec,
                request.target_start,
                request.target_end,
                request.compare_start,
                request.compare_end,
                min_valid_ratio=request.min_valid_ratio,
            )
        except StockAnalyzerError as exc:
            skipped.append((spec.label, str(exc)))
            continue
        pairs.append(pair)
        score = compute_indicator_score(similarity_service, pair, mode=request.score_mode)
        scores.append(score)
        if not score.is_valid:
            skipped.append((spec.label, score.message))

    valid_keys = [score.spec_key for score in scores if score.is_valid]
    weights, changed = normalize_weights(input_weights, valid_keys)

    total: float | None = None
    if weights:
        weighted = 0.0
        for score in scores:
            if not score.is_valid:
                continue
            value = score.score_for(request.score_mode)
            if value is None:
                continue
            weighted += weights.get(score.spec_key, 0.0) * value
        total = weighted if valid_keys else None

    trading_days = max((pair.length for pair in pairs), default=0)
    return MultiIndicatorResult(
        target_start=request.target_start,
        target_end=request.target_end,
        compare_start=request.compare_start,
        compare_end=request.compare_end,
        pairs=tuple(pairs),
        scores=tuple(scores),
        weights=weights,
        weight_changed=changed,
        total_score=total,
        score_mode=request.score_mode,
        skipped=tuple(skipped),
        trading_days=trading_days,
        input_weights=input_weights,
    )


def analyze_dataset(
    dataset: MarketDataset,
    request: MultiIndicatorRequest,
    *,
    similarity_service: object,
) -> MultiIndicatorResult:
    """对 :class:`MarketDataset` 做多指标联动分析（只读其 frame 副本）。"""
    if not isinstance(dataset, MarketDataset):
        raise StockAnalyzerError("多指标分析需要原始行情数据")
    frame = dataset.frame.copy()
    return analyze_frame(frame, request, similarity_service=similarity_service)
