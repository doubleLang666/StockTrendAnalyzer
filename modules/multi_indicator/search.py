"""多指标历史检索（设计文档第 3 节：与历史检索联动）。

做法与单指标检索一致——按交易日滑动，窗口长度等于目标区间的交易日数，
逐个窗口与目标区间比较；区别是每个窗口都要把选中的多个指标分别评分，
再按权重合成一个综合分排序。

两条硬性要求：

1. **防数据泄漏**：候选窗口的结束日必须早于目标区间起始日，检索过程只使用
   目标区间之前的历史数据；
2. **不做"看着精确"的假结果**：有效交易日比例低于门槛的窗口直接跳过，
   最终返回的每一条都用 :func:`modules.multi_indicator.analysis.analyze_frame`
   重新精算一遍，界面上的分数与"手动分析该区间"得到的数字完全一致。

排序阶段用向量化计算（避免几千个窗口逐个调用 Pearson），只在最后对前 N 名
精算，既快又稳。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.exceptions import InsufficientDataError, ProcessingError
from modules.multi_indicator.analysis import (
    MultiIndicatorRequest,
    MultiIndicatorResult,
    analyze_frame,
    default_weights,
)
from modules.multi_indicator.preparation import DEFAULT_MIN_VALID_RATIO, transform_values
from modules.multi_indicator.scoring import (
    SCORE_MODES,
    IndicatorScore,
    normalize_weights,
)
from modules.multi_indicator.specs import KIND_RATE, get_indicator, indicator_series

__all__ = [
    "MultiSearchRequest",
    "MultiSearchResult",
    "search_history",
]

_STD_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class MultiSearchRequest:
    """一次多指标历史检索的输入条件。"""

    target_start: Any
    target_end: Any
    scope_start: Any
    scope_end: Any
    keys: tuple[str, ...]
    weights: Mapping[str, float] = field(default_factory=dict)
    score_mode: str = "combined"
    top_n: int = 5
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO
    max_overlap_ratio: float = 0.5


@dataclass(frozen=True, slots=True)
class MultiSearchResult:
    """一条检索结果：一个历史区间 + 它的多指标明细。"""

    rank: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    trading_days: int
    total_score: float
    grade: str
    weights: Mapping[str, float]
    weight_changed: bool
    scores: tuple[IndicatorScore, ...]
    analysis: MultiIndicatorResult | None = None

    @property
    def label(self) -> str:
        return f"{self.start_date.date()} ~ {self.end_date.date()}"

    @property
    def valid_scores(self) -> tuple[IndicatorScore, ...]:
        return tuple(score for score in self.scores if score.is_valid)

    @property
    def valid_count(self) -> int:
        return len(self.valid_scores)

    def describe(self) -> str:
        parts = [
            f"{score.label} {score.score_for('combined'):.1f}"
            for score in self.valid_scores
            if score.score_for("combined") is not None
        ]
        detail = "｜".join(parts) if parts else "没有有效指标"
        return (
            f"综合相似度 {self.total_score:.2f}（{self.grade}）｜{self.label}"
            f"｜{self.trading_days} 个交易日｜{detail}"
        )


def search_history(
    frame: pd.DataFrame,
    request: MultiSearchRequest,
    *,
    similarity_service: object,
    max_candidates: int = 4000,
) -> tuple[MultiSearchResult, ...]:
    """在历史区间内寻找与目标区间最相似的多指标窗口。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ProcessingError("没有可用于检索的行情数据")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ProcessingError(
            "多指标检索需要真实交易日期索引",
            detail="相对交易日数据没有真实日期，无法在历史区间内滑动",
        )
    if not request.keys:
        raise ProcessingError("请至少选择一个指标")
    if request.score_mode not in SCORE_MODES:
        raise ProcessingError(f"未知的评分模式：{request.score_mode}")
    top_n = max(1, int(request.top_n))

    index = frame.index
    target_positions = _positions(index, request.target_start, request.target_end, "目标区间")
    target_start_pos, target_end_pos = target_positions
    days = target_end_pos - target_start_pos + 1
    if days < 2:
        raise InsufficientDataError("目标区间太短", detail="至少需要 2 个交易日才能做多指标检索")

    scope_start_pos, scope_end_pos = _positions(index, request.scope_start, request.scope_end, "历史范围")
    # 候选窗口必须完整落在历史范围内，并且结束日早于目标区间起始日（防数据泄漏）
    last_start = min(scope_end_pos, target_start_pos - 1) - days + 1
    first_start = max(scope_start_pos, 0)
    if last_start < first_start:
        raise InsufficientDataError(
            "历史范围太短",
            detail="历史范围内没有足够长的、且早于目标区间的候选窗口",
        )
    starts = np.arange(first_start, last_start + 1, dtype=int)
    if starts.size > max_candidates:
        starts = starts[-max_candidates:]

    target_dates = index[target_start_pos : target_end_pos + 1]

    weights_input = dict(request.weights) if request.weights else default_weights(list(request.keys))
    score_mode = request.score_mode
    min_ratio = float(request.min_valid_ratio)

    scores_by_key: dict[str, np.ndarray] = {}
    valid_by_key: dict[str, np.ndarray] = {}
    ratios_by_key: dict[str, np.ndarray] = {}
    for key in request.keys:
        spec = get_indicator(key)
        series = indicator_series(frame, spec).astype("float64")
        series_values = series.to_numpy()
        target_values = series_values[target_start_pos : target_end_pos + 1]
        target_series = pd.Series(target_values, index=target_dates)
        target_scored, transform, _note = transform_values(spec, target_series)
        target_array = target_scored.to_numpy(dtype=float)

        windows = np.lib.stride_tricks.sliding_window_view(series_values, days)[starts]
        window_scored = _normalize_windows(spec, windows, transform)
        metrics = _metrics(target_array, window_scored)
        score_array = _mode_scores(metrics, score_mode)
        valid = np.isfinite(score_array) & (metrics["ratio"] >= min_ratio)
        scores_by_key[key] = score_array
        valid_by_key[key] = valid
        ratios_by_key[key] = metrics["ratio"]

    keys = tuple(key for key in request.keys if key in scores_by_key)
    if not keys:
        return ()

    raw_weights = np.array([max(0.0, float(weights_input.get(key, 0.0))) for key in keys])
    if raw_weights.sum() <= 0:
        raw_weights = np.ones_like(raw_weights)
    valid_matrix = np.vstack([valid_by_key[key] for key in keys])
    score_matrix = np.vstack([np.where(valid_by_key[key], scores_by_key[key], 0.0) for key in keys])

    effective = raw_weights[:, None] * valid_matrix
    denominator = effective.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.where(denominator > 0, (score_matrix * effective).sum(axis=0) / denominator, np.nan)
    usable = np.isfinite(composite)
    if not usable.any():
        raise InsufficientDataError(
            "没有合格的候选区间",
            detail="所有候选窗口的有效交易日比例都不足，或指标数据缺失",
        )

    order = np.argsort(-np.where(usable, composite, -np.inf))
    # 相邻窗口高度重叠时只保留排名更高的一条，避免结果被同一段行情刷屏
    min_gap = max(1, int(round(days * (1.0 - float(request.max_overlap_ratio)))))
    picked: list[int] = []
    for position in order:
        if not usable[position]:
            break
        start_pos = int(starts[position])
        if all(abs(start_pos - int(starts[other])) >= min_gap for other in picked):
            picked.append(int(position))
        if len(picked) >= top_n:
            break

    results: list[MultiSearchResult] = []
    for rank, position in enumerate(picked, start=1):
        start_pos = int(starts[position])
        window_dates = index[start_pos : start_pos + days]
        detail = analyze_frame(
            frame,
            MultiIndicatorRequest(
                target_start=target_dates[0],
                target_end=target_dates[-1],
                compare_start=window_dates[0],
                compare_end=window_dates[-1],
                keys=keys,
                weights={key: float(weights_input.get(key, 0.0)) for key in keys},
                score_mode=score_mode,
                min_valid_ratio=min_ratio,
            ),
            similarity_service=similarity_service,
        )
        if detail.total_score is None:
            continue
        valid_keys = tuple(score.spec_key for score in detail.valid_scores)
        actual, changed = normalize_weights(weights_input, valid_keys)
        results.append(
            MultiSearchResult(
                rank=rank,
                start_date=window_dates[0],
                end_date=window_dates[-1],
                trading_days=days,
                total_score=float(detail.total_score),
                grade=detail.grade,
                weights=actual,
                weight_changed=changed or detail.weight_changed,
                scores=detail.scores,
                analysis=detail,
            )
        )

    results.sort(key=lambda item: item.total_score, reverse=True)
    return tuple(
        MultiSearchResult(
            rank=position,
            start_date=item.start_date,
            end_date=item.end_date,
            trading_days=item.trading_days,
            total_score=item.total_score,
            grade=item.grade,
            weights=item.weights,
            weight_changed=item.weight_changed,
            scores=item.scores,
            analysis=item.analysis,
        )
        for position, item in enumerate(results, start=1)
    )


# ----------------------------------------------------------------------
# 向量化工具
# ----------------------------------------------------------------------
def _positions(index: pd.DatetimeIndex, start: Any, end: Any, label: str) -> tuple[int, int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ProcessingError(f"{label}的起始日期晚于结束日期")
    subset = index[(index >= start_ts) & (index <= end_ts)]
    if subset.empty:
        raise InsufficientDataError(f"{label}内没有可用数据", detail=f"{start_ts.date()} ~ {end_ts.date()}")
    return int(index.get_loc(subset[0])), int(index.get_loc(subset[-1]))


def _normalize_windows(spec, windows: np.ndarray, transform: str) -> np.ndarray:
    """把每个候选窗口按"该窗口自己的首个有效值"换算到同一口径。"""
    rows = windows.shape[0]
    finite = np.isfinite(windows)
    if not finite.any(axis=1).all():
        # 存在整行都没有数值的窗口，保留 NaN 交给后续有效性判断
        pass

    if spec.kind == KIND_RATE:
        return windows + 100.0

    if transform == "raw":
        # 目标区间首个有效值非正，退回原始数值比较（与 preparation 的规则一致）
        return windows.copy()

    if transform.startswith("log"):
        positive = np.where(windows > 0, windows, np.nan)
        base = _first_column(positive)
        logged = np.log1p(np.where(windows >= 0, windows, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            return logged / np.log1p(base)[:, None] * 100.0

    base = _first_column(np.where(finite, windows, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = windows / base[:, None] * 100.0
    scaled[~np.isfinite(base) | (base <= 0)] = np.nan
    if scaled.shape[0] != rows:  # 保持形状（防御性）
        raise AssertionError("窗口数量与得分数量不一致")
    return scaled


def _first_column(values: np.ndarray) -> np.ndarray:
    """每行第一个有限值；整行无效时返回 NaN。"""
    finite = np.isfinite(values)
    has_value = finite.any(axis=1)
    first = np.argmax(finite, axis=1)
    return np.where(has_value, values[np.arange(values.shape[0]), first], np.nan)


def _metrics(target: np.ndarray, windows: np.ndarray) -> dict[str, np.ndarray]:
    """对每一行窗口计算 Pearson / RMSE / MAE 与合并标准差、有效比例。"""
    target_row = target[None, :]
    valid = np.isfinite(windows) & np.isfinite(target_row)
    counts = valid.sum(axis=1)
    ratio = counts / windows.shape[1]

    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(valid, target_row, np.nan)
        w = np.where(valid, windows, np.nan)
        mean_t = np.nanmean(t, axis=1)
        mean_w = np.nanmean(w, axis=1)
        dt = t - mean_t[:, None]
        dw = w - mean_w[:, None]
        cov = np.nanmean(dt * dw, axis=1)
        std_t = np.sqrt(np.nanmean(dt * dt, axis=1))
        std_w = np.sqrt(np.nanmean(dw * dw, axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            pearson = cov / (std_t * std_w)
        diff = w - t
        rmse = np.sqrt(np.nanmean(diff * diff, axis=1))
        mae = np.nanmean(np.abs(diff), axis=1)

        merged = np.concatenate([t, w], axis=1)
        merged_mean = np.nanmean(merged, axis=1)
        merged_var = np.nanmean((merged - merged_mean[:, None]) ** 2, axis=1)
        scale = np.sqrt(merged_var)

    pearson = np.where(np.isfinite(pearson), pearson, np.nan)
    return {"pearson": pearson, "rmse": rmse, "mae": mae, "scale": scale, "ratio": ratio}


def _mode_scores(metrics: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """与 :func:`modules.multi_indicator.scoring.error_scores` 完全一致的向量化版本。"""
    pearson = metrics["pearson"]
    rmse = metrics["rmse"]
    mae = metrics["mae"]
    scale = metrics["scale"]

    pearson_score = np.clip((pearson + 1.0) * 50.0, 0.0, 100.0)
    pearson_score = np.where(np.isfinite(pearson_score), pearson_score, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        nrmse = rmse / scale
        nmae = mae / scale
        rmse_score = np.clip(100.0 / (1.0 + nrmse), 0.0, 100.0)
        mae_score = np.clip(100.0 / (1.0 + nmae), 0.0, 100.0)

    flat = ~np.isfinite(scale) | (scale <= _STD_EPSILON)
    exact = flat & np.isfinite(rmse) & np.isfinite(mae) & (rmse <= _STD_EPSILON) & (mae <= _STD_EPSILON)
    impossible = flat & ~exact
    rmse_score = np.where(exact, 100.0, rmse_score)
    mae_score = np.where(exact, 100.0, mae_score)

    combined = np.nanmean(np.vstack([pearson_score, rmse_score, mae_score]), axis=0)
    if exact.any():
        # 完全平直且误差为 0：误差两项记满分，综合分按三项等权
        hundred = np.full_like(pearson_score, 100.0)
        exact_combined = np.nanmean(np.vstack([pearson_score, hundred, hundred]), axis=0)
        combined = np.where(exact, exact_combined, combined)

    chosen = {"pearson": pearson_score, "rmse": rmse_score, "mae": mae_score, "combined": combined}[mode]
    return np.where(impossible, np.nan, chosen)
