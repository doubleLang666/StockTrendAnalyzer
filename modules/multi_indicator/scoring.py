"""多指标联动分析：单指标评分（对应设计文档 2.3 / 2.4）。

每个指标仍然调用现有的 ``SimilarityService``（Pearson / RMSE / MAE 三种算法不变），
本模块只负责把三种**量纲不同**的结果统一到 0–100 分：

* ``PearsonScore = (Pearson + 1) / 2 × 100``
* ``RMSEScore = 100 / (1 + NRMSE)``，``MAEScore = 100 / (1 + NMAE)``
* ``NRMSE = RMSE / S``、``NMAE = MAE / S``，``S`` 是两段序列**合并后的总体标准差**（ddof=0）

这样误差分与数据量纲无关，不同指标之间才可以比较、加权。原始的 Pearson / RMSE / MAE
数值会原样保留在 :class:`IndicatorScore` 里，界面上与评分分列显示（文档 2.4）。

若两段序列合并标准差为 0（序列完全平直），则误差为 0 时给 100 分，误差不为 0 时无法
用相对误差衡量，判为无效而不是给出一个看着精确的假分数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from core.data_models import SimilarityResult
from core.exceptions import ProcessingError
from modules.multi_indicator.preparation import SeriesPair

__all__ = [
    "ALGORITHM_LABELS",
    "GRADE_THRESHOLDS",
    "IndicatorScore",
    "SCORE_MODES",
    "combined_series",
    "compute_indicator_score",
    "error_scores",
    "grade_label",
    "normalize_weights",
    "score_mode_label",
]

#: 单指标评分模式（文档 2.4）：三种单算法 + 三项等权综合。
SCORE_MODES: tuple[str, ...] = ("pearson", "rmse", "mae", "combined")

ALGORITHM_LABELS = {
    "pearson": "Pearson 相关系数",
    "rmse": "RMSE",
    "mae": "MAE",
    "combined": "综合评分（Pearson/RMSE/MAE 等权）",
}

#: 综合相似度分级（文档 2.5），阈值可由调用方覆盖。
GRADE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (90.0, "高度相似"),
    (80.0, "较高相似"),
    (65.0, "中等相似"),
    (50.0, "一般相似"),
    (0.0, "较低相似"),
)

_STD_EPSILON = 1e-12


def score_mode_label(mode: str) -> str:
    return ALGORITHM_LABELS.get(mode, ALGORITHM_LABELS["combined"])


def grade_label(score: float, thresholds: Iterable[tuple[float, str]] = GRADE_THRESHOLDS) -> str:
    """把 0–100 的综合分映射为文字分级。"""
    for threshold, label in thresholds:
        if score >= threshold:
            return label
    return "较低相似"


def combined_series(pair: SeriesPair) -> np.ndarray:
    """返回两段序列合并后的有效数值（用于计算误差归一化的尺度 S）。"""
    left, right = pair.scored_values()
    return np.concatenate([left, right])


def _safe_score(value: float) -> float:
    return float(min(100.0, max(0.0, value)))


def error_scores(
    pearson: float | None,
    rmse: float,
    mae: float,
    scale: float,
) -> tuple[float | None, float | None, float | None, float, float, float | None] | None:
    """把三种算法的原始数值换算成 0–100 分。

    返回 ``(pearson_score, rmse_score, mae_score, nrmse, nmae, combined)``；
    当两段合并标准差为 0 且误差不为 0 时返回 ``None``（无法衡量相对误差）。
    单指标分析与多指标历史检索共用这一套公式，保证两条路径的分数完全一致。
    """
    pearson_score = None if pearson is None or not np.isfinite(pearson) else _safe_score((pearson + 1.0) * 50.0)
    if scale <= _STD_EPSILON:
        if rmse <= _STD_EPSILON and mae <= _STD_EPSILON:
            components = [value for value in (pearson_score, 100.0, 100.0) if value is not None]
            return (pearson_score, 100.0, 100.0, 0.0, 0.0, sum(components) / len(components))
        return None

    nrmse = float(rmse) / scale
    nmae = float(mae) / scale
    rmse_score = _safe_score(100.0 / (1.0 + nrmse))
    mae_score = _safe_score(100.0 / (1.0 + nmae))
    components = [value for value in (pearson_score, rmse_score, mae_score) if value is not None]
    combined = sum(components) / len(components) if components else None
    return (pearson_score, rmse_score, mae_score, nrmse, nmae, combined)


@dataclass(frozen=True, slots=True)
class IndicatorScore:
    """一个指标在两种算法上的完整评分结果。"""

    spec_key: str
    label: str
    #: 原始算法数值（未评分化），键为 pearson/rmse/mae
    results: Mapping[str, SimilarityResult] = field(default_factory=dict)
    pearson_score: float | None = None
    rmse_score: float | None = None
    mae_score: float | None = None
    combined_score: float | None = None
    scale: float | None = None
    nrmse: float | None = None
    nmae: float | None = None
    sample_size: int = 0
    total_points: int = 0
    valid_ratio: float = 0.0
    status: str = "valid"
    message: str = ""
    transform: str = ""
    min_valid_ratio: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def effective_weight_basis(self) -> bool:
        return self.is_valid

    def score_for(self, mode: str) -> float | None:
        """按评分模式取分；无效指标返回 ``None``。"""
        if not self.is_valid:
            return None
        if mode == "pearson":
            return self.pearson_score
        if mode == "rmse":
            return self.rmse_score
        if mode == "mae":
            return self.mae_score
        return self.combined_score

    def raw_value(self, algorithm: str) -> float | None:
        result = self.results.get(algorithm)
        if result is None or not result.is_valid:
            return None
        return result.raw_value

    def raw_label(self, algorithm: str) -> str:
        result = self.results.get(algorithm)
        return result.label if result is not None else "-"


def compute_indicator_score(
    service: object,
    pair: SeriesPair,
    *,
    mode: str = "combined",
) -> IndicatorScore:
    """计算单个指标的三算法结果与 0–100 评分。

    ``pair`` 无效（有效点不足）时返回 ``status="invalid"`` 的 :class:`IndicatorScore`，
    不会产生任何评分数字，避免"看着精确"的假结果。
    """
    if mode not in SCORE_MODES:
        raise ProcessingError(f"未知的评分模式：{mode}")

    if not pair.is_valid:
        return IndicatorScore(
            spec_key=pair.spec.key,
            label=pair.spec.label,
            sample_size=pair.valid_count,
            total_points=pair.length,
            valid_ratio=pair.valid_ratio,
            status="invalid",
            message=pair.reason,
            transform=pair.transform_label,
            min_valid_ratio=pair.min_valid_ratio,
        )

    left, right = pair.scored_values()
    # 序列已按指标口径变换与清洗过，这里关闭服务内部的"首值归一化"，
    # 保证 RMSE/MAE 与下面计算的尺度 S 使用同一套数值（否则尺度不一致，评分会随量纲漂移）。
    results: dict[str, SimilarityResult] = {}
    for algorithm in ("pearson", "rmse", "mae"):
        results[algorithm] = service.compare(  # type: ignore[attr-defined]
            left, right, algorithm=algorithm, min_valid_points=2, normalize=False
        )

    pearson_result = results["pearson"]
    rmse_result = results["rmse"]
    mae_result = results["mae"]
    if not (rmse_result.is_valid and mae_result.is_valid):
        failed = rmse_result if not rmse_result.is_valid else mae_result
        return IndicatorScore(
            spec_key=pair.spec.key,
            label=pair.spec.label,
            results=results,
            sample_size=int(min(rmse_result.sample_size, mae_result.sample_size)),
            total_points=pair.length,
            valid_ratio=pair.valid_ratio,
            status="invalid",
            message=failed.message or "误差算法未能给出有效结果",
            transform=pair.transform_label,
            min_valid_ratio=pair.min_valid_ratio,
        )

    merged = combined_series(pair)
    scale = float(merged.std(ddof=0)) if merged.size else 0.0
    rmse = float(rmse_result.raw_value)
    mae = float(mae_result.raw_value)
    pearson = float(pearson_result.raw_value) if pearson_result.is_valid else None

    scored = error_scores(pearson, rmse, mae, scale)
    if scored is None:
        return IndicatorScore(
            spec_key=pair.spec.key,
            label=pair.spec.label,
            results=results,
            pearson_score=None if pearson is None else _safe_score((pearson + 1.0) * 50.0),
            scale=scale,
            sample_size=rmse_result.sample_size,
            total_points=pair.length,
            valid_ratio=pair.valid_ratio,
            status="invalid",
            message="两段序列合并标准差为 0，无法计算相对误差评分",
            transform=pair.transform_label,
            min_valid_ratio=pair.min_valid_ratio,
        )

    pearson_score, rmse_score, mae_score, nrmse, nmae, combined = scored
    return IndicatorScore(
        spec_key=pair.spec.key,
        label=pair.spec.label,
        results=results,
        pearson_score=pearson_score,
        rmse_score=rmse_score,
        mae_score=mae_score,
        combined_score=combined,
        scale=scale,
        nrmse=nrmse,
        nmae=nmae,
        sample_size=int(min(rmse_result.sample_size, mae_result.sample_size)),
        total_points=pair.length,
        valid_ratio=pair.valid_ratio,
        status="valid",
        message="",
        transform=pair.transform_label,
        min_valid_ratio=pair.min_valid_ratio,
    )



def normalize_weights(weights: Mapping[str, float], valid_keys: Iterable[str]) -> tuple[dict[str, float], bool]:
    """把权重限制在有效指标上并重新归一化。

    返回 ``(实际权重, 是否发生了重新归一化)``。没有任何有效指标或权重全为 0 时
    返回空字典（调用方据此提示"没有可参与计算的指标"）。
    """
    keys = [key for key in valid_keys]
    if not keys:
        return {}, False
    positive = {key: max(0.0, float(weights.get(key, 0.0))) for key in keys}
    total = sum(positive.values())
    if total <= 0:
        # 全部被置零时退化为等权，避免整体无结果。
        equal = 1.0 / len(keys)
        return {key: equal for key in keys}, True
    normalized = {key: value / total for key, value in positive.items()}
    changed = len(keys) != len(weights) or any(
        abs(normalized[key] - max(0.0, float(weights.get(key, 0.0)))) > 1e-9 for key in keys
    )
    return normalized, changed
