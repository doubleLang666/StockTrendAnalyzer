"""相似度服务：统一注册与调用各相似度算法（设计文档 3.4）。

本模块只负责"两条序列之间的相似度"，不导入文件、不绘图、不涉及界面。
新增算法时实现 :class:`~modules.similarity.base.SimilarityAlgorithm` 并调用
:meth:`SimilarityService.register` 即可，调用方无需修改。
"""

from __future__ import annotations

from typing import Any

from core.constants import (
    DEFAULT_MIN_VALID_POINTS,
    ALGORITHM_LABELS,
    Algorithm,
    SortDirection,
)
from core.data_models import SimilarityResult
from core.exceptions import AlgorithmError, UnknownAlgorithmError
from modules.similarity.base import SimilarityAlgorithm
from modules.similarity.cosine import CosineAlgorithm
from modules.similarity.dtw import DtwAlgorithm
from modules.similarity.mae import MaeAlgorithm
from modules.similarity.pearson import PearsonAlgorithm
from modules.similarity.rmse import RmseAlgorithm

__all__ = ["SimilarityService"]


class SimilarityService:
    """相似度计算服务（满足 :class:`core.interfaces.SimilarityService` 协议）。"""

    def __init__(self, config: Any | None = None) -> None:
        self._config = config
        self._algorithms: dict[str, SimilarityAlgorithm] = {}
        for strategy in (
            PearsonAlgorithm(),
            RmseAlgorithm(),
            MaeAlgorithm(),
            DtwAlgorithm(),
            CosineAlgorithm(),
        ):
            self.register(strategy)

    # -- 注册与查询 -------------------------------------------------------
    def register(self, algorithm: SimilarityAlgorithm) -> None:
        """注册（或替换）一个算法实现。"""
        if not getattr(algorithm, "key", ""):
            raise AlgorithmError("相似度算法缺少 key 标识")
        self._algorithms[algorithm.key] = algorithm

    def available_algorithms(self) -> tuple[str, ...]:
        """返回已注册算法标识，按注册顺序。"""
        return tuple(self._algorithms)

    def label_of(self, algorithm: Algorithm | str | None) -> str:
        key = self._normalize_key(algorithm)
        return self._algorithms[key].label

    def direction_of(self, algorithm: Algorithm | str | None) -> SortDirection:
        return self._algorithms[self._normalize_key(algorithm)].direction

    @property
    def default_algorithm(self) -> str:
        """配置中指定的默认算法；非法或缺失时退回 Pearson。"""
        raw = self._setting("similarity.algorithm", Algorithm.PEARSON.value)
        try:
            key = Algorithm(raw).value
        except ValueError:
            return Algorithm.PEARSON.value
        return key if key in self._algorithms else Algorithm.PEARSON.value

    # -- 计算 -------------------------------------------------------------
    def compare(
        self,
        series_a: object,
        series_b: object,
        algorithm: Algorithm | str | None = None,
        **params: Any,
    ) -> SimilarityResult:
        """比较两条序列；无效输入以带状态的 :class:`SimilarityResult` 返回。"""
        key = self._normalize_key(algorithm)
        strategy = self._algorithms[key]
        options = dict(params)
        normalize = bool(options.pop("normalize", self._setting("similarity.normalize", True)))
        min_points = int(
            options.pop("min_valid_points", self._setting("similarity.min_valid_points", DEFAULT_MIN_VALID_POINTS))
        )
        return strategy.compare(
            series_a,
            series_b,
            min_valid_points=min_points,
            normalize=normalize,
            **options,
        )

    def compare_all(self, series_a: object, series_b: object, **params: Any) -> tuple[SimilarityResult, ...]:
        """用全部已注册算法依次比较，便于界面对比不同指标。"""
        return tuple(self.compare(series_a, series_b, key, **params) for key in self._algorithms)

    # -- 内部工具 ---------------------------------------------------------
    def _normalize_key(self, algorithm: Algorithm | str | None) -> str:
        if algorithm is None or algorithm == "":
            return self.default_algorithm
        try:
            key = Algorithm(algorithm).value
        except ValueError as exc:
            raise UnknownAlgorithmError(
                f"未知的相似度算法：{algorithm}",
                detail=f"可用算法：{'、'.join(ALGORITHM_LABELS[k] for k in self._algorithms)}",
            ) from exc
        if key not in self._algorithms:
            raise UnknownAlgorithmError(f"相似度算法尚未注册：{key}")
        return key

    def _setting(self, dotted: str, default: Any) -> Any:
        if self._config is None:
            return default
        try:
            value = self._config.get(dotted)
        except Exception:  # pragma: no cover - 配置结构异常时保持可用
            return default
        return default if value is None else value
