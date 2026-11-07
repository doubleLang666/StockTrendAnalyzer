"""Pearson 相关系数（设计文档 3.4.2）。"""

from __future__ import annotations

import numpy as np

from core.constants import Algorithm, SortDirection
from modules.similarity.base import SimilarityAlgorithm

__all__ = ["PearsonAlgorithm"]


class PearsonAlgorithm(SimilarityAlgorithm):
    """皮尔逊相关系数：衡量两条曲线的线性相关程度，取值 -1 ~ 1。"""

    key = Algorithm.PEARSON.value
    direction = SortDirection.DESC
    detects_constant = True
    needs_normalization = False  # 相关系数对缩放不敏感，无需归一化

    def _compute(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.corrcoef(left, right)[0, 1])
