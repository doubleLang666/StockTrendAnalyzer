"""RMSE 均方根误差（设计文档 3.4.2）。"""

from __future__ import annotations

import numpy as np

from core.constants import Algorithm, SortDirection
from modules.similarity.base import SimilarityAlgorithm

__all__ = ["RmseAlgorithm"]


class RmseAlgorithm(SimilarityAlgorithm):
    """均方根误差：越小越相似；价格水平差异大时先按起始值归一化。"""

    key = Algorithm.RMSE.value
    direction = SortDirection.ASC
    needs_normalization = True

    def _compute(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sqrt(np.mean((left - right) ** 2)))
