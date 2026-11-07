"""MAE 平均绝对误差（设计文档 3.4.2）。"""

from __future__ import annotations

import numpy as np

from core.constants import Algorithm, SortDirection
from modules.similarity.base import SimilarityAlgorithm

__all__ = ["MaeAlgorithm"]


class MaeAlgorithm(SimilarityAlgorithm):
    """平均绝对误差：越小越相似；对离群点不如 RMSE 敏感。"""

    key = Algorithm.MAE.value
    direction = SortDirection.ASC
    needs_normalization = True

    def _compute(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean(np.abs(left - right)))
