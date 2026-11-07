"""余弦相似度（设计文档 3.4.2 的扩展算法）。

余弦相似度衡量两条序列"方向"的一致性：把序列当作多维向量求夹角余弦，
结果落在 ``[-1, 1]``，1 表示同向（形状相同、缩放不影响结果），
-1 表示反向。它天然对被比较序列的整体缩放不敏感，因此不需要先归一化。
"""

from __future__ import annotations

import numpy as np

from core.constants import Algorithm, SortDirection
from core.exceptions import AlgorithmError
from modules.similarity.base import SimilarityAlgorithm

__all__ = ["CosineAlgorithm"]


class CosineAlgorithm(SimilarityAlgorithm):
    """余弦相似度（越大越相似）。"""

    key = Algorithm.COSINE.value
    detects_constant = True
    needs_normalization = False
    direction = SortDirection.DESC

    def _compute(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            raise AlgorithmError("曲线范数为 0，余弦相似度无定义")
        return float(np.dot(left, right) / (left_norm * right_norm))
