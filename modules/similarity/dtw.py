"""DTW 动态时间规整（设计文档 3.4.2 的扩展算法）。

与 Pearson / RMSE / MAE 不同，DTW **允许两条曲线长度不一致**：它通过寻找一条
单调路径把两条序列"弯曲"对齐，再累加逐点距离，因此对时间轴上的伸缩不敏感。

约定：

* 输入为两条数值序列，内部先各自剔除无效点（NaN / inf）；
* 距离按路径长度归一化后开方，含义接近"平均每步差异"，与价格水平无关
  （``normalize=True`` 时再按起始值归一化到 100，便于跨区间比较）；
* 输出越小越相似（``SortDirection.ASC``）；
* 计算量随长度平方增长，超过 :data:`MAX_CELLS` 个单元格时明确报错，
  而不是让界面长时间无响应。
"""

from __future__ import annotations

import math

import numpy as np

from core.constants import Algorithm, SortDirection
from core.exceptions import AlgorithmError
from modules.similarity.base import SimilarityAlgorithm

__all__ = ["DtwAlgorithm", "dtw_distance", "MAX_CELLS"]

#: DTW 距离矩阵允许的最大单元格数（约 2000 × 2000），超出则拒绝计算。
MAX_CELLS: int = 4_000_000


def dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    """返回两条序列的归一化 DTW 距离（越小越相似）。

    采用"反对角线"推进的动态规划：``acc[i, j]`` 只依赖
    ``(i-1, j)``、``(i, j-1)`` 与 ``(i-1, j-1)``，而它们都位于更靠前的
    反对角线上，因此每条反对角线可以一次性向量化计算，避免 Python 双层循环。
    """
    n, m = int(left.size), int(right.size)
    if n == 0 or m == 0:
        raise AlgorithmError("DTW 需要两条非空序列")
    if n * m > MAX_CELLS:
        raise AlgorithmError(
            f"DTW 计算量过大（{n}×{m} 个单元格，上限 {MAX_CELLS}），请缩短比较区间"
        )

    costs = (left[:, None] - right[None, :]) ** 2

    acc = np.full((n + 1, m + 1), np.inf, dtype="float64")
    acc[0, 0] = 0.0
    for diagonal in range(2, n + m + 1):
        rows = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        if rows.size == 0:
            continue
        columns = diagonal - rows
        previous = np.minimum(acc[rows - 1, columns], acc[rows, columns - 1])
        previous = np.minimum(previous, acc[rows - 1, columns - 1])
        acc[rows, columns] = costs[rows - 1, columns - 1] + previous

    total = float(acc[n, m])
    if not math.isfinite(total):  # pragma: no cover - 防御性兜底
        raise AlgorithmError("DTW 路径累积距离不是有限值")
    return math.sqrt(total / (n + m))


class DtwAlgorithm(SimilarityAlgorithm):
    """DTW 动态时间规整距离（越小越相似，支持长度不一致）。"""

    key = Algorithm.DTW.value
    detects_constant = False
    needs_normalization = True
    direction = SortDirection.ASC
    supports_length_mismatch = True

    def _compute(self, left: np.ndarray, right: np.ndarray) -> float:
        return dtw_distance(left, right)
