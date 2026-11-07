"""相似度算法的统一接口（对应设计文档 3.4.3）。

所有算法都继承 :class:`SimilarityAlgorithm`，由基类统一完成

* 长度一致性检查；
* NaN / 非有限值掩码；
* 有效数据点数量检查；
* 可选的起始值归一化；
* 常数序列检查；
* 生成 :class:`~core.data_models.SimilarityResult`。

子类只需实现 :meth:`SimilarityAlgorithm._compute`，不得在历史检索等
调用方里重复编写算法公式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
import pandas as pd

from core.constants import (
    ALGORITHM_LABELS,
    DEFAULT_MIN_VALID_POINTS,
    START_NORMALIZED_BASE,
    CalculationStatus,
    SortDirection,
)
from core.data_models import SimilarityResult
from core.exceptions import AlgorithmError

__all__ = ["SimilarityAlgorithm", "as_float_array", "start_normalized"]


def as_float_array(values: object) -> np.ndarray:
    """把 Series / 列表 / ndarray 统一转成一维 float64 数组。"""
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype="float64", copy=True)
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1:
        raise AlgorithmError("相似度算法只接受一维数值序列")
    return array.copy()


def start_normalized(values: np.ndarray) -> np.ndarray | None:
    """按首个有效值归一化到 100；首个有效值为 0 时返回 None。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values.copy()
    base = float(finite[0])
    if base == 0.0:
        return None
    return values / base * START_NORMALIZED_BASE


class SimilarityAlgorithm(ABC):
    """相似度算法基类（策略模式）。"""

    #: 算法在结果中使用的唯一标识，例如 ``"pearson"``。
    key: str = ""
    #: 该算法是否受"曲线为常数"影响（相关系数会无定义，误差类不受影响）。
    detects_constant: bool = False
    #: 该算法是否需要在计算前做起始值归一化才有意义（误差类需要）。
    needs_normalization: bool = False
    #: 该算法默认排序方向。
    direction: SortDirection = SortDirection.DESC
    #: 该算法是否天生支持长度不一致的曲线（DTW 通过路径弯曲对齐，不需要等长）。
    supports_length_mismatch: bool = False

    @property
    def label(self) -> str:
        return ALGORITHM_LABELS.get(self.key, self.key)

    # -- 对外接口 ---------------------------------------------------------
    def compare(
        self,
        series_a: object,
        series_b: object,
        *,
        min_valid_points: int = DEFAULT_MIN_VALID_POINTS,
        normalize: bool = False,
        **params: object,
    ) -> SimilarityResult:
        """比较两条数值序列，返回完整结果；任何异常都以无效结果表达。"""
        left = as_float_array(series_a)
        right = as_float_array(series_b)
        unequal = left.size != right.size

        if unequal and not self.supports_length_mismatch:
            return self._invalid(
                CalculationStatus.LENGTH_MISMATCH,
                f"两条曲线长度不一致（{left.size} 与 {right.size} 个数据点），未做任何截断或补值",
            )

        min_points = max(2, int(min_valid_points))
        if unequal:
            # 支持不等长的算法（DTW）：各自剔除无效点，只在两条有效子序列上对齐。
            a = left[np.isfinite(left)]
            b = right[np.isfinite(right)]
            sample_size = int(min(a.size, b.size))
            if a.size < min_points or b.size < min_points:
                return self._invalid(
                    CalculationStatus.INSUFFICIENT_DATA,
                    f"有效数据点不足：{a.size} 与 {b.size} 个（每条曲线至少需要 {min_points} 个）",
                    sample_size=sample_size,
                )
        else:
            mask = np.isfinite(left) & np.isfinite(right)
            sample_size = int(mask.sum())
            if sample_size < min_points:
                return self._invalid(
                    CalculationStatus.INSUFFICIENT_DATA,
                    f"有效数据点不足：{sample_size} 个（至少需要 {min_points} 个）",
                    sample_size=sample_size,
                )

            a = left[mask]
            b = right[mask]

        normalized = bool(normalize and self.needs_normalization)
        if normalized:
            base_a = start_normalized(a)
            base_b = start_normalized(b)
            if base_a is None or base_b is None:
                return self._invalid(
                    CalculationStatus.INVALID_PARAMETER,
                    "起始值为 0，无法归一化后再比较",
                    sample_size=sample_size,
                )
            a, b = base_a, base_b

        if self.detects_constant and (np.ptp(a) == 0.0 or np.ptp(b) == 0.0):
            return self._invalid(
                CalculationStatus.CONSTANT_SERIES,
                "其中一条曲线在所选区间内为常数，相关系数无定义",
                sample_size=sample_size,
                normalized=normalized,
            )

        try:
            raw_value = float(self._compute(a, b))
        except AlgorithmError as exc:
            return self._invalid(
                CalculationStatus.FAILED,
                exc.user_message(),
                sample_size=sample_size,
                normalized=normalized,
            )
        except Exception as exc:  # pragma: no cover - 防御性兜底
            return self._invalid(
                CalculationStatus.FAILED,
                f"算法计算失败：{exc}",
                sample_size=sample_size,
                normalized=normalized,
            )

        result = SimilarityResult(
            algorithm=self.key,
            raw_value=raw_value,
            direction=self.direction,
            display_value=round(raw_value, 6),
            sample_size=sample_size,
            normalized=normalized,
            params=dict(params),
        )
        if not result.is_valid:  # 非有限数值等，由数据模型统一标记为失败
            return result
        return result

    # -- 子类实现 ---------------------------------------------------------
    @abstractmethod
    def _compute(self, left: np.ndarray, right: np.ndarray) -> float:
        """在已经完成掩码与归一化的等长数组上计算指标。"""

    # -- 内部工具 ---------------------------------------------------------
    def _invalid(
        self,
        status: CalculationStatus,
        message: str,
        *,
        sample_size: int = 0,
        normalized: bool = False,
    ) -> SimilarityResult:
        return SimilarityResult(
            algorithm=self.key,
            raw_value=None,
            direction=self.direction,
            status=status,
            message=message,
            sample_size=sample_size,
            normalized=normalized,
        )


#: 避免 linters 误报未使用（SortDirection 参与类型注解）。
_SORT_DIRECTIONS: Final = SortDirection
