"""相似度分析模块（设计文档 3.4）。

对外只暴露统一接口：算法实现 + 相似度服务。
"""

from __future__ import annotations

from modules.similarity.base import SimilarityAlgorithm, as_float_array, start_normalized
from modules.similarity.cosine import CosineAlgorithm
from modules.similarity.dtw import DtwAlgorithm, dtw_distance
from modules.similarity.mae import MaeAlgorithm
from modules.similarity.pearson import PearsonAlgorithm
from modules.similarity.rmse import RmseAlgorithm
from modules.similarity.service import SimilarityService

__all__ = [
    "SimilarityAlgorithm",
    "SimilarityService",
    "PearsonAlgorithm",
    "RmseAlgorithm",
    "MaeAlgorithm",
    "DtwAlgorithm",
    "CosineAlgorithm",
    "dtw_distance",
    "as_float_array",
    "start_normalized",
]
