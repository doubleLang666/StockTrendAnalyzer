"""多指标联动分析模块。

对应毕业设计设计文档《指标分析功能升级设计文档》：

* :mod:`specs` 指标注册与可用性判断（2.1）；
* :mod:`preparation` 区间对齐与尺度变换（2.2 / 4）；
* :mod:`scoring` 三种算法的 0–100 评分与权重归一化（2.3 / 2.4）；
* :mod:`analysis` 一次分析的编排与综合分（2.4 / 2.5）。

模块只做计算，不依赖 Qt；界面在 :mod:`ui.multi_indicator_page`。
原有的单指标相似度功能（``modules.similarity`` / ``modules.trend_analysis``）保持不变。
"""

from __future__ import annotations

from modules.multi_indicator.analysis import (
    MultiIndicatorRequest,
    MultiIndicatorResult,
    analyze_dataset,
    analyze_frame,
    default_weights,
)
from modules.multi_indicator.preparation import (
    DEFAULT_MIN_VALID_RATIO,
    SeriesPair,
    align_pair,
    prepare_pair,
    transform_values,
)
from modules.multi_indicator.scoring import (
    ALGORITHM_LABELS,
    GRADE_THRESHOLDS,
    SCORE_MODES,
    IndicatorScore,
    compute_indicator_score,
    grade_label,
    normalize_weights,
    score_mode_label,
)
from modules.multi_indicator.search import (
    MultiSearchRequest,
    MultiSearchResult,
    search_history,
)
from modules.multi_indicator.specs import (
    DEFAULT_INDICATOR_KEYS,
    INDICATORS,
    IndicatorSpec,
    available_indicators,
    get_indicator,
    indicator_label,
    indicator_series,
    select_default_indicators,
    unavailable_reason,
)

__all__ = [
    "ALGORITHM_LABELS",
    "DEFAULT_INDICATOR_KEYS",
    "DEFAULT_MIN_VALID_RATIO",
    "GRADE_THRESHOLDS",
    "INDICATORS",
    "IndicatorScore",
    "IndicatorSpec",
    "MultiIndicatorRequest",
    "MultiIndicatorResult",
    "MultiSearchRequest",
    "MultiSearchResult",
    "SCORE_MODES",
    "SeriesPair",
    "align_pair",
    "analyze_dataset",
    "analyze_frame",
    "available_indicators",
    "compute_indicator_score",
    "default_weights",
    "get_indicator",
    "grade_label",
    "indicator_label",
    "indicator_series",
    "normalize_weights",
    "prepare_pair",
    "score_mode_label",
    "search_history",
    "select_default_indicators",
    "transform_values",
    "unavailable_reason",
]
