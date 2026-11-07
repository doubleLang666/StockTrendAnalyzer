"""全局常量与枚举。

本模块只存放不依赖任何其他项目模块的静态定义，
供 core / modules / ui 各层共享，避免魔法字符串散落各处。
"""

from __future__ import annotations

from enum import Enum
from typing import Final

APP_NAME: Final[str] = "特定时间范围内的股票趋势共性分析系统"
APP_NAME_SHORT: Final[str] = "股票趋势共性分析系统"
APP_VERSION: Final[str] = "0.1.0"

# ---------------------------------------------------------------------------
# 行情字段
# ---------------------------------------------------------------------------
FIELD_DATE: Final[str] = "date"
FIELD_OPEN: Final[str] = "open"
FIELD_HIGH: Final[str] = "high"
FIELD_LOW: Final[str] = "low"
FIELD_CLOSE: Final[str] = "close"
FIELD_VOLUME: Final[str] = "volume"
FIELD_AMOUNT: Final[str] = "amount"
FIELD_TURNOVER: Final[str] = "turnover"

#: 标准字段顺序，DataFrame 列顺序以此为准。
STANDARD_FIELDS: Final[tuple[str, ...]] = (
    FIELD_OPEN,
    FIELD_HIGH,
    FIELD_LOW,
    FIELD_CLOSE,
    FIELD_VOLUME,
    FIELD_AMOUNT,
    FIELD_TURNOVER,
)

#: 中文标签，用于表格表头、图例与提示信息。
FIELD_LABELS: Final[dict[str, str]] = {
    FIELD_OPEN: "开盘价",
    FIELD_HIGH: "最高价",
    FIELD_LOW: "最低价",
    FIELD_CLOSE: "收盘价",
    FIELD_VOLUME: "成交量",
    FIELD_AMOUNT: "成交额",
    FIELD_TURNOVER: "换手率",
}

#: 至少需要存在的字段，否则无法构成一条有效的行情记录。
REQUIRED_FIELDS: Final[tuple[str, ...]] = (FIELD_DATE, FIELD_CLOSE)

PRICE_FIELDS: Final[tuple[str, ...]] = (
    FIELD_OPEN,
    FIELD_HIGH,
    FIELD_LOW,
    FIELD_CLOSE,
)

#: 券商导出表头别名 -> 标准字段。键统一按去空白、小写处理。
FIELD_ALIASES: Final[dict[str, str]] = {
    # 日期
    "date": FIELD_DATE,
    "日期": FIELD_DATE,
    "交易日期": FIELD_DATE,
    "交易日": FIELD_DATE,
    "时间": FIELD_DATE,
    # 开高低收
    "open": FIELD_OPEN,
    "开盘": FIELD_OPEN,
    "开盘价": FIELD_OPEN,
    "high": FIELD_HIGH,
    "最高": FIELD_HIGH,
    "最高价": FIELD_HIGH,
    "low": FIELD_LOW,
    "最低": FIELD_LOW,
    "最低价": FIELD_LOW,
    "close": FIELD_CLOSE,
    "收盘": FIELD_CLOSE,
    "收盘价": FIELD_CLOSE,
    "收": FIELD_CLOSE,
    # 量额
    "volume": FIELD_VOLUME,
    "vol": FIELD_VOLUME,
    "成交量": FIELD_VOLUME,
    "amount": FIELD_AMOUNT,
    "成交额": FIELD_AMOUNT,
    "成交金额": FIELD_AMOUNT,
    "turnover": FIELD_TURNOVER,
    "换手率": FIELD_TURNOVER,
    "换手": FIELD_TURNOVER,
}

# ---------------------------------------------------------------------------
# 文件解析
# ---------------------------------------------------------------------------
#: 编码探测候选顺序：越靠前优先级越高。
ENCODING_CANDIDATES: Final[tuple[str, ...]] = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "big5",
)

DELIMITER_CANDIDATES: Final[tuple[str, ...]] = (",", "\t", ";", "|")

#: 常见日期格式，按优先级尝试。
DATE_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y%m%d %H%M%S",
)

#: 视为缺失值的原始文本标记（大小写不敏感）。
MISSING_TOKENS: Final[frozenset[str]] = frozenset(
    {"", "-", "--", "---", "nan", "none", "null", "n/a", "na", "空", "无"}
)


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------
class Normalization(str, Enum):
    """归一化方式。"""

    NONE = "none"
    START = "start"      # 起始值归一化：区间首个有效值 -> 100
    MIN_MAX = "min_max"  # Min-Max 归一化：默认映射到 [0, 1]


class Alignment(str, Enum):
    """时间对齐方式。"""

    REAL_DATE = "real_date"        # 真实交易日期
    RELATIVE_DAY = "relative_day"  # 相对交易日（起始日 = 0）


class DataKind(str, Enum):
    """数据形态。"""

    ABSOLUTE = "absolute"  # 原始数值
    RELATIVE = "relative"  # 归一化或变化率数据


class DerivedMethod(str, Enum):
    """二级数据计算方法。"""

    PCT_CHANGE = "pct_change"  # 增长率 / 涨跌幅，百分比
    DIFF = "diff"              # 差分
    LOG_RETURN = "log_return"  # 对数收益率


#: 二级数据计算方法的中文名称。
DERIVED_METHOD_LABELS: Final[dict[str, str]] = {
    DerivedMethod.PCT_CHANGE.value: "变化率（%）",
    DerivedMethod.DIFF.value: "差分",
    DerivedMethod.LOG_RETURN.value: "对数收益率",
}

#: 归一化方式的中文名称。
NORMALIZATION_LABELS: Final[dict[str, str]] = {
    Normalization.NONE.value: "不归一化",
    Normalization.START.value: "起始值归一化（首个有效值 = 100）",
    Normalization.MIN_MAX.value: "Min-Max 归一化",
}

#: 时间对齐方式的中文名称。
ALIGNMENT_LABELS: Final[dict[str, str]] = {
    Alignment.REAL_DATE.value: "真实交易日期",
    Alignment.RELATIVE_DAY.value: "相对交易日（起始日 = 第 0 日）",
}

#: 数据形态的中文名称。
DATA_KIND_LABELS: Final[dict[str, str]] = {
    DataKind.ABSOLUTE.value: "绝对数据",
    DataKind.RELATIVE.value: "相对数据",
}


class Algorithm(str, Enum):
    """相似度算法标识。"""

    PEARSON = "pearson"
    RMSE = "rmse"
    MAE = "mae"
    DTW = "dtw"
    COSINE = "cosine"


ALGORITHM_LABELS: Final[dict[str, str]] = {
    Algorithm.PEARSON.value: "Pearson 相关系数",
    Algorithm.RMSE.value: "RMSE 均方根误差",
    Algorithm.MAE.value: "MAE 平均绝对误差",
    Algorithm.DTW.value: "DTW 动态时间规整距离",
    Algorithm.COSINE.value: "余弦相似度",
}


class SortDirection(str, Enum):
    """排序方向。"""

    DESC = "desc"  # 越大越相似（相关系数）
    ASC = "asc"    # 越小越相似（误差类）


#: 每种算法默认的排序方向：相关系数越大越相似，误差类越小越相似。
ALGORITHM_DIRECTIONS: Final[dict[str, SortDirection]] = {
    Algorithm.PEARSON.value: SortDirection.DESC,
    Algorithm.RMSE.value: SortDirection.ASC,
    Algorithm.MAE.value: SortDirection.ASC,
    Algorithm.DTW.value: SortDirection.ASC,
    Algorithm.COSINE.value: SortDirection.DESC,
}


class CalculationStatus(str, Enum):
    """相似度计算状态。"""

    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"
    CONSTANT_SERIES = "constant_series"
    LENGTH_MISMATCH = "length_mismatch"
    INVALID_PARAMETER = "invalid_parameter"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_TOP_N: Final[int] = 3
DEFAULT_MIN_VALID_POINTS: Final[int] = 5
DEFAULT_MAX_OVERLAP_RATIO: Final[float] = 0.5
DEFAULT_MIN_MAX_LOW: Final[float] = 0.0
DEFAULT_MIN_MAX_HIGH: Final[float] = 1.0
START_NORMALIZED_BASE: Final[float] = 100.0
