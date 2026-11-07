"""统一数据模型。

设计原则（对应设计文档 4.3 / 第五章）：

* 原始数据（MarketDataset）与处理结果（ProcessedDataset）严格分离；
* 原始数据只读，任何处理都必须返回新对象；
* 缺失值统一使用 ``NaN``，绝不自动填充为 0；
* 每个模型自带可追溯的元信息（来源、字段、配置）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

import numpy as np
import pandas as pd

from core.constants import (
    ALGORITHM_DIRECTIONS,
    DEFAULT_MAX_OVERLAP_RATIO,
    DEFAULT_MIN_MAX_HIGH,
    DEFAULT_MIN_MAX_LOW,
    DEFAULT_MIN_VALID_POINTS,
    DEFAULT_TOP_N,
    FIELD_CLOSE,
    FIELD_DATE,
    FIELD_LABELS,
    STANDARD_FIELDS,
    Algorithm,
    Alignment,
    CalculationStatus,
    DataKind,
    DerivedMethod,
    Normalization,
    SortDirection,
)
from core.exceptions import ProcessingError, ValidationError

__all__ = [
    "MarketDataset",
    "ProcessingConfig",
    "DerivedField",
    "ProcessedDataset",
    "SimilarityConfig",
    "SimilarityResult",
    "SearchResult",
    "DataIssue",
    "ImportReport",
    "IssueSeverity",
    "field_label",
]


def field_label(name: str) -> str:
    """返回字段的中文名称，未知字段原样返回。"""
    return FIELD_LABELS.get(name, name)


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    return stamp.date()


class MarketDataset:
    """单份行情数据集（原始数据，只读语义）。

    内部持有 ``pandas.DataFrame``：

    * 索引：``DatetimeIndex``，名称为 ``date``，升序、唯一、已归一化到日；
    * 列：仅包含标准字段（见 ``core.constants.STANDARD_FIELDS``）的子集，
      缺失字段不出现在列中（不以全空列冒充缺失字段）。

    通过 :attr:`frame` 取得的是数据块的浅副本（pandas 写时复制），
    修改它不会影响本对象，但也不应用它做处理，处理请使用
    :mod:`modules.data_processing`。
    """

    __slots__ = (
        "_name",
        "_frame",
        "_source_path",
        "_source_encoding",
        "_declared_fields",
        "_notes",
    )

    def __init__(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        source_path: str | Path | None = None,
        source_encoding: str | None = None,
        declared_fields: Iterable[str] | None = None,
        notes: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError("行情数据必须是 pandas.DataFrame")
        self._name = (str(name or "").strip() or "未命名数据")
        self._frame = self._normalize(frame)
        self._source_path = Path(source_path) if source_path is not None else None
        self._source_encoding = source_encoding
        self._declared_fields = tuple(declared_fields or self._frame.columns)
        self._notes = tuple(notes or ())

    # ------------------------------------------------------------------
    # 构造期校验
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy(deep=True)

        # 允许把日期放在普通列里传入。
        if FIELD_DATE in data.columns:
            data = data.set_index(FIELD_DATE)

        if data.empty:
            raise ValidationError("行情数据为空，没有任何记录")

        try:
            index = pd.to_datetime(data.index)
        except (ValueError, TypeError) as exc:  # pragma: no cover - 依输入而定
            raise ValidationError(f"日期索引无法转换为标准日期：{exc}") from exc

        index = pd.DatetimeIndex(index)
        if index.tz is not None:
            index = index.tz_localize(None)
        index = index.normalize()
        index.name = FIELD_DATE

        if index.has_duplicates:
            duplicated = index[index.duplicated()].unique()
            sample = ", ".join(str(item.date()) for item in duplicated[:3])
            raise ValidationError(f"存在重复交易日期：{sample}")

        data.index = index
        data = data.sort_index()

        data.columns = [str(col).strip().lower() for col in data.columns]
        unknown = [col for col in data.columns if col not in STANDARD_FIELDS]
        if unknown:
            raise ValidationError(f"存在未识别的字段：{', '.join(unknown)}")
        if FIELD_CLOSE not in data.columns:
            raise ValidationError("缺少收盘价字段，无法构成有效行情数据")

        ordered = [name for name in STANDARD_FIELDS if name in data.columns]
        return data.loc[:, ordered]

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def source_encoding(self) -> str | None:
        return self._source_encoding

    @property
    def declared_fields(self) -> tuple[str, ...]:
        return self._declared_fields

    @property
    def notes(self) -> tuple[str, ...]:
        return self._notes

    @property
    def frame(self) -> pd.DataFrame:
        """返回数据块的浅副本（写时复制）。

        调用方不得把它当作处理结果就地修改，处理逻辑请走数据处理器。
        """
        return self._frame.copy(deep=False)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self._frame.columns)

    @property
    def date_index(self) -> pd.DatetimeIndex:
        return self._frame.index

    @property
    def record_count(self) -> int:
        return int(len(self._frame))

    @property
    def start_date(self) -> date | None:
        return None if self._frame.empty else self._frame.index[0].date()

    @property
    def end_date(self) -> date | None:
        return None if self._frame.empty else self._frame.index[-1].date()

    def __len__(self) -> int:
        return int(len(self._frame))

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"MarketDataset(name={self._name!r}, records={self.record_count}, "
            f"fields={self.fields})"
        )

    # ------------------------------------------------------------------
    # 数据访问
    # ------------------------------------------------------------------
    def has_field(self, name: str) -> bool:
        return name in self._frame.columns

    def require_field(self, name: str) -> str:
        if name not in self._frame.columns:
            raise ValidationError(f"数据「{self._name}」不包含字段：{field_label(name)}")
        return name

    def series(self, name: str) -> pd.Series:
        """返回指定字段的序列副本；缺失字段抛出 ValidationError。"""
        self.require_field(name)
        return self._frame[name].copy()

    def missing_counts(self) -> dict[str, int]:
        """各字段的缺失值个数。"""
        return {
            str(col): int(self._frame[col].isna().sum())
            for col in self._frame.columns
        }

    def missing_ratio(self) -> dict[str, float]:
        total = self.record_count or 1
        return {key: value / total for key, value in self.missing_counts().items()}

    def has_missing(self) -> bool:
        return bool(self._frame.isna().any().any())

    def summary(self) -> dict[str, Any]:
        """供 UI 展示的数据概览。"""
        return {
            "name": self._name,
            "source": str(self._source_path) if self._source_path else "",
            "encoding": self._source_encoding or "",
            "record_count": self.record_count,
            "field_count": len(self.fields),
            "fields": [field_label(name) for name in self.fields],
            "start_date": self.start_date,
            "end_date": self.end_date,
            "missing_counts": self.missing_counts(),
        }

    # ------------------------------------------------------------------
    # 派生
    # ------------------------------------------------------------------
    def select(self, fields: Iterable[str]) -> pd.DataFrame:
        """按字段顺序返回 DataFrame 深副本。"""
        ordered: list[str] = []
        for name in fields:
            self.require_field(name)
            ordered.append(name)
        return self._frame.loc[:, ordered].copy(deep=True)

    def slice_dates(self, start: Any = None, end: Any = None) -> "MarketDataset":
        """按日期区间（闭区间）返回新的数据集，原对象不变。"""
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValidationError("日期区间的起始日期晚于结束日期")

        mask = np.ones(self.record_count, dtype=bool)
        if start_ts is not None:
            mask &= self._frame.index >= start_ts.normalize()
        if end_ts is not None:
            mask &= self._frame.index <= end_ts.normalize()

        sliced = self._frame.loc[mask]
        if sliced.empty:
            raise ValidationError("指定日期区间内没有有效记录")
        return MarketDataset(
            self._name,
            sliced,
            source_path=self._source_path,
            source_encoding=self._source_encoding,
            declared_fields=self._declared_fields,
            notes=self._notes,
        )

    def with_frame(self, frame: pd.DataFrame, *, notes: Iterable[str] | None = None) -> "MarketDataset":
        """以同一元信息构建新数据集（用于仅重排列等轻量变换）。"""
        return MarketDataset(
            self._name,
            frame,
            source_path=self._source_path,
            source_encoding=self._source_encoding,
            declared_fields=self._declared_fields,
            notes=tuple(notes) if notes is not None else self._notes,
        )

    def with_name(self, name: str) -> "MarketDataset":
        return MarketDataset(
            name,
            self._frame,
            source_path=self._source_path,
            source_encoding=self._source_encoding,
            declared_fields=self._declared_fields,
            notes=self._notes,
        )


# ---------------------------------------------------------------------------
# 处理配置与结果
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DerivedField:
    """二级数据定义，例如「收盘价的日涨跌幅」。"""

    base_field: str
    method: DerivedMethod | str = DerivedMethod.PCT_CHANGE

    def __post_init__(self) -> None:
        method = DerivedMethod(self.method)
        object.__setattr__(self, "method", method)

    @property
    def output_name(self) -> str:
        return f"{self.base_field}_{self.method.value}"

    @property
    def label(self) -> str:
        method_labels = {
            DerivedMethod.PCT_CHANGE: "变化率",
            DerivedMethod.DIFF: "差分",
            DerivedMethod.LOG_RETURN: "对数收益率",
        }
        return f"{field_label(self.base_field)}{method_labels[self.method]}"


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """数据处理配置（对应设计文档 3.2.2）。

    所有参数都会写入 :class:`ProcessedDataset` 的追溯信息。
    """

    fields: tuple[str, ...] = ()
    start_date: date | None = None
    end_date: date | None = None
    derived: tuple[DerivedField, ...] = ()
    normalization: Normalization | str = Normalization.NONE
    alignment: Alignment | str = Alignment.REAL_DATE
    data_kind: DataKind | str = DataKind.ABSOLUTE
    min_max_low: float = DEFAULT_MIN_MAX_LOW
    min_max_high: float = DEFAULT_MIN_MAX_HIGH

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(dict.fromkeys(self.fields)))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "normalization", Normalization(self.normalization))
        object.__setattr__(self, "alignment", Alignment(self.alignment))
        object.__setattr__(self, "data_kind", DataKind(self.data_kind))
        object.__setattr__(self, "start_date", _coerce_date(self.start_date))
        object.__setattr__(self, "end_date", _coerce_date(self.end_date))
        self.validate()

    def validate(self) -> None:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValidationError("起始日期不能晚于结束日期")
        if self.min_max_high <= self.min_max_low:
            raise ValidationError("Min-Max 归一化的上界必须大于下界")

    def with_fields(self, fields: Iterable[str]) -> "ProcessingConfig":
        return replace(self, fields=tuple(fields))

    def describe(self) -> dict[str, Any]:
        """可写入日志 / 导出结果的配置快照。"""
        return {
            "fields": list(self.fields),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "derived": [
                {
                    "base_field": item.base_field,
                    "method": item.method.value,
                    "output_name": item.output_name,
                }
                for item in self.derived
            ],
            "normalization": self.normalization.value,
            "alignment": self.alignment.value,
            "data_kind": self.data_kind.value,
            "min_max_range": [self.min_max_low, self.min_max_high],
        }


class ProcessedDataset:
    """处理后的数据集。

    * ``real_date`` 对齐时索引为 ``DatetimeIndex``；
    * ``relative_day`` 对齐时索引为整数交易日序号（0, 1, 2, ...）。

    原始数据不在本对象中，仅通过 :attr:`provenance` 记录来源。
    """

    __slots__ = ("_name", "_config", "_frame", "_value_columns", "_provenance")

    def __init__(
        self,
        name: str,
        frame: pd.DataFrame,
        config: ProcessingConfig,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise ProcessingError("处理结果必须是 pandas.DataFrame")
        if frame.empty:
            raise ProcessingError("处理结果为空，请检查字段与日期范围")
        self._name = str(name or "未命名数据")
        self._config = config
        self._frame = frame.copy(deep=True)
        self._value_columns = tuple(self._frame.columns)
        base = {"source_name": self._name, "config": config.describe()}
        if provenance:
            base.update(dict(provenance))
        self._provenance = base

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> ProcessingConfig:
        return self._config

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=False)

    @property
    def value_columns(self) -> tuple[str, ...]:
        return self._value_columns

    @property
    def index(self) -> pd.Index:
        return self._frame.index

    @property
    def alignment(self) -> Alignment:
        return self._config.alignment

    @property
    def data_kind(self) -> DataKind:
        return self._config.data_kind

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def __len__(self) -> int:
        return int(len(self._frame))

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"ProcessedDataset(name={self._name!r}, rows={len(self)}, "
            f"columns={self.value_columns}, alignment={self.alignment.value})"
        )

    def has_column(self, name: str) -> bool:
        return name in self._frame.columns

    def column(self, name: str) -> pd.Series:
        """返回数值列副本；列不存在时抛出 ProcessingError。"""
        if name not in self._frame.columns:
            raise ProcessingError(f"处理结果中不存在数据列：{name}")
        return self._frame[name].copy()

    def column_label(self, name: str) -> str:
        for item in self._config.derived:
            if item.output_name == name:
                return item.label
        return field_label(name)

    def window(self, start: int, stop: int) -> pd.DataFrame:
        """按位置切片，返回 DataFrame 深副本（左闭右开）。"""
        return self._frame.iloc[start:stop].copy(deep=True)


# ---------------------------------------------------------------------------
# 相似度
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    """相似度计算与历史检索配置（对应设计文档 3.4、3.3.4）。"""

    algorithm: Algorithm | str = Algorithm.PEARSON
    #: 计算误差类指标前是否先做起始值归一化，避免价格水平差异掩盖形态相似。
    normalize: bool = True
    top_n: int = DEFAULT_TOP_N
    min_valid_points: int = DEFAULT_MIN_VALID_POINTS
    max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithm", Algorithm(self.algorithm))
        object.__setattr__(self, "params", dict(self.params))
        if self.top_n < 1:
            raise ValidationError("返回结果数量必须大于 0")
        if self.min_valid_points < 2:
            raise ValidationError("最少有效数据点不得少于 2")
        if not 0.0 <= self.max_overlap_ratio <= 1.0:
            raise ValidationError("重叠率阈值必须位于 0 到 1 之间")

    @property
    def direction(self) -> SortDirection:
        return (
            SortDirection.DESC
            if self.algorithm is Algorithm.PEARSON
            else SortDirection.ASC
        )

    def describe(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "normalize": self.normalize,
            "top_n": self.top_n,
            "min_valid_points": self.min_valid_points,
            "max_overlap_ratio": self.max_overlap_ratio,
            "params": dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """一次相似度计算的完整结果（对应设计文档 3.4.3）。"""

    algorithm: Algorithm | str
    raw_value: float | None = None
    direction: SortDirection | str | None = None
    display_value: float | None = None
    status: CalculationStatus | str = CalculationStatus.OK
    message: str = ""
    sample_size: int = 0
    normalized: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithm", Algorithm(self.algorithm))
        direction = self.direction
        if direction is None:
            direction = ALGORITHM_DIRECTIONS.get(self.algorithm.value, SortDirection.DESC)
        object.__setattr__(self, "direction", SortDirection(direction))
        object.__setattr__(self, "status", CalculationStatus(self.status))
        object.__setattr__(self, "params", dict(self.params))
        if self.raw_value is not None and not np.isfinite(self.raw_value):
            object.__setattr__(self, "raw_value", None)
            object.__setattr__(self, "status", CalculationStatus.FAILED)
            object.__setattr__(self, "message", self.message or "计算结果不是有限数值")

    @property
    def is_valid(self) -> bool:
        return self.status is CalculationStatus.OK and self.raw_value is not None

    @property
    def label(self) -> str:
        from core.constants import ALGORITHM_LABELS

        return ALGORITHM_LABELS.get(self.algorithm.value, self.algorithm.value)

    def sort_key(self) -> float:
        """返回用于排序的数值；无效结果排在最末。"""
        if not self.is_valid:
            return float("-inf") if self.direction is SortDirection.DESC else float("inf")
        value = float(self.raw_value)  # type: ignore[arg-type]
        return -value if self.direction is SortDirection.DESC else value


@dataclass(frozen=True, slots=True)
class SearchResult:
    """历史区间检索的单条结果（对应设计文档 3.3.4）。"""

    rank: int
    start_date: date
    end_date: date
    trading_days: int
    field: str
    result: SimilarityResult
    series: pd.Series
    overlap_ratio: float = 0.0

    @property
    def score(self) -> float | None:
        return self.result.raw_value

    @property
    def algorithm(self) -> Algorithm:
        return self.result.algorithm

    def describe(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_days": self.trading_days,
            "field": self.field,
            "algorithm": self.result.algorithm.value,
            "score": self.result.raw_value,
            "status": self.result.status.value,
            "overlap_ratio": self.overlap_ratio,
        }


class IssueSeverity(str, Enum):
    """数据问题严重程度。"""

    INFO = "info"        # 仅提示，例如自动排序
    WARNING = "warning"  # 需要用户关注，例如缺失值、重复日期
    ERROR = "error"      # 数据明显异常，例如负价格、最高价低于最低价


ISSUE_SEVERITY_LABELS: Final[dict[str, str]] = {
    IssueSeverity.INFO.value: "提示",
    IssueSeverity.WARNING.value: "警告",
    IssueSeverity.ERROR.value: "错误",
}


@dataclass(frozen=True, slots=True)
class DataIssue:
    """一条数据问题记录（导入校验、后续处理校验共用）。

    设计文档 3.1.4 要求异常数据不得静默删除，因此所有问题都进入本结构，
    由界面完整展示给用户。
    """

    code: str
    message: str
    severity: IssueSeverity | str = IssueSeverity.WARNING
    field: str | None = None
    count: int = 0
    samples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", IssueSeverity(self.severity))
        object.__setattr__(self, "samples", tuple(str(item) for item in self.samples))

    @property
    def severity_label(self) -> str:
        return ISSUE_SEVERITY_LABELS.get(self.severity.value, self.severity.value)

    @property
    def field_label(self) -> str:
        return field_label(self.field) if self.field else ""

    def describe(self) -> str:
        """拼接成一行可直接展示的文本。"""
        parts = [self.message]
        if self.count:
            parts.append(f"共 {self.count} 处")
        if self.samples:
            parts.append("示例：" + "、".join(self.samples[:5]))
        return "；".join(parts)


@dataclass(frozen=True, slots=True)
class ImportReport:
    """一次文件导入的完整报告（编码、映射、行数、问题清单）。"""

    source_path: Path | None = None
    dataset_name: str = ""
    code: str = ""
    encoding: str = ""
    delimiter_label: str = ""
    header_line: int = 0
    mapping: Mapping[str, str] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    ignored_columns: tuple[str, ...] = ()
    total_rows: int = 0
    imported_rows: int = 0
    dropped_rows: int = 0
    issues: tuple[DataIssue, ...] = ()
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", dict(self.mapping))
        object.__setattr__(self, "missing_fields", tuple(self.missing_fields))
        object.__setattr__(self, "ignored_columns", tuple(self.ignored_columns))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is IssueSeverity.WARNING)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def summary_text(self) -> str:
        if self.is_clean:
            return "数据校验通过，未发现异常。"
        if self.error_count:
            return f"发现 {self.error_count} 类错误、{self.warning_count} 类警告，请查看校验结果。"
        if self.warning_count:
            return f"发现 {self.warning_count} 类需要注意的问题，请查看校验结果。"
        return f"数据校验通过，另有 {len(self.issues)} 条提示。"

    def describe(self) -> dict[str, Any]:
        return {
            "source": str(self.source_path) if self.source_path else "",
            "name": self.dataset_name,
            "code": self.code,
            "encoding": self.encoding,
            "delimiter": self.delimiter_label,
            "header_line": self.header_line,
            "mapping": dict(self.mapping),
            "missing_fields": list(self.missing_fields),
            "ignored_columns": list(self.ignored_columns),
            "total_rows": self.total_rows,
            "imported_rows": self.imported_rows,
            "dropped_rows": self.dropped_rows,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
