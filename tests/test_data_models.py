"""core.data_models 单元测试：数据结构、校验与只读语义。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.constants import (
    Algorithm,
    Alignment,
    CalculationStatus,
    DataKind,
    DerivedMethod,
    Normalization,
    SortDirection,
)
from core.data_models import (
    DerivedField,
    MarketDataset,
    ProcessedDataset,
    ProcessingConfig,
    SearchResult,
    SimilarityConfig,
    SimilarityResult,
)
from core.exceptions import ProcessingError, ValidationError


# ---------------------------------------------------------------------------
# MarketDataset
# ---------------------------------------------------------------------------
def test_dataset_normalizes_index_and_columns(sample_frame: pd.DataFrame) -> None:
    shuffled = sample_frame.iloc[::-1]
    dataset = MarketDataset("测试数据", shuffled)

    assert dataset.record_count == 12
    assert dataset.name == "测试数据"
    assert dataset.date_index.is_monotonic_increasing
    assert dataset.date_index.name == "date"
    assert dataset.fields == ("open", "high", "low", "close", "volume")
    assert dataset.start_date == date(2024, 1, 2)
    assert dataset.end_date == date(2024, 1, 17)


def test_dataset_accepts_date_column(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.reset_index()
    dataset = MarketDataset("含日期列", frame)
    assert dataset.record_count == 12
    assert "date" not in dataset.fields


def test_dataset_rejects_duplicate_dates(sample_frame: pd.DataFrame) -> None:
    frame = pd.concat([sample_frame, sample_frame.iloc[[0]]])
    with pytest.raises(ValidationError, match="重复交易日期"):
        MarketDataset("重复", frame)


def test_dataset_rejects_empty_frame() -> None:
    with pytest.raises(ValidationError, match="为空"):
        MarketDataset("空数据", pd.DataFrame({"close": []}))


def test_dataset_requires_close_column(sample_frame: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="收盘价"):
        MarketDataset("无收盘", sample_frame.drop(columns=["close"]))


def test_dataset_rejects_unknown_column(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.assign(pe=12.3)
    with pytest.raises(ValidationError, match="未识别的字段"):
        MarketDataset("未知字段", frame)


def test_dataset_frame_is_isolated_copy(sample_frame: pd.DataFrame) -> None:
    dataset = MarketDataset("只读", sample_frame)
    view = dataset.frame
    view.loc[:, "close"] = 0.0
    assert dataset.series("close").iloc[0] != 0.0


def test_dataset_series_returns_copy(sample_frame: pd.DataFrame) -> None:
    dataset = MarketDataset("只读", sample_frame)
    series = dataset.series("close")
    series.iloc[0] = -999.0
    assert dataset.series("close").iloc[0] != -999.0
    with pytest.raises(ValidationError, match="不包含字段"):
        dataset.series("turnover")


def test_dataset_missing_summary(sample_frame: pd.DataFrame) -> None:
    dataset = MarketDataset("缺失", sample_frame)
    assert dataset.has_missing() is True
    assert dataset.missing_counts()["volume"] == 1
    assert dataset.missing_counts()["close"] == 0
    assert dataset.missing_ratio()["volume"] == pytest.approx(1 / 12)

    summary = dataset.summary()
    assert summary["record_count"] == 12
    assert summary["field_count"] == 5
    assert "收盘价" in summary["fields"]


def test_dataset_slice_dates(sample_frame: pd.DataFrame) -> None:
    dataset = MarketDataset("切片", sample_frame)
    sliced = dataset.slice_dates("2024-01-03", "2024-01-09")
    assert sliced.record_count == 5
    assert sliced.start_date == date(2024, 1, 3)
    assert dataset.record_count == 12  # 原对象不受影响

    with pytest.raises(ValidationError, match="晚于"):
        dataset.slice_dates("2024-01-09", "2024-01-03")
    with pytest.raises(ValidationError, match="没有有效记录"):
        dataset.slice_dates("2030-01-01", "2030-02-01")


def test_dataset_select_is_deep_copy(sample_frame: pd.DataFrame) -> None:
    dataset = MarketDataset("选择", sample_frame)
    selected = dataset.select(["close", "volume"])
    selected.iloc[0, 0] = 0.0
    assert dataset.series("close").iloc[0] != 0.0


# ---------------------------------------------------------------------------
# ProcessingConfig / DerivedField
# ---------------------------------------------------------------------------
def test_processing_config_defaults_and_describe() -> None:
    config = ProcessingConfig()
    assert config.normalization is Normalization.NONE
    assert config.alignment is Alignment.REAL_DATE
    assert config.data_kind is DataKind.ABSOLUTE
    described = config.describe()
    assert described["normalization"] == "none"
    assert described["alignment"] == "real_date"
    assert described["derived"] == []


def test_processing_config_validates_dates() -> None:
    with pytest.raises(ValidationError, match="晚于"):
        ProcessingConfig(start_date=date(2024, 3, 1), end_date=date(2024, 1, 1))
    with pytest.raises(ValidationError, match="上界"):
        ProcessingConfig(min_max_low=1.0, min_max_high=0.0)


def test_derived_field_output_name() -> None:
    field = DerivedField("close", DerivedMethod.PCT_CHANGE)
    assert field.output_name == "close_pct_change"
    assert field.label == "收盘价变化率"

    diff = DerivedField("volume", "diff")
    assert diff.method is DerivedMethod.DIFF
    assert diff.output_name == "volume_diff"


# ---------------------------------------------------------------------------
# ProcessedDataset
# ---------------------------------------------------------------------------
def test_processed_dataset_holds_config_and_provenance() -> None:
    config = ProcessingConfig(fields=("close",), normalization=Normalization.START)
    frame = pd.DataFrame({"close": [100.0, 101.0, 99.0]}, index=pd.RangeIndex(3))
    processed = ProcessedDataset("派生", frame, config, provenance={"warnings": ["x"]})

    assert processed.value_columns == ("close",)
    assert processed.alignment is Alignment.REAL_DATE
    assert processed.provenance["config"]["normalization"] == "start"
    assert processed.provenance["warnings"] == ["x"]
    assert processed.column_label("close") == "收盘价"

    with pytest.raises(ProcessingError, match="不存在数据列"):
        processed.column("amount")


def test_processed_dataset_rejects_empty_frame() -> None:
    with pytest.raises(ProcessingError, match="为空"):
        ProcessedDataset("空", pd.DataFrame(), ProcessingConfig())


# ---------------------------------------------------------------------------
# SimilarityConfig / SimilarityResult / SearchResult
# ---------------------------------------------------------------------------
def test_similarity_config_direction_and_validation() -> None:
    pearson = SimilarityConfig(algorithm=Algorithm.PEARSON)
    assert pearson.direction is SortDirection.DESC

    rmse = SimilarityConfig(algorithm="rmse")
    assert rmse.direction is SortDirection.ASC
    assert rmse.describe()["algorithm"] == "rmse"

    with pytest.raises(ValidationError, match="大于 0"):
        SimilarityConfig(top_n=0)
    with pytest.raises(ValidationError, match="不得少于 2"):
        SimilarityConfig(min_valid_points=1)
    with pytest.raises(ValidationError, match="重叠率"):
        SimilarityConfig(max_overlap_ratio=1.5)


def test_similarity_result_validity_and_sort_key() -> None:
    ok = SimilarityResult(algorithm="pearson", raw_value=0.93, display_value=0.965)
    assert ok.is_valid
    assert ok.label == "Pearson 相关系数"
    assert ok.sort_key() == pytest.approx(-0.93)

    bad = SimilarityResult(algorithm="rmse", raw_value=np.nan, message="NaN")
    assert bad.status is CalculationStatus.FAILED
    assert bad.is_valid is False
    assert bad.sort_key() == pytest.approx(float("inf"))


def test_search_result_describe() -> None:
    series = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2024-01-02", periods=3))
    result = SearchResult(
        rank=1,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
        trading_days=3,
        field="close",
        result=SimilarityResult(algorithm="pearson", raw_value=0.88),
        series=series,
        overlap_ratio=0.0,
    )
    described = result.describe()
    assert described["rank"] == 1
    assert described["trading_days"] == 3
    assert described["score"] == pytest.approx(0.88)
    assert result.algorithm is Algorithm.PEARSON
