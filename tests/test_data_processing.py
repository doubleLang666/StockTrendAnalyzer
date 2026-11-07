"""数据处理模块测试：二级数据、归一化、对齐、服务编排与界面。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import AppConfig
from app.service_registry import ServiceRegistry
from core.constants import Alignment, DataKind, DerivedMethod, Normalization
from core.data_models import DerivedField, MarketDataset, ProcessingConfig
from core.exceptions import InsufficientDataError, ProcessingError
from modules.data_processing import (
    DataProcessor,
    apply_alignment,
    compute_derived,
    normalize_series,
    select_frame,
)
from modules.data_processing.selector import numeric_fields

_DATES = pd.bdate_range("2024-01-02", periods=6)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 10.2, np.nan, 11.5, 11.0, 12.0],
            "high": [10.5, 10.8, np.nan, 12.0, 11.4, 12.6],
            "low": [9.8, 10.0, np.nan, 11.0, 10.6, 11.8],
            "close": [10.0, 10.5, np.nan, 11.8, 11.2, 12.4],
            "volume": [1000.0, 1200.0, np.nan, 1500.0, 900.0, 1100.0],
        },
        index=pd.DatetimeIndex(_DATES, name="date"),
    )


def _dataset() -> MarketDataset:
    return MarketDataset(name="测试指数", frame=_frame(), source_path=Path("sample.txt"))


# ---------------------------------------------------------------------------
# 二级数据
# ---------------------------------------------------------------------------
def test_pct_change_uses_previous_valid_value() -> None:
    series = pd.Series([10.0, 11.0, np.nan, 12.1], index=_DATES[:4])
    result = compute_derived(series, DerivedMethod.PCT_CHANGE)

    assert np.isnan(result.iloc[0])  # 首条记录没有前一有效值
    assert result.iloc[1] == pytest.approx(10.0)
    assert np.isnan(result.iloc[2])  # 当前值缺失
    assert result.iloc[3] == pytest.approx(10.0)  # 对比的是 11.0，中间缺失不影响


def test_pct_change_skips_zero_previous_value() -> None:
    series = pd.Series([0.0, 5.0, 6.0], index=_DATES[:3])
    result = compute_derived(series, DerivedMethod.PCT_CHANGE)

    assert np.isnan(result.iloc[1])  # 前一有效值为 0，不做除法
    assert result.iloc[2] == pytest.approx(20.0)


def test_diff_and_log_return() -> None:
    series = pd.Series([10.0, 12.0, 9.0], index=_DATES[:3])

    diff = compute_derived(series, DerivedMethod.DIFF)
    assert np.isnan(diff.iloc[0])
    assert diff.iloc[1] == pytest.approx(2.0)
    assert diff.iloc[2] == pytest.approx(-3.0)

    log_return = compute_derived(series, DerivedMethod.LOG_RETURN)
    assert log_return.iloc[1] == pytest.approx(np.log(12.0 / 10.0))
    assert log_return.iloc[2] == pytest.approx(np.log(9.0 / 12.0))


def test_unknown_derived_method_raises() -> None:
    with pytest.raises(ProcessingError):
        compute_derived(pd.Series([1.0, 2.0]), "moving_average")


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def test_start_normalization_sets_first_valid_value_to_100() -> None:
    series = pd.Series([np.nan, 4.0, 6.0], index=_DATES[:3])
    outcome = normalize_series(series, Normalization.START)

    assert outcome.base_value == pytest.approx(4.0)
    assert np.isnan(outcome.series.iloc[0])
    assert outcome.series.iloc[1] == pytest.approx(100.0)
    assert outcome.series.iloc[2] == pytest.approx(150.0)


def test_start_normalization_rejects_zero_base() -> None:
    series = pd.Series([0.0, 1.0], index=_DATES[:2])
    with pytest.raises(ProcessingError):
        normalize_series(series, Normalization.START)


def test_min_max_normalization_maps_to_default_range() -> None:
    series = pd.Series([2.0, 4.0, 6.0], index=_DATES[:3])
    outcome = normalize_series(series, Normalization.MIN_MAX)

    assert outcome.series.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert outcome.extra["data_min"] == pytest.approx(2.0)
    assert outcome.extra["data_max"] == pytest.approx(6.0)


def test_min_max_normalization_custom_range_and_constant_series() -> None:
    series = pd.Series([5.0, 5.0], index=_DATES[:2])
    outcome = normalize_series(series, Normalization.MIN_MAX, low=-1.0, high=1.0)

    assert outcome.series.tolist() == pytest.approx([-1.0, -1.0])
    assert outcome.note  # 明确说明取值恒定，而不是除零


def test_min_max_invalid_range_raises() -> None:
    with pytest.raises(ProcessingError):
        normalize_series(pd.Series([1.0, 2.0]), Normalization.MIN_MAX, low=1.0, high=1.0)


# ---------------------------------------------------------------------------
# 对齐
# ---------------------------------------------------------------------------
def test_alignment_modes() -> None:
    frame = _frame().loc[:, ["close"]]

    real = apply_alignment(frame, Alignment.REAL_DATE)
    assert isinstance(real.index, pd.DatetimeIndex)
    assert real.index.name == "date"

    relative = apply_alignment(frame, Alignment.RELATIVE_DAY)
    assert isinstance(relative.index, pd.RangeIndex)
    assert relative.index.tolist() == [0, 1, 2, 3, 4, 5]
    assert relative["close"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 数据选择
# ---------------------------------------------------------------------------
def test_numeric_fields_excludes_date() -> None:
    assert numeric_fields(_dataset()) == ("open", "high", "low", "close", "volume")


def test_select_frame_clips_dates_and_reports_missing_fields() -> None:
    frame, notes = select_frame(
        _dataset(),
        fields=("close", "amount"),
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 4),
    )

    assert list(frame.columns) == ["close"]
    assert len(frame) == 2
    assert any("不存在" in note for note in notes)
    assert any("裁剪" in note for note in notes)


def test_select_frame_empty_range_raises() -> None:
    with pytest.raises(InsufficientDataError):
        select_frame(
            _dataset(),
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
        )


# ---------------------------------------------------------------------------
# 服务编排
# ---------------------------------------------------------------------------
def test_processor_combines_selection_derived_normalization_and_alignment() -> None:
    dataset = _dataset()
    config = ProcessingConfig(
        fields=("close",),
        derived=(DerivedField("close", DerivedMethod.PCT_CHANGE),),
        normalization=Normalization.START,
        alignment=Alignment.RELATIVE_DAY,
    )

    result = DataProcessor().process(dataset, config)

    assert result.value_columns == ("close", "close_pct_change")
    assert result.alignment is Alignment.RELATIVE_DAY
    assert result.data_kind is DataKind.RELATIVE
    assert result.index.tolist() == [0, 1, 2, 3, 4, 5]
    assert result.column("close").iloc[0] == pytest.approx(100.0)
    assert np.isnan(result.column("close_pct_change").iloc[0])

    provenance = result.provenance
    assert provenance["normalization"]["close"]["base_value"] == pytest.approx(10.0)
    assert provenance["result_rows"] == 6
    assert provenance["alignment_axis"] == "相对交易日"
    assert result.column_label("close_pct_change") == "收盘价变化率"


def test_processor_keeps_absolute_data_when_no_transform() -> None:
    result = DataProcessor().process(
        _dataset(),
        ProcessingConfig(fields=("close",), alignment=Alignment.REAL_DATE),
    )

    assert result.data_kind is DataKind.ABSOLUTE
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.column("close").iloc[0] == pytest.approx(10.0)


def test_processor_does_not_modify_source_dataset() -> None:
    dataset = _dataset()
    before = dataset.frame.copy(deep=True)

    DataProcessor().process(
        dataset,
        ProcessingConfig(fields=("close",), normalization=Normalization.MIN_MAX),
    )

    pd.testing.assert_frame_equal(dataset.frame, before)


def test_processor_notes_missing_derived_base() -> None:
    result = DataProcessor().process(
        _dataset(),
        ProcessingConfig(
            fields=("close",),
            derived=(DerivedField("volume", DerivedMethod.PCT_CHANGE),),
        ),
    )

    assert result.value_columns == ("close",)
    assert any("二级数据跳过" in note for note in result.provenance["notes"])


def test_processor_marks_relative_kind_when_config_declares_absolute() -> None:
    result = DataProcessor().process(
        _dataset(),
        ProcessingConfig(
            fields=("close",),
            normalization=Normalization.MIN_MAX,
            data_kind=DataKind.ABSOLUTE,
        ),
    )

    assert result.data_kind is DataKind.RELATIVE
    assert any("结果按相对数据标记" in note for note in result.provenance["notes"])


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
_SAMPLE = (
    "                  测试指数 (000001)\n"
    "\n"
    "      时间\t开盘\t最高\t最低\t收盘\t成交量\n"
    "2024/01/02\t9.80\t10.20\t9.70\t10.00\t900\n"
    "2024/01/03\t10.00\t11.00\t9.50\t10.50\t1,000\n"
    "2024/01/04\t10.50\t10.80\t10.20\t10.60\t1,100\n"
)


def test_processing_page_generates_result(qapp, tmp_path: Path) -> None:
    from modules.data_import.importer import TxtImporter
    from ui.import_page import ImportPage
    from ui.main_window import MainWindow
    from ui.processing_page import ProcessingPage

    sample = tmp_path / "sample.txt"
    sample.write_bytes(_SAMPLE.encode("gbk"))

    registry = ServiceRegistry()
    registry.register("importer", TxtImporter())
    registry.register("processor", DataProcessor())
    window = MainWindow(AppConfig.load(tmp_path / "config.json"), registry)

    import_page = window.tabs.widget(0)
    processing_page = window.tabs.widget(1)
    assert isinstance(import_page, ImportPage)
    assert isinstance(processing_page, ProcessingPage)

    import_page.import_paths([str(sample)])
    assert processing_page.combo_dataset.count() == 1
    assert processing_page.current_dataset is not None

    processing_page.combo_normalization.setCurrentIndex(
        [Normalization.NONE, Normalization.START, Normalization.MIN_MAX].index(Normalization.START)
    )
    processing_page.combo_alignment.setCurrentIndex(
        [Alignment.REAL_DATE, Alignment.RELATIVE_DAY].index(Alignment.RELATIVE_DAY)
    )
    result = processing_page.process_current()

    assert result is not None
    assert result.alignment is Alignment.RELATIVE_DAY
    assert processing_page.table_result.model().rowCount() == 3
    assert "相对数据" in processing_page.label_result.text()
    assert "归一化基准值" in processing_page.text_notes.toPlainText()

    window.close()
