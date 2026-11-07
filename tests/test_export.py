"""第六阶段：分析结果导出（CSV）的测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.constants import Alignment, CalculationStatus, DataKind, Normalization
from core.data_models import (
    ProcessedDataset,
    ProcessingConfig,
    SearchResult,
    SimilarityResult,
)
from core.exceptions import ExportError
from modules.export.csv_exporter import CsvExporter


def _processed() -> ProcessedDataset:
    frame = pd.DataFrame(
        {"close": [1.0, 2.5, np.nan], "volume": [1000.0, 2000.0, 3000.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    config = ProcessingConfig(
        fields=("close", "volume"),
        normalization=Normalization.START,
        alignment=Alignment.RELATIVE_DAY,
        data_kind=DataKind.RELATIVE,
    )
    return ProcessedDataset("测试指数", frame, config, provenance={"source_rows": 3})


def _search_results() -> tuple[SearchResult, ...]:
    series = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2023-01-02", periods=3))
    return (
        SearchResult(
            rank=1,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 4),
            trading_days=3,
            field="close",
            result=SimilarityResult(algorithm="pearson", raw_value=0.8644, sample_size=3),
            series=series,
            overlap_ratio=0.0,
        ),
        SearchResult(
            rank=2,
            start_date=date(2023, 2, 1),
            end_date=date(2023, 2, 3),
            trading_days=3,
            field="close",
            result=SimilarityResult(algorithm="pearson", raw_value=0.8123, sample_size=3),
            series=series,
            overlap_ratio=0.0,
        ),
    )


def test_export_processed_writes_metadata_header_and_values(tmp_path: Path) -> None:
    target = CsvExporter().export_processed(_processed(), tmp_path / "处理结果")

    assert target.suffix == ".csv"
    assert target.name == "处理结果.csv"
    text = target.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    assert lines[0].startswith("# 数据集：测试指数")
    assert "# 对齐方式：相对交易日" in lines
    assert "日期,收盘价,成交量" in lines
    assert "2024-01-02,1,1000" in lines
    assert "2024-01-03,2.5,2000" in lines
    assert "2024-01-04,,3000" in lines  # NaN 导出为空，不填 0


def test_export_uses_bom_for_excel(tmp_path: Path) -> None:
    target = CsvExporter().export_processed(_processed(), tmp_path / "bom.csv")
    assert target.read_bytes().startswith(b"\xef\xbb\xbf")


def test_export_frame_uses_date_label_for_datetime_index(tmp_path: Path) -> None:
    frame = pd.DataFrame({"close": [1.5]}, index=pd.to_datetime(["2024-05-06"]))
    target = CsvExporter().export_frame(frame, tmp_path / "frame.csv")
    lines = target.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "日期,收盘价"
    assert lines[1] == "2024-05-06,1.5"


def test_export_search_results_contains_parameters(tmp_path: Path) -> None:
    target = CsvExporter().export_search_results(
        _search_results(),
        tmp_path / "检索.csv",
        field="close",
        algorithm="pearson",
    )
    lines = target.read_text(encoding="utf-8-sig").splitlines()
    assert "# 检索字段：收盘价" in lines
    assert "# 算法：Pearson 相关系数" in lines
    assert "排名,起始日期,结束日期,交易日,字段,算法,原始值,状态,说明" in lines
    assert "1,2023-01-02,2023-01-04,3,收盘价,Pearson 相关系数,0.864400,ok," in lines


def test_export_similarity_results_lists_every_algorithm(tmp_path: Path) -> None:
    results = (
        SimilarityResult(algorithm="pearson", raw_value=0.8413, sample_size=64),
        SimilarityResult(
            algorithm="rmse",
            raw_value=None,
            status=CalculationStatus.INSUFFICIENT_DATA,
            message="有效数据点不足：1 个（至少需要 5 个）",
            sample_size=1,
        ),
    )
    target = CsvExporter().export_similarity_results(
        results, tmp_path / "相似度.csv", curve_a="A 区间", curve_b="B 区间"
    )
    lines = target.read_text(encoding="utf-8-sig").splitlines()
    assert "# 曲线 A：A 区间" in lines
    assert "算法,原始值,有效点,先归一化,状态,说明" in lines
    assert "Pearson 相关系数,0.841300,64,否,ok," in lines
    assert "RMSE 均方根误差,,1,否,insufficient_data,有效数据点不足" in lines[4]


def test_export_dispatch_routes_by_payload_type(tmp_path: Path) -> None:
    exporter = CsvExporter()
    assert exporter.export(_processed(), tmp_path / "a.csv").exists()
    assert exporter.export(_search_results(), tmp_path / "b.csv").exists()
    assert exporter.export(_search_results()[0].result, tmp_path / "c.csv").exists()


def test_export_rejects_empty_and_unknown_payloads(tmp_path: Path) -> None:
    exporter = CsvExporter()
    with pytest.raises(ExportError):
        exporter.export([], tmp_path / "empty.csv")
    with pytest.raises(ExportError):
        exporter.export(42, tmp_path / "unknown.csv")
    with pytest.raises(ExportError):
        exporter.export_processed(_processed(), tmp_path / "缺失目录" / "x.csv")
