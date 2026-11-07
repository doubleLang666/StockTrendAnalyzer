"""用真实券商文件做一次端到端导出校验（文件不存在时自动跳过）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from core.constants import Alignment, Normalization
from core.data_models import ProcessingConfig, SimilarityConfig
from modules.data_import.importer import TxtImporter
from modules.data_processing.processor import DataProcessor
from modules.export.csv_exporter import CsvExporter
from modules.similarity.service import SimilarityService
from modules.trend_analysis.segments import Segment
from modules.trend_analysis.trend_service import TrendService

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "399006.txt"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="缺少真实行情样例 data/399006.txt")
def test_export_real_processed_search_and_similarity(tmp_path: Path) -> None:
    dataset = TxtImporter().load_file(DATA_FILE)
    processed = DataProcessor().process(
        dataset,
        ProcessingConfig(
            fields=("open", "close", "volume"),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 6, 28),
            normalization=Normalization.START,
            alignment=Alignment.RELATIVE_DAY,
        ),
    )

    exporter = CsvExporter()
    processed_path = exporter.export(processed, tmp_path / "处理数据.csv")
    lines = processed_path.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("# 数据集：")
    assert "# 对齐方式：相对交易日" in lines
    header = next(line for line in lines if line.startswith("相对交易日,"))
    assert header == "相对交易日,开盘价,收盘价,成交量"
    data_rows = [line for line in lines if line and not line.startswith("#")][1:]
    assert len(data_rows) == 117
    assert data_rows[0].split(",")[1:3] == ["100", "100"]

    similarity = SimilarityService()
    trend = TrendService(None, lambda: similarity)
    results = trend.search_similar_periods(
        Segment(date(2024, 1, 2), date(2024, 3, 1)),
        dataset,
        SimilarityConfig(algorithm="pearson", top_n=3, params={"field": "close"}),
    )
    search_path = exporter.export_search_results(
        results,
        tmp_path / "检索结果.csv",
        field="close",
        target="2024-01-02 ~ 2024-03-01",
        algorithm="pearson",
    )
    search_lines = search_path.read_text(encoding="utf-8-sig").splitlines()
    assert "# 检索字段：收盘价" in search_lines
    assert len([line for line in search_lines if line and not line.startswith("#")]) == 4

    close = dataset.series("close")
    left = close.loc["2023-01-03":"2023-04-10"].reset_index(drop=True)
    right = close.loc["2024-01-02":"2024-04-10"].reset_index(drop=True)
    values = similarity.compare_all(left, right)
    assert [item.algorithm.value for item in values] == ["pearson", "rmse", "mae", "dtw", "cosine"]

    similarity_path = exporter.export_similarity_results(
        values, tmp_path / "相似度对比.csv", curve_a="A 区间", curve_b="B 区间"
    )
    similarity_lines = similarity_path.read_text(encoding="utf-8-sig").splitlines()
    assert "# 曲线 A：A 区间" in similarity_lines
    assert len([line for line in similarity_lines if line and not line.startswith("#")]) == 6  # 表头 + 5 行
