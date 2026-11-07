"""数据导入模块测试：编码、分隔符、字段映射、解析、校验与导入服务。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.constants import FIELD_TURNOVER
from core.data_models import IssueSeverity, MarketDataset
from core.exceptions import EncodingDetectionError, ValidationError
from modules.data_import.dialect import WHITESPACE, detect_delimiter, split_lines, split_row
from modules.data_import.encoding import detect_encoding
from modules.data_import.field_mapping import build_mapping, parse_title
from modules.data_import.importer import TxtImporter
from modules.data_import.parser import parse_date, parse_number, parse_table
from modules.data_import.validation import validate_frame

#: 与券商文件同构的最小样例：标题行 + 空行 + 制表符表头 + 数据行（故意乱序）。
SAMPLE_TEXT = (
    "                  测试指数 (000001)\n"
    "\n"
    "      时间\t开盘\t最高\t最低\t收盘\t成交量\t成交额\n"
    "2024/01/03\t10.00\t11.00\t9.50\t10.50\t1,000\t10,500\n"
    "2024/01/02\t9.80\t10.20\t9.70\t10.00\t900\t9,000\n"
    "2024/01/04\t10.50\t11.20\t10.40\t\t800\t8,400\n"
)

REAL_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "399006.txt"


# ---------------------------------------------------------------------------
# 编码与分隔符
# ---------------------------------------------------------------------------
def test_detect_encoding_utf8_and_gbk() -> None:
    text = "时间\t开盘\n2024/01/02\t10.00\n"
    encoding, decoded = detect_encoding(text.encode("utf-8"))
    assert encoding in {"utf-8", "utf-8-sig"}
    assert decoded == text

    encoding, decoded = detect_encoding(text.encode("gbk"))
    assert encoding in {"gbk", "gb18030"}
    assert decoded == text


def test_detect_encoding_rejects_garbage() -> None:
    with pytest.raises(EncodingDetectionError):
        detect_encoding(b"")


def test_delimiter_and_row_split() -> None:
    assert detect_delimiter("时间\t开盘\t收盘") == "\t"
    assert detect_delimiter("date,open,close") == ","
    assert detect_delimiter("时间   开盘   收盘") == WHITESPACE

    assert split_row("  a \t b \t ", "\t") == ["a", "b"]
    assert split_row("10.0   11.0   9.5", WHITESPACE) == ["10.0", "11.0", "9.5"]
    assert split_lines("a\r\nb\rc\n") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 字段映射与标题
# ---------------------------------------------------------------------------
def test_field_mapping_ignores_indicator_columns() -> None:
    headers = ["时间", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "MA5", "MACD.DIF"]
    mapping = build_mapping(headers)
    assert mapping.has_date
    assert mapping.columns["close"] == "收盘"
    assert mapping.columns["date"] == "时间"
    assert "MA5" in mapping.ignored
    assert FIELD_TURNOVER in mapping.missing


def test_field_alias_with_unit_suffix() -> None:
    mapping = build_mapping(["日期", "开盘价(元)", "最高价(元)", "最低价(元)", "收盘价(元)", "成交量(手)"])
    assert mapping.columns["close"] == "收盘价(元)"
    assert mapping.columns["volume"] == "成交量(手)"


def test_parse_title() -> None:
    assert parse_title("                  创业板指 (399006)") == ("创业板指", "399006")
    assert parse_title("上证指数 000001") == ("上证指数", "000001")
    assert parse_title("无代码标题") == ("无代码标题", "")


# ---------------------------------------------------------------------------
# 单元格解析
# ---------------------------------------------------------------------------
def test_parse_date_formats() -> None:
    assert parse_date("2024/01/03") == pd.Timestamp("2024-01-03")
    assert parse_date("20240103") == pd.Timestamp("2024-01-03")
    assert parse_date("2024/13/45") is None
    assert parse_date("") is None


def test_parse_number_keeps_missing_as_none() -> None:
    assert parse_number("1,234.5") == 1234.5
    assert parse_number("1.23%") == 1.23
    assert parse_number("--") is None
    assert parse_number("abc") is None


# ---------------------------------------------------------------------------
# 解析整表
# ---------------------------------------------------------------------------
def test_parse_table_sorts_and_keeps_nan() -> None:
    parsed = parse_table(SAMPLE_TEXT)

    assert parsed.title_name == "测试指数"
    assert parsed.title_code == "000001"
    assert parsed.delimiter == "\t"
    assert parsed.total_rows == 3
    assert parsed.imported_rows == 3
    assert list(parsed.frame.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert parsed.frame.index.is_monotonic_increasing
    assert parsed.frame.index[0] == pd.Timestamp("2024-01-02")

    # 缺失值必须保持 NaN，不允许填 0。
    assert parsed.frame.loc["2024-01-04", "close"] != parsed.frame.loc["2024-01-04", "close"]
    assert parsed.frame.loc["2024-01-03", "close"] == pytest.approx(10.5)
    assert parsed.frame.loc["2024-01-03", "volume"] == pytest.approx(1000.0)

    codes = {issue.code for issue in parsed.issues}
    assert "unsorted" in codes


def test_parse_table_drops_invalid_date_rows() -> None:
    text = SAMPLE_TEXT.replace(
        "2024/01/04",
        "2024/13/45",
    )
    parsed = parse_table(text)
    assert parsed.imported_rows == 2
    assert parsed.dropped_rows == 1
    invalid = [issue for issue in parsed.issues if issue.code == "invalid_date"]
    assert invalid and invalid[0].severity is IssueSeverity.ERROR


def test_parse_table_reports_duplicate_dates() -> None:
    text = (
        "标题 (000002)\n"
        "时间\t开盘\t最高\t最低\t收盘\n"
        "2024/01/02\t1\t2\t0.5\t1.5\n"
        "2024/01/02\t1\t2\t0.5\t1.6\n"
    )
    parsed = parse_table(text)
    assert parsed.imported_rows == 1
    assert any(issue.code == "duplicate_date" for issue in parsed.issues)


def test_parse_table_requires_header() -> None:
    with pytest.raises(ValidationError):
        parse_table("这不是行情文件\n只有两行文字\n")


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def test_validate_frame_flags_logic_conflicts(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.copy()
    frame.iloc[0, frame.columns.get_loc("high")] = -1.0  # 负价格（同时 <= low）
    frame.iloc[1, frame.columns.get_loc("low")] = 99.0   # 最低价高于开盘/收盘
    issues = validate_frame(frame)
    codes = {issue.code for issue in issues}
    assert "negative_price" in codes
    assert "high_below_low" in codes
    assert "missing_values" in codes  # 成交量最后一行为 NaN


# ---------------------------------------------------------------------------
# 导入服务
# ---------------------------------------------------------------------------
def test_importer_reads_gbk_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(SAMPLE_TEXT.encode("gbk"))

    importer = TxtImporter()
    result = importer.import_file(path)

    assert isinstance(result.dataset, MarketDataset)
    assert result.dataset.name == "测试指数"
    assert result.dataset.source_encoding in {"gbk", "gb18030"}
    assert result.report.code == "000001"
    assert result.report.imported_rows == 3
    assert result.report.delimiter_label == "制表符"
    assert result.report.header_line == 3
    assert "close" in result.report.mapping

    dataset = importer.load_file(path)
    assert isinstance(dataset, MarketDataset)
    assert dataset.missing_counts()["close"] == 1


def test_importer_load_files_returns_all(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(SAMPLE_TEXT.encode("gbk"))
    second.write_bytes(SAMPLE_TEXT.replace("测试指数", "样本指数").encode("utf-8"))

    importer = TxtImporter()
    datasets = importer.load_files([first, second])
    assert len(datasets) == 2
    assert {item.name for item in datasets} == {"测试指数", "样本指数"}


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="缺少真实券商样例数据")
def test_importer_reads_real_bohai_file() -> None:
    result = TxtImporter().import_file(REAL_SAMPLE)
    dataset = result.dataset

    assert dataset.name == "创业板指"
    assert result.report.code == "399006"
    assert result.report.imported_rows > 3000
    assert dataset.frame.index.is_monotonic_increasing
    assert not dataset.frame.index.has_duplicates
    assert {"open", "high", "low", "close", "volume"}.issubset(dataset.frame.columns)
    assert dataset.frame["close"].notna().all()
