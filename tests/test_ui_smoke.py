"""UI 冒烟测试：主窗口可以在 offscreen 平台构建、页签齐全、导入页可用。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

from app.config import AppConfig
from app.service_registry import ServiceRegistry
from core.constants import APP_NAME
from modules.data_import.importer import TxtImporter
from ui.import_page import ImportPage
from ui.main_window import MainWindow

_SAMPLE = (
    "                  测试指数 (000001)\n"
    "\n"
    "      时间\t开盘\t最高\t最低\t收盘\t成交量\n"
    "2024/01/02\t9.80\t10.20\t9.70\t10.00\t900\n"
    "2024/01/03\t10.00\t11.00\t9.50\t10.50\t1,000\n"
)


def _state_texts(window: MainWindow) -> list[str]:
    """按页签顺序取出占位状态文本（findChildren 的顺序不保证与页签一致）。"""
    texts: list[str] = []
    for index in range(window.tabs.count()):
        page = window.tabs.widget(index)
        label = page.findChild(QLabel, "placeholderState")
        texts.append(label.text() if label is not None else "")
    return texts


def test_main_window_builds_five_tabs(qapp, tmp_path: Path) -> None:
    config = AppConfig.load(tmp_path / "config.json")
    window = MainWindow(config, ServiceRegistry())

    assert window.windowTitle().startswith(APP_NAME)
    assert window.tabs.count() == 5
    assert [window.tabs.tabText(i) for i in range(5)] == [
        "数据导入",
        "数据处理",
        "趋势分析",
        "相似度分析",
        "多指标分析",
    ]

    states = _state_texts(window)
    assert len(states) == 5
    assert all("尚未接入" in text for text in states[:4])
    assert "第二阶段实现" in states[0]
    # 多指标分析页不依赖注册表，始终是真实控件
    assert states[4] == ""

    window.close()
    assert (tmp_path / "config.json").exists()  # 关闭时保存窗口尺寸


def test_import_page_replaces_placeholder(qapp, tmp_path: Path) -> None:
    registry = ServiceRegistry()
    registry.register("importer", TxtImporter())
    window = MainWindow(AppConfig.load(tmp_path / "config.json"), registry)

    first = window.tabs.widget(0)
    assert isinstance(first, ImportPage)
    states = _state_texts(window)
    assert states[0] == ""  # 导入页不再有占位提示
    assert all("尚未接入" in text for text in states[1:4])
    window.close()


def test_import_page_loads_file(qapp, tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(_SAMPLE.encode("gbk"))

    registry = ServiceRegistry()
    registry.register("importer", TxtImporter())
    window = MainWindow(AppConfig.load(tmp_path / "config.json"), registry)

    page = window.tabs.widget(0)
    assert isinstance(page, ImportPage)
    page.import_paths([str(sample)])

    assert len(page.results) == 1
    assert page.results[0].dataset.name == "测试指数"
    assert page.table_view.model().rowCount() == 2
    assert page.list_datasets.count() == 1

    page.clear()
    assert page.results == ()
    window.close()


def test_build_registry_registers_importer(qapp, tmp_path: Path) -> None:
    from app.application import build_registry
    from modules.data_processing.processor import DataProcessor
    from modules.export.csv_exporter import CsvExporter
    from modules.similarity.service import SimilarityService
    from modules.trend_analysis.trend_service import TrendService

    registry = build_registry(AppConfig.load(tmp_path / "config.json"))
    assert registry.available() == ("importer", "processor", "trend", "similarity", "exporter")
    assert isinstance(registry.get("importer"), TxtImporter)
    assert isinstance(registry.get("processor"), DataProcessor)
    assert isinstance(registry.get("similarity"), SimilarityService)
    assert isinstance(registry.get("trend"), TrendService)
    assert isinstance(registry.get("exporter"), CsvExporter)


def test_all_pages_use_real_widgets_when_registered(qapp, tmp_path: Path) -> None:
    from app.application import build_registry
    from ui.multi_indicator_page import MultiIndicatorPage
    from ui.processing_page import ProcessingPage
    from ui.similarity_page import SimilarityPage
    from ui.trend_page import TrendPage

    config = AppConfig.load(tmp_path / "config.json")
    window = MainWindow(AppConfig.load(tmp_path / "config.json"), build_registry(config))
    assert isinstance(window.tabs.widget(0), ImportPage)
    assert isinstance(window.tabs.widget(1), ProcessingPage)
    assert isinstance(window.tabs.widget(2), TrendPage)
    assert isinstance(window.tabs.widget(3), SimilarityPage)
    assert isinstance(window.tabs.widget(4), MultiIndicatorPage)
    window.close()
