"""第三阶段/第六阶段截图脚本：在原生窗口平台渲染数据处理页（含导出按钮）。

沙箱子进程无法写工作区，因此截图统一输出到系统临时目录。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(r"F:\毕业设计\StockTrendAnalyzer")
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QDate
from PySide6.QtWidgets import QApplication

from app.application import configure_matplotlib
from app.config import AppConfig
from core.constants import Alignment, Normalization
from modules.data_import.importer import TxtImporter
from modules.data_processing.processor import DataProcessor
from modules.export.csv_exporter import CsvExporter
from ui.processing_page import ProcessingPage

DATA_FILE = PROJECT_ROOT / "data" / "399006.txt"
OUT_DIR = Path(tempfile.gettempdir()) / "dsh-shots"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = AppConfig.load()
    configure_matplotlib(config)
    app = QApplication.instance() or QApplication(sys.argv)

    dataset = TxtImporter(config).load_file(DATA_FILE)
    page = ProcessingPage(DataProcessor(config), lambda: (dataset,), exporter=CsvExporter())
    page.resize(1400, 880)
    page.refresh()

    for index in range(page.list_fields.count()):
        item = page.list_fields.item(index)
        if item.text() not in ("开盘价", "收盘价", "成交量"):
            item.setCheckState(item.checkState().__class__.Unchecked)

    page.date_start.setDate(QDate(2024, 1, 2))
    page.date_end.setDate(QDate(2024, 6, 28))
    start_index = page.combo_normalization.findData(Normalization.START.value)
    page.combo_normalization.setCurrentIndex(start_index)
    relative_index = page.combo_alignment.findData(Alignment.RELATIVE_DAY.value)
    page.combo_alignment.setCurrentIndex(relative_index)
    page.process_current()
    app.processEvents()

    path = OUT_DIR / "processing_page.png"
    print("saved", page.grab().save(str(path)), path)


if __name__ == "__main__":
    main()
