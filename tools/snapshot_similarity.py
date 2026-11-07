"""第五阶段截图脚本：在原生窗口平台渲染相似度分析页。

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
from modules.data_import.importer import TxtImporter
from modules.similarity.service import SimilarityService
from ui.similarity_page import SimilarityPage

OUT_DIR = Path(tempfile.gettempdir()) / "dsh-shots"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = AppConfig.load()
    configure_matplotlib(config)
    app = QApplication.instance() or QApplication(sys.argv)

    dataset = TxtImporter(config).load_file(PROJECT_ROOT / "data" / "399006.txt")
    page = SimilarityPage(SimilarityService(config), lambda: (dataset,), None)
    page.resize(1400, 880)
    page.show()
    page.refresh()

    # 曲线 A：2023 年初一段；曲线 B：2024 年初一段（长度不同，按最短截断）
    page.box_a.date_start.setDate(QDate(2023, 1, 3))
    page.box_a.date_end.setDate(QDate(2023, 4, 10))
    page.box_b.date_start.setDate(QDate(2024, 1, 2))
    page.box_b.date_end.setDate(QDate(2024, 4, 10))
    page.compare_current()
    app.processEvents()
    path = OUT_DIR / "similarity_page.png"
    print("saved", page.grab().save(str(path)), path)
    print("status:", page.label_status.text())
    for item in page.results:
        print(item.algorithm.value, item.raw_value, item.status.value, item.message)


if __name__ == "__main__":
    main()
