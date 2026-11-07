"""第四阶段截图脚本：在原生窗口平台渲染趋势分析页。

沙箱子进程无法写工作区，因此截图统一输出到系统临时目录。
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(r"F:\毕业设计\StockTrendAnalyzer")
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import QApplication

from app.application import configure_matplotlib
from app.config import AppConfig
from core.constants import Alignment, Normalization
from core.data_models import ProcessingConfig
from modules.data_import.importer import TxtImporter
from modules.data_processing.processor import DataProcessor
from modules.similarity.service import SimilarityService
from modules.trend_analysis.trend_service import TrendService
from ui.trend_page import TrendPage

OUT_DIR = Path(tempfile.gettempdir()) / "dsh-shots"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = AppConfig.load()
    configure_matplotlib(config)
    app = QApplication.instance() or QApplication(sys.argv)

    dataset = TxtImporter(config).load_file(PROJECT_ROOT / "data" / "399006.txt")
    processed = DataProcessor(config).process(
        dataset,
        ProcessingConfig(
            fields=("close", "volume"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 28),
            normalization=Normalization.START,
            alignment=Alignment.RELATIVE_DAY,
        ),
    )

    similarity = SimilarityService(config)
    page = TrendPage(
        TrendService(config, lambda: similarity),
        lambda: (dataset,),
        lambda: (processed,),
        lambda: similarity,
    )
    page.resize(1400, 880)
    page.show()
    page.refresh()

    # 1) 处理结果单曲线
    page.combo_source.setCurrentIndex(0)
    for row in range(page.list_fields.count()):
        item = page.list_fields.item(row)
        checked = item.data(Qt.ItemDataRole.UserRole) == "close"
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    page.plot_current()
    app.processEvents()
    path1 = OUT_DIR / "trend_page.png"
    print("saved", page.grab().save(str(path1)), path1)

    # 2) 原始数据多周期比较
    page.combo_source.setCurrentIndex(page.combo_source.findText("原始数据 · 创业板指"))
    page.clear_ranges()
    for start, end in (
        (QDate(2020, 1, 2), QDate(2020, 3, 2)),
        (QDate(2022, 5, 5), QDate(2022, 7, 6)),
        (QDate(2025, 1, 2), QDate(2025, 3, 3)),
    ):
        page.date_start.setDate(start)
        page.date_end.setDate(end)
        page.add_current_range()
    page.plot_comparison()
    app.processEvents()
    path2 = OUT_DIR / "trend_page_compare.png"
    print("saved", page.grab().save(str(path2)), path2)
    print("status:", page.label_status.text())

    # 3) 历史区间检索（第五阶段）：目标区间 + Top 3
    page.combo_source.setCurrentIndex(page.combo_source.findText("原始数据 · 创业板指"))
    page.date_start.setDate(QDate(2024, 1, 2))
    page.date_end.setDate(QDate(2024, 3, 1))
    page.search_current()
    page.scroll_area.ensureWidgetVisible(page.table_results)
    app.processEvents()
    path3 = OUT_DIR / "trend_page_search.png"
    print("saved", page.grab().save(str(path3)), path3)
    print("search:", page.label_status.text())
    print("results:", [(r.rank, r.start_date, r.end_date, round(r.score, 4)) for r in page.search_results])


if __name__ == "__main__":
    main()
