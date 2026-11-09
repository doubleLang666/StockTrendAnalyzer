"""主窗口。

第一阶段只提供界面骨架：菜单、四个模块页签与状态栏。
每个页签在没有对应服务时显示占位说明，避免给出"功能已可用"的错觉。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.service_registry import ServiceRegistry
from core.constants import APP_NAME, APP_VERSION
from ui.import_page import ImportPage
from ui.multi_indicator_page import MultiIndicatorPage
from ui.processing_page import ProcessingPage
from ui.similarity_page import SimilarityPage
from ui.trend_page import TrendPage

__all__ = ["MainWindow"]

_PLACEHOLDER_STYLE = """
QLabel#placeholderTitle { font-size: 18px; font-weight: 600; color: #1f2933; }
QLabel#placeholderBody  { color: #52606d; }
QLabel#placeholderState { color: #9aa5b1; }
QFrame#placeholderCard  { background: #f8fafc; border: 1px solid #e4e7eb; border-radius: 8px; }
"""


class PlaceholderPage(QWidget):
    """模块占位页：标题 + 说明 + 当前接入状态。"""

    def __init__(
        self,
        title: str,
        body: str,
        state: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        card = QFrame(self)
        card.setObjectName("placeholderCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(10)

        title_label = QLabel(title, card)
        title_label.setObjectName("placeholderTitle")
        body_label = QLabel(body, card)
        body_label.setObjectName("placeholderBody")
        body_label.setWordWrap(True)
        state_label = QLabel(state, card)
        state_label.setObjectName("placeholderState")
        state_label.setWordWrap(True)

        card_layout.addWidget(title_label)
        card_layout.addWidget(body_label)
        card_layout.addWidget(state_label)
        card_layout.addStretch(1)

        layout.addWidget(card)
        layout.addStretch(1)
        self.setStyleSheet(_PLACEHOLDER_STYLE)


class MainWindow(QMainWindow):
    PAGE_SPECS: tuple[tuple[str, str, str, str], ...] = (
        (
            "数据导入",
            "导入国内券商交易软件导出的 TXT 行情文件，完成编码识别、字段映射、"
            "数据校验与预览。可同时导入多份文件（股票、指数或不同指标）。",
            "importer",
            "第二阶段实现",
        ),
        (
            "数据处理",
            "选择字段、日期范围与数据形式，计算二级数据（涨跌幅、变化率），"
            "执行起始值/Min-Max 归一化与时间对齐。原始数据始终保持不变。",
            "processor",
            "第三阶段实现",
        ),
        (
            "趋势分析",
            "绘制单条/多条历史曲线，将同一指标的多个历史区间对齐到同一张图中比较，"
            "并支持按目标区间检索历史相似片段。",
            "trend",
            "第四阶段实现",
        ),
        (
            "相似度分析",
            "以 Pearson 相关系数、RMSE、MAE 等数学方法衡量两条历史曲线的相似程度，"
            "按统一接口排序并输出 Top N 结果。",
            "similarity",
            "第五阶段实现",
        ),
        (
            "多指标分析",
            "把价格、成交量、换手率、涨跌幅等多个指标放在同一套 0–100 评分体系里"
            "做联动分析：逐指标计算 Pearson/RMSE/MAE，按权重合成综合相似度。",
            "multi",
            "第六阶段实现",
        ),
    )

    def __init__(
        self,
        config: AppConfig,
        registry: ServiceRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._registry = registry

        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        width = int(config.get("app.window.width", 1280))
        height = int(config.get("app.window.height", 820))
        self.resize(max(width, 960), max(height, 680))
        self.setMinimumSize(880, 600)

        self._build_actions()
        self._build_menus()
        self._build_central()
        self._build_status_bar()
        self._restore_recent_files()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_actions(self) -> None:
        self.act_import = QAction("导入行情数据(&O)…", self)
        self.act_import.setShortcut(QKeySequence("Ctrl+O"))
        self.act_import.setStatusTip("导入券商导出的 TXT 行情文件")
        self.act_import.triggered.connect(self._on_import)

        self.act_reset_layout = QAction("重置窗口布局(&R)", self)
        self.act_reset_layout.triggered.connect(self._on_reset_layout)

        self.act_quit = QAction("退出(&Q)", self)
        self.act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        self.act_quit.triggered.connect(self.close)

        self.act_about = QAction("关于(&A)", self)
        self.act_about.triggered.connect(self._on_about)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("文件(&F)")
        file_menu.addAction(self.act_import)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        view_menu = menu_bar.addMenu("视图(&V)")
        view_menu.addAction(self.act_reset_layout)

        help_menu = menu_bar.addMenu("帮助(&H)")
        help_menu.addAction(self.act_about)

    def _build_central(self) -> None:
        self._tabs = QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._pages: dict[str, PlaceholderPage] = {}
        self._import_page: ImportPage | None = None
        self._processing_page: ProcessingPage | None = None
        self._trend_page: TrendPage | None = None
        self._similarity_page: SimilarityPage | None = None
        self._multi_page: MultiIndicatorPage | None = None

        for title, body, service_name, stage in self.PAGE_SPECS:
            if service_name == "importer":
                importer = self._registry.try_get("importer")
                if importer is not None:
                    page = ImportPage(importer, self._config, self)
                    page.datasets_changed.connect(self._on_datasets_changed)
                    self._import_page = page
                    self._tabs.addTab(page, title)
                    continue
            elif service_name == "processor":
                processor = self._registry.try_get("processor")
                if processor is not None:
                    page = ProcessingPage(
                        processor,
                        self._imported_datasets,
                        self,
                        exporter=self._registry.try_get("exporter"),
                    )
                    page.processed_changed.connect(self._on_processed_changed)
                    self._processing_page = page
                    self._tabs.addTab(page, title)
                    continue
            elif service_name == "trend":
                trend = self._registry.try_get("trend")
                if trend is not None:
                    page = TrendPage(
                        trend,
                        self._imported_datasets,
                        self._processed_datasets,
                        lambda: self._registry.try_get("similarity"),
                        self,
                        exporter=self._registry.try_get("exporter"),
                        config=self._config,
                    )
                    page.plotted.connect(self._on_plotted)
                    self._trend_page = page
                    self._tabs.addTab(page, title)
                    continue
            elif service_name == "similarity":
                similarity = self._registry.try_get("similarity")
                if similarity is not None:
                    page = SimilarityPage(
                        similarity,
                        self._imported_datasets,
                        self._processed_datasets,
                        self,
                        exporter=self._registry.try_get("exporter"),
                    )
                    page.compared.connect(self._on_compared)
                    self._similarity_page = page
                    self._tabs.addTab(page, title)
                    continue
            elif service_name == "multi":
                page = MultiIndicatorPage(
                    self._imported_datasets,
                    lambda: self._registry.try_get("similarity"),
                    self,
                    config=self._config,
                )
                self._multi_page = page
                self._tabs.addTab(page, title)
                continue

            available = self._registry.is_available(service_name)
            state = (
                "当前状态：模块已接入。"
                if available
                else f"当前状态：{stage}，界面骨架已就绪，功能尚未接入。"
            )
            page = PlaceholderPage(title, body, state, self)
            self._pages[service_name] = page
            self._tabs.addTab(page, title)

        self.setCentralWidget(self._tabs)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        bar.showMessage("就绪")
        self._version_label = QLabel(f"v{APP_VERSION}", self)
        self._version_label.setStyleSheet("color: #7b8794; padding-right: 8px;")
        bar.addPermanentWidget(self._version_label)

    # ------------------------------------------------------------------
    # 槽函数
    # ------------------------------------------------------------------
    def _on_import(self) -> None:
        if self._import_page is not None:
            self._tabs.setCurrentWidget(self._import_page)
            self._import_page.choose_files()
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择行情数据文件",
            str(self._config.get("last_open_dir", str(self._default_data_dir()))),
            "行情文本 (*.txt *.csv);;所有文件 (*.*)",
        )
        if not files:
            return
        selected_dir = str(Path(files[0]).resolve().parent)
        self._config.set("last_open_dir", selected_dir)
        QMessageBox.information(
            self,
            "数据导入尚未接入",
            "数据导入模块将在第二阶段实现。\n\n"
            "届时此处会完成：编码识别、字段映射、数据校验与多文件预览。",
        )

    def _on_datasets_changed(self) -> None:
        results = self._import_page.results if self._import_page is not None else ()
        if self._processing_page is not None:
            self._processing_page.refresh()
        if self._trend_page is not None:
            self._trend_page.refresh()
        if self._similarity_page is not None:
            self._similarity_page.refresh()
        if self._multi_page is not None:
            self._multi_page.refresh()
        if not results:
            self.statusBar().showMessage("已清空导入数据", 5000)
            return
        name = results[-1].dataset.name
        self.statusBar().showMessage(f"已导入 {len(results)} 份数据，当前：{name}", 8000)

    def _imported_datasets(self) -> tuple[object, ...]:
        if self._import_page is None:
            return ()
        return tuple(result.dataset for result in self._import_page.results)

    def _processed_datasets(self) -> tuple[object, ...]:
        if self._processing_page is None:
            return ()
        result = self._processing_page.processed
        return (result,) if result is not None else ()

    def _on_processed_changed(self) -> None:
        result = self._processing_page.processed if self._processing_page is not None else None
        if self._trend_page is not None:
            self._trend_page.refresh()
        if self._similarity_page is not None:
            self._similarity_page.refresh()
        if result is None:
            return
        self.statusBar().showMessage(
            f"已生成处理数据：{result.name}，{len(result)} 条记录",
            8000,
        )

    def _on_plotted(self) -> None:
        self.statusBar().showMessage("已更新趋势图", 4000)

    def _on_compared(self) -> None:
        self.statusBar().showMessage("已完成相似度计算", 4000)

    def _on_reset_layout(self) -> None:
        self.resize(1280, 820)
        self._tabs.setCurrentIndex(0)
        self.statusBar().showMessage("已重置窗口布局", 5000)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "关于",
            f"<h3>{APP_NAME}</h3>"
            f"<p>版本 {APP_VERSION}　·　本科毕业设计</p>"
            "<p>本地化运行，不接入实时行情，不提供走势预测与投资建议。</p>"
            "<p>所有分析结果均由明确的数学方法与程序逻辑产生。</p>",
        )

    # ------------------------------------------------------------------
    # 杂项
    # ------------------------------------------------------------------
    def _default_data_dir(self) -> str:
        from app import paths

        return str(paths.data_dir())

    def _restore_recent_files(self) -> None:
        """启动时按配置恢复上次导入的数据（文件仍存在时才导入）。"""
        if self._import_page is None or not self._config.get("import.restore_recent", True):
            return
        recent = self._config.get("import.recent_files") or []
        if not isinstance(recent, (list, tuple)):
            return
        existing = [Path(item) for item in recent if str(item) and Path(item).is_file()]
        if not existing:
            return
        self._import_page.import_paths(existing)
        if existing:
            self.statusBar().showMessage(f"已恢复上次导入的 {len(existing)} 份数据", 5000)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._config.get("app.window.remember_geometry", True):
            self._config.set("app.window.width", self.width())
            self._config.set("app.window.height", self.height())
            try:
                self._config.save()
            except OSError:  # pragma: no cover - 无写权限时静默忽略
                pass
        super().closeEvent(event)

    # 供测试/后续阶段使用
    @property
    def tabs(self) -> QTabWidget:
        return self._tabs

    @property
    def registry(self) -> ServiceRegistry:
        return self._registry
