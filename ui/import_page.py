"""数据导入页：文件导入、校验结果显示与数据预览。

界面职责（设计文档 3.1.5）：

* 支持一次导入多份文件，左侧列出已导入的数据集；
* 显示数据名称、代码、日期范围、字段数、记录数与缺失情况；
* 右侧以表格预览，行 = 交易日，列 = 标准字段，缺失值留白；
* 校验问题单独列出，异常数据不会被静默删除。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.data_models import ImportReport, MarketDataset, field_label
from core.exceptions import StockAnalyzerError
from modules.data_import.importer import ImportResult
from ui.table_model import MarketTableModel

__all__ = ["ImportPage"]

_ISSUE_COLUMNS = ("级别", "字段", "问题描述", "数量", "示例")


class ImportPage(QWidget):
    """第二阶段的数据导入页。"""

    #: 已导入数据集发生变化（新增/移除/清空）时发出。
    datasets_changed = Signal()

    def __init__(
        self,
        importer: object,
        config: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._importer = importer
        self._config = config
        self._results: list[ImportResult] = []

        self._model = MarketTableModel(self)
        self._build_ui()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.act_import = QAction("导入文件(&O)…", self)
        self.act_import.setToolTip("导入券商导出的 TXT 行情文件（可多选，快捷键 Ctrl+O）")
        self.act_import.triggered.connect(self.choose_files)

        self.act_remove = QAction("移除选中(&R)", self)
        self.act_remove.triggered.connect(self._remove_selected)

        self.act_clear = QAction("清空列表(&C)", self)
        self.act_clear.triggered.connect(self.clear)

        toolbar = QToolBar(self)
        toolbar.addAction(self.act_import)
        toolbar.addAction(self.act_remove)
        toolbar.addAction(self.act_clear)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 860])
        self._splitter = splitter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(splitter, 1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 8, 12)
        layout.setSpacing(6)

        layout.addWidget(QLabel("已导入数据", panel))

        self.list_datasets = QListWidget(panel)
        self.list_datasets.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_datasets.currentRowChanged.connect(self._on_current_changed)
        self.list_datasets.setMinimumHeight(120)
        layout.addWidget(self.list_datasets, 1)

        self.label_detail = QLabel("尚未导入数据。", panel)
        self.label_detail.setWordWrap(True)
        self.label_detail.setObjectName("importDetail")
        self.label_detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.label_detail)

        layout.addWidget(QLabel("校验结果", panel))
        self.table_issues = QTableWidget(0, len(_ISSUE_COLUMNS), panel)
        self.table_issues.setHorizontalHeaderLabels(list(_ISSUE_COLUMNS))
        self.table_issues.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_issues.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_issues.verticalHeader().setVisible(False)
        header = self.table_issues.horizontalHeader()
        header.setStretchLastSection(False)
        for column in (0, 1, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_issues.setMinimumHeight(140)
        layout.addWidget(self.table_issues, 1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 12, 12)
        layout.setSpacing(6)

        self.label_preview = QLabel("数据预览", panel)
        self.label_preview.setObjectName("previewTitle")
        layout.addWidget(self.label_preview)

        self.table_view = QTableView(panel)
        self.table_view.setModel(self._model)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_view.verticalHeader().setDefaultSectionSize(22)
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table_view, 1)

        self.label_footer = QLabel("", panel)
        self.label_footer.setObjectName("previewFooter")
        layout.addWidget(self.label_footer)

        return panel

    # ------------------------------------------------------------------
    # 导入
    # ------------------------------------------------------------------
    def choose_files(self) -> None:
        start_dir = ""
        if self._config is not None:
            start_dir = str(self._config.get("last_open_dir", "") or "")
        if not start_dir:
            from app import paths

            start_dir = str(paths.data_dir())
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择行情数据文件",
            start_dir,
            "行情文本 (*.txt *.csv);;所有文件 (*.*)",
        )
        if not files:
            return
        if self._config is not None:
            self._config.set("last_open_dir", str(Path(files[0]).resolve().parent))
        self.import_paths(files)

    def import_paths(self, paths: Sequence[str | Path]) -> None:
        """逐个导入文件；单个文件失败不影响其他文件。"""
        if not self._importer:  # pragma: no cover - 构造时已保证
            return
        failures: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            try:
                result = self._importer.import_file(path)  # type: ignore[attr-defined]
            except StockAnalyzerError as exc:
                failures.append(f"{path.name}：{exc.user_message()}")
                continue
            self._add_result(result)

        if failures:
            QMessageBox.warning(
                self,
                "部分文件导入失败",
                "\n".join(failures),
            )
        self._remember_recent_paths()
        self.datasets_changed.emit()

    def _remember_recent_paths(self) -> None:
        """把当前已导入文件的来源路径记到配置里，重启后可自动恢复。"""
        if self._config is None:
            return
        sources = [
            str(result.report.source_path)
            for result in self._results
            if result.report.source_path is not None
        ]
        self._config.set("import.recent_files", sources[-10:])
        try:
            self._config.save()
        except OSError:  # pragma: no cover - 无写权限时忽略
            pass

    def _add_result(self, result: ImportResult) -> None:
        source = result.report.source_path
        for index, existing in enumerate(self._results):
            if source is not None and existing.report.source_path == source:
                self._results[index] = result
                self._refresh_list(select=index)
                return
        self._results.append(result)
        self._refresh_list(select=len(self._results) - 1)

    # ------------------------------------------------------------------
    # 列表与详情
    # ------------------------------------------------------------------
    def _refresh_list(self, select: int | None = None) -> None:
        self.list_datasets.blockSignals(True)
        self.list_datasets.clear()
        for result in self._results:
            report = result.report
            suffix = f"  ·  {report.code}" if report.code else ""
            item = QListWidgetItem(f"{result.dataset.name}{suffix}")
            item.setToolTip(f"{report.imported_rows} 条记录 · {Path(str(report.source_path)).name}")
            if report.has_errors:
                item.setForeground(Qt.GlobalColor.darkRed)
            self.list_datasets.addItem(item)
        self.list_datasets.blockSignals(False)

        if not self._results:
            self._on_current_changed(-1)
            return
        target = len(self._results) - 1 if select is None else max(0, min(select, len(self._results) - 1))
        self.list_datasets.setCurrentRow(target)
        self._on_current_changed(target)

    def _on_current_changed(self, row: int) -> None:
        if not (0 <= row < len(self._results)):
            self._model.set_frame(None)
            self.label_detail.setText("尚未导入数据。")
            self.label_preview.setText("数据预览")
            self.label_footer.setText("")
            self._fill_issues(())
            return

        result = self._results[row]
        self._model.set_frame(result.dataset.frame)
        self.label_detail.setText(self._detail_text(result.dataset, result.report))
        self.label_preview.setText(f"数据预览 —— {result.dataset.name}")
        self.label_footer.setText(self._footer_text(result.dataset, result.report))
        self._fill_issues(result.report.issues)

    @staticmethod
    def _detail_text(dataset: MarketDataset, report: ImportReport) -> str:
        frame = dataset.frame
        start = frame.index[0].date() if len(frame) else "-"
        end = frame.index[-1].date() if len(frame) else "-"
        missing = dataset.missing_counts()
        missing_text = "、".join(
            f"{field} {count}" for field, count in missing.items() if count
        ) or "无"
        return (
            "<table cellspacing='0' cellpadding='2'>"
            f"<tr><td><b>名称</b></td><td>{dataset.name}</td></tr>"
            f"<tr><td><b>代码</b></td><td>{report.code or '未识别'}</td></tr>"
            f"<tr><td><b>来源</b></td><td>{Path(str(report.source_path)).name}</td></tr>"
            f"<tr><td><b>编码</b></td><td>{report.encoding}（{report.delimiter_label}分隔）</td></tr>"
            f"<tr><td><b>日期范围</b></td><td>{start} ~ {end}</td></tr>"
            f"<tr><td><b>记录数</b></td><td>{report.imported_rows} / 扫描 {report.total_rows} 行"
            f"（丢弃 {report.dropped_rows} 行）</td></tr>"
            f"<tr><td><b>字段数</b></td><td>{len(frame.columns)}</td></tr>"
            f"<tr><td><b>缺失值</b></td><td>{missing_text}</td></tr>"
            "</table>"
        )

    @staticmethod
    def _footer_text(dataset: MarketDataset, report: ImportReport) -> str:
        parts = [report.summary_text()]
        if report.missing_fields:
            parts.append("缺失字段：" + "、".join(field_label(item) for item in report.missing_fields))
        if report.ignored_columns:
            ignored = report.ignored_columns
            shown = "、".join(ignored[:5])
            if len(ignored) > 5:
                shown += f" 等 {len(ignored)} 列"
            parts.append("忽略列：" + shown)
        return "　|　".join(parts)

    def _fill_issues(self, issues) -> None:  # type: ignore[no-untyped-def]
        self.table_issues.setRowCount(0)
        if not issues:
            return
        self.table_issues.setRowCount(len(issues))
        for row, issue in enumerate(issues):
            values = (
                issue.severity_label,
                issue.field_label or "-",
                issue.message,
                str(issue.count) if issue.count else "-",
                "、".join(issue.samples[:3]) if issue.samples else "-",
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(issue.describe())
                if column == 0:
                    colors = {"错误": "#b42318", "警告": "#b54708", "提示": "#475467"}
                    color = colors.get(text)
                    if color:
                        item.setForeground(QColor(color))
                self.table_issues.setItem(row, column, item)

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------
    def _remove_selected(self) -> None:
        row = self.list_datasets.currentRow()
        if not (0 <= row < len(self._results)):
            return
        self._results.pop(row)
        self._refresh_list(select=min(row, len(self._results) - 1))
        self.datasets_changed.emit()

    def clear(self) -> None:
        if not self._results:
            return
        self._results.clear()
        self._refresh_list(select=None)
        self.datasets_changed.emit()

    # ------------------------------------------------------------------
    # 对外只读接口
    # ------------------------------------------------------------------
    @property
    def results(self) -> tuple[ImportResult, ...]:
        return tuple(self._results)

    @property
    def datasets(self) -> tuple[MarketDataset, ...]:
        return tuple(result.dataset for result in self._results)

    @property
    def current_result(self) -> ImportResult | None:
        row = self.list_datasets.currentRow()
        if 0 <= row < len(self._results):
            return self._results[row]
        return None
