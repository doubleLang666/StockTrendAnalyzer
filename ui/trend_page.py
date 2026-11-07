"""趋势分析页（设计文档 3.3.2 / 3.3.3）。

界面职责仅限收集参数并展示 Figure：

* 选择数据源（处理结果或原始数据）、字段（可多选）、日期范围；
* 切换绝对/相对显示与真实/相对交易日（仅对原始数据生效）；
* 把若干历史区间加入比较列表，生成多周期对比图；
* 以当前区间为目标检索相似历史区间（3.3.4），并可叠加显示结果曲线；
* 保存当前图片。

所有计算都在 :class:`~modules.trend_analysis.trend_service.TrendService` 中完成，
相似度算法由 :mod:`modules.similarity` 提供。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.constants import (
    ALGORITHM_LABELS,
    ALIGNMENT_LABELS,
    NORMALIZATION_LABELS,
    Algorithm,
    Alignment,
    Normalization,
)
from core.data_models import MarketDataset, ProcessedDataset, SearchResult, SimilarityConfig, field_label
from core.exceptions import StockAnalyzerError
from modules.data_processing.normalization import normalize_series
from modules.trend_analysis.plotting import CurveData, PlotRequest, build_figure
from modules.trend_analysis.segments import Segment, compare_periods
from modules.trend_analysis.trend_service import TrendService, series_from_source, source_frame

__all__ = ["TrendPage"]

_NORMALIZATION_ORDER = (Normalization.NONE, Normalization.START, Normalization.MIN_MAX)
_ALIGNMENT_ORDER = (Alignment.REAL_DATE, Alignment.RELATIVE_DAY)
_ALGORITHM_SHORT = {
    Algorithm.PEARSON.value: "Pearson",
    Algorithm.RMSE.value: "RMSE",
    Algorithm.MAE.value: "MAE",
}


class TrendPage(QWidget):
    """趋势分析页签。"""

    plotted = Signal()

    def __init__(
        self,
        trend: TrendService,
        datasets_provider: Callable[[], Sequence[MarketDataset]] | None = None,
        processed_provider: Callable[[], Sequence[ProcessedDataset]] | None = None,
        similarity_provider: Callable[[], object | None] | None = None,
        parent: QWidget | None = None,
        exporter: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._trend = trend
        self._exporter = exporter
        self._datasets_provider = datasets_provider
        self._processed_provider = processed_provider
        self._similarity_provider = similarity_provider
        self._sources: list[tuple[str, MarketDataset | ProcessedDataset]] = []
        self._segments: list[Segment] = []
        self._search_results: list[SearchResult] = []
        self._search_target: Segment | None = None
        self._search_field: str | None = None

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        box_source = QGroupBox("数据与字段", left)
        source_layout = QVBoxLayout(box_source)
        self.combo_source = QComboBox(box_source)
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        self.list_fields = QListWidget(box_source)
        self.list_fields.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_fields.setMaximumHeight(130)
        self.list_fields.setToolTip("勾选需要绘制的字段，可同时选择多个")
        self.date_start = QDateEdit(box_source)
        self.date_end = QDateEdit(box_source)
        for widget in (self.date_start, self.date_end):
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
        dates = QHBoxLayout()
        dates.addWidget(QLabel("起始", box_source))
        dates.addWidget(self.date_start)
        dates.addWidget(QLabel("结束", box_source))
        dates.addWidget(self.date_end)
        self.edit_title = QLineEdit(box_source)
        self.edit_title.setPlaceholderText("图表标题（留空自动生成）")
        source_layout.addWidget(QLabel("数据源", box_source))
        source_layout.addWidget(self.combo_source)
        source_layout.addWidget(QLabel("字段（可多选）", box_source))
        source_layout.addWidget(self.list_fields)
        source_layout.addLayout(dates)
        source_layout.addWidget(self.edit_title)

        box_mode = QGroupBox("显示方式", left)
        mode_layout = QVBoxLayout(box_mode)
        self.combo_normalization = QComboBox(box_mode)
        for method in _NORMALIZATION_ORDER:
            self.combo_normalization.addItem(NORMALIZATION_LABELS[method.value], method.value)
        self.combo_alignment = QComboBox(box_mode)
        for alignment in _ALIGNMENT_ORDER:
            self.combo_alignment.addItem(ALIGNMENT_LABELS[alignment.value], alignment.value)
        self.combo_normalization.currentIndexChanged.connect(lambda _index: self._update_hint())
        mode_layout.addWidget(QLabel("绝对 / 相对数据", box_mode))
        mode_layout.addWidget(self.combo_normalization)
        mode_layout.addWidget(QLabel("时间轴", box_mode))
        mode_layout.addWidget(self.combo_alignment)
        self.label_hint = QLabel("", box_mode)
        self.label_hint.setWordWrap(True)
        self.label_hint.setStyleSheet("color: #52606d;")
        mode_layout.addWidget(self.label_hint)

        self.btn_draw = QPushButton("绘制曲线", left)
        self.btn_draw.clicked.connect(self.plot_current)
        self.btn_save = QPushButton("保存图片…", left)
        self.btn_save.clicked.connect(self.save_figure)

        box_compare = QGroupBox("多周期比较", left)
        compare_layout = QVBoxLayout(box_compare)
        self.table_ranges = QTableWidget(0, 3, box_compare)
        self.table_ranges.setHorizontalHeaderLabels(["起始", "结束", "交易日"])
        self.table_ranges.verticalHeader().setVisible(False)
        self.table_ranges.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_ranges.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_ranges.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_ranges.setMaximumHeight(120)
        buttons = QHBoxLayout()
        self.btn_add_range = QPushButton("添加当前区间", box_compare)
        self.btn_add_range.clicked.connect(self.add_current_range)
        self.btn_remove_range = QPushButton("移除选中", box_compare)
        self.btn_remove_range.clicked.connect(self.remove_selected_range)
        self.btn_clear_ranges = QPushButton("清空", box_compare)
        self.btn_clear_ranges.clicked.connect(self.clear_ranges)
        for button in (self.btn_add_range, self.btn_remove_range, self.btn_clear_ranges):
            buttons.addWidget(button)
        self.check_truncate = QCheckBox("长度不同时按最短区间截断（会在图中注明）", box_compare)
        self.check_truncate.setChecked(True)
        self.btn_compare = QPushButton("绘制比较图", box_compare)
        self.btn_compare.clicked.connect(self.plot_comparison)

        compare_layout.addWidget(self.table_ranges)
        compare_layout.addLayout(buttons)
        compare_layout.addWidget(self.check_truncate)
        compare_layout.addWidget(self.btn_compare)

        box_search = QGroupBox("历史区间检索（3.3.4）", left)
        search_layout = QVBoxLayout(box_search)
        options = QHBoxLayout()
        self.combo_algorithm = QComboBox(box_search)
        for key, label in ALGORITHM_LABELS.items():
            self.combo_algorithm.addItem(label, key)
        self.spin_top_n = QSpinBox(box_search)
        self.spin_top_n.setRange(1, 10)
        self.spin_top_n.setValue(3)
        self.spin_top_n.setToolTip("返回的相似区间数量（默认前三名）")
        options.addWidget(self.combo_algorithm, 1)
        options.addWidget(QLabel("返回", box_search))
        options.addWidget(self.spin_top_n)
        self.check_search_normalize = QCheckBox("误差类指标先按起始值归一化", box_search)
        self.check_search_normalize.setChecked(True)
        self.check_search_normalize.setToolTip(
            "RMSE / MAE 对价格水平敏感，先归一化才能比较形态；Pearson 对缩放不敏感，不受影响"
        )
        self.btn_search = QPushButton("检索相似区间", box_search)
        self.btn_search.clicked.connect(self.search_current)
        self.btn_search.setToolTip("以当前起始/结束日期为目标区间，用所选算法在历史上滑动检索")
        self.check_overlay = QCheckBox("叠加显示所选结果（目标曲线 + 结果曲线）", box_search)
        self.check_overlay.setChecked(True)
        self.table_results = QTableWidget(0, 6, box_search)
        self.table_results.setHorizontalHeaderLabels(["排名", "起始日期", "结束日期", "交易日", "算法", "指标值"])
        self.table_results.verticalHeader().setVisible(False)
        self.table_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        results_header = self.table_results.horizontalHeader()
        results_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        results_header.setStretchLastSection(False)
        for column, width in enumerate((34, 84, 84, 50, 60, 56)):
            self.table_results.setColumnWidth(column, width)
        self.table_results.setMaximumHeight(168)
        self.table_results.itemSelectionChanged.connect(self._on_result_selected)

        search_layout.addLayout(options)
        search_layout.addWidget(self.check_search_normalize)
        search_layout.addWidget(self.btn_search)
        search_layout.addWidget(self.table_results)
        search_layout.addWidget(self.check_overlay)

        self.btn_export_results = QPushButton("导出检索结果…（CSV）", box_search)
        self.btn_export_results.setToolTip("把当前 Top N 结果连同算法与参数导出为 CSV")
        self.btn_export_results.setEnabled(False)
        self.btn_export_results.clicked.connect(self.export_results)
        search_layout.addWidget(self.btn_export_results)

        left_layout.addWidget(box_source)
        left_layout.addWidget(box_mode)
        left_layout.addWidget(self.btn_draw)
        left_layout.addWidget(self.btn_save)
        left_layout.addWidget(box_compare)
        left_layout.addWidget(box_search)
        left_layout.addStretch(1)

        scroll = QScrollArea(splitter)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left)
        self.scroll_area = scroll

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        self._figure = Figure(figsize=(7.2, 4.8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self._figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, right)
        self.label_status = QLabel("选择数据源后点击“绘制曲线”", right)
        self.label_status.setWordWrap(True)
        self.label_status.setStyleSheet("color: #52606d;")
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)
        right_layout.addWidget(self.label_status)

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 900])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # 数据源
    # ------------------------------------------------------------------
    def _collect_sources(self) -> list[tuple[str, MarketDataset | ProcessedDataset]]:
        sources: list[tuple[str, MarketDataset | ProcessedDataset]] = []
        if self._processed_provider is not None:
            for item in self._processed_provider():
                sources.append((f"处理结果 · {item.name}", item))
        if self._datasets_provider is not None:
            for item in self._datasets_provider():
                sources.append((f"原始数据 · {item.name}", item))
        return sources

    def refresh(self) -> None:
        """重建数据源下拉（处理结果变化后由主窗口调用）。"""
        current = self.combo_source.currentText()
        self._sources = self._collect_sources()
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        for label, _source in self._sources:
            self.combo_source.addItem(label)
        if current:
            index = self.combo_source.findText(current)
            if index >= 0:
                self.combo_source.setCurrentIndex(index)
        self.combo_source.blockSignals(False)
        self._on_source_changed()

    def current_source(self) -> MarketDataset | ProcessedDataset | None:
        index = self.combo_source.currentIndex()
        if index < 0 or index >= len(self._sources):
            return None
        return self._sources[index][1]

    def _on_source_changed(self) -> None:
        source = self.current_source()
        self.list_fields.clear()
        if source is None:
            self._set_dates_enabled(False)
            return

        frame, _name = source_frame(source)
        labels = (
            {column: source.column_label(column) for column in frame.columns}
            if isinstance(source, ProcessedDataset)
            else {column: field_label(column) for column in frame.columns}
        )
        for column in frame.columns:
            item = QListWidgetItem(labels.get(column, str(column)))
            item.setData(Qt.ItemDataRole.UserRole, column)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_fields.addItem(item)

        is_processed = isinstance(source, ProcessedDataset)
        if is_processed:
            config = source.config
            self._set_combo_value(self.combo_normalization, config.normalization.value)
            self._set_combo_value(self.combo_alignment, config.alignment.value)
        self.combo_normalization.setEnabled(not is_processed)
        self.combo_alignment.setEnabled(not is_processed)

        self._apply_date_bounds(frame)
        self._update_hint()

    def _apply_date_bounds(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            first = date(2000, 1, 1)
            self._set_dates_enabled(False)
            self.date_start.setDate(QDate(first))
            self.date_end.setDate(QDate(first))
            self.label_hint.setText("该数据使用相对交易日索引，日期范围筛选不可用")
            return
        self._set_dates_enabled(True)
        start = frame.index[0].date()
        end = frame.index[-1].date()
        for widget in (self.date_start, self.date_end):
            widget.setMinimumDate(QDate(start))
            widget.setMaximumDate(QDate(end))
        self.date_start.setDate(QDate(start))
        self.date_end.setDate(QDate(end))

    def _set_dates_enabled(self, enabled: bool) -> None:
        self.date_start.setEnabled(enabled)
        self.date_end.setEnabled(enabled)
        self.btn_add_range.setEnabled(enabled)
        self.btn_compare.setEnabled(enabled)
        self.btn_search.setEnabled(enabled)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _checked_fields(self) -> list[str]:
        fields: list[str] = []
        for row in range(self.list_fields.count()):
            item = self.list_fields.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                fields.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return fields

    def _update_hint(self) -> None:
        source = self.current_source()
        if source is None:
            return
        fields = self._checked_fields()
        if isinstance(source, ProcessedDataset):
            self.label_hint.setText("该数据来自“数据处理”页签的处理结果，已按其中的设置归一化/对齐，此处不再重复处理")
            return
        if len(fields) > 1 and self.combo_normalization.currentData() == Normalization.NONE.value:
            self.label_hint.setText("同时绘制多个字段时量纲可能不同，建议选择起始值归一化或 Min-Max 归一化")
        else:
            self.label_hint.setText("")

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    def _effective_normalization(self, source: MarketDataset | ProcessedDataset) -> Normalization:
        if isinstance(source, ProcessedDataset):
            return Normalization.NONE
        return Normalization(str(self.combo_normalization.currentData()))

    def _effective_alignment(self, source: MarketDataset | ProcessedDataset) -> Alignment:
        if isinstance(source, ProcessedDataset):
            return source.config.alignment
        return Alignment(str(self.combo_alignment.currentData()))

    def _axis_label(self, alignment: Alignment, is_processed: bool) -> str:
        if is_processed and alignment is Alignment.RELATIVE_DAY:
            return "相对交易日（第 N 日）"
        return "交易日期" if alignment is Alignment.REAL_DATE else "相对交易日（第 N 日）"

    def _date_value(self, widget: QDateEdit) -> date | None:
        if not widget.isEnabled():
            return None
        return widget.date().toPython()

    def plot_current(self) -> None:
        source = self.current_source()
        if source is None:
            QMessageBox.information(self, "提示", "请先在“数据导入”或“数据处理”页签准备数据。")
            return
        fields = self._checked_fields()
        if not fields:
            QMessageBox.information(self, "提示", "请至少勾选一个字段。")
            return

        normalization = self._effective_normalization(source)
        alignment = self._effective_alignment(source)
        start = self._date_value(self.date_start)
        end = self._date_value(self.date_end)
        curves: list[CurveData] = []
        notes: list[str] = []
        try:
            for field in fields:
                series, extra_notes = series_from_source(
                    source,
                    field,
                    start=start,
                    end=end,
                    normalization=normalization,
                    alignment=alignment,
                )
                notes.extend(extra_notes)
                label = (
                    source.column_label(field) if isinstance(source, ProcessedDataset) else field_label(field)
                )
                curves.append(CurveData(label=label, values=series))
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法绘图", exc.user_message())
            return

        title = self.edit_title.text().strip() or self._default_title(source, fields)
        y_label = field_label(fields[0]) if len(fields) == 1 else "数值"
        request = PlotRequest(
            curves=tuple(curves),
            title=title,
            x_label=self._axis_label(alignment, isinstance(source, ProcessedDataset)),
            y_label=y_label,
            legend=True,
        )
        try:
            build_figure(request, figure=self._figure)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法绘图", exc.user_message())
            return
        self.canvas.draw_idle()
        self.plotted.emit()
        message = f"已绘制 {len(curves)} 条曲线：{title}"
        if notes:
            message += "｜" + "；".join(dict.fromkeys(notes))
        self.label_status.setText(message)

    def _default_title(self, source: MarketDataset | ProcessedDataset, fields: Sequence[str]) -> str:
        names = "、".join(
            source.column_label(field) if isinstance(source, ProcessedDataset) else field_label(field)
            for field in fields
        )
        start = self._date_value(self.date_start)
        end = self._date_value(self.date_end)
        if start and end:
            return f"{source.name} · {names}（{start} ~ {end}）"
        return f"{source.name} · {names}"

    # ------------------------------------------------------------------
    # 多周期比较
    # ------------------------------------------------------------------
    def add_current_range(self) -> None:
        start = self._date_value(self.date_start)
        end = self._date_value(self.date_end)
        if start is None or end is None:
            QMessageBox.information(self, "提示", "该数据无法按日期选择区间。")
            return
        if start > end:
            QMessageBox.warning(self, "区间无效", "起始日期不能晚于结束日期。")
            return
        segment = Segment(start=start, end=end)
        if any(item.start == segment.start and item.end == segment.end for item in self._segments):
            QMessageBox.information(self, "提示", "该区间已存在于比较列表中。")
            return
        self._segments.append(segment)
        self._refresh_ranges()

    def remove_selected_range(self) -> None:
        rows = sorted({index.row() for index in self.table_ranges.selectedIndexes()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self._segments):
                del self._segments[row]
        self._refresh_ranges()

    def clear_ranges(self) -> None:
        self._segments.clear()
        self._refresh_ranges()

    def _refresh_ranges(self) -> None:
        self.table_ranges.setRowCount(len(self._segments))
        for row, segment in enumerate(self._segments):
            self.table_ranges.setItem(row, 0, QTableWidgetItem(segment.start.isoformat()))
            self.table_ranges.setItem(row, 1, QTableWidgetItem(segment.end.isoformat()))
            self.table_ranges.setItem(row, 2, QTableWidgetItem("—"))

    def plot_comparison(self) -> None:
        source = self.current_source()
        if source is None:
            QMessageBox.information(self, "提示", "请先选择数据源。")
            return
        if not self._segments:
            QMessageBox.information(self, "提示", "请先添加至少一个比较区间。")
            return
        fields = self._checked_fields()
        if not fields:
            QMessageBox.information(self, "提示", "请至少勾选一个字段。")
            return
        field = fields[0]
        frame, _name = source_frame(source)
        if not isinstance(frame.index, pd.DatetimeIndex):
            QMessageBox.information(
                self,
                "无法比较",
                "当前数据使用相对交易日索引，无法按日期选择历史区间。\n"
                "请在“数据处理”页签选择真实交易日期后再比较。",
            )
            return

        normalization = self._effective_normalization(source)
        alignment = self._effective_alignment(source)
        try:
            comparison = compare_periods(
                frame,
                field,
                tuple(self._segments),
                normalization=normalization,
                alignment=alignment,
                truncate=self.check_truncate.isChecked(),
            )
            request = PlotRequest(
                curves=comparison.curves(),
                title=self.edit_title.text().strip()
                or f"{source.name} · {field_label(field)} 多周期比较",
                x_label=self._axis_label(alignment, isinstance(source, ProcessedDataset)),
                y_label=field_label(field),
                legend=True,
            )
            build_figure(request, figure=self._figure)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法比较", exc.user_message())
            return

        self.canvas.draw_idle()
        self.plotted.emit()
        detail = "｜".join(comparison.notes)
        field_note = f"比较字段：{field_label(field)}"
        if len(fields) > 1:
            field_note += "（同时勾选多个字段时，比较图只使用第一个）"
        self.label_status.setText(f"{field_note}｜{comparison.length_summary}｜{detail}")
        for row, segment in enumerate(comparison.segments):
            if row < self.table_ranges.rowCount():
                self.table_ranges.setItem(row, 2, QTableWidgetItem(str(segment.trading_days)))

    # ------------------------------------------------------------------
    # 历史区间检索（3.3.4）
    # ------------------------------------------------------------------
    def _similarity_service(self) -> object | None:
        if self._similarity_provider is None:
            return None
        return self._similarity_provider()

    def search_current(self) -> None:
        """以当前日期范围为目标区间，检索历史上最相似的若干区间。"""
        source = self.current_source()
        if source is None:
            QMessageBox.information(self, "提示", "请先选择数据源。")
            return
        service = self._similarity_service()
        if service is None:
            QMessageBox.warning(self, "相似度模块尚未接入", "历史区间检索需要相似度算法服务。")
            return
        fields = self._checked_fields()
        if not fields:
            QMessageBox.information(self, "提示", "请至少勾选一个字段。")
            return
        start = self._date_value(self.date_start)
        end = self._date_value(self.date_end)
        if start is None or end is None:
            QMessageBox.information(self, "提示", "该数据无法按日期选择区间。")
            return
        if start > end:
            QMessageBox.warning(self, "区间无效", "起始日期不能晚于结束日期。")
            return
        frame, _name = source_frame(source)
        if not isinstance(frame.index, pd.DatetimeIndex):
            QMessageBox.information(
                self,
                "无法检索",
                "历史区间检索需要真实交易日期，请选择原始数据或使用真实日期对齐的处理结果。",
            )
            return

        field = fields[0]
        config = SimilarityConfig(
            algorithm=str(self.combo_algorithm.currentData()),
            normalize=self.check_search_normalize.isChecked(),
            top_n=self.spin_top_n.value(),
            params={"field": field},
        )
        try:
            results = self._trend.search_similar_periods(Segment(start=start, end=end), source, config)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法检索", exc.user_message())
            return

        self._search_results = list(results)
        self._search_target = Segment(start=start, end=end)
        self._search_field = field
        self._fill_results()

        algo_label = ALGORITHM_LABELS.get(config.algorithm.value, config.algorithm.value)
        params = (
            f"算法：{algo_label}｜字段：{field_label(field)}｜目标区间：{start} ~ {end}"
            f"｜重叠率上限：{config.max_overlap_ratio:.0%}｜有效点下限：{config.min_valid_points}"
        )
        if not results:
            self._figure.clear()
            self.canvas.draw_idle()
            self.label_status.setText(f"{params}｜未找到满足条件的相似区间")
            return
        self.table_results.selectRow(0)
        self._on_result_selected()
        self.label_status.setText(f"检索到 {len(results)} 个相似区间｜{params}")

    def _fill_results(self) -> None:
        results = self._search_results
        self.table_results.blockSignals(True)
        self.table_results.setRowCount(len(results))
        for row, item in enumerate(results):
            score = f"{item.score:.4f}" if item.score is not None else "—"
            values = (
                str(item.rank),
                item.start_date.isoformat(),
                item.end_date.isoformat(),
                str(item.trading_days),
                _ALGORITHM_SHORT.get(item.algorithm.value, item.algorithm.value),
                score,
            )
            for column, text in enumerate(values):
                cell = QTableWidgetItem(text)
                if column in (0, 3, 5):
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if column == 4:
                    cell.setToolTip(ALGORITHM_LABELS.get(item.algorithm.value, item.algorithm.value))
                self.table_results.setItem(row, column, cell)
        self.table_results.blockSignals(False)
        self.btn_export_results.setEnabled(bool(results))

    def export_results(self) -> None:
        """把当前检索结果导出为 CSV（含算法、目标区间与参数）。"""
        if not self._search_results:
            QMessageBox.information(self, "没有可导出的结果", "请先检索相似区间。")
            return
        export = getattr(self._exporter, "export_search_results", None)
        if export is None:
            QMessageBox.warning(self, "无法导出", "结果导出模块尚未接入。")
            return
        field = self._search_field or "结果"
        default = str(Path.cwd() / f"相似区间检索_{field}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出检索结果", default, "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            saved = export(
                self._search_results,
                path,
                field=self._search_field,
                target=self._search_target,
                algorithm=str(self.combo_algorithm.currentData()),
                metadata=[
                    f"返回条数上限：{self.spin_top_n.value()}",
                    f"误差类归一化：{'是' if self.check_search_normalize.isChecked() else '否'}",
                ],
            )
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "导出失败", exc.user_message())
            return
        self.label_status.setText(f"已导出检索结果：{saved}")

    def _selected_result(self) -> SearchResult | None:
        rows = {index.row() for index in self.table_results.selectedIndexes()}
        if not rows:
            return None
        row = min(rows)
        if 0 <= row < len(self._search_results):
            return self._search_results[row]
        return None

    def _on_result_selected(self) -> None:
        result = self._selected_result()
        if result is None:
            return
        label = (
            f"第 {result.rank} 名：{result.start_date} ~ {result.end_date}，"
            f"{result.trading_days} 个交易日，{result.result.label} = {result.score:.4f}"
            if result.score is not None
            else f"第 {result.rank} 名：结果无效（{result.result.message}）"
        )
        if not self.check_overlay.isChecked():
            self.label_status.setText(label)
            return
        try:
            self._plot_overlay(result)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法叠加曲线", exc.user_message())
            return
        self.label_status.setText(f"{label}｜曲线已按首个有效值归一化叠加显示")

    def _plot_overlay(self, result: SearchResult) -> None:
        source = self.current_source()
        segment = getattr(self, "_search_target", None)
        if source is None or segment is None:
            return
        field = result.field
        target, _notes = series_from_source(
            source,
            field,
            start=segment.start,
            end=segment.end,
            normalization=Normalization.START,
            alignment=Alignment.RELATIVE_DAY,
        )
        window = normalize_series(result.series, Normalization.START).series
        target = target.copy()
        window = window.copy()
        target.index = pd.RangeIndex(len(target))
        window.index = pd.RangeIndex(len(window))
        curves = (
            CurveData(label=f"目标区间 {segment.start} ~ {segment.end}", values=target),
            CurveData(
                label=f"第 {result.rank} 名 {result.start_date} ~ {result.end_date}",
                values=window,
            ),
        )
        build_figure(
            PlotRequest(
                curves=curves,
                title=f"{source.name} · {field_label(field)} 相似区间叠加",
                x_label="相对交易日（第 N 日）",
                y_label="归一化数值（首个有效值 = 100）",
                legend=True,
            ),
            figure=self._figure,
        )
        self.canvas.draw_idle()
        self.plotted.emit()

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def save_figure(self) -> None:
        if not self._figure.axes:
            QMessageBox.information(self, "提示", "请先绘制曲线。")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "保存图片", f"{self.edit_title.text().strip() or '趋势分析'}.png", "PNG 图片 (*.png)"
        )
        if not path:
            return
        try:
            self._figure.savefig(path, dpi=150, bbox_inches="tight")
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", f"无法写入文件：{exc}")
            return
        self.label_status.setText(f"图片已保存：{path}")

    # 供测试使用
    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def segments(self) -> tuple[Segment, ...]:
        return tuple(self._segments)

    @property
    def search_results(self) -> tuple[SearchResult, ...]:
        return tuple(self._search_results)
