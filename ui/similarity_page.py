"""相似度分析页（设计文档 3.4）。

界面只负责取两条曲线并展示结果，所有指标由
:class:`modules.similarity.service.SimilarityService` 计算。
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
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.constants import Alignment, Normalization
from core.data_models import MarketDataset, ProcessedDataset, SimilarityResult, field_label
from core.exceptions import StockAnalyzerError
from modules.similarity.service import SimilarityService
from modules.trend_analysis.plotting import CurveData, PlotRequest, build_figure
from modules.trend_analysis.trend_service import source_frame

__all__ = ["SimilarityPage"]


class _CurveBox(QGroupBox):
    """一条待比较曲线的选择框：数据源 + 字段 + 日期范围。"""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self.combo_source = QComboBox(self)
        self.combo_field = QComboBox(self)
        self.date_start = QDateEdit(self)
        self.date_end = QDateEdit(self)
        for widget in (self.date_start, self.date_end):
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
        dates = QHBoxLayout()
        dates.addWidget(QLabel("起始", self))
        dates.addWidget(self.date_start)
        dates.addWidget(QLabel("结束", self))
        dates.addWidget(self.date_end)
        layout.addWidget(QLabel("数据源", self))
        layout.addWidget(self.combo_source)
        layout.addWidget(QLabel("字段", self))
        layout.addWidget(self.combo_field)
        layout.addLayout(dates)

        self._sources: list[tuple[str, MarketDataset | ProcessedDataset]] = []
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)

    # -- 数据源 -----------------------------------------------------------
    def set_sources(self, sources: Sequence[tuple[str, MarketDataset | ProcessedDataset]]) -> None:
        keep = self.combo_source.currentText()
        self._sources = list(sources)
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        for label, _source in self._sources:
            self.combo_source.addItem(label)
        if keep:
            index = self.combo_source.findText(keep)
            if index >= 0:
                self.combo_source.setCurrentIndex(index)
        self.combo_source.blockSignals(False)
        self._on_source_changed()

    def current_source(self) -> MarketDataset | ProcessedDataset | None:
        index = self.combo_source.currentIndex()
        if index < 0 or index >= len(self._sources):
            return None
        return self._sources[index][1]

    def field(self) -> str | None:
        data = self.combo_field.currentData()
        return None if data is None else str(data)

    def _on_source_changed(self) -> None:
        source = self.current_source()
        self.combo_field.blockSignals(True)
        self.combo_field.clear()
        if source is not None:
            frame, _name = source_frame(source)
            for column in frame.columns:
                label = (
                    source.column_label(column)
                    if isinstance(source, ProcessedDataset)
                    else field_label(column)
                )
                self.combo_field.addItem(label, column)
        self.combo_field.blockSignals(False)
        if source is None:
            self.date_start.setEnabled(False)
            self.date_end.setEnabled(False)
            return
        frame, _name = source_frame(source)
        if isinstance(frame.index, pd.DatetimeIndex):
            start = frame.index[0].date()
            end = frame.index[-1].date()
            for widget in (self.date_start, self.date_end):
                widget.setEnabled(True)
                widget.setMinimumDate(QDate(start))
                widget.setMaximumDate(QDate(end))
            self.date_start.setDate(QDate(start))
            self.date_end.setDate(QDate(end))
        else:
            self.date_start.setEnabled(False)
            self.date_end.setEnabled(False)

    def dates(self) -> tuple[date | None, date | None]:
        if not self.date_start.isEnabled():
            return None, None
        return self.date_start.date().toPython(), self.date_end.date().toPython()

    def label(self) -> str:
        source = self.current_source()
        if source is None:
            return "（未选择）"
        field = self.field()
        text = f"{source.name}"
        if field:
            text += f" · {field_label(field)}"
        start, end = self.dates()
        if start and end:
            text += f"（{start} ~ {end}）"
        return text


class SimilarityPage(QWidget):
    """相似度分析：用全部已注册算法比较两条历史区间曲线。"""

    compared = Signal()

    def __init__(
        self,
        similarity: SimilarityService,
        datasets_provider: Callable[[], Sequence[MarketDataset]] | None = None,
        processed_provider: Callable[[], Sequence[ProcessedDataset]] | None = None,
        parent: QWidget | None = None,
        exporter: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._similarity = similarity
        self._exporter = exporter
        self._datasets_provider = datasets_provider
        self._processed_provider = processed_provider
        self._results: tuple[SimilarityResult, ...] = ()
        self._plot_series: tuple[pd.Series, pd.Series] | None = None

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)
        self.box_a = _CurveBox("曲线 A", left)
        self.box_b = _CurveBox("曲线 B", left)
        self.check_truncate = QCheckBox("长度不一致时按较短的曲线截断（会在说明中注明）", left)
        self.check_truncate.setChecked(True)
        self.btn_compare = QPushButton("计算相似度", left)
        self.btn_compare.clicked.connect(self.compare_current)
        self.label_hint = QLabel(
            "Pearson / 余弦衡量形态相关程度（越大越相似）；RMSE / MAE / DTW 衡量数值差异（越小越相似）。"
            "误差类与 DTW 默认先按起始值归一化，避免价格水平掩盖形态；"
            "DTW 允许两条曲线长度不同，其他算法要求等长。",
            left,
        )
        self.label_hint.setWordWrap(True)
        self.label_hint.setStyleSheet("color: #52606d;")
        left_layout.addWidget(self.box_a)
        left_layout.addWidget(self.box_b)
        left_layout.addWidget(self.check_truncate)
        left_layout.addWidget(self.btn_compare)
        left_layout.addWidget(self.label_hint)
        left_layout.addStretch(1)

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        self.table_results = QTableWidget(0, 4, right)
        self.table_results.setHorizontalHeaderLabels(["算法", "原始值", "有效点", "状态说明"])
        self.table_results.verticalHeader().setVisible(False)
        self.table_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        results_header = self.table_results.horizontalHeader()
        results_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        results_header.setStretchLastSection(True)
        self.table_results.setMaximumHeight(230)
        self.table_results.setMinimumHeight(150)
        self.btn_export = QPushButton("导出结果…（CSV）", right)
        self.btn_export.setToolTip("导出当前两条曲线的全部算法结果（含状态与说明）")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_current)
        self._figure = Figure(figsize=(7.2, 4.4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self._figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, right)
        self.label_status = QLabel("选择两条曲线后点击“计算相似度”", right)
        self.label_status.setWordWrap(True)
        self.label_status.setStyleSheet("color: #52606d;")
        right_layout.addWidget(self.table_results)
        right_layout.addWidget(self.btn_export)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)
        right_layout.addWidget(self.label_status)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 900])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # 数据源与计算
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
        sources = self._collect_sources()
        self.box_a.set_sources(sources)
        self.box_b.set_sources(sources)

    def _series_of(self, box: _CurveBox) -> pd.Series | None:
        source = box.current_source()
        field = box.field()
        if source is None or field is None:
            return None
        start, end = box.dates()
        series, _notes = _series_from_source(source, field, start, end)
        return series

    def compare_current(self) -> None:
        if self.box_a.current_source() is None or self.box_b.current_source() is None:
            QMessageBox.information(self, "提示", "请先选择两条曲线。")
            return
        try:
            left = self._series_of(self.box_a)
            right = self._series_of(self.box_b)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法取值", exc.user_message())
            return
        if left is None or right is None or left.empty or right.empty:
            QMessageBox.information(self, "提示", "所选区间没有可用数据。")
            return

        note = ""
        if len(left) != len(right):
            original_left, original_right = len(left), len(right)
            if not self.check_truncate.isChecked():
                note = f"两条曲线长度不一致（{original_left} 与 {original_right}），未做任何截断"
            else:
                size = min(original_left, original_right)
                left = left.iloc[:size].copy()
                right = right.iloc[:size].copy()
                note = f"两条曲线长度不一致（{original_left} 与 {original_right}），已按较短的 {size} 个数据点截断后比较"

        self._plot_series = (left, right)
        self._results = self._similarity.compare_all(left, right)
        self._fill_results()
        self._plot_curves(left, right)
        self.compared.emit()

        valid = [item for item in self._results if item.is_valid]
        message = f"A：{self.box_a.label()}｜B：{self.box_b.label()}｜有效采样点：{valid[0].sample_size if valid else 0}"
        if note:
            message += "｜" + note
        self.label_status.setText(message)

    def _fill_results(self) -> None:
        self.table_results.setRowCount(len(self._results))
        self.btn_export.setEnabled(bool(self._results))
        for row, item in enumerate(self._results):
            value = f"{item.raw_value:.6f}" if item.raw_value is not None else "—"
            status_text = "正常" if item.is_valid else item.message
            if item.is_valid and item.normalized:
                status_text += "（已归一化）"
            cells = (item.label, value, str(item.sample_size), status_text)
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if column in (1, 2):
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table_results.setItem(row, column, cell)

    def export_current(self) -> None:
        """把当前两条曲线的全部算法结果导出为 CSV。"""
        if not self._results:
            QMessageBox.information(self, "没有可导出的结果", "请先计算相似度。")
            return
        export = getattr(self._exporter, "export_similarity_results", None)
        if export is None:
            QMessageBox.warning(self, "无法导出", "结果导出模块尚未接入。")
            return
        field = self.box_a.field() or "结果"
        default = str(Path.cwd() / f"相似度对比_{field}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出相似度结果", default, "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            saved = export(
                self._results,
                path,
                curve_a=self.box_a.label(),
                curve_b=self.box_b.label(),
                metadata=[f"曲线 A：{self.box_a.label()}", f"曲线 B：{self.box_b.label()}"],
            )
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "导出失败", exc.user_message())
            return
        self.label_status.setText(f"已导出相似度结果：{saved}")

    def _plot_curves(self, left: pd.Series, right: pd.Series) -> None:
        curves = (
            CurveData(label=f"A：{self.box_a.label()}", values=_normalized(left), style={"color": "C0"}),
            CurveData(label=f"B：{self.box_b.label()}", values=_normalized(right), style={"color": "C1"}),
        )
        build_figure(
            PlotRequest(
                curves=curves,
                title="两条曲线对比（按首个有效值归一化）",
                x_label="比较点序号（第 N 个数据点）",
                y_label="归一化数值（首个有效值 = 100）",
            ),
            figure=self._figure,
        )
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 供测试使用
    # ------------------------------------------------------------------
    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def results(self) -> tuple[SimilarityResult, ...]:
        return self._results


def _series_from_source(
    source: MarketDataset | ProcessedDataset,
    field: str,
    start: date | None,
    end: date | None,
) -> tuple[pd.Series, tuple[str, ...]]:
    """取数：原始数据按日期切片并归一化到相对交易日；处理结果直接取值。"""
    from modules.trend_analysis.trend_service import series_from_source

    if isinstance(source, ProcessedDataset):
        return series_from_source(source, field)
    return series_from_source(
        source,
        field,
        start=start,
        end=end,
        normalization=Normalization.START,
        alignment=Alignment.RELATIVE_DAY,
    )


def _normalized(series: pd.Series) -> pd.Series:
    finite = series.dropna()
    if finite.empty:
        values = series.astype("float64")
    else:
        base = float(finite.iloc[0])
        values = series.astype("float64")
        if base != 0.0:
            values = values / base * 100.0
    values = values.copy()
    values.index = pd.RangeIndex(len(values))
    return values
