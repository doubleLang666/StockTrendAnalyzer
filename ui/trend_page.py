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

from dataclasses import replace
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
    DataKind,
    Normalization,
)
from core.data_models import MarketDataset, ProcessedDataset, SearchResult, SimilarityConfig, field_label
from core.exceptions import StockAnalyzerError
from modules.charting import (
    CandleChartController,
    ChartConfig,
    CursorInfo,
    WindowInfo,
    build_candlestick_figure,
    build_dual_candle_figure,
    build_normalized_figure,
    prepare_candle_data,
)
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
        config: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._trend = trend
        self._exporter = exporter
        self._config = config
        self._chart_config = ChartConfig.from_app_config(config)
        self._datasets_provider = datasets_provider
        self._processed_provider = processed_provider
        self._similarity_provider = similarity_provider
        self._sources: list[tuple[str, MarketDataset | ProcessedDataset]] = []
        self._segments: list[Segment] = []
        self._search_results: list[SearchResult] = []
        self._search_target: Segment | None = None
        self._search_field: str | None = None
        self._chart_result: object | None = None
        self._controller: CandleChartController | None = None
        self._cursor_visible = False

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

        box_chart = QGroupBox("K线图（蜡烛图）", left)
        chart_layout = QVBoxLayout(box_chart)
        self.combo_chart_type = QComboBox(box_chart)
        self.combo_chart_type.addItem("折线图（多字段）", "line")
        self.combo_chart_type.addItem("K线图（蜡烛图）", "candle")
        self.combo_chart_type.setToolTip(
            "折线图沿用原有曲线绘制；K线图由开盘/最高/最低/收盘价生成蜡烛，并附加成交量副图"
        )
        self.combo_chart_type.currentIndexChanged.connect(self._on_chart_type_changed)
        chart_layout.addWidget(QLabel("图表类型", box_chart))
        chart_layout.addWidget(self.combo_chart_type)

        ma_row = QHBoxLayout()
        ma_row.addWidget(QLabel("均线", box_chart))
        self.check_ma5 = QCheckBox("MA5", box_chart)
        self.check_ma10 = QCheckBox("MA10", box_chart)
        self.check_ma20 = QCheckBox("MA20", box_chart)
        for window, widget in ((5, self.check_ma5), (10, self.check_ma10), (20, self.check_ma20)):
            widget.setChecked(window in self._chart_config.ma_windows)
            widget.setToolTip(f"是否显示 {window} 日简单移动平均线（由本地数据计算）")
            ma_row.addWidget(widget)
        ma_row.addStretch(1)
        chart_layout.addLayout(ma_row)

        volume_row = QHBoxLayout()
        self.check_volume = QCheckBox("成交量副图", box_chart)
        self.check_volume.setChecked(self._chart_config.show_volume)
        self.check_volume.setToolTip("数据不含成交量字段时自动隐藏该副图，不做任何估算")
        volume_row.addWidget(self.check_volume)
        volume_row.addStretch(1)
        chart_layout.addLayout(volume_row)

        indicator_row = QHBoxLayout()
        indicator_row.addWidget(QLabel("指标", box_chart))
        self.check_macd = QCheckBox("MACD", box_chart)
        self.check_macd.setChecked(self._chart_config.show_macd)
        self.check_macd.setToolTip("MACD(12,26,9) 副图：DIF、DEA 与柱状值，由本地收盘价计算")
        self.check_rsi = QCheckBox("RSI", box_chart)
        self.check_rsi.setChecked(self._chart_config.show_rsi)
        self.check_rsi.setToolTip("RSI(6,12,24) 副图：0~100，含 30/70 参考线，由本地收盘价计算")
        self.check_boll = QCheckBox("BOLL", box_chart)
        self.check_boll.setChecked(self._chart_config.show_boll)
        self.check_boll.setToolTip("BOLL(20,2) 叠加在主图：中轨与上下轨，由本地收盘价计算")
        for widget in (self.check_macd, self.check_rsi, self.check_boll):
            indicator_row.addWidget(widget)
        indicator_row.addStretch(1)
        chart_layout.addLayout(indicator_row)

        for widget in (
            self.check_ma5,
            self.check_ma10,
            self.check_ma20,
            self.check_volume,
            self.check_macd,
            self.check_rsi,
            self.check_boll,
        ):
            widget.toggled.connect(self._on_chart_option_changed)

        bars_row = QHBoxLayout()
        bars_row.addWidget(QLabel("初始显示", box_chart))
        self.spin_initial_bars = QSpinBox(box_chart)
        self.spin_initial_bars.setRange(10, 2000)
        self.spin_initial_bars.setSingleStep(20)
        self.spin_initial_bars.setValue(max(10, min(2000, self._chart_config.initial_bars)))
        self.spin_initial_bars.setSuffix(" 根")
        self.spin_initial_bars.setToolTip("打开图表时默认显示最近多少个交易日，可用工具条缩放查看更早数据")
        bars_row.addWidget(self.spin_initial_bars, 1)
        chart_layout.addLayout(bars_row)

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
        self.combo_compare_mode = QComboBox(box_compare)
        self.combo_compare_mode.addItem("多区间折线对比", "line")
        self.combo_compare_mode.addItem("K线并排对比（模式 A）", "candle_side")
        self.combo_compare_mode.addItem("归一化走势对比（模式 B）", "normalized")
        self.combo_compare_mode.setToolTip(
            "折线：同一字段的多段走势；K线并排：前两个区间的蜡烛图等宽并排；"
            "归一化：每个所选字段各一行，把两段区间都缩放到首个有效值 = 100"
        )
        self.btn_compare = QPushButton("绘制比较图", box_compare)
        self.btn_compare.clicked.connect(self.plot_comparison)

        compare_layout.addWidget(self.table_ranges)
        compare_layout.addLayout(buttons)
        compare_layout.addWidget(self.check_truncate)
        compare_layout.addWidget(self.combo_compare_mode)
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
        left_layout.addWidget(box_chart)
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
        self.label_chart_info = QLabel("", right)
        self.label_chart_info.setWordWrap(True)
        self.label_chart_info.setStyleSheet("color: #52606d;")
        self.label_chart_info.setVisible(False)

        self.chart_controls = QWidget(right)
        controls_layout = QHBoxLayout(self.chart_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self.btn_zoom_in = QPushButton("放大", self.chart_controls)
        self.btn_zoom_out = QPushButton("缩小", self.chart_controls)
        self.btn_zoom_reset = QPushButton("恢复默认", self.chart_controls)
        self.btn_zoom_all = QPushButton("显示全部", self.chart_controls)
        for button, tip in (
            (self.btn_zoom_in, "以窗口中心为锚点放大（也可在图上滚动滚轮）"),
            (self.btn_zoom_out, "缩小，显示更多交易日"),
            (self.btn_zoom_reset, "回到初始显示区间"),
            (self.btn_zoom_all, "显示全部历史区间"),
        ):
            button.setToolTip(tip)
            button.setEnabled(False)
            controls_layout.addWidget(button)
        controls_layout.addStretch(1)
        self.chart_controls.setVisible(False)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_zoom_reset.clicked.connect(self._zoom_reset)
        self.btn_zoom_all.clicked.connect(self._zoom_all)

        right_layout.addWidget(self.label_chart_info)
        right_layout.addWidget(self.chart_controls)
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

    def _on_chart_type_changed(self) -> None:
        is_candle = self.combo_chart_type.currentData() == "candle"
        self.btn_draw.setText("绘制K线图" if is_candle else "绘制曲线")
        self.list_fields.setEnabled(not is_candle)
        self.chart_controls.setVisible(is_candle)
        self.toolbar.setVisible(not is_candle)
        if not is_candle:
            self._close_controller()
            self.label_chart_info.setVisible(False)
        self._sync_zoom_buttons()
        self.label_status.setText(
            "K线图取开盘/最高/最低/收盘价绘制蜡烛，均线与成交量由本地数据计算；"
            "滚轮缩放横轴、按住左键拖动可平移"
            if is_candle
            else "折线图按左侧勾选的字段绘制"
        )

    def _zoom_buttons(self) -> tuple[QPushButton, ...]:
        return (self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_reset, self.btn_zoom_all)

    def _sync_zoom_buttons(self) -> None:
        enabled = self._controller is not None and self.combo_chart_type.currentData() == "candle"
        for button in self._zoom_buttons():
            button.setEnabled(enabled)

    def _close_controller(self) -> None:
        """断开旧图表的事件连接，避免重绘后旧控制器继续响应鼠标。"""
        if self._controller is not None:
            self._controller.close()
            self._controller = None
        self._chart_result = None
        self._cursor_visible = False
        for button in self._zoom_buttons():
            button.setEnabled(False)

    def plot_current(self) -> None:
        """按当前图表类型分派：折线图保持原有逻辑，K线图走 charting 模块。"""
        if self.combo_chart_type.currentData() == "candle":
            self.plot_candles()
            return
        self._close_controller()
        self.chart_controls.setVisible(False)
        self.toolbar.setVisible(True)
        self._plot_lines()

    def _plot_lines(self) -> None:
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
        self._set_chart_notice(notes)
        message = f"已绘制 {len(curves)} 条曲线：{title}"
        if notes:
            message += "｜" + "；".join(dict.fromkeys(notes))
        self.label_status.setText(message)

    # ------------------------------------------------------------------
    # K 线图（专业行情图）
    # ------------------------------------------------------------------
    def plot_candles(self) -> None:
        source = self.current_source()
        if source is None:
            QMessageBox.information(self, "提示", "请先在“数据导入”页签导入原始行情数据。")
            return
        if isinstance(source, ProcessedDataset) and (
            source.config.normalization != Normalization.NONE
            or source.config.data_kind == DataKind.RELATIVE
        ):
            QMessageBox.warning(
                self,
                "无法绘制K线",
                "当前选择的是已归一化或相对化的处理结果，价格已被缩放，画出的K线不是真实价格。\n"
                "请在数据源中选择“原始数据”。",
            )
            return
        try:
            frame, name = source_frame(source)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法绘制K线", exc.user_message())
            return
        frame = self._slice_frame(frame, self._date_value(self.date_start), self._date_value(self.date_end))
        if frame.empty:
            QMessageBox.information(self, "提示", "所选日期范围内没有数据。")
            return
        try:
            candle = prepare_candle_data(frame, name)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法绘制K线", exc.user_message())
            return
        self._render_candles(candle)

    def _render_candles(self, candle) -> bool:
        """把整理好的行情画成 K 线并接上交互控制器。"""
        try:
            result = build_candlestick_figure(candle, self._candle_config(), figure=self._figure)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法绘制K线", exc.user_message())
            return False

        self._close_controller()
        self._chart_result = result
        self.toolbar.setVisible(False)
        self.chart_controls.setVisible(True)
        self._controller = CandleChartController(
            result,
            self.canvas,
            on_cursor=self._on_cursor_info,
            on_window=self._on_window_info,
        )
        self._sync_zoom_buttons()
        self.canvas.draw_idle()
        self.plotted.emit()
        self.check_volume.setEnabled(candle.has_volume)
        self.check_volume.setToolTip(
            "数据不含成交量字段时自动隐藏该副图，不做任何估算"
            if candle.has_volume
            else "该数据不含有效成交量，已隐藏成交量副图"
        )
        message = result.header
        if candle.notes:
            message += "｜" + "；".join(dict.fromkeys(candle.notes))
        self.label_status.setText(message)
        return True

    # ------------------------------------------------------------------
    # K 线交互：信息栏与缩放按钮
    # ------------------------------------------------------------------
    def _zoom_in(self) -> None:
        if self._controller is not None:
            self._controller.zoom_in()

    def _zoom_out(self) -> None:
        if self._controller is not None:
            self._controller.zoom_out()

    def _zoom_reset(self) -> None:
        if self._controller is not None:
            self._controller.reset()

    def _zoom_all(self) -> None:
        if self._controller is not None:
            self._controller.show_all()

    def _on_window_info(self, info: WindowInfo) -> None:
        if self._cursor_visible:
            return
        self._set_info_text(info.text(), None)

    def _on_cursor_info(self, info: CursorInfo | None) -> None:
        self._cursor_visible = info is not None
        if info is None:
            if self._controller is not None:
                self._set_info_text(self._controller.window_info().text(), None)
            return
        colors = {"up": "#c8102e", "down": "#1a7f4b", "flat": "#52606d"}
        self._set_info_text(info.text(), colors.get(info.direction))

    def _set_info_text(self, text: str, color: str | None) -> None:
        self.label_chart_info.setText(text)
        self.label_chart_info.setStyleSheet(
            f"color: {color}; font-weight: 600;" if color else "color: #52606d;"
        )
        self.label_chart_info.setVisible(bool(text))

    def _set_chart_notice(self, notes: object) -> None:
        """把数据质量说明放在图表上方；没有说明时不占版面。

        股票名称、周期与时间区间由图形自身的信息栏（suptitle）显示，
        这里只提示被忽略或修正的数据，避免同一段文字出现两遍。
        """
        text = "；".join(dict.fromkeys(notes)) if notes else ""
        self.label_chart_info.setText(text)
        self.label_chart_info.setStyleSheet("color: #9a6700; font-weight: 600;")
        self.label_chart_info.setVisible(bool(text))

    def _on_chart_option_changed(self, _checked: bool = False) -> None:
        """指标/均线/成交量开关变动后立刻重绘，已能看到效果就不必再点按钮。"""
        if self._chart_result is None or self.combo_chart_type.currentData() != "candle":
            return
        self.plot_current()

    def _candle_config(self) -> ChartConfig:
        windows = [
            window
            for window, widget in ((5, self.check_ma5), (10, self.check_ma10), (20, self.check_ma20))
            if widget.isChecked()
        ]
        return replace(
            self._chart_config,
            ma_windows=tuple(windows),
            show_volume=self.check_volume.isChecked(),
            show_macd=self.check_macd.isChecked(),
            show_rsi=self.check_rsi.isChecked(),
            show_boll=self.check_boll.isChecked(),
            initial_bars=self.spin_initial_bars.value(),
        )

    def _slice_frame(
        self, frame: pd.DataFrame, start: date | None, end: date | None
    ) -> pd.DataFrame:
        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end)]
        return frame

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
        mode = self.combo_compare_mode.currentData() or "line"
        if mode == "candle_side":
            self._plot_candle_comparison(fields)
            return
        if mode == "normalized":
            self._plot_normalized_comparison(fields)
            return

        self._close_controller()
        self.chart_controls.setVisible(False)
        self.toolbar.setVisible(True)
        self.label_chart_info.setVisible(False)
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
    # 历史区间对比：模式 A（K线并排）与模式 B（归一化走势）
    # ------------------------------------------------------------------
    def _comparison_frame(self) -> tuple[pd.DataFrame, str] | None:
        source = self.current_source()
        if source is None:
            QMessageBox.information(self, "提示", "请先选择数据源。")
            return None
        frame, name = source_frame(source)
        if not isinstance(frame.index, pd.DatetimeIndex):
            QMessageBox.information(
                self,
                "无法比较",
                "当前数据使用相对交易日索引，无法按日期选择历史区间。",
            )
            return None
        return frame, name

    def _comparison_segments(self, count: int) -> list[Segment] | None:
        segments = list(self._segments)[:count]
        missing = count - len(segments)
        if missing > 0:
            QMessageBox.information(
                self,
                "提示",
                f"该对比方式需要 {count} 个比较区间，当前只有 {len(segments)} 个。\n"
                "请先在“多周期比较”里添加区间。",
            )
            return None
        return segments

    def _plot_candle_comparison(self, fields: list[str]) -> None:
        frame_name = self._comparison_frame()
        if frame_name is None:
            return
        frame, name = frame_name
        if isinstance(self.current_source(), ProcessedDataset):
            processed = self.current_source()
            if processed.config.normalization != Normalization.NONE or processed.config.data_kind == DataKind.RELATIVE:
                QMessageBox.warning(
                    self,
                    "无法绘制K线",
                    "当前数据已被归一化或相对化，价格不是真实价格，无法做 K 线并排对比。\n"
                    "请在数据源中选择原始数据。",
                )
                return
        segments = self._comparison_segments(2)
        if segments is None:
            return

        candles = []
        for segment in segments:
            sub = self._slice_frame(frame, segment.start, segment.end)
            if sub.empty:
                QMessageBox.information(
                    self,
                    "提示",
                    f"区间 {segment.label()} 内没有数据。",
                )
                return
            candles.append(
                prepare_candle_data(
                    sub,
                    f"{segment.start.strftime('%Y-%m-%d')} ~ {segment.end.strftime('%Y-%m-%d')}",
                )
            )

        self._close_controller()
        self.chart_controls.setVisible(False)
        self.toolbar.setVisible(True)
        self.label_chart_info.setVisible(True)
        try:
            result = build_dual_candle_figure(
                candles[0],
                candles[1],
                self._candle_config(),
                figure=self._figure,
                labels=(candles[0].name, candles[1].name),
            )
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法比较", exc.user_message())
            return

        self.canvas.draw_idle()
        self.plotted.emit()
        self._set_info_text(
            f"模式 A：{candles[0].name} ↔ {candles[1].name}（横轴等宽，各 {result.bars} 根）",
            None,
        )
        self.label_status.setText(
            f"K线并排对比｜{name}｜较长区间按较短区间截断到 {result.bars} 个交易日"
            "（长度不同时只比较前 N 根）"
        )

    def _plot_normalized_comparison(self, fields: list[str]) -> None:
        frame_name = self._comparison_frame()
        if frame_name is None:
            return
        frame, name = frame_name
        segments = self._comparison_segments(2)
        if segments is None:
            return
        chosen = fields[:3]
        if len(fields) > 3:
            self.label_status.setText("归一化对比最多同时展示 3 个字段，已按勾选顺序取前 3 个。")

        panels: list[tuple[str, list[tuple[str, pd.Series]]]] = []
        for field in chosen:
            if field not in frame.columns:
                continue
            curves: list[tuple[str, pd.Series]] = []
            for segment in segments:
                sub = self._slice_frame(frame, segment.start, segment.end)
                if sub.empty or field not in sub.columns:
                    continue
                curves.append(
                    (
                        f"{segment.start.strftime('%Y-%m-%d')} ~ {segment.end.strftime('%Y-%m-%d')}",
                        sub[field],
                    )
                )
            if curves:
                panels.append((field_label(field), curves))

        if not panels:
            QMessageBox.information(self, "提示", "所选区间内没有可用于对比的数据。")
            return

        self._close_controller()
        self.chart_controls.setVisible(False)
        self.toolbar.setVisible(True)
        self.label_chart_info.setVisible(True)
        try:
            build_normalized_figure(panels, self._candle_config(), figure=self._figure)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法比较", exc.user_message())
            return

        self.canvas.draw_idle()
        self.plotted.emit()
        self._set_info_text(
            f"模式 B：{name}｜{len(panels)} 个指标 × 2 段区间，均归一化到首个有效值 = 100",
            None,
        )
        self.label_status.setText(
            "归一化走势对比｜"
            + "、".join(label for label, _curves in panels)
            + "｜注意：曲线为归一化数值，不代表真实价格"
        )


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
        if self.combo_chart_type.currentData() == "candle":
            self._locate_result(result, label)
            return
        if not self.check_overlay.isChecked():
            self.label_status.setText(label)
            return
        try:
            self._plot_overlay(result)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法叠加曲线", exc.user_message())
            return
        self.label_status.setText(f"{label}｜曲线已按首个有效值归一化叠加显示")

    def _locate_result(self, result: SearchResult, label: str) -> None:
        """K线模式下把检索到的区间定位并高亮出来（验收第 9 项）。"""
        if self._controller is None:
            frame_name = self._comparison_frame()
            if frame_name is None:
                return
            frame, name = frame_name
            try:
                candle = prepare_candle_data(frame, name)
            except StockAnalyzerError as exc:
                QMessageBox.warning(self, "无法定位", exc.user_message())
                return
            if not self._render_candles(candle):
                return
        if self._controller is None:  # pragma: no cover - 防御性分支
            return
        score = f"，{result.result.label} = {result.score:.4f}" if result.score is not None else ""
        highlighted = self._controller.locate(
            result.start_date,
            result.end_date,
            label=f"第 {result.rank} 名 {result.start_date} ~ {result.end_date}{score}",
        )
        self.canvas.draw_idle()
        if highlighted is None:
            self.label_status.setText(f"{label}｜该区间不在当前K线数据的日期范围内，无法定位")
            return
        self.label_status.setText(f"{label}｜已定位并高亮该区间的K线（主副图同步）")
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
        self._close_controller()
        self.chart_controls.setVisible(False)
        self.toolbar.setVisible(True)
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
