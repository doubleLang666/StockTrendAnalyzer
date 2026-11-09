"""多指标联动分析页面（设计文档第 5 节界面的第一、二区）。

阶段一职责：选数据源与两段区间 → 勾选指标 → 对每个指标分别计算
Pearson / RMSE / MAE 与 0–100 评分 → 表格 + 综合分摘要。

界面风格与趋势分析页保持一致（左侧条件区可滚动、右侧结果区），
只读取"原始数据"，处理结果（归一化/相对化）不能用于多指标分析。
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.data_models import MarketDataset
from core.exceptions import StockAnalyzerError
from modules.multi_indicator import (
    SCORE_MODES,
    MultiIndicatorRequest,
    MultiIndicatorResult,
    MultiSearchRequest,
    MultiSearchResult,
    analyze_dataset,
    default_weights,
    score_mode_label,
    search_history as run_search_history,
    select_default_indicators,
    unavailable_reason,
)
from modules.multi_indicator.charts import (
    build_indicator_panels,
    build_overlay_figure,
    build_score_radar,
)
from modules.multi_indicator.specs import (
    INDICATORS,
    KIND_RATE,
    KIND_TURNOVER,
    KIND_VOLUME,
    IndicatorSpec,
)

__all__ = ["MultiIndicatorPage"]

_MODE_CHOICES: tuple[tuple[str, str], ...] = tuple(
    (score_mode_label(mode), mode) for mode in SCORE_MODES
)

_TABLE_HEADERS = (
    "指标",
    "变换口径",
    "权重",
    "Pearson",
    "RMSE",
    "MAE",
    "Pearson 分",
    "RMSE 分",
    "MAE 分",
    "三项均分",
    "有效点",
    "状态",
)

_SEARCH_HEADERS = (
    "排名",
    "区间",
    "交易日",
    "综合评分",
    "分级",
    "有效指标",
)

#: 绩效着色：与分级阈值一致（90/80/65/50）。
_SCORE_COLORS: tuple[tuple[float, str], ...] = (
    (90.0, "#1a7f4b"),
    (80.0, "#2f7d32"),
    (65.0, "#9a6700"),
    (50.0, "#b35c00"),
    (0.0, "#a12b2b"),
)


def _score_color(score: float | None) -> str:
    if score is None:
        return "#7b8794"
    for threshold, color in _SCORE_COLORS:
        if score >= threshold:
            return color
    return "#a12b2b"


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _format_raw(spec: IndicatorSpec, value: float) -> str:
    """把原始数值按指标类型格式化，供悬停提示使用。"""
    if value is None or pd.isna(value):
        return "—"
    kind = spec.kind
    if kind == KIND_VOLUME:
        if abs(value) >= 1e8:
            return f"{value / 1e8:.2f} 亿"
        if abs(value) >= 1e4:
            return f"{value / 1e4:.1f} 万"
        return f"{value:.0f}"
    if kind in (KIND_RATE, KIND_TURNOVER):
        return f"{value:+.2f}%"
    return f"{value:.2f}"


class MultiIndicatorPage(QWidget):
    """多指标联动分析（阶段一：指标选择 + 三算法计算 + 结果表）。"""

    def __init__(
        self,
        datasets_provider,
        similarity_provider,
        parent: QWidget | None = None,
        config: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._datasets_provider = datasets_provider
        self._similarity_provider = similarity_provider
        self._config = config
        self._dataset: MarketDataset | None = None
        self._result: MultiIndicatorResult | None = None
        self._search_results: tuple[MultiSearchResult, ...] = ()
        self._specs: tuple[IndicatorSpec, ...] = ()

        self._build_ui()
        self._hover_ids = [
            self.canvas_overlay.mpl_connect("motion_notify_event", self._on_chart_hover),
            self.canvas_panels.mpl_connect("motion_notify_event", self._on_chart_hover),
        ]
        self.refresh()
        for canvas in self._canvases:
            self._draw_message(canvas, "点击「开始分析」后在这里查看图表。")

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 950])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(splitter)

    def _build_controls(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 6, 0)

        scroll = QScrollArea(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(scroll)
        inner.setMinimumWidth(0)
        left = QVBoxLayout(inner)
        left.setContentsMargins(4, 4, 8, 4)

        # 数据源
        box_source = QGroupBox("数据源", inner)
        form_source = QFormLayout(box_source)
        self.combo_source = QComboBox(box_source)
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        form_source.addRow("原始数据", self.combo_source)
        left.addWidget(box_source)

        # 目标区间
        box_target = QGroupBox("目标区间", inner)
        form_target = QFormLayout(box_target)
        self.date_target_start = self._make_date_edit(box_target)
        self.date_target_end = self._make_date_edit(box_target)
        form_target.addRow("起始", self.date_target_start)
        form_target.addRow("结束", self.date_target_end)
        left.addWidget(box_target)

        # 对比区间
        box_compare = QGroupBox("对比区间", inner)
        form_compare = QFormLayout(box_compare)
        self.date_compare_start = self._make_date_edit(box_compare)
        self.date_compare_end = self._make_date_edit(box_compare)
        form_compare.addRow("起始", self.date_compare_start)
        form_compare.addRow("结束", self.date_compare_end)
        left.addWidget(box_compare)

        # 指标
        box_indicators = QGroupBox("参与分析的指标", inner)
        box_indicators_layout = QVBoxLayout(box_indicators)
        self.list_indicators = QListWidget(box_indicators)
        self.list_indicators.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_indicators.setMaximumHeight(170)
        self.list_indicators.itemChanged.connect(lambda _item: self._on_indicator_toggled())
        box_indicators_layout.addWidget(self.list_indicators)
        self.label_hint = QLabel(box_indicators)
        self.label_hint.setWordWrap(True)
        self.label_hint.setStyleSheet("color: #7b8794; font-size: 12px;")
        box_indicators_layout.addWidget(self.label_hint)
        left.addWidget(box_indicators)

        # 权重
        box_weights = QGroupBox("指标权重（%）", inner)
        weights_layout = QVBoxLayout(box_weights)
        self.weight_grid = QGridLayout()
        self.weight_grid.setColumnStretch(1, 1)
        weights_layout.addLayout(self.weight_grid)
        self._weight_inputs: dict[str, QDoubleSpinBox] = {}
        self._weight_labels: dict[str, QLabel] = {}
        for row, spec in enumerate(INDICATORS):
            label = QLabel(spec.label, box_weights)
            spin = QDoubleSpinBox(box_weights)
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(5.0)
            spin.setSuffix(" %")
            spin.setMinimumWidth(72)
            spin.setMaximumWidth(120)
            spin.valueChanged.connect(lambda _value: self._update_weight_total())
            self.weight_grid.addWidget(label, row, 0)
            self.weight_grid.addWidget(spin, row, 1)
            self._weight_inputs[spec.key] = spin
            self._weight_labels[spec.key] = label
        self.label_weight_total = QLabel(box_weights)
        self.label_weight_total.setWordWrap(True)
        weights_layout.addWidget(self.label_weight_total)
        btn_reset = QPushButton("恢复默认权重", box_weights)
        btn_reset.clicked.connect(self._reset_weights)
        weights_layout.addWidget(btn_reset)
        left.addWidget(box_weights)

        # 评分设置
        box_score = QGroupBox("评分设置", inner)
        form_score = QFormLayout(box_score)
        self.combo_mode = QComboBox(box_score)
        for label, mode in _MODE_CHOICES:
            self.combo_mode.addItem(label, mode)
        form_score.addRow("评分模式", self.combo_mode)
        self.spin_min_ratio = QSpinBox(box_score)
        self.spin_min_ratio.setRange(50, 100)
        self.spin_min_ratio.setValue(80)
        self.spin_min_ratio.setSuffix(" %")
        self.spin_min_ratio.setToolTip(
            "两段区间中同时有效的交易日占比低于该比例时，该指标不参与评分（不填 0、不插值）"
        )
        form_score.addRow("最低有效比例", self.spin_min_ratio)
        left.addWidget(box_score)

        self.btn_analyze = QPushButton("开始分析", inner)
        self.btn_analyze.setMinimumHeight(32)
        self.btn_analyze.clicked.connect(self.analyze)
        left.addWidget(self.btn_analyze)

        # 历史检索（多指标）
        box_search = QGroupBox("历史检索（多指标）", inner)
        form_search = QFormLayout(box_search)
        self.date_scope_start = self._make_date_edit(box_search)
        self.date_scope_end = self._make_date_edit(box_search)
        form_search.addRow("历史起", self.date_scope_start)
        form_search.addRow("历史止", self.date_scope_end)
        self.spin_top_n = QSpinBox(box_search)
        self.spin_top_n.setRange(1, 20)
        self.spin_top_n.setValue(5)
        self.spin_top_n.setSuffix(" 条")
        form_search.addRow("返回数量", self.spin_top_n)
        self.btn_search = QPushButton("检索历史区间", box_search)
        self.btn_search.clicked.connect(self.search_history)
        form_search.addRow(self.btn_search)
        hint = QLabel(
            "只使用目标区间之前的历史数据；综合分按上方权重合成，"
            "相邻重叠的候选区间会自动去重。",
            box_search,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #7b8794; font-size: 11px;")
        form_search.addRow(hint)
        left.addWidget(box_search)

        left.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        return container

    def _build_results(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label_total = QLabel("尚未分析", container)
        self.label_total.setWordWrap(True)
        self.label_total.setTextFormat(Qt.TextFormat.RichText)
        self.label_total.setMinimumHeight(56)
        self.label_total.setStyleSheet(
            "font-size: 15px; padding: 8px 10px; background: #f7f9fb; "
            "border: 1px solid #d9e1e8; border-radius: 6px;"
        )
        layout.addWidget(self.label_total)

        # 图表区：三种视图共用右侧面板，表格放在下方，中间可拖动分隔
        splitter = QSplitter(Qt.Orientation.Vertical, container)
        chart_box = QWidget(splitter)
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs_views = QTabWidget(chart_box)
        self.canvas_overlay = FigureCanvasQTAgg(Figure(figsize=(6.0, 3.1), dpi=100))
        self.canvas_panels = FigureCanvasQTAgg(Figure(figsize=(6.0, 3.1), dpi=100))
        self.canvas_radar = FigureCanvasQTAgg(Figure(figsize=(6.0, 3.1), dpi=100))
        self._canvases = (self.canvas_overlay, self.canvas_panels, self.canvas_radar)
        for title, canvas in (
            ("归一化叠加（模式 B）", self.canvas_overlay),
            ("单指标独立图（模式 A）", self.canvas_panels),
            ("指标雷达图", self.canvas_radar),
        ):
            page = QWidget(self.tabs_views)
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            toolbar = NavigationToolbar2QT(canvas, page)
            page_layout.addWidget(toolbar)
            page_layout.addWidget(canvas, 1)
            self.tabs_views.addTab(page, title)
        chart_layout.addWidget(self.tabs_views)
        self.tabs_views.setMinimumHeight(250)
        splitter.addWidget(chart_box)

        table_box = QWidget(splitter)
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_results = QTableWidget(0, len(_TABLE_HEADERS), table_box)
        self.table_results.setHorizontalHeaderLabels(list(_TABLE_HEADERS))
        self.table_results.verticalHeader().setVisible(False)
        self.table_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table_results.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table_layout.addWidget(self.table_results)

        # 历史检索结果单独一格，双击某条结果即可载入它的指标明细
        search_box = QWidget(splitter)
        search_layout = QVBoxLayout(search_box)
        search_layout.setContentsMargins(0, 0, 0, 0)
        self.table_search = QTableWidget(0, len(_SEARCH_HEADERS), search_box)
        self.table_search.setHorizontalHeaderLabels(list(_SEARCH_HEADERS))
        self.table_search.verticalHeader().setVisible(False)
        self.table_search.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_search.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_search.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        search_header = self.table_search.horizontalHeader()
        search_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        search_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        search_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_search.cellDoubleClicked.connect(self._on_search_selected)
        self.table_search.itemSelectionChanged.connect(self._on_search_selection_changed)
        search_layout.addWidget(self.table_search)
        self.label_search_note = QLabel("尚未检索：设置目标区间与历史范围后点击「检索历史区间」。", search_box)
        self.label_search_note.setWordWrap(True)
        self.label_search_note.setStyleSheet("color: #7b8794; font-size: 11px;")
        search_layout.addWidget(self.label_search_note)

        self.tabs_results = QTabWidget(splitter)
        self.tabs_results.addTab(table_box, "分析结果")
        self.tabs_results.addTab(search_box, "历史检索")
        self.tabs_results.currentChanged.connect(self._on_result_tab_changed)
        self._results_splitter = splitter
        splitter.addWidget(self.tabs_results)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([430, 250])
        layout.addWidget(splitter, 1)

        self.label_notes = QLabel(container)
        self.label_notes.setWordWrap(True)
        self.label_notes.setStyleSheet("color: #52606d; font-size: 12px;")
        layout.addWidget(self.label_notes)

        self.label_status = QLabel("就绪", container)
        self.label_status.setWordWrap(True)
        self.label_status.setStyleSheet("color: #7b8794;")
        layout.addWidget(self.label_status)
        return container

    @staticmethod
    def _make_date_edit(parent: QWidget) -> QDateEdit:
        edit = QDateEdit(parent)
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("yyyy-MM-dd")
        return edit

    # ------------------------------------------------------------------
    # 数据源
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """重新收集原始数据集。"""
        current = self.combo_source.currentData()
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        for dataset in self._datasets_provider():
            if isinstance(dataset, MarketDataset):
                self.combo_source.addItem(f"原始数据 · {dataset.name}", dataset)
        self.combo_source.blockSignals(False)
        if self.combo_source.count() == 0:
            self._dataset = None
            self.label_total.setText("尚未分析：请先在「数据导入」页导入行情数据。")
            self.label_status.setText("没有可用的原始数据。")
            self.btn_analyze.setEnabled(False)
            self._fill_indicators(None)
            return
        self.btn_analyze.setEnabled(True)
        if current is not None:
            index = self.combo_source.findData(current)
            if index >= 0:
                self.combo_source.setCurrentIndex(index)
        self._on_source_changed()

    def _current_dataset(self) -> MarketDataset | None:
        data = self.combo_source.currentData()
        return data if isinstance(data, MarketDataset) else None

    def _on_source_changed(self) -> None:
        dataset = self._current_dataset()
        self._dataset = dataset
        self._fill_indicators(dataset)
        if dataset is None:
            return
        index = dataset.frame.index
        if not isinstance(index, pd.DatetimeIndex) or index.empty:
            self.label_status.setText("该数据没有真实交易日期，无法做多指标分析。")
            self.btn_analyze.setEnabled(False)
            return
        count = len(index)
        span = max(2, min(60, count // 3))
        target_end = index[-1]
        target_start = index[max(0, count - span)]
        compare_end = index[max(0, count - span - 1)]
        compare_start = index[max(0, count - span * 2)]
        self._set_date(self.date_target_start, target_start)
        self._set_date(self.date_target_end, target_end)
        self._set_date(self.date_compare_start, compare_start)
        self._set_date(self.date_compare_end, compare_end)
        # 历史检索范围默认覆盖全部历史，引擎自己会排除目标区间之后的数据
        self._set_date(self.date_scope_start, index[0])
        self._set_date(self.date_scope_end, index[-1])
        self.label_status.setText(
            f"已选择 {dataset.name}：共 {count} 个交易日，默认使用最近两段等长区间，可自行调整。"
        )

    @staticmethod
    def _set_date(edit: QDateEdit, value: object) -> None:
        timestamp = pd.Timestamp(value)
        edit.setDate(QDate(timestamp.year, timestamp.month, timestamp.day))

    @staticmethod
    def _date_value(edit: QDateEdit) -> date:
        return edit.date().toPython()

    # ------------------------------------------------------------------
    # 指标选择
    # ------------------------------------------------------------------
    def _fill_indicators(self, dataset: MarketDataset | None) -> None:
        self.list_indicators.blockSignals(True)
        self.list_indicators.clear()
        self._specs = ()
        if dataset is None:
            self.list_indicators.blockSignals(False)
            self.label_hint.setText("")
            self._rebuild_weight_rows(None)
            self._reset_weights()
            return

        available = tuple(
            spec for spec in INDICATORS if unavailable_reason(spec, dataset.frame.columns) is None
        )
        self._specs = available
        defaults = set(select_default_indicators(available))
        for spec in INDICATORS:
            reason = unavailable_reason(spec, dataset.frame.columns)
            item = QListWidgetItem(spec.label)
            item.setData(Qt.ItemDataRole.UserRole, spec.key)
            item.setToolTip(spec.description if reason is None else reason)
            if reason is not None:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if spec.key in defaults else Qt.CheckState.Unchecked
                )
            self.list_indicators.addItem(item)

        missing = [
            f"{spec.label}（{unavailable_reason(spec, dataset.frame.columns)}）"
            for spec in INDICATORS
            if unavailable_reason(spec, dataset.frame.columns) is not None
        ]
        if missing:
            self.label_hint.setText("不可用：" + "；".join(missing))
        else:
            self.label_hint.setText("全部指标可用。")
        self.list_indicators.blockSignals(False)
        self._rebuild_weight_rows(dataset)
        self._reset_weights()

    def _rebuild_weight_rows(self, dataset: MarketDataset | None) -> None:
        """按当前数据的可用指标刷新权重行的可用状态与提示。"""
        for spec in INDICATORS:
            reason = unavailable_reason(spec, dataset.frame.columns) if dataset is not None else "无数据"
            label = self._weight_labels[spec.key]
            spin = self._weight_inputs[spec.key]
            enabled = reason is None
            label.setEnabled(enabled)
            spin.setToolTip(spec.description if enabled else (reason or ""))
            spin.setEnabled(enabled)

    def _checked_specs(self) -> tuple[IndicatorSpec, ...]:
        specs: list[IndicatorSpec] = []
        for row in range(self.list_indicators.count()):
            item = self.list_indicators.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                key = item.data(Qt.ItemDataRole.UserRole)
                spec = next((spec for spec in INDICATORS if spec.key == key), None)
                if spec is not None:
                    specs.append(spec)
        return tuple(specs)

    def checked_keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self._checked_specs())

    def _on_indicator_toggled(self) -> None:
        available = {spec.key for spec in self._specs}
        checked = set(self.checked_keys())
        for spec in INDICATORS:
            self._weight_inputs[spec.key].setEnabled(spec.key in available and spec.key in checked)
        self._update_weight_total()

    def _update_weight_total(self) -> None:
        keys = self.checked_keys()
        total = sum(self._weight_inputs[key].value() for key in keys)
        self._weight_total = total
        if not keys:
            self.label_weight_total.setText("请至少勾选一个指标。")
            return
        if abs(total - 100.0) < 0.05:
            self.label_weight_total.setText(f"合计：{total:.1f}%（共 {len(keys)} 个指标）")
            self.label_weight_total.setStyleSheet("color: #1a7f4b; font-size: 12px;")
        else:
            self.label_weight_total.setText(
                f"合计：{total:.1f}%（与 100% 不一致，计算时会按比例归一化）"
            )
            self.label_weight_total.setStyleSheet("color: #9a6700; font-size: 12px;")

    def _reset_weights(self) -> None:
        available = {spec.key for spec in self._specs}
        checked = set(self.checked_keys())
        defaults = default_weights([spec.key for spec in self._specs if spec.key in checked])
        for spec in INDICATORS:
            spin = self._weight_inputs[spec.key]
            if spec.key in available:
                spin.setValue(defaults.get(spec.key, 0.0) * 100.0)
                spin.setEnabled(spec.key in checked)
            else:
                spin.setValue(0.0)
                spin.setEnabled(False)
        self._update_weight_total()

    def weights_fraction(self) -> dict[str, float]:
        """返回 0–1 的权重（界面里输入的是百分数）。"""
        return {key: self._weight_inputs[key].value() / 100.0 for key in self.checked_keys()}

    # ------------------------------------------------------------------
    # 分析
    # ------------------------------------------------------------------
    def analyze(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            QMessageBox.information(self, "多指标分析", "请先选择一档原始数据。")
            return
        keys = self.checked_keys()
        if not keys:
            QMessageBox.information(self, "多指标分析", "请至少勾选一个指标。")
            return
        service = self._similarity_provider()
        if service is None:
            QMessageBox.warning(self, "多指标分析", "相似度服务未接入，无法计算。")
            return

        request = MultiIndicatorRequest(
            target_start=self._date_value(self.date_target_start),
            target_end=self._date_value(self.date_target_end),
            compare_start=self._date_value(self.date_compare_start),
            compare_end=self._date_value(self.date_compare_end),
            keys=keys,
            weights=self.weights_fraction(),
            score_mode=self.combo_mode.currentData() or "combined",
            min_valid_ratio=self.spin_min_ratio.value() / 100.0,
        )
        try:
            result = analyze_dataset(dataset, request, similarity_service=service)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "多指标分析", str(exc))
            self.label_status.setText(f"分析失败：{exc}")
            return

        self._apply_result(result)
        self.tabs_results.setCurrentIndex(0)

    def _apply_result(self, result: MultiIndicatorResult) -> None:
        """把一次分析结果铺到界面上（手工分析与历史检索共用）。"""
        self._result = result
        self._fill_table(result)
        self._fill_summary(result)
        self._render_charts(result)
        self.label_status.setText(result.summary_text())

    # ------------------------------------------------------------------
    # 历史检索
    # ------------------------------------------------------------------
    def search_history(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            QMessageBox.information(self, "多指标分析", "请先选择一档原始数据。")
            return
        keys = self.checked_keys()
        if not keys:
            QMessageBox.information(self, "多指标分析", "请至少勾选一个指标。")
            return
        service = self._similarity_provider()
        if service is None:
            QMessageBox.warning(self, "多指标分析", "相似度服务未接入，无法计算。")
            return

        request = MultiSearchRequest(
            target_start=self._date_value(self.date_target_start),
            target_end=self._date_value(self.date_target_end),
            scope_start=self._date_value(self.date_scope_start),
            scope_end=self._date_value(self.date_scope_end),
            keys=keys,
            weights=self.weights_fraction(),
            score_mode=self.combo_mode.currentData() or "combined",
            top_n=self.spin_top_n.value(),
            min_valid_ratio=self.spin_min_ratio.value() / 100.0,
        )
        try:
            results = run_search_history(dataset.frame, request, similarity_service=service)
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "多指标分析", str(exc))
            self.label_status.setText(f"检索失败：{exc}")
            return

        self._search_results = results
        self._fill_search_table(results)
        self.tabs_results.setCurrentIndex(1)
        if not results:
            self.label_search_note.setText(
                "没有找到可用的历史区间：可能历史范围过短，或有效交易日达不到门槛。"
            )
            self.label_status.setText("历史检索完成：没有命中符合条件的区间。")
            return
        best = results[0]
        self.label_search_note.setText(
            f"共 {len(results)} 条结果，按综合分从高到低排列；双击某一行可载入它的指标明细。"
            f"当前最高：{best.label}（{best.total_score:.2f}，{best.grade}）。"
        )
        self.label_status.setText(
            f"历史检索完成：{len(results)} 条｜目标区间 {self.date_target_start.date().toString('yyyy-MM-dd')}"
            f" ~ {self.date_target_end.date().toString('yyyy-MM-dd')}｜只使用目标区间之前的数据"
        )

    def _on_result_tab_changed(self, index: int) -> None:
        """切到检索页时把下半区放大，方便一次看更多候选区间。"""
        splitter = getattr(self, "_results_splitter", None)
        if splitter is None:
            return
        if index == 1:
            splitter.setSizes([260, 500])
        else:
            splitter.setSizes([430, 250])

    def _fill_search_table(self, results: Sequence[MultiSearchResult]) -> None:
        self.table_search.setRowCount(len(results))
        for row, item in enumerate(results):
            values = (
                str(item.rank),
                item.label,
                str(item.trading_days),
                f"{item.total_score:.2f}",
                item.grade,
                f"{item.valid_count}/{len(item.scores)}",
            )
            for column, text in enumerate(values):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 3:
                    cell.setForeground(QColor(_score_color(item.total_score)))
                self.table_search.setItem(row, column, cell)
        self.table_search.resizeRowsToContents()

    def _on_search_selection_changed(self) -> None:
        row = self.table_search.currentRow()
        if 0 <= row < len(self._search_results):
            item = self._search_results[row]
            self.label_search_note.setText(
                f"当前选中：第 {item.rank} 名 {item.label}｜综合 {item.total_score:.2f}（{item.grade}）"
                f"｜{item.describe()}"
            )

    def _on_search_selected(self, row: int, column: int) -> None:
        if not (0 <= row < len(self._search_results)):
            return
        detail = self._search_results[row].analysis
        if detail is None:
            return
        # 让左侧条件区与图表保持一致：对比区间改成被检索到的历史区间
        target = self._search_results[row]
        self._set_date(self.date_compare_start, target.start_date)
        self._set_date(self.date_compare_end, target.end_date)
        self._apply_result(detail)
        self.tabs_results.setCurrentIndex(0)
        self.label_status.setText(
            f"已载入历史检索第 {row + 1} 名：{self._search_results[row].label}"
            "（表格与图表显示的是该区间与被检索目标区间的对比）"
        )

    def _fill_table(self, result: MultiIndicatorResult) -> None:
        self.table_results.setRowCount(len(result.scores))
        for row, score in enumerate(result.scores):
            pair = result.pair_of(score.spec_key)
            values = [
                score.label,
                "涨跌幅：100 表示不变" if pair is not None and pair.spec.kind == "rate" else "首个有效值 = 100",
                f"{result.weight_of(score.spec_key):.1%}" if score.is_valid else "—",
                _fmt(score.raw_value("pearson")),
                _fmt(score.raw_value("rmse")),
                _fmt(score.raw_value("mae")),
                _fmt(score.pearson_score, 2),
                _fmt(score.rmse_score, 2),
                _fmt(score.mae_score, 2),
                _fmt(score.combined_score, 2),
                f"{score.sample_size}/{score.total_points}",
                "有效" if score.is_valid else score.message,
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column in (3, 4, 5):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if column in (6, 7, 8, 9):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    score_value = (
                        score.pearson_score
                        if column == 6
                        else score.rmse_score
                        if column == 7
                        else score.mae_score
                        if column == 8
                        else score.combined_score
                    )
                    item.setToolTip("0–100 分，越大越相似")
                    if score_value is not None:
                        item.setForeground(QColor(_score_color(score_value)))
                if column == 11 and not score.is_valid:
                    item.setToolTip(score.message)
                self.table_results.setItem(row, column, item)
        self.table_results.resizeRowsToContents()

    def _fill_summary(self, result: MultiIndicatorResult) -> None:
        if result.total_score is None:
            self.label_total.setText(
                "<b style='color:#a12b2b;'>没有可参与计算的指标</b><br/>"
                "请检查区间选择与「最低有效比例」，或更换数据源。"
            )
            self.label_notes.setText("")
            return
        color = _score_color(result.total_score)
        participants = "、".join(score.label for score in result.valid_scores)
        self.label_total.setText(
            f"<span style='font-size:26px; font-weight:700; color:{color};'>"
            f"{result.total_score:.2f}</span>"
            f"<span style='font-size:14px; color:#52606d;'> / 100　{result.grade}</span>"
            f"<br/><span style='color:#52606d;'>评分模式：{result.mode_label}　|　"
            f"有效交易日：{result.trading_days}　|　参与指标：{participants}</span>"
        )

        notes: list[str] = []
        if result.weight_changed:
            rows = []
            for label, given, actual in result.weight_rows():
                rows.append(f"{label} {given:.1%} → {actual:.1%}")
            notes.append("部分指标无效，权重已按有效指标重新归一化：" + "；".join(rows))
        for label, reason in result.skipped:
            notes.append(f"未参与评分：{label} —— {reason}")
        if not notes:
            notes.append(
                "误差评分已将 RMSE/MAE 除以两段序列合并标准差，不同量纲的指标可以直接比较；"
                "权重默认按文档示例（收盘价 40%、成交量 30%、换手率 20%、涨跌幅 10%）归一化。"
            )
        self.label_notes.setText("　".join(notes))

    # ------------------------------------------------------------------
    # 图表
    # ------------------------------------------------------------------
    def _chart_pairs(self, result: MultiIndicatorResult) -> list:
        pairs = []
        for score in result.valid_scores:
            pair = result.pair_of(score.spec_key)
            if pair is not None:
                pairs.append(pair)
        return pairs

    def _render_charts(self, result: MultiIndicatorResult) -> None:
        pairs = self._chart_pairs(result)
        if not pairs:
            self._draw_message(self.canvas_overlay, "没有可参与绘图的指标。")
            self._draw_message(self.canvas_panels, "没有可参与绘图的指标。")
            self._draw_message(self.canvas_radar, "没有可参与绘图的指标。")
            return
        self._safe_draw(
            self.canvas_overlay,
            lambda: build_overlay_figure(pairs, figure=self.canvas_overlay.figure),
        )
        self._safe_draw(
            self.canvas_panels,
            lambda: build_indicator_panels(pairs, figure=self.canvas_panels.figure),
        )
        self._safe_draw(
            self.canvas_radar,
            lambda: build_score_radar(result, figure=self.canvas_radar.figure),
        )

    def _safe_draw(self, canvas, builder) -> None:
        try:
            builder()
        except StockAnalyzerError as exc:
            self._draw_message(canvas, str(exc))
            return
        canvas.draw_idle()

    @staticmethod
    def _draw_message(canvas, text: str) -> None:
        figure = canvas.figure
        figure.clear()
        figure.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color="#7b8794", wrap=True)
        canvas.draw_idle()

    def _on_chart_hover(self, event) -> None:
        result = self._result
        if result is None or event.inaxes is None or event.xdata is None:
            return
        index = int(round(float(event.xdata)))
        parts = []
        for pair in self._chart_pairs(result):
            if not 0 <= index < pair.length:
                continue
            left = _format_raw(pair.spec, pair.target_raw.iloc[index])
            right = _format_raw(pair.spec, pair.compare_raw.iloc[index])
            parts.append(f"{pair.spec.label} 目标 {left} / 对比 {right}")
        if parts:
            self.label_status.setText(f"第 {index + 1} 个交易日｜" + "　｜　".join(parts))

    # ------------------------------------------------------------------
    # 对外
    # ------------------------------------------------------------------
    @property
    def result(self) -> MultiIndicatorResult | None:
        return self._result
