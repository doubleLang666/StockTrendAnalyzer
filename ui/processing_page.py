"""数据处理页：字段与日期选择、二级数据、归一化、时间对齐（设计文档 3.2）。

界面只负责收集配置并展示结果，所有计算都交给
:class:`~modules.data_processing.processor.DataProcessor`，界面中不含任何公式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.constants import (
    ALIGNMENT_LABELS,
    DATA_KIND_LABELS,
    DEFAULT_MIN_MAX_HIGH,
    DEFAULT_MIN_MAX_LOW,
    DERIVED_METHOD_LABELS,
    NORMALIZATION_LABELS,
    Alignment,
    DataKind,
    DerivedMethod,
    Normalization,
)
from core.data_models import (
    DerivedField,
    MarketDataset,
    ProcessedDataset,
    ProcessingConfig,
    field_label,
)
from core.exceptions import StockAnalyzerError
from modules.data_processing.selector import numeric_fields
from ui.table_model import ProcessedTableModel

__all__ = ["ProcessingPage"]

_NO_DERIVED = "__none__"
_INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def _safe_filename(name: str) -> str:
    """把数据集名称清理成可用作文件名的文本。"""
    cleaned = "".join("_" if char in _INVALID_FILENAME_CHARS else char for char in name or "")
    return cleaned.strip().strip(".")


class ProcessingPage(QWidget):
    """第三阶段的数据处理页。"""

    #: 每次生成处理结果后发出。
    processed_changed = Signal()

    def __init__(
        self,
        processor: object,
        datasets_provider: Callable[[], Sequence[MarketDataset]] | None = None,
        parent: QWidget | None = None,
        exporter: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._processor = processor
        self._exporter = exporter
        self._provider = datasets_provider or (lambda: ())
        self._datasets: list[MarketDataset] = []
        self._processed: ProcessedDataset | None = None
        self._model = ProcessedTableModel(self)
        self._build_ui()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 850])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _build_config_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 8, 12)
        layout.setSpacing(8)

        # --- 数据选择 -------------------------------------------------
        select_box = QGroupBox("数据选择", panel)
        select_form = QFormLayout(select_box)
        select_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.combo_dataset = QComboBox(select_box)
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        select_form.addRow("数据", self.combo_dataset)

        self.list_fields = QListWidget(select_box)
        self.list_fields.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_fields.setMaximumHeight(120)
        select_form.addRow("字段", self.list_fields)

        field_buttons = QHBoxLayout()
        self.btn_all_fields = QPushButton("全选", select_box)
        self.btn_all_fields.clicked.connect(lambda: self._check_all(True))
        self.btn_none_fields = QPushButton("全不选", select_box)
        self.btn_none_fields.clicked.connect(lambda: self._check_all(False))
        field_buttons.addWidget(self.btn_all_fields)
        field_buttons.addWidget(self.btn_none_fields)
        field_buttons.addStretch(1)
        select_form.addRow("", self._wrap(field_buttons, select_box))

        self.date_start = QDateEdit(select_box)
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        select_form.addRow("起始日期", self.date_start)

        self.date_end = QDateEdit(select_box)
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        select_form.addRow("结束日期", self.date_end)
        layout.addWidget(select_box)

        # --- 二级数据 -------------------------------------------------
        derived_box = QGroupBox("二级数据（可选）", panel)
        derived_form = QFormLayout(derived_box)
        self.combo_derived_method = QComboBox(derived_box)
        self.combo_derived_method.addItem("不计算", _NO_DERIVED)
        for method in DerivedMethod:
            self.combo_derived_method.addItem(DERIVED_METHOD_LABELS[method.value], method.value)
        self.combo_derived_method.currentIndexChanged.connect(
            lambda _index: self._on_derived_method_changed()
        )
        derived_form.addRow("计算方式", self.combo_derived_method)

        self.combo_derived_base = QComboBox(derived_box)
        self.combo_derived_base.setEnabled(False)
        derived_form.addRow("基于字段", self.combo_derived_base)

        derived_hint = QLabel(
            "变化率 =（当日值 − 前一有效值）÷ 前一有效值 × 100%，"
            "如日涨跌幅取「收盘价」、成交量变化率取「成交量」。\n"
            "首个记录没有前一有效值，结果保持为空；前一有效值为 0 时不计算。",
            derived_box,
        )
        derived_hint.setWordWrap(True)
        derived_hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        derived_form.addRow(derived_hint)
        layout.addWidget(derived_box)

        # --- 归一化 ---------------------------------------------------
        norm_box = QGroupBox("归一化（可选）", panel)
        norm_form = QFormLayout(norm_box)
        self.combo_normalization = QComboBox(norm_box)
        for norm in Normalization:
            self.combo_normalization.addItem(NORMALIZATION_LABELS[norm.value], norm.value)
        self.combo_normalization.currentIndexChanged.connect(
            lambda _index: self._on_normalization_changed()
        )
        norm_form.addRow("方式", self.combo_normalization)

        range_row = QHBoxLayout()
        self.spin_low = QDoubleSpinBox(norm_box)
        self.spin_low.setRange(-1e6, 1e6)
        self.spin_low.setDecimals(4)
        self.spin_low.setValue(DEFAULT_MIN_MAX_LOW)
        range_row.addWidget(self.spin_low)
        range_row.addWidget(QLabel("~", norm_box))
        self.spin_high = QDoubleSpinBox(norm_box)
        self.spin_high.setRange(-1e6, 1e6)
        self.spin_high.setDecimals(4)
        self.spin_high.setValue(DEFAULT_MIN_MAX_HIGH)
        range_row.addWidget(self.spin_high)
        range_row.addStretch(1)
        norm_form.addRow("映射区间", self._wrap(range_row, norm_box))
        self._on_normalization_changed()
        layout.addWidget(norm_box)

        # --- 对齐 -----------------------------------------------------
        align_box = QGroupBox("时间对齐", panel)
        align_form = QFormLayout(align_box)
        self.combo_alignment = QComboBox(align_box)
        for alignment in Alignment:
            self.combo_alignment.addItem(ALIGNMENT_LABELS[alignment.value], alignment.value)
        align_form.addRow("方式", self.combo_alignment)
        layout.addWidget(align_box)

        self.btn_process = QPushButton("生成处理数据", panel)
        self.btn_process.setMinimumHeight(34)
        self.btn_process.clicked.connect(self.process_current)
        layout.addWidget(self.btn_process)

        self.btn_export = QPushButton("导出数据…（CSV）", panel)
        self.btn_export.setToolTip("把当前处理结果导出为 CSV，Excel 可直接打开")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_current)
        layout.addWidget(self.btn_export)

        self.label_status = QLabel("请先在「数据导入」页导入行情数据。", panel)
        self.label_status.setWordWrap(True)
        self.label_status.setObjectName("processingStatus")
        layout.addWidget(self.label_status)
        layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(400)
        return scroll

    def _build_result_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 12, 12)
        layout.setSpacing(6)

        self.label_result = QLabel("处理结果", panel)
        self.label_result.setObjectName("processingResultTitle")
        layout.addWidget(self.label_result)

        self.table_result = QTableView(panel)
        self.table_result.setModel(self._model)
        self.table_result.setAlternatingRowColors(True)
        self.table_result.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_result.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_result.verticalHeader().setDefaultSectionSize(22)
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_result.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table_result, 3)

        self.text_notes = QPlainTextEdit(panel)
        self.text_notes.setReadOnly(True)
        self.text_notes.setPlaceholderText("处理说明与追溯信息会显示在这里。")
        self.text_notes.setMaximumHeight(150)
        layout.addWidget(self.text_notes, 1)

        return panel

    @staticmethod
    def _wrap(layout: QHBoxLayout, parent: QWidget) -> QWidget:
        holder = QWidget(parent)
        holder.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return holder

    # ------------------------------------------------------------------
    # 数据集
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """重新读取已导入的数据集列表（数据导入页变化后调用）。"""
        datasets = [item for item in self._provider() if isinstance(item, MarketDataset)]
        current_name = self.current_dataset.name if self._datasets else None
        self._datasets = datasets
        self.combo_dataset.blockSignals(True)
        self.combo_dataset.clear()
        for dataset in datasets:
            label = dataset.name
            if dataset.start_date and dataset.end_date:
                label += f"（{dataset.start_date:%Y-%m-%d} ~ {dataset.end_date:%Y-%m-%d}）"
            self.combo_dataset.addItem(label)
        index = 0
        if current_name:
            for position, dataset in enumerate(datasets):
                if dataset.name == current_name:
                    index = position
                    break
        if datasets:
            self.combo_dataset.setCurrentIndex(index)
        self.combo_dataset.blockSignals(False)
        self._on_dataset_changed(self.combo_dataset.currentIndex())

    @property
    def datasets(self) -> tuple[MarketDataset, ...]:
        return tuple(self._datasets)

    @property
    def current_dataset(self) -> MarketDataset | None:
        index = self.combo_dataset.currentIndex()
        if 0 <= index < len(self._datasets):
            return self._datasets[index]
        return None

    @property
    def processed(self) -> ProcessedDataset | None:
        return self._processed

    def _on_dataset_changed(self, index: int) -> None:
        dataset = self.current_dataset
        self.list_fields.clear()
        self.combo_derived_base.clear()
        if dataset is None:
            self.btn_process.setEnabled(False)
            self.label_status.setText("请先在「数据导入」页导入行情数据。")
            return

        fields = numeric_fields(dataset)
        for name in fields:
            item = QListWidgetItem(field_label(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_fields.addItem(item)
        self.combo_derived_base.addItems([field_label(name) for name in fields])
        if "close" in fields:
            self.combo_derived_base.setCurrentIndex(fields.index("close"))

        if dataset.start_date:
            self.date_start.setDate(QDate.fromString(dataset.start_date.isoformat(), "yyyy-MM-dd"))
        if dataset.end_date:
            self.date_end.setDate(QDate.fromString(dataset.end_date.isoformat(), "yyyy-MM-dd"))
        self.btn_process.setEnabled(bool(fields))
        self.label_status.setText(
            f"共 {dataset.record_count} 条记录，"
            f"{dataset.start_date:%Y-%m-%d} ~ {dataset.end_date:%Y-%m-%d}"
        )

    def _check_all(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.list_fields.count()):
            self.list_fields.item(row).setCheckState(state)

    def _on_derived_method_changed(self) -> None:
        enabled = self.combo_derived_method.currentData() != _NO_DERIVED
        self.combo_derived_base.setEnabled(enabled)
        if enabled:
            self._ensure_derived_base_checked()

    def _ensure_derived_base_checked(self) -> None:
        """二级数据的基字段必须处于勾选状态，否则自动勾上。"""
        name = self._derived_base_field()
        for row in range(self.list_fields.count()):
            item = self.list_fields.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == name:
                item.setCheckState(Qt.CheckState.Checked)

    def _on_normalization_changed(self) -> None:
        enabled = self.combo_normalization.currentData() == Normalization.MIN_MAX.value
        self.spin_low.setEnabled(enabled)
        self.spin_high.setEnabled(enabled)

    # ------------------------------------------------------------------
    # 处理
    # ------------------------------------------------------------------
    def _selected_fields(self) -> tuple[str, ...]:
        names: list[str] = []
        for row in range(self.list_fields.count()):
            item = self.list_fields.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return tuple(names)

    def _derived_base_field(self) -> str | None:
        index = self.combo_derived_base.currentIndex()
        fields = numeric_fields(self.current_dataset) if self.current_dataset else ()
        if 0 <= index < len(fields):
            return fields[index]
        return None

    def build_config(self) -> ProcessingConfig:
        """把界面上的选择转换成 :class:`ProcessingConfig`。"""
        derived: tuple[DerivedField, ...] = ()
        method_value = self.combo_derived_method.currentData()
        base_field = self._derived_base_field()
        if method_value != _NO_DERIVED and base_field:
            derived = (DerivedField(base_field=base_field, method=method_value),)

        normalization = Normalization(self.combo_normalization.currentData())
        min_max_low = float(self.spin_low.value())
        min_max_high = float(self.spin_high.value())
        alignment = Alignment(self.combo_alignment.currentData())

        data_kind = (
            DataKind.RELATIVE
            if normalization is not Normalization.NONE or derived
            else DataKind.ABSOLUTE
        )

        start = self.date_start.date().toPython()
        end = self.date_end.date().toPython()
        return ProcessingConfig(
            fields=self._selected_fields(),
            start_date=start,
            end_date=end,
            derived=derived,
            normalization=normalization,
            alignment=alignment,
            data_kind=data_kind,
            min_max_low=min_max_low,
            min_max_high=min_max_high,
        )

    def process_current(self) -> ProcessedDataset | None:
        dataset = self.current_dataset
        if dataset is None:
            QMessageBox.information(self, "尚未选择数据", "请先在「数据导入」页导入行情数据。")
            return None
        try:
            config = self.build_config()
            result = self._processor.process(dataset, config)  # type: ignore[attr-defined]
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "无法处理数据", exc.user_message())
            self.label_status.setText(exc.user_message())
            return None
        self._show_result(result)
        self.processed_changed.emit()
        return result

    def _show_result(self, result: ProcessedDataset) -> None:
        self._processed = result
        self.btn_export.setEnabled(True)
        labels = {name: result.column_label(name) for name in result.value_columns}
        self._model.set_data(result.frame, labels)
        self.table_result.resizeColumnsToContents()

        config = result.config
        header = (
            f"处理结果 —— {result.name}　共 {len(result)} 条记录　"
            f"{DATA_KIND_LABELS[result.data_kind.value]}　"
            f"{ALIGNMENT_LABELS[result.alignment.value]}"
        )
        self.label_result.setText(header)
        self.table_result.horizontalHeader().setStretchLastSection(True)
        self.label_status.setText(f"已生成处理数据：{len(result)} 条记录，{len(result.value_columns)} 个数据列。")

        notes = result.provenance.get("notes") or ()
        labels = [result.column_label(name) for name in result.value_columns]
        lines = [
            f"来源：{result.name}",
            f"原始记录：{result.provenance.get('source_range', '未知')}"
            f"（{result.provenance.get('source_rows', '?')} 条）",
            f"结果记录：{result.provenance.get('applied_range', '未知')}",
            f"字段：{'、'.join(labels)}",
            f"归一化：{NORMALIZATION_LABELS[config.normalization.value]}",
            f"对齐：{ALIGNMENT_LABELS[config.alignment.value]}",
        ]
        if config.derived:
            lines.append("二级数据：" + "、".join(item.label for item in config.derived))
        normalization_info = result.provenance.get("normalization") or {}
        for name, params in normalization_info.items():
            label = result.column_label(name)
            base = params.get("base_value")
            if base is not None:
                lines.append(f"　{label} 的归一化基准值：{base:.6g}")
            if params.get("data_min") is not None:
                lines.append(
                    f"　{label} 的原始区间：[{params['data_min']:.6g}, {params['data_max']:.6g}]"
                )
        if notes:
            lines.append("")
            lines.append("说明：")
            lines.extend(f"· {note}" for note in notes)
        self.text_notes.setPlainText("\n".join(lines))

    def clear(self) -> None:
        self._processed = None
        self._model.set_data(None)
        self.label_result.setText("处理结果")
        self.text_notes.clear()
        self.btn_export.setEnabled(False)

    # ------------------------------------------------------------------
    # 导出（第六阶段）
    # ------------------------------------------------------------------
    def export_current(self) -> None:
        """把当前处理结果导出为 CSV。"""
        if self._processed is None:
            QMessageBox.information(self, "没有可导出的数据", "请先生成处理数据。")
            return
        exporter = self._exporter
        if exporter is None:
            QMessageBox.warning(self, "无法导出", "结果导出模块尚未接入。")
            return
        name = _safe_filename(self._processed.name) or "处理结果"
        default = str(Path.cwd() / f"{name}_处理数据.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出处理结果", default, "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            saved = exporter.export(self._processed, path)  # type: ignore[attr-defined]
        except StockAnalyzerError as exc:
            QMessageBox.warning(self, "导出失败", exc.user_message())
            return
        self.label_status.setText(f"已导出处理数据：{saved}")
