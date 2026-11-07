"""应用装配层：创建 QApplication、装配服务、打开主窗口。

UI 层不得越过本层直接 new 业务模块；业务模块也不得 import 本模块。
"""

from __future__ import annotations

import sys
import traceback
from typing import Sequence

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from app.config import AppConfig
from app.service_registry import ServiceRegistry
from core.constants import APP_NAME, APP_VERSION
from core.exceptions import StockAnalyzerError

__all__ = ["configure_matplotlib", "build_registry", "create_application", "run"]


def configure_matplotlib(config: AppConfig) -> None:
    """统一 Matplotlib 中文字体与线宽，避免图中出现方块字。"""
    from app import paths

    paths.set_cache_environment()
    import matplotlib

    families = config.get("plotting.font_families", ["Microsoft YaHei", "SimHei"])
    rc = matplotlib.rcParams
    rc["font.sans-serif"] = list(families)
    rc["font.family"] = "sans-serif"
    rc["axes.unicode_minus"] = False
    rc["figure.dpi"] = float(config.get("plotting.figure_dpi", 100))
    rc["lines.linewidth"] = float(config.get("plotting.line_width", 1.6))
    rc["axes.grid"] = True
    rc["grid.alpha"] = 0.25
    rc["figure.autolayout"] = False


def build_registry(config: AppConfig) -> ServiceRegistry:
    """装配业务服务。

    已接入：
        importer   -> modules.data_import.importer.TxtImporter
        processor  -> modules.data_processing.processor.DataProcessor
        trend      -> modules.trend_analysis.trend_service.TrendService
        similarity -> modules.similarity.service.SimilarityService
        exporter   -> modules.export.csv_exporter.CsvExporter
    """
    from modules.data_import.importer import TxtImporter
    from modules.data_processing.processor import DataProcessor
    from modules.export.csv_exporter import CsvExporter
    from modules.similarity.service import SimilarityService
    from modules.trend_analysis.trend_service import TrendService

    registry = ServiceRegistry()
    registry.register_factory("importer", lambda _registry: TxtImporter(config))
    registry.register_factory("processor", lambda _registry: DataProcessor(config))
    registry.register_factory("similarity", lambda _registry: SimilarityService(config))
    registry.register_factory("exporter", lambda _registry: CsvExporter())
    registry.register_factory(
        "trend",
        lambda registry: TrendService(config, lambda: registry.try_get("similarity")),
    )
    return registry


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """创建或复用 QApplication，并统一字体与元信息。"""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        app = existing
    else:
        app = QApplication(list(argv) if argv is not None else sys.argv)

    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName("毕业设计")
    app.setApplicationVersion(APP_VERSION)
    app.setFont(QFont("Microsoft YaHei UI", 10))

    from app import paths

    icon = paths.resource_path("app.ico")
    if icon.exists():
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(icon)))
    return app


def _install_exception_hook(silent: bool = False) -> None:
    """把未捕获异常转成用户可读的对话框，而不是让程序静默退出。

    ``silent=True``（自检模式）不弹窗，直接把错误写到 stderr 并以非零码退出，
    避免无界面环境下被模态对话框卡住。
    """

    def hook(exc_type, exc_value, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        traceback.print_exception(exc_type, exc_value, exc_tb)
        if silent:
            import os

            os._exit(2)
        if issubclass(exc_type, StockAnalyzerError):
            text = exc_value.user_message()
        else:
            text = f"{exc_type.__name__}: {exc_value}"
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("程序异常")
        box.setText(text)
        box.setDetailedText("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        box.exec()

    sys.excepthook = hook


def run(argv: Sequence[str] | None = None) -> int:
    """程序主入口，返回进程退出码。

    支持 ``--self-test``：以无界面方式构建主窗口后自动退出，
    用于打包后快速验证依赖是否齐全（退出码 0 表示正常）。
    """
    from app import paths

    arguments = list(argv) if argv is not None else sys.argv
    self_test = "--self-test" in arguments

    # 顺序很重要：必须先设置缓存目录，再 import Matplotlib / 界面层，
    # 否则 Matplotlib 会在导入时就把只读的临时目录当成配置目录。
    paths.ensure_runtime_dirs()
    paths.set_cache_environment()
    config = AppConfig.load()
    configure_matplotlib(config)

    from ui.main_window import MainWindow

    app = create_application(arguments)
    _install_exception_hook(silent=self_test)

    registry = build_registry(config)
    window = MainWindow(config, registry)
    window.show()
    if self_test:
        from PySide6.QtCore import QTimer

        _write_self_test_report(window, config)
        QTimer.singleShot(150, app.quit)
    return int(app.exec())


def _write_self_test_report(window, config: AppConfig) -> None:  # type: ignore[no-untyped-def]
    """把自检结果写到程序目录，便于打包后在无控制台环境里核对。"""
    import json

    from app import paths

    report: dict[str, object] = {
        "window_title": window.windowTitle(),
        "tabs": [window.tabs.tabText(index) for index in range(window.tabs.count())],
        "application_dir": str(paths.application_dir()),
        "data_dir": str(paths.data_dir()),
        "config_path": str(config.path) if config.path else None,
        "icon_present": paths.resource_path("app.ico").exists(),
    }

    import_page = getattr(window, "_import_page", None)
    if import_page is not None:
        sources = sorted(paths.data_dir().glob("*.txt"))
        if sources:
            import_page.import_paths(sources)
        results = import_page.results
        report["imported_files"] = [str(result.report.source_path) for result in results]
        report["imported_rows"] = [result.report.imported_rows for result in results]
        report["datasets"] = [result.dataset.name for result in results]

    try:
        target = paths.application_dir() / "self_test_report.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _say(f"自检报告已写入：{target}")
    except OSError:  # pragma: no cover - 只读目录
        pass
    _say(json.dumps(report, ensure_ascii=False, default=str))


def _say(message: str) -> None:
    """在无控制台的窗体程序里也能安全输出的打印。"""
    stream = sys.stdout
    if stream is None:  # pragma: no cover - pythonw / 窗体模式
        return
    try:
        print(message, flush=True)
    except (OSError, ValueError):  # pragma: no cover
        pass
