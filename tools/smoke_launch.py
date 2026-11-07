"""启动冒烟检查：无显示环境下构建主窗口，确认能正常启动并退出（开发期工具）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QTimer  # noqa: E402

from app.application import build_registry, create_application, run  # noqa: E402
from app.config import AppConfig  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    qapp = create_application(sys.argv)
    config = AppConfig.load()
    window = MainWindow(config, build_registry(config))
    window.show()
    print(f"启动成功：窗口标题={window.windowTitle()!r}，页签数={window.tabs.count()}")
    QTimer.singleShot(200, qapp.quit)
    qapp.exec()
    print("退出成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
