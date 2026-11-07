"""pytest 公共配置与夹具。

注意：必须在导入 PySide6 之前设置 ``QT_QPA_PLATFORM=offscreen``，
否则无显示环境下创建窗口会失败。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

_TEMP_ROOT = Path(tempfile.gettempdir()) / "pytest-stock-trend"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMP_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """整个测试会话共享一个 QApplication。"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """替代 pytest 内置实现。

    内置 tmp_path 用 ``os.mkdir(mode=0o700)`` 创建基础目录，在受限沙箱下会生成
    当前进程无法再次访问的目录；这里改用默认权限，并在用例结束后清理。
    """
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", request.node.name)[:60]
    path = _TEMP_ROOT / f"{safe_name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """12 个交易日的最小行情数据，成交量含缺失值。"""
    index = pd.bdate_range("2024-01-02", periods=12, name="date")
    return pd.DataFrame(
        {
            "open": [10.0 + i * 0.5 for i in range(12)],
            "high": [10.6 + i * 0.5 for i in range(12)],
            "low": [9.8 + i * 0.5 for i in range(12)],
            "close": [10.2 + i * 0.6 for i in range(12)],
            "volume": [1000.0 + i * 25 for i in range(11)] + [float("nan")],
        },
        index=index,
    )
