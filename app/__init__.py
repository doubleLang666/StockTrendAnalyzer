"""应用协调层：装配 QApplication、服务注册表与主窗口。"""

from __future__ import annotations

from app.config import AppConfig, DEFAULT_CONFIG, PROJECT_ROOT
from app.service_registry import SERVICE_LABELS, SERVICE_NAMES, ServiceRegistry

__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    "ServiceRegistry",
    "SERVICE_NAMES",
    "SERVICE_LABELS",
]
