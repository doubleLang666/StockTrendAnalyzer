"""服务注册表：UI 通过它获取各业务模块，而不是直接 import 实现类。"""

from __future__ import annotations

from typing import Any, Callable

from core.exceptions import ServiceNotRegisteredError

__all__ = ["ServiceRegistry", "SERVICE_NAMES"]

#: 约定的服务名称，与 core.interfaces 中的协议一一对应。
SERVICE_NAMES: tuple[str, ...] = (
    "importer",
    "processor",
    "trend",
    "similarity",
    "exporter",
)

SERVICE_LABELS: dict[str, str] = {
    "importer": "数据导入",
    "processor": "数据处理",
    "trend": "趋势分析",
    "similarity": "相似度分析",
    "exporter": "结果导出",
}


class ServiceRegistry:
    """轻量服务容器（不含反射、不做自动装配）。"""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._factories: dict[str, Callable[["ServiceRegistry"], Any]] = {}

    # ------------------------------------------------------------------
    def register(self, name: str, service: Any, *, replace: bool = False) -> None:
        if name in self._providers and not replace:
            raise ValueError(f"服务「{name}」已注册")
        self._providers[name] = service

    def register_factory(
        self,
        name: str,
        factory: Callable[["ServiceRegistry"], Any],
        *,
        replace: bool = False,
    ) -> None:
        """延迟创建：首次 :meth:`get` 时实例化。"""
        if name in self._factories and not replace:
            raise ValueError(f"服务工厂「{name}」已注册")
        self._factories[name] = factory

    # ------------------------------------------------------------------
    def get(self, name: str) -> Any:
        if name not in self._providers and name in self._factories:
            self._providers[name] = self._factories[name](self)
        if name not in self._providers:
            label = SERVICE_LABELS.get(name, name)
            raise ServiceNotRegisteredError(
                f"{label}模块尚未接入",
                detail=f"服务名：{name}",
            )
        return self._providers[name]

    def try_get(self, name: str) -> Any | None:
        try:
            return self.get(name)
        except ServiceNotRegisteredError:
            return None

    def is_available(self, name: str) -> bool:
        return name in self._providers or name in self._factories

    def available(self) -> tuple[str, ...]:
        return tuple(
            name for name in SERVICE_NAMES if name in self._providers or name in self._factories
        )

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in SERVICE_NAMES if not self.is_available(name))

    def clear(self) -> None:
        self._providers.clear()
        self._factories.clear()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.is_available(name)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"ServiceRegistry(available={self.available()})"
