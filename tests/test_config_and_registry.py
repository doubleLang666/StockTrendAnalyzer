"""app.config 与 app.service_registry 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import AppConfig, DEFAULT_CONFIG
from app.service_registry import ServiceRegistry
from core.exceptions import ConfigurationError, ServiceNotRegisteredError


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------
def test_config_uses_defaults_when_file_missing(tmp_path: Path) -> None:
    config = AppConfig.load(tmp_path / "config.json")
    assert config.get("app.window.width") == 1280
    assert config.get("similarity.top_n") == 3
    assert config.get("不存在的键", "兜底") == "兜底"
    assert any("内置默认配置" in item for item in config.warnings)


def test_config_deep_merges_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"similarity": {"top_n": 5}, "app": {"window": {"width": 1000}}}),
        encoding="utf-8",
    )
    config = AppConfig.load(path)
    assert config.get("similarity.top_n") == 5
    assert config.get("similarity.algorithm") == "pearson"  # 默认值保留
    assert config.get("app.window.width") == 1000
    assert config.get("app.window.height") == 820
    assert config.warnings == ()


def test_config_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{ 不是合法 json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="无法读取"):
        AppConfig.load(path)


def test_config_set_and_save_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig.load(path)
    config.set("app.window.width", 1440)
    config.set("新增.深层.键", True)
    config.save()

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["app"]["window"]["width"] == 1440
    assert reloaded["新增"]["深层"]["键"] is True
    assert reloaded["similarity"]["top_n"] == DEFAULT_CONFIG["similarity"]["top_n"]


def test_config_section_is_copy(tmp_path: Path) -> None:
    config = AppConfig.load(tmp_path / "config.json")
    section = config.section("similarity")
    section["top_n"] = 99
    assert config.get("similarity.top_n") == 3


# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------
class _DummyService:
    def __init__(self) -> None:
        self.calls = 0


def test_registry_get_unregistered_raises() -> None:
    registry = ServiceRegistry()
    with pytest.raises(ServiceNotRegisteredError, match="数据导入模块尚未接入"):
        registry.get("importer")
    assert registry.try_get("importer") is None
    assert registry.is_available("importer") is False


def test_registry_register_and_lookup() -> None:
    registry = ServiceRegistry()
    service = _DummyService()
    registry.register("importer", service)
    assert registry.get("importer") is service
    assert "importer" in registry
    assert registry.available() == ("importer",)
    assert "processor" in registry.missing()

    with pytest.raises(ValueError, match="已注册"):
        registry.register("importer", _DummyService())


def test_registry_lazy_factory_creates_once() -> None:
    registry = ServiceRegistry()
    created: list[_DummyService] = []

    def factory(_registry: ServiceRegistry) -> _DummyService:
        service = _DummyService()
        created.append(service)
        return service

    registry.register_factory("similarity", factory)
    first = registry.get("similarity")
    second = registry.get("similarity")
    assert first is second
    assert len(created) == 1


def test_registry_clear() -> None:
    registry = ServiceRegistry()
    registry.register("processor", _DummyService())
    registry.clear()
    assert registry.available() == ()
