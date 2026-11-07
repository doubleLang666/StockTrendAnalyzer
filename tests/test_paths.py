"""app.paths 的单元测试：开发环境与打包环境的路径解析。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app import paths
from app.config import AppConfig


# ---------------------------------------------------------------------------
# 打包（frozen）环境模拟
# ---------------------------------------------------------------------------
def test_application_dir_follows_executable_when_frozen(monkeypatch, tmp_path: Path) -> None:
    exe = tmp_path / "程序" / "StockTrendAnalyzer.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert paths.is_frozen() is True
    assert paths.application_dir() == exe.parent
    assert paths.config_path() == exe.parent / "config.json"
    assert paths.data_dir() == exe.parent / "data"


def test_bundle_dir_uses_meipass_when_present(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "resources").mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    assert paths.bundle_dir() == bundle
    assert paths.resources_dir() == bundle / "resources"
    assert paths.resource_path("app.ico") == bundle / "resources" / "app.ico"


def test_resource_path_required_flag(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "resources").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    with pytest.raises(FileNotFoundError, match="app.ico"):
        paths.resource_path("app.ico", required=True)


def test_development_paths_point_to_source_root() -> None:
    root = Path(__file__).resolve().parents[1]
    assert paths.is_frozen() is False
    assert paths.application_dir() == root
    assert paths.bundle_dir() == root
    assert paths.resources_dir() == root / "resources"
    assert paths.data_dir() == root / "data"


# ---------------------------------------------------------------------------
# 目录创建与配置落地
# ---------------------------------------------------------------------------
def test_ensure_runtime_dirs_creates_missing_directories(monkeypatch, tmp_path: Path) -> None:
    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    monkeypatch.setattr(paths, "application_dir", lambda: exe_dir)

    created = paths.ensure_runtime_dirs()

    assert set(created) == {exe_dir / "data", exe_dir / "exports", exe_dir / ".cache"}
    assert (exe_dir / "data").is_dir()
    assert (exe_dir / ".cache").is_dir()
    assert paths.ensure_runtime_dirs() == ()  # 第二次调用不再重复创建


def test_ensure_config_file_copies_bundled_default(monkeypatch, tmp_path: Path) -> None:
    exe_dir = tmp_path / "app"
    bundle = tmp_path / "bundle"
    exe_dir.mkdir()
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"similarity": {"top_n": 7}}), encoding="utf-8")
    monkeypatch.setattr(paths, "application_dir", lambda: exe_dir)
    monkeypatch.setattr(paths, "bundle_dir", lambda: bundle)

    target = paths.ensure_config_file()

    assert target == exe_dir / "config.json"
    assert json.loads(target.read_text(encoding="utf-8"))["similarity"]["top_n"] == 7
    # 已存在时不覆盖用户修改
    target.write_text(json.dumps({"similarity": {"top_n": 9}}), encoding="utf-8")
    paths.ensure_config_file()
    assert json.loads(target.read_text(encoding="utf-8"))["similarity"]["top_n"] == 9


def test_config_load_uses_writable_location(monkeypatch, tmp_path: Path) -> None:
    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    monkeypatch.setattr(paths, "application_dir", lambda: exe_dir)
    monkeypatch.setattr(paths, "ensure_config_file", lambda: exe_dir / "config.json")

    config = AppConfig.load()

    assert config.path == exe_dir / "config.json"
    assert any("内置默认配置" in item for item in config.warnings)


def test_set_cache_environment_points_to_writable_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "application_dir", lambda: tmp_path)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    paths.set_cache_environment()

    import os

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / ".cache" / "matplotlib")
