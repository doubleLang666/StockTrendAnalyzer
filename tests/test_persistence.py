"""数据与会话持久化测试：关闭再打开后数据能否恢复。"""

from __future__ import annotations

import json
from pathlib import Path

from app.application import build_registry
from app.config import AppConfig
from core.constants import APP_NAME
from ui.main_window import MainWindow

_SAMPLE = (
    "                  测试指数 (000001)\n"
    "\n"
    "      时间\t开盘\t最高\t最低\t收盘\t成交量\n"
    "2024/01/02\t9.80\t10.20\t9.70\t10.00\t900\n"
    "2024/01/03\t10.00\t11.00\t9.50\t10.50\t1,000\n"
)


def _write_sample(directory: Path, name: str = "sample.txt") -> Path:
    path = directory / name
    path.write_bytes(_SAMPLE.encode("gbk"))
    return path


def test_imported_file_paths_are_remembered(qapp, tmp_path: Path) -> None:
    data_file = _write_sample(tmp_path)
    config = AppConfig.load(tmp_path / "config.json")
    window = MainWindow(config, build_registry(config))

    window._import_page.import_paths([data_file])
    window.close()

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["import"]["recent_files"] == [str(data_file)]
    assert saved["app"]["window"]["width"] >= 960


def test_recent_files_are_restored_on_restart(qapp, tmp_path: Path) -> None:
    data_file = _write_sample(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"import": {"recent_files": [str(data_file)]}}), encoding="utf-8"
    )

    window = MainWindow(AppConfig.load(config_path), build_registry(AppConfig.load(config_path)))

    assert len(window._import_page.results) == 1
    assert window._import_page.results[0].dataset.name == "测试指数"
    assert window._processing_page.datasets  # 下游页面同步到数据
    window.close()


def test_missing_recent_files_are_skipped(qapp, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"import": {"recent_files": [str(tmp_path / "不存在.txt")]}}), encoding="utf-8"
    )

    window = MainWindow(AppConfig.load(config_path), build_registry(AppConfig.load(config_path)))

    assert window._import_page.results == ()
    assert window.windowTitle().startswith(APP_NAME)
    window.close()


def test_restore_can_be_disabled_in_config(qapp, tmp_path: Path) -> None:
    data_file = _write_sample(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {"import": {"restore_recent": False, "recent_files": [str(data_file)]}}
        ),
        encoding="utf-8",
    )

    window = MainWindow(AppConfig.load(config_path), build_registry(AppConfig.load(config_path)))

    assert window._import_page.results == ()
    window.close()
