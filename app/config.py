"""应用级配置：加载/保存 ``config.json``。

配置与业务逻辑分离（设计文档 8.4）。所有键都有内置默认值，
文件缺失或字段不全时使用默认值并在 ``warnings`` 中记录。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.constants import APP_NAME, APP_VERSION
from core.exceptions import ConfigurationError
from app import paths

__all__ = ["AppConfig", "DEFAULT_CONFIG", "PROJECT_ROOT"]

# 开发时是源码根目录，打包后是 exe 所在目录（见 app/paths.py）
PROJECT_ROOT = paths.application_dir()

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": APP_NAME,
        "version": APP_VERSION,
        "window": {
            "width": 1280,
            "height": 820,
            "remember_geometry": True,
        },
    },
    "import": {
        "encodings": ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5"],
        "delimiters": [",", "\t", ";", "|"],
        "date_formats": [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y%m%d",
            "%Y.%m.%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ],
        "missing_tokens": ["", "-", "--", "---", "nan", "none", "null", "n/a", "na"],
        "header_scan_rows": 10,
        "restore_recent": True,
        "recent_files": [],
    },
    "processing": {
        "default_normalization": "none",
        "default_alignment": "real_date",
        "default_data_kind": "absolute",
        "min_max_range": [0.0, 1.0],
    },
    "similarity": {
        "algorithm": "pearson",
        "normalize": True,
        "top_n": 3,
        "min_valid_points": 5,
        "max_overlap_ratio": 0.5,
    },
    "plotting": {
        "figure_dpi": 100,
        "line_width": 1.6,
        "max_curves": 8,
        "font_families": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class AppConfig:
    """只读配置视图 + 少量可写项（如窗口尺寸）。"""

    def __init__(self, data: dict[str, Any], path: Path | None = None, warnings: list[str] | None = None) -> None:
        self._data = data
        self._path = path
        self._warnings = list(warnings or [])

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        if path is not None:
            target = Path(path)
        else:
            paths.ensure_runtime_dirs()
            target = paths.ensure_config_file()
        warnings: list[str] = []
        raw: dict[str, Any] = {}
        if target.exists():
            try:
                text = target.read_text(encoding="utf-8-sig")
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("根节点必须是 JSON 对象")
                raw = parsed
            except (OSError, ValueError) as exc:
                raise ConfigurationError(f"配置文件无法读取：{target.name}", detail=str(exc)) from exc
        else:
            warnings.append(f"未找到 {target.name}，使用内置默认配置")
        return cls(_deep_merge(DEFAULT_CONFIG, raw), target, warnings)

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return deepcopy(node)

    def section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return deepcopy(value) if isinstance(value, dict) else {}

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    # ------------------------------------------------------------------
    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._data, ensure_ascii=False, indent=2)
        self._path.write_text(text + "\n", encoding="utf-8")
