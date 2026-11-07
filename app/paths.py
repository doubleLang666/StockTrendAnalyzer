"""统一的路径管理。

开发环境与 PyInstaller 打包后的运行环境差异集中在这里处理：

* ``application_dir()``：程序"自己的工作目录"。
  开发时是源码根目录，打包后是 exe 所在目录（可写，用来放 config.json、data/）。
* ``bundle_dir()``：随程序分发的只读资源目录。
  开发时同样是源码根，打包后是 PyInstaller 解包出的临时目录（``sys._MEIPASS``）。
* ``data_dir()``：用户行情数据默认目录（exe/源码旁的 ``data``）。
* ``cache_dir()``：可写的缓存目录（Matplotlib 字体缓存等）。

这样程序里不再出现任何本机绝对路径，也不依赖当前工作目录。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = [
    "application_dir",
    "bundle_dir",
    "cache_dir",
    "config_path",
    "data_dir",
    "ensure_config_file",
    "ensure_runtime_dirs",
    "export_dir",
    "is_frozen",
    "resource_path",
    "resources_dir",
    "set_cache_environment",
]


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包出的可执行文件中。"""
    return bool(getattr(sys, "frozen", False))


def application_dir() -> Path:
    """程序目录：exe 所在目录（打包后）或源码根目录（开发时）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def bundle_dir() -> Path:
    """只读资源根目录：``sys._MEIPASS``（打包后）或源码根目录（开发时）。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[1]


def resources_dir() -> Path:
    return bundle_dir() / "resources"


def resource_path(name: str, *, required: bool = False) -> Path:
    """返回 ``resources/<name>``；``required=True`` 时找不到抛 FileNotFoundError。"""
    candidate = resources_dir() / name
    if required and not candidate.exists():
        raise FileNotFoundError(f"缺少资源文件：{candidate.name}")
    return candidate


def data_dir() -> Path:
    """用户数据目录，位于程序目录下的 ``data``。"""
    return application_dir() / "data"


def export_dir() -> Path:
    """导出结果的默认目录，位于程序目录下的 ``exports``。"""
    return application_dir() / "exports"


def cache_dir() -> Path:
    """可写缓存目录，位于程序目录下的 ``.cache``。"""
    return application_dir() / ".cache"


def config_path() -> Path:
    """可写配置文件路径：exe/源码旁的 ``config.json``。"""
    return application_dir() / "config.json"


def _bundled_config_path() -> Path:
    return bundle_dir() / "config.json"


def ensure_runtime_dirs() -> tuple[Path, ...]:
    """确保 data / exports / cache 目录存在，返回实际创建的目录。

    目录不可写时（例如装在只读位置）静默跳过，由调用方使用默认值兜底。
    """
    created: list[Path] = []
    for target in (data_dir(), export_dir(), cache_dir()):
        if target.exists():
            continue
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - 只读目录
            continue
        created.append(target)
    return tuple(created)


def ensure_config_file() -> Path:
    """首次运行时把随程序分发的默认 config.json 复制到可写位置。

    开发环境下若源码根已有 config.json，则直接返回它；打包后 exe 旁的
    config.json 不存在时，从 ``sys._MEIPASS`` 复制一份，实现"开箱即用"。
    """
    target = config_path()
    if target.exists():
        return target
    bundled = _bundled_config_path()
    if bundled.exists() and bundled.resolve() != target.resolve():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundled, target)
        except OSError:  # pragma: no cover - 目录只读时退化为使用默认配置
            return target
    return target


def set_cache_environment() -> None:
    """把 Matplotlib 等第三方库的缓存目录指向可写位置。

    PyInstaller 单文件/单目录模式下 ``_MEIPASS`` 可能只读，Matplotlib 首次
    运行会尝试写字体缓存，不重定向会在打包后报错。
    """
    matplotlib_cache = cache_dir() / "matplotlib"
    # 直接覆盖：PyInstaller 的钩子可能已经设了一个只读的临时目录。
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    os.environ.setdefault("MPLBACKEND", "QtAgg")
