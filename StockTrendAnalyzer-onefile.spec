# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 单文件打包配置（发送给别人时用这个版本）。

单文件版把所有依赖打进一个 exe，运行时自解压到系统临时目录，
因此**不存在 `_internal` 目录**，也就不会再出现"只解压出 exe、
缺少 `_internal\\base_library.zip`"导致的启动失败。

用法（项目根目录）：
    ..\\.venv\\Scripts\\python.exe -m PyInstaller StockTrendAnalyzer-onefile.spec --noconfirm

输出：dist\\StockTrendAnalyzer.exe（单个文件）
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path(SPECPATH)  # noqa: F821 - PyInstaller 注入
RESOURCES = PROJECT_ROOT / "resources"
CONFIG = PROJECT_ROOT / "config.json"
ICON = RESOURCES / "app.ico"
VERSION_FILE = PROJECT_ROOT / "packaging" / "version_info.txt"

datas = [
    (str(RESOURCES), "resources"),
    (str(CONFIG), "."),
]

hiddenimports = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    "matplotlib.projections.polar",
    "PySide6.QtPrintSupport",
    *collect_submodules("modules"),
    *collect_submodules("core"),
    *collect_submodules("app"),
]

# 与目录版一致：不要排除 unittest（pyparsing 依赖它）。
excludes = [
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "scipy",
    "openpyxl",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "PySide6.Qt3DCore",
    "PySide6.QtBluetooth",
    "PySide6.QtNetworkAuth",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
]

analysis = Analysis(  # noqa: F821
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(analysis.pure)  # noqa: F821

# 单文件模式：binaries / datas 直接交给 EXE，不再生成 COLLECT 目录。
exe = EXE(  # noqa: F821
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="StockTrendAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
    version=str(VERSION_FILE) if VERSION_FILE.exists() else None,
)
