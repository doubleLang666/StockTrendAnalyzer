# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（单目录模式）。

用法（在项目根目录 F:\\毕业设计\\StockTrendAnalyzer 下执行）：
    ..\\.venv\\Scripts\\python.exe -m PyInstaller StockTrendAnalyzer.spec --noconfirm

输出：dist\\StockTrendAnalyzer\\
    StockTrendAnalyzer.exe   主程序
    _internal\\               依赖库（PySide6 / Matplotlib / pandas …）
    config.json              默认配置（首次运行复制到 exe 旁可写位置）
    resources\\app.ico        窗口与任务栏图标

关于路径：程序运行时不再依赖本文件的任何绝对路径，
打包结果可以整体拷贝到任意目录运行（见 app/paths.py）。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path(SPECPATH)  # noqa: F821 - PyInstaller 注入
RESOURCES = PROJECT_ROOT / "resources"
CONFIG = PROJECT_ROOT / "config.json"
ICON = RESOURCES / "app.ico"
VERSION_FILE = PROJECT_ROOT / "packaging" / "version_info.txt"

# 随程序分发的只读资源。config.json 打进包内作为“出厂默认配置”，
# 首次运行时由 app.paths.ensure_config_file() 复制到 exe 目录。
datas = [
    (str(RESOURCES), "resources"),
    (str(CONFIG), "."),
]

# 显式声明动态导入的模块：
#  * Matplotlib 的后端是按名字延迟导入的，静态分析容易漏掉；
#  * pandas / numpy 的部分可选子模块同样容易漏。
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

# 明确不用的重型依赖，排除后安装包明显变小。
# 注意：不要排除 unittest / doctest —— PyInstaller 打包后的 pyparsing 会
# import unittest（matplotlib 的依赖链 matplotlib → pyparsing → testing），
# 排除后会以 ModuleNotFoundError: No module named 'unittest' 启动失败。
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

exe = EXE(  # noqa: F821
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="StockTrendAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # 窗体程序，不弹黑框
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
    version=str(VERSION_FILE) if VERSION_FILE.exists() else None,
)

collect = COLLECT(  # noqa: F821
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StockTrendAnalyzer",
)
