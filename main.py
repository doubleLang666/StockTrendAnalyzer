"""程序入口。

用法（源码）：
    .venv\\Scripts\\python.exe StockTrendAnalyzer\\main.py

打包后由 PyInstaller 直接调用 ``main()``，路径处理见 ``app/paths.py``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 开发环境下把源码根加入 sys.path；打包后模块已随 exe 分发，无需处理。
_SOURCE_ROOT = Path(__file__).resolve().parent
if not getattr(sys, "frozen", False) and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


def main(argv: list[str] | None = None) -> int:
    from app.application import run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
