"""开发环境自检脚本（非产品代码）。

用途：当测试报出 "Permission denied" 之类的写入错误时，运行本脚本确认
当前进程能否在临时目录中创建文件与目录。

    .venv\\Scripts\\python.exe tools\\probe_tmp.py
"""

from __future__ import annotations

import os
import tempfile

ROOT = os.path.join(tempfile.gettempdir(), "pytest-stock-trend")


def main() -> int:
    os.makedirs(ROOT, exist_ok=True)
    print("临时目录:", ROOT)

    ok = True

    try:
        fd, path = tempfile.mkstemp(dir=ROOT, suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(path, encoding="utf-8") as handle:
            handle.read()
        os.remove(path)
        print("文件创建/读取/删除: 正常")
    except OSError as exc:
        ok = False
        print("文件创建/读取/删除: 失败 ->", exc)

    try:
        sub = os.path.join(ROOT, "dir-probe")
        os.makedirs(sub, exist_ok=True)
        os.listdir(sub)
        os.rmdir(sub)
        print("目录创建/列举/删除: 正常")
    except OSError as exc:
        ok = False
        print("目录创建/列举/删除: 失败 ->", exc)

    print("结论:", "可正常写入" if ok else "存在写入限制，请检查沙箱/权限设置")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
