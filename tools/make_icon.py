"""生成应用图标 ``resources/app.ico``（可重复执行）。

用法：
    python tools/make_icon.py [输出路径]

默认输出到系统临时目录，便于在只读工作区环境下先生成再复制。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = (16, 24, 32, 48, 64, 128, 256)
BACKGROUND = (23, 55, 94, 255)
GRID = (58, 96, 140, 255)
LINE = (255, 176, 59, 255)
ACCENT = (255, 255, 255, 255)


def _draw(size: int) -> Image.Image:
    image = Image.new("RGBA", (size * 4, size * 4), BACKGROUND)
    draw = ImageDraw.Draw(image)
    scale = size * 4

    # 网格
    step = scale / 4
    for index in range(1, 4):
        position = round(index * step)
        width = max(1, scale // 96)
        draw.line([(position, 0), (position, scale)], fill=GRID, width=width)
        draw.line([(0, position), (scale, position)], fill=GRID, width=width)

    # 上升折线（业绩趋势意象）
    points = [(0.12, 0.76), (0.32, 0.56), (0.50, 0.64), (0.72, 0.34), (0.90, 0.20)]
    coords = [(round(x * scale), round(y * scale)) for x, y in points]
    draw.line(coords, fill=LINE, width=max(2, scale // 14), joint="curve")

    # 顶点圆点
    radius = max(2, scale // 20)
    cx, cy = coords[-1]
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=ACCENT)

    return image.resize((size, size), Image.LANCZOS)


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else Path(tempfile.gettempdir()) / "app.ico"
    target.parent.mkdir(parents=True, exist_ok=True)
    images = [_draw(size) for size in SIZES]
    images[-1].save(target, format="ICO", sizes=[(size, size) for size in SIZES])
    png = target.with_suffix(".png")
    images[-1].save(png, format="PNG")
    print(f"已生成图标：{target}")
    print(f"已生成预览：{png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
