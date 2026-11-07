"""文件编码识别。

设计文档 3.1.3 要求进行编码识别或让用户手动选择编码。这里不引入第三方
编码探测库，而是按候选编码逐个尝试解码，再用"解码是否成功 + 中文字符占比"
打分，选出最可能的一种。券商导出的 TXT 多为 GBK/GB18030 或 UTF-8(BOM)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from core.constants import ENCODING_CANDIDATES
from core.exceptions import EncodingDetectionError

__all__ = ["EncodingCandidate", "detect_encoding", "decode_bytes", "ENCODING_LABELS"]

#: 供界面展示的编码名称（Python 编码名 -> 常见叫法）。
ENCODING_LABELS: dict[str, str] = {
    "utf-8-sig": "UTF-8 (含 BOM)",
    "utf-8": "UTF-8",
    "gb18030": "GB18030",
    "gbk": "GBK",
    "big5": "BIG5",
}

#: 常见中文字符范围（基本区、扩展 A、中文标点、全角符号），用于给候选编码打分。
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")

#: 能无损解码即可直接认定的 UTF-8 族编码。
_UTF8_FAMILY = frozenset({"utf-8", "utf-8-sig"})


@dataclass(frozen=True, slots=True)
class EncodingCandidate:
    """一种候选编码及其得分。"""

    encoding: str
    score: float
    decoded: str

    @property
    def label(self) -> str:
        return ENCODING_LABELS.get(self.encoding, self.encoding)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_PATTERN.findall(text)) / len(text)


def _score(decoded: str) -> float:
    """解码结果打分：出现替换字符直接判负，中文字符占比越高越可信。"""
    if "\ufffd" in decoded:
        return -1.0
    return _cjk_ratio(decoded)


def decode_bytes(raw: bytes, encoding: str) -> str:
    """按指定编码解码，失败时抛出 :class:`EncodingDetectionError`。"""
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise EncodingDetectionError(
            f"无法按 {ENCODING_LABELS.get(encoding, encoding)} 解码文件",
            detail=str(exc),
        ) from exc


def detect_encoding(
    raw: bytes,
    candidates: Sequence[str] | None = None,
) -> tuple[str, str]:
    """探测编码，返回 ``(编码名, 解码后的文本)``。

    判定顺序：

    1. UTF-8 族（含 BOM）：严格解码成功即可认定——非 UTF-8 文本几乎不可能
       同时是合法的 UTF-8 字节序列；
    2. 其余候选（GB18030 / GBK / BIG5 等）按"中文字符占比"打分取最高。

    全部失败时抛出 :class:`EncodingDetectionError`。
    """
    if not raw:
        raise EncodingDetectionError("文件内容为空，无法识别编码")

    ordered: tuple[str, ...] = tuple(candidates or ENCODING_CANDIDATES)

    # 第 1 步：UTF-8 是强特征，能无损解码就直接采用。
    for encoding in ordered:
        if encoding not in _UTF8_FAMILY:
            continue
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return encoding, text

    # 第 2 步：其余编码按中文占比打分（同分取候选顺序靠前者）。
    best: EncodingCandidate | None = None
    for encoding in ordered:
        if encoding in _UTF8_FAMILY:
            continue
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        score = _score(text)
        if score < 0:
            continue
        if best is None or score > best.score + 1e-9:
            best = EncodingCandidate(encoding=encoding, score=score, decoded=text)

    if best is None:
        raise EncodingDetectionError(
            "无法识别文件编码",
            detail="已尝试的编码：" + "、".join(ENCODING_LABELS.get(item, item) for item in ordered),
        )

    return best.encoding, best.decoded
