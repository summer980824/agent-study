"""Unicode-aware text preprocessing helpers.

A small pipeline for normalizing raw text into clean paragraphs:

* :func:`normalize_text` — unify Unicode representation and newlines
* :func:`clean_text` — strip control / invisible characters and tidy whitespace
* :func:`split_text` — split into paragraphs, dropping empty ones
* :func:`preprocess_text` — run the whole pipeline, returning paragraphs
"""

from __future__ import annotations

import re
import unicodedata

# C0/C1 control characters minus the ones we keep: \n and \r (\r is
# unified into \n by ``normalize_text`` before cleaning happens), plus
# invisible format characters that pollute embeddings and tokenizers:
# BOM (ZERO WIDTH NO-BREAK SPACE) and ZERO WIDTH SPACE.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff\u200b]")
# Inline whitespace: ASCII space, tab, NO-BREAK SPACE, and the ideographic
# space common in CJK documents.
_INLINE_WHITESPACE = re.compile(r"[ \t\xa0\u3000]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_PARAGRAPH_BOUNDARY = re.compile(r"\n{2,}")


def normalize_text(text: str) -> str:
    """统一 Unicode 表示和换行符。

    Applies NFC so canonically-equivalent strings compare equal, and
    converts ``\\r\\n`` / ``\\r`` line endings to ``\\n``.
    """
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_text(text: str) -> str:
    """清理控制字符、多余空白和冗余空行。

    Removes C0/C1 control characters and invisible format characters
    (BOM, zero-width space), keeping newline; collapses
    runs of inline whitespace (spaces, tabs, no-break spaces, ideographic
    spaces) to a single
    space, collapses runs of blank
    lines to a single blank line, and trims leading/trailing newlines.
    """
    text = _CONTROL_CHARS.sub("", text)
    lines = [
        _INLINE_WHITESPACE.sub(" ", line).strip(" ")
        for line in text.split("\n")
    ]
    text = "\n".join(lines)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip("\n")


def split_text(text: str) -> list[str]:
    """按段落边界拆分文本，过滤空段落。

    Paragraphs are separated by one or more blank lines; paragraphs that
    are empty or whitespace-only are dropped.
    """
    return [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BOUNDARY.split(text)
        if paragraph.strip()
    ]


def preprocess_text(text: str) -> list[str]:
    """串联上述步骤，返回处理后的段落列表。"""
    return split_text(clean_text(normalize_text(text)))
