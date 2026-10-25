"""Unit tests for ``agent_study.text_cleaning``.

Each stage of the pipeline is tested on its own — NFC normalization,
newline unification, control-character removal, whitespace collapsing,
redundant blank-line collapsing, paragraph splitting — plus an
end-to-end ``preprocess_text`` pass over a messy input.
"""

from __future__ import annotations

from agent_study.text_cleaning import (
    clean_prompt_text,
    clean_text,
    normalize_text,
    preprocess_text,
    split_text,
)


class TestNormalizeText:
    def test_combining_sequence_becomes_composed(self):
        decomposed = "cafe\u0301"  # e + combining acute
        assert normalize_text(decomposed) == "café"  # precomposed form
        assert normalize_text(decomposed) == normalize_text("café")

    def test_nfc_does_not_fold_full_width_characters(self):
        # Width folding (Ａ -> A) is NFKC's job, not NFC's; NFC must leave
        # full-width characters untouched.
        assert normalize_text("ＡＢＣ") == "ＡＢＣ"

    def test_crlf_becomes_lf(self):
        assert normalize_text("a\r\nb\r\nc") == "a\nb\nc"

    def test_lone_cr_becomes_lf(self):
        assert normalize_text("a\rb\r\nc") == "a\nb\nc"

    def test_normalization_applies_before_newline_unification(self):
        # Combining marks on a CR should not break newline detection.
        assert normalize_text("x\u0301\r\ny") == "x\u0301\ny"


class TestCleanText:
    def test_removes_null_and_bell(self):
        assert clean_text("a\x00b\x07c") == "abc"

    def test_removes_c1_control_characters(self):
        assert clean_text("a\x80b\x9fc") == "abc"

    def test_keeps_newline(self):
        assert clean_text("a\nb\nc") == "a\nb\nc"

    def test_collapses_tabs_to_a_single_space(self):
        assert clean_text("a\t\tb") == "a b"

    def test_collapses_ideographic_space(self):
        assert clean_text("中文　　文本") == "中文 文本"

    def test_removes_vertical_tab_and_form_feed(self):
        assert clean_text("a\x0bb\x0cc") == "abc"

    def test_removes_bom(self):
        assert clean_text("\ufeffhello") == "hello"

    def test_removes_zero_width_space(self):
        assert clean_text("he\u200bllo") == "hello"

    def test_collapses_inline_whitespace(self):
        assert clean_text("a   b\t\t c") == "a b c"

    def test_strips_each_line(self):
        assert clean_text("  hello \n  world  ") == "hello\nworld"

    def test_collapses_redundant_blank_lines(self):
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_trims_leading_and_trailing_newlines(self):
        assert clean_text("\n\n\nhello\n\n") == "hello"

    def test_empty_input_stays_empty(self):
        assert clean_text("") == ""


class TestCleanPromptText:
    def test_preserves_all_internal_whitespace(self):
        raw = " \r\na  b\t \r\n\r\n\r\n  c  \r\n "
        assert clean_prompt_text(raw) == "a  b\t \n\n\n  c"

    def test_normalizes_and_removes_control_characters(self):
        assert clean_prompt_text(" \x00cafe\u0301\ufeff\r\n") == "café"


class TestSplitText:
    def test_splits_on_blank_lines(self):
        assert split_text("one\n\ntwo\n\nthree") == ["one", "two", "three"]

    def test_treats_multiple_blank_lines_as_one_boundary(self):
        assert split_text("one\n\n\n\ntwo") == ["one", "two"]

    def test_filters_empty_paragraphs(self):
        assert split_text("\n\n\none\n\n  \n\ntwo\n\n") == ["one", "two"]

    def test_single_paragraph_without_blank_lines(self):
        assert split_text("only one") == ["only one"]

    def test_all_empty_input_returns_empty_list(self):
        assert split_text("  \n\n\t\n ") == []


class TestPreprocessText:
    def test_full_pipeline_on_messy_input(self):
        raw = "\r\n\x00第一段   内容\r\n\r\n\r\n  \r\n第二段\t\t内容\x07\r\n"
        assert preprocess_text(raw) == ["第一段 内容", "第二段 内容"]

    def test_normalizes_unicode_inside_paragraphs(self):
        assert preprocess_text("cafe\u0301\n\nplain") == ["café", "plain"]

    def test_returns_empty_list_for_blank_input(self):
        assert preprocess_text("   \n\n  ") == []

    def test_pipeline_is_composable_in_the_documented_order(self):
        raw = "hello\u001b\r\n\r\n\r\nworld\r\n"
        assert preprocess_text(raw) == split_text(clean_text(normalize_text(raw)))
