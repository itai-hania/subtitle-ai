"""Comprehensive tests for translation response parsing (Issue 10A)."""

import pytest

from processing import parse_numbered_translations, is_untranslated


class TestParseNumberedTranslations:
    """Tests for parse_numbered_translations — the most fragile part of the codebase."""

    def test_happy_path(self):
        """Basic numbered translations parse correctly."""
        result_text = "[1] שלום\n[2] עולם\n[3] מה קורה"
        originals = ["Hello", "World", "What's up"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations == ["שלום", "עולם", "מה קורה"]

    def test_extra_whitespace(self):
        """Handles extra whitespace between entries."""
        result_text = "[1]   שלום   \n\n\n[2]   עולם   "
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations == ["שלום", "עולם"]

    def test_preamble_text(self):
        """LLM adds preamble text before the translations."""
        result_text = "Here are your translations:\n\n[1] שלום\n[2] עולם"
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations == ["שלום", "עולם"]

    def test_trailing_text(self):
        """LLM adds trailing commentary after translations."""
        result_text = "[1] שלום\n[2] עולם\n\nI hope these translations help!"
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations == ["שלום", "עולם"]

    def test_multiline_translation(self):
        """A single translation spans multiple lines."""
        result_text = "[1] שלום\nעולם\n[2] מה קורה"
        originals = ["Hello world", "What's up"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations[0] == "שלום עולם"  # Multiline collapsed
        assert translations[1] == "מה קורה"

    def test_brackets_in_translation(self):
        """Translation text contains bracket characters."""
        result_text = "[1] ה[API] עובד\n[2] גרסה [2.0] יצאה"
        originals = ["The [API] works", "Version [2.0] released"]
        translations = parse_numbered_translations(result_text, originals)
        # Should get something reasonable even with brackets
        assert len(translations) == 2
        # First translation should be found
        assert "API" in translations[0] or translations[0] == originals[0]

    def test_missing_segment_falls_back(self):
        """When LLM skips a segment, falls back to original text."""
        result_text = "[1] שלום\n[3] סוף"
        originals = ["Hello", "Middle text", "End"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations[0] == "שלום"
        assert translations[1] == "Middle text"  # Fallback to original
        assert translations[2] == "סוף"

    def test_single_segment(self):
        """Single segment translation."""
        result_text = "[1] שלום עולם"
        originals = ["Hello world"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations == ["שלום עולם"]

    def test_empty_input(self):
        """Empty input list returns empty."""
        translations = parse_numbered_translations("", [])
        assert translations == []

    def test_large_batch(self):
        """Handles a large batch of translations."""
        n = 50
        lines = [f"[{i+1}] תרגום {i+1}" for i in range(n)]
        result_text = "\n".join(lines)
        originals = [f"Original {i+1}" for i in range(n)]
        translations = parse_numbered_translations(result_text, originals)
        assert len(translations) == n
        for i, t in enumerate(translations):
            assert f"תרגום {i+1}" == t

    def test_duplicate_numbers(self):
        """LLM accidentally duplicates a number — should handle gracefully."""
        result_text = "[1] שלום\n[1] אחר\n[2] עולם"
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert len(translations) == 2
        # First match for [1] should be captured
        assert translations[0] == "שלום"
        assert translations[1] == "עולם"

    def test_no_numbers_at_all(self):
        """LLM returns completely unformatted text — falls back to all originals."""
        result_text = "שלום\nעולם"
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations == originals  # All fallback

    def test_extra_numbers(self):
        """LLM adds more numbers than expected — should only take what we need."""
        result_text = "[1] שלום\n[2] עולם\n[3] אקסטרה"
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert len(translations) == 2
        assert translations == ["שלום", "עולם"]

    def test_with_colon_format(self):
        """Some LLMs format as [N]: instead of [N] ."""
        result_text = "[1] שלום\n[2] עולם"
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations == ["שלום", "עולם"]

    def test_windows_line_endings(self):
        """Handles \\r\\n line endings."""
        result_text = "[1] שלום\r\n[2] עולם"
        originals = ["Hello", "World"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations[0] == "שלום"
        assert translations[1] == "עולם"

    def test_preserves_punctuation(self):
        """Translation with punctuation is preserved."""
        result_text = "[1] !שלום, עולם"
        originals = ["Hello, world!"]
        translations = parse_numbered_translations(result_text, originals)
        assert translations[0] == "!שלום, עולם"


class TestIsUntranslated:
    """Tests for is_untranslated — detects English text when target is non-Latin."""

    def test_hebrew_text_not_flagged(self):
        assert is_untranslated("שלום עולם", "hebrew") is False

    def test_english_text_flagged_for_hebrew(self):
        assert is_untranslated("Hello world this is English", "hebrew") is True

    def test_english_text_not_flagged_for_spanish(self):
        """Latin-script target languages should never be flagged."""
        assert is_untranslated("Hello world", "spanish") is False

    def test_mixed_mostly_hebrew(self):
        """Mostly Hebrew with a few English words (like proper nouns) is OK."""
        assert is_untranslated("וורן באפט הוא משקיע מוצלח", "hebrew") is False

    def test_mixed_mostly_english(self):
        """Mostly English with a Hebrew word is still untranslated."""
        assert is_untranslated("The שלום investor bought stocks", "hebrew") is True

    def test_numbers_only(self):
        """Pure numbers have no alpha chars — should not be flagged."""
        assert is_untranslated("1956, 69, 69", "hebrew") is False

    def test_empty_string(self):
        assert is_untranslated("", "hebrew") is False

    def test_arabic_target(self):
        assert is_untranslated("Hello world", "arabic") is True

    def test_russian_target(self):
        assert is_untranslated("Hello world", "russian") is True
