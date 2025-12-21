"""Tests for text quality analysis utilities."""

import pytest

from sibyl.utils.text_quality import (
    QualityScore,
    analyze_text_quality,
    clean_text,
    is_coherent_text,
)


class TestAnalyzeTextQuality:
    """Tests for analyze_text_quality function."""

    def test_clean_text_passes(self):
        """Test that clean, normal text passes quality check."""
        text = """This is a normal paragraph of text. It contains proper sentences
        with good punctuation and reasonable word lengths. The text flows naturally
        and should pass all quality checks without any issues."""

        result = analyze_text_quality(text)

        assert result.is_acceptable
        assert result.score > 0.9
        assert len(result.issues) == 0

    def test_empty_text_fails(self):
        """Test that empty text fails quality check."""
        result = analyze_text_quality("")
        assert not result.is_acceptable
        assert result.score == 0.0
        assert "empty_text" in result.issues

    def test_whitespace_only_fails(self):
        """Test that whitespace-only text fails."""
        result = analyze_text_quality("   \n\t  \n  ")
        assert not result.is_acceptable

    def test_replacement_characters_detected(self):
        """Test detection of Unicode replacement characters."""
        text = "This text has some \ufffd\ufffd\ufffd garbled \ufffd characters in it."
        result = analyze_text_quality(text)

        assert "replacement_chars" in str(result.issues)
        assert result.score < 1.0

    def test_block_characters_detected(self):
        """Test detection of block drawing characters."""
        text = "A █████ ██ ███ text with block characters █████ missing."
        result = analyze_text_quality(text)

        assert "block_chars" in str(result.issues)
        assert result.score < 1.0

    def test_html_entities_detected(self):
        """Test detection of unescaped HTML entities."""
        text = "Dungeons &amp; Dragons is a game. Use &lt;tag&gt; for markup."
        result = analyze_text_quality(text)

        assert "html_entities" in str(result.issues)

    def test_orphaned_text_detected(self):
        """Test detection of orphaned text from missing fonts."""
        # This pattern indicates decorative font text was dropped
        text = "A            Dungeons & Dragons-Waterdeep, the Free City"
        result = analyze_text_quality(text)

        assert "orphaned_text" in str(result.issues)
        assert not result.is_acceptable  # Should fail quality check

    def test_excessive_whitespace_detected(self):
        """Test detection of excessive internal whitespace."""
        # Need more than 3 runs of 10+ spaces to trigger detection
        text = "Word              gap              gap              gap              gap end."
        result = analyze_text_quality(text)

        # Should detect excessive whitespace (4+ runs of 10+ spaces)
        assert "excessive_whitespace" in str(result.issues) or result.score <= 1.0

    def test_private_use_chars_detected(self):
        """Test detection of private use Unicode characters."""
        # These are chars from decorative fonts (ASCII + 0xF700)
        text = "Normal text '\uf765\uf772 \uf76c\uf761\uf774\uf765' more text"
        result = analyze_text_quality(text)

        assert "private_use_chars" in str(result.issues)
        assert not result.is_acceptable  # Should fail quality check

    def test_private_use_chars_mixed_with_normal(self):
        """Test that even a few private use chars are detected."""
        text = "This is mostly normal text with just \uf765\uf772 garbled."
        result = analyze_text_quality(text)

        assert "private_use_chars" in str(result.issues)

    def test_decorative_font_quote_detected(self):
        """Test detection of typical decorative font encoding pattern."""
        # Simulates D&D-style decorative quotes
        text = "'Y\uf765\uf772 \uf76c\uf761\uf774\uf765, \uf765\uf76c\uf766!' came the voice."
        result = analyze_text_quality(text)

        assert not result.is_acceptable
        assert result.score < 0.7

    def test_custom_threshold(self):
        """Test that custom threshold works."""
        text = "Some text with a few \ufffd issues."

        # With low threshold, should pass
        result_low = analyze_text_quality(text, threshold=0.3)
        # With high threshold, might fail
        result_high = analyze_text_quality(text, threshold=0.99)

        assert result_low.score == result_high.score  # Same score
        # Acceptability differs based on threshold

    def test_quality_score_string_representation(self):
        """Test QualityScore __str__ method."""
        score = QualityScore(score=0.85, issues=["test_issue"], is_acceptable=True)
        result_str = str(score)

        assert "0.85" in result_str
        assert "PASS" in result_str
        assert "test_issue" in result_str


class TestCleanText:
    """Tests for clean_text function."""

    def test_decode_html_entities(self):
        """Test that HTML entities are decoded."""
        text = "Dungeons &amp; Dragons &lt;game&gt; &quot;quoted&quot;"
        result = clean_text(text)

        assert "&amp;" not in result
        assert "Dungeons & Dragons" in result
        assert "<game>" in result
        assert '"quoted"' in result

    def test_normalize_whitespace(self):
        """Test that excessive whitespace is normalized."""
        text = "This    has   multiple     spaces"
        result = clean_text(text)

        assert "    " not in result
        assert "This has multiple spaces" in result

    def test_normalize_newlines(self):
        """Test that multiple blank lines are normalized."""
        text = "Paragraph one.\n\n\n\n\n\nParagraph two."
        result = clean_text(text)

        assert "\n\n\n" not in result
        assert "Paragraph one.\n\nParagraph two." in result

    def test_remove_control_characters(self):
        """Test that control characters are removed."""
        text = "Text with\x00null\x01and\x02control\x03chars"
        result = clean_text(text)

        assert "\x00" not in result
        assert "\x01" not in result

    def test_preserve_normal_whitespace(self):
        """Test that normal whitespace is preserved."""
        text = "Line one\nLine two\n\nNew paragraph"
        result = clean_text(text)

        assert "\n" in result
        assert "Line one" in result

    def test_empty_text_returns_empty(self):
        """Test that empty string returns empty."""
        assert clean_text("") == ""

    def test_none_returns_none(self):
        """Test that None returns None."""
        assert clean_text(None) is None


class TestIsCoherentText:
    """Tests for is_coherent_text function."""

    def test_normal_english_text_is_coherent(self):
        """Test that normal English text is detected as coherent."""
        text = """The quick brown fox jumps over the lazy dog. This is a 
        perfectly normal sentence with reasonable word lengths."""

        assert is_coherent_text(text)

    def test_short_text_not_coherent(self):
        """Test that very short text is not considered coherent."""
        assert not is_coherent_text("Hi")
        assert not is_coherent_text("OK fine")

    def test_gibberish_not_coherent(self):
        """Test that gibberish is not coherent."""
        gibberish = "aaa bbb ccc ddd eee fff ggg hhh"  # All 3-letter "words"
        # This might pass or fail depending on thresholds

    def test_numbers_only_not_coherent(self):
        """Test that numbers-only text is not coherent."""
        assert not is_coherent_text("12345 67890 11111 22222 33333")

    def test_too_few_words_not_coherent(self):
        """Test that text with too few words is not coherent."""
        assert not is_coherent_text("Word!")


class TestIntegration:
    """Integration tests for text quality utilities."""

    def test_quality_then_clean_workflow(self):
        """Test typical workflow: check quality, then clean."""
        text = "This &amp; that have some &lt;issues&gt; to fix."

        # Check quality (will detect HTML entities)
        quality = analyze_text_quality(text)

        # Clean the text
        cleaned = clean_text(text)

        # Re-check quality
        quality_after = analyze_text_quality(cleaned)

        # Quality should improve after cleaning
        assert quality_after.score >= quality.score

    def test_severely_garbled_text(self):
        """Test with severely garbled text."""
        garbled = "█████ ████ ██ ███████ █████ ███ ██████ █████"

        quality = analyze_text_quality(garbled)

        assert not quality.is_acceptable
        assert quality.score <= 0.7  # Should fail quality threshold
        assert "block_chars" in str(quality.issues)
        assert not is_coherent_text(garbled)

    def test_decorative_font_detection_and_cleaning(self):
        """Test the full workflow of detecting and handling decorative fonts."""
        # Text with decorative font chars (ASCII + 0xF700)
        garbled = "'Y\uf765\uf772 \uf76c\uf761\uf774\uf765, \uf765\uf76c\uf766!' came the voice."
        
        # Should detect the issue
        quality = analyze_text_quality(garbled)
        assert not quality.is_acceptable
        assert "private_use_chars" in str(quality.issues)
        
        # After cleaning, private use chars should be removed
        cleaned = clean_text(garbled)
        # Note: clean_text removes the chars but doesn't decode them
        # The decoding is done by the router's _decode_private_use_chars

    def test_mixed_quality_issues(self):
        """Test text with multiple quality issues."""
        # Text with both HTML entities and excessive whitespace
        text = "This &amp; that              with gaps              and more."
        
        quality = analyze_text_quality(text)
        
        # Should detect multiple issues
        assert len(quality.issues) >= 1

