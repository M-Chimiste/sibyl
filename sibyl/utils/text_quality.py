"""Text quality analysis and cleaning utilities."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Result of text quality analysis."""

    score: float  # 0.0 (garbage) to 1.0 (clean)
    issues: list[str]  # List of detected issues
    is_acceptable: bool  # Whether quality is above threshold

    def __str__(self) -> str:
        status = "PASS" if self.is_acceptable else "FAIL"
        issues_str = ", ".join(self.issues) if self.issues else "none"
        return f"Quality: {self.score:.2f} ({status}) - Issues: {issues_str}"


def analyze_text_quality(text: str, threshold: float = 0.7) -> QualityScore:
    """Analyze text quality to detect extraction issues.

    Checks for common signs of failed text extraction:
    - Replacement characters (�)
    - Box drawing / block characters (█, ▓, etc.)
    - Private use Unicode characters
    - Excessive whitespace or missing text
    - HTML entities that should be decoded
    - Very low character diversity

    Args:
        text: Text to analyze
        threshold: Quality score threshold (0.0-1.0)

    Returns:
        QualityScore with score, issues, and acceptability
    """
    if not text or not text.strip():
        return QualityScore(score=0.0, issues=["empty_text"], is_acceptable=False)

    issues: list[str] = []
    penalties: list[float] = []

    # Check for replacement characters (�)
    replacement_count = text.count("\ufffd")
    if replacement_count > 0:
        ratio = replacement_count / len(text)
        if ratio > 0.01:  # More than 1%
            issues.append(f"replacement_chars ({replacement_count})")
            penalties.append(min(0.5, ratio * 10))

    # Check for block/box drawing characters (common in garbled PDFs)
    block_chars = re.findall(r"[█▓▒░▀▄▌▐▲▼◄►■□▪▫●○◘◙◦]", text)
    if block_chars:
        ratio = len(block_chars) / len(text)
        if ratio > 0.005:  # More than 0.5%
            issues.append(f"block_chars ({len(block_chars)})")
            penalties.append(min(0.4, ratio * 20))

    # Check for private use Unicode characters
    # These often indicate custom font glyphs that weren't properly extracted
    private_use = sum(1 for c in text if unicodedata.category(c) == "Co")
    if private_use > 0:
        ratio = private_use / len(text)
        if ratio > 0.005:  # Even a small amount is problematic
            issues.append(f"private_use_chars ({private_use})")
            # Strong penalty - these indicate font extraction failures
            penalties.append(min(0.5, ratio * 20 + 0.2))

    # Check for HTML entities (sign of improper decoding)
    html_entities = re.findall(r"&(?:amp|lt|gt|quot|nbsp|apos|#\d+|#x[0-9a-fA-F]+);", text)
    if html_entities:
        ratio = len(html_entities) / max(1, len(text.split()))
        if ratio > 0.02:  # More than 2% of words
            issues.append(f"html_entities ({len(html_entities)})")
            penalties.append(min(0.2, ratio * 2))

    # Check for excessive consecutive whitespace (missing text)
    # Look for runs of 10+ spaces that aren't at line starts
    excessive_spaces = re.findall(r"(?<!\n)\s{10,}(?!\n)", text)
    if len(excessive_spaces) > 3:
        issues.append(f"excessive_whitespace ({len(excessive_spaces)} runs)")
        penalties.append(min(0.3, len(excessive_spaces) * 0.02))

    # Check for orphaned text patterns (signs of missing decorative fonts)
    # e.g., "A            Word" suggests text was dropped between A and Word
    # This is a strong indicator of font rendering issues
    orphaned_patterns = re.findall(
        r"(?:^|[.!?]\s+)([A-Z])\s{5,}[A-Z]",  # Single capital letter + gap + capital
        text
    )
    if orphaned_patterns:
        issues.append(f"orphaned_text ({len(orphaned_patterns)} occurrences)")
        # Strong penalty - this usually means significant content is missing
        penalties.append(min(0.5, len(orphaned_patterns) * 0.35))

    # Check for very low character diversity (repeated garbage)
    if len(text) > 100:
        unique_chars = len(set(text))
        diversity_ratio = unique_chars / min(len(text), 500)
        if diversity_ratio < 0.05:  # Less than 5% unique chars
            issues.append(f"low_diversity ({unique_chars} unique)")
            penalties.append(0.4)

    # Check for control characters (except normal whitespace)
    control_chars = sum(
        1 for c in text
        if unicodedata.category(c) == "Cc" and c not in "\n\r\t "
    )
    if control_chars > 0:
        ratio = control_chars / len(text)
        if ratio > 0.001:
            issues.append(f"control_chars ({control_chars})")
            penalties.append(min(0.3, ratio * 50))

    # Calculate final score
    total_penalty = sum(penalties)
    score = max(0.0, 1.0 - total_penalty)

    return QualityScore(
        score=score,
        issues=issues,
        is_acceptable=score >= threshold,
    )


def clean_text(text: str) -> str:
    """Clean common text extraction issues.

    Applies fixes for:
    - HTML entity decoding
    - Normalization of Unicode
    - Removal of excessive whitespace
    - Removal of control characters

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return text

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Remove control characters (keep normal whitespace)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Cc" or c in "\n\r\t "
    )

    # Normalize excessive internal whitespace (preserve paragraph breaks)
    # Replace runs of spaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_coherent_text(text: str, min_word_length: float = 3.0) -> bool:
    """Quick check if text appears to be coherent natural language.

    Args:
        text: Text to check
        min_word_length: Minimum average word length for natural text

    Returns:
        True if text appears coherent
    """
    if not text or len(text) < 20:
        return False

    # Extract words (sequences of letters)
    words = re.findall(r"[a-zA-Z]+", text)
    if len(words) < 5:
        return False

    # Check average word length (garbled text often has wrong lengths)
    avg_length = sum(len(w) for w in words) / len(words)
    if avg_length < min_word_length or avg_length > 15:
        return False

    # Check for reasonable ratio of letters to total characters
    letters = sum(1 for c in text if c.isalpha())
    letter_ratio = letters / len(text)
    if letter_ratio < 0.4:  # Less than 40% letters
        return False

    return True

