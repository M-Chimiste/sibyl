"""Tests for table utility functions."""

import pytest

from sibyl.utils.tables import merge_split_tables


class TestMergeSplitTables:
    """Tests for the merge_split_tables function."""

    def test_merge_simple_4_column_table(self):
        """Test merging a simple 4-column table with repeated headers."""
        markdown = """| Name | Value | Name | Value |
|------|-------|------|-------|
| A    | 1     | C    | 3     |
| B    | 2     | D    | 4     |
"""
        result, count = merge_split_tables(markdown)

        assert count == 1
        assert "| Name | Value |" in result
        # Should have 4 data rows now
        lines = [l for l in result.strip().split('\n') if l.startswith('|')]
        assert len(lines) == 6  # header + separator + 4 data rows

    def test_merge_xp_table_format(self):
        """Test merging a table like the D&D XP by Challenge Rating table."""
        markdown = """## Experience Points by Challenge Rating

| Challenge | XP | Challenge | XP |
|-----------|--------|-----------|--------|
| 0 | 0 or 10 | 14 | 11,500 |
| 1/8 | 25 | 15 | 13,000 |
| 1/4 | 50 | 16 | 15,000 |
"""
        result, count = merge_split_tables(markdown)

        assert count == 1
        # Header should now be 2 columns
        assert "| Challenge | XP |" in result
        # Should preserve the heading
        assert "## Experience Points by Challenge Rating" in result
        # Check that data is merged
        assert "| 0 | 0 or 10 |" in result
        assert "| 14 | 11,500 |" in result

    def test_no_merge_different_headers(self):
        """Test that tables with different headers are not merged."""
        markdown = """| Name | Value | Other | Data |
|------|-------|-------|------|
| A    | 1     | X     | Y    |
"""
        result, count = merge_split_tables(markdown)

        assert count == 0
        assert result == markdown

    def test_no_merge_odd_columns(self):
        """Test that tables with odd number of columns are not merged."""
        markdown = """| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
"""
        result, count = merge_split_tables(markdown)

        assert count == 0
        assert result == markdown

    def test_no_merge_2_column_table(self):
        """Test that 2-column tables are not affected."""
        markdown = """| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |
"""
        result, count = merge_split_tables(markdown)

        assert count == 0
        assert result == markdown

    def test_merge_6_column_table(self):
        """Test merging a 6-column table (3+3)."""
        markdown = """| A | B | C | A | B | C |
|---|---|---|---|---|---|
| 1 | 2 | 3 | 4 | 5 | 6 |
| 7 | 8 | 9 | 10 | 11 | 12 |
"""
        result, count = merge_split_tables(markdown)

        assert count == 1
        assert "| A | B | C |" in result
        # Should have 4 data rows (2 original rows × 2 halves)
        lines = [l for l in result.strip().split('\n') if l.startswith('|') and '---' not in l]
        # header + 4 data rows = 5 lines
        assert len(lines) == 5

    def test_preserve_empty_right_cells(self):
        """Test that rows with empty right cells only add left cells."""
        markdown = """| Name | Value | Name | Value |
|------|-------|------|-------|
| A    | 1     | C    | 3     |
| B    | 2     |      |       |
"""
        result, count = merge_split_tables(markdown)

        assert count == 1
        # Should have 3 data rows (A, B, C) - empty right row ignored
        lines = [l for l in result.strip().split('\n') if l.startswith('|') and '---' not in l and l != '| Name | Value |']
        assert len(lines) == 3

    def test_case_insensitive_header_matching(self):
        """Test that header matching is case-insensitive."""
        markdown = """| Name | VALUE | name | value |
|------|-------|------|-------|
| A    | 1     | B    | 2     |
"""
        result, count = merge_split_tables(markdown)

        assert count == 1

    def test_multiple_tables_in_document(self):
        """Test merging multiple split tables in one document."""
        markdown = """# First Table

| A | B | A | B |
|---|---|---|---|
| 1 | 2 | 3 | 4 |

Some text between tables.

# Second Table

| X | Y | X | Y |
|---|---|---|---|
| a | b | c | d |
"""
        result, count = merge_split_tables(markdown)

        assert count == 2
        assert "# First Table" in result
        assert "# Second Table" in result
        assert "Some text between tables." in result

    def test_preserves_surrounding_content(self):
        """Test that content before and after tables is preserved."""
        markdown = """# Introduction

Some intro text here.

| A | B | A | B |
|---|---|---|---|
| 1 | 2 | 3 | 4 |

## Conclusion

Final thoughts.
"""
        result, count = merge_split_tables(markdown)

        assert count == 1
        assert "# Introduction" in result
        assert "Some intro text here." in result
        assert "## Conclusion" in result
        assert "Final thoughts." in result

    def test_preserves_table_alignment(self):
        """Test that column alignment markers are preserved."""
        markdown = """| Left | Right | Left | Right |
|:-----|------:|:-----|------:|
| A    | 1     | B    | 2     |
"""
        result, count = merge_split_tables(markdown)

        assert count == 1
        # Left alignment marker should be preserved
        assert ":--" in result or "---" in result

    def test_empty_markdown(self):
        """Test with empty markdown string."""
        result, count = merge_split_tables("")

        assert count == 0
        assert result == ""

    def test_no_tables(self):
        """Test markdown with no tables."""
        markdown = """# Just a heading

Some paragraph text.

- A list item
- Another item
"""
        result, count = merge_split_tables(markdown)

        assert count == 0
        assert result == markdown

