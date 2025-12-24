"""Utility functions for table processing and normalization."""

from __future__ import annotations

import re


def merge_split_tables(markdown: str) -> tuple[str, int]:
    """
    Detect and merge horizontally-split tables in markdown.
    
    PDFs often display tables in multiple columns to save space, e.g.:
    | Col1 | Col2 | Col1 | Col2 |
    
    This function detects such tables by identifying repeated column headers
    and merges them into a single tall table:
    | Col1 | Col2 |
    
    Args:
        markdown: Markdown content containing tables
        
    Returns:
        Tuple of (processed markdown, number of tables merged)
        
    Example:
        >>> md = '''
        ... | Name | Value | Name | Value |
        ... |------|-------|------|-------|
        ... | A    | 1     | C    | 3     |
        ... | B    | 2     | D    | 4     |
        ... '''
        >>> result, count = merge_split_tables(md)
        >>> print(result)
        | Name | Value |
        |------|-------|
        | A    | 1     |
        | B    | 2     |
        | C    | 3     |
        | D    | 4     |
    """
    # Pattern to match markdown tables
    # Note: Last data row may or may not have trailing newline
    table_pattern = re.compile(
        r'(\|[^\n]+\|\n)'              # Header row
        r'(\|[-:\s|]+\|\n)'            # Separator row  
        r'((?:\|[^\n]+\|(?:\n|$))+)',  # Data rows (newline or end of string)
        re.MULTILINE
    )
    
    merged_count = 0
    
    def process_table(match: re.Match) -> str:
        nonlocal merged_count
        header_line = match.group(1)
        separator_line = match.group(2)
        data_lines = match.group(3)
        
        # Parse header columns
        headers = [h.strip() for h in header_line.strip().strip('|').split('|')]
        
        # Check if this is a horizontally-split table
        # Look for repeated column patterns (e.g., [A, B, A, B] or [A, B, C, A, B, C])
        num_cols = len(headers)
        
        if num_cols < 4 or num_cols % 2 != 0:
            return match.group(0)  # Not a candidate for merging
        
        half = num_cols // 2
        left_headers = headers[:half]
        right_headers = headers[half:]
        
        # Normalize headers for comparison (lowercase, strip whitespace)
        def normalize(h: str) -> str:
            return h.lower().strip()
        
        left_normalized = [normalize(h) for h in left_headers]
        right_normalized = [normalize(h) for h in right_headers]
        
        # Check if headers match (allowing for minor variations)
        if left_normalized != right_normalized:
            return match.group(0)  # Headers don't match, not a split table
        
        # Parse separator to preserve alignment
        separators = [s.strip() for s in separator_line.strip().strip('|').split('|')]
        left_separators = separators[:half]
        
        # Parse data rows
        rows: list[list[str]] = []
        for line in data_lines.strip().split('\n'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if len(cells) == num_cols:
                left_cells = cells[:half]
                right_cells = cells[half:]
                rows.append(left_cells)
                # Only add right cells if they have content
                if any(c.strip() for c in right_cells):
                    rows.append(right_cells)
        
        # Build merged table
        new_header = '| ' + ' | '.join(left_headers) + ' |'
        new_separator = '| ' + ' | '.join(left_separators) + ' |'
        new_rows = '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)
        
        merged_count += 1
        return f'{new_header}\n{new_separator}\n{new_rows}\n'
    
    result = table_pattern.sub(process_table, markdown)
    return result, merged_count

