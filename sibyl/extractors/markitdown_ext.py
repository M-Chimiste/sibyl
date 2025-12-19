"""MarkItDown extractor for Office formats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from markitdown import MarkItDown

from sibyl.extractors.base import ExtractionResult, Extractor
from sibyl.models import PageResult, TableResult

if TYPE_CHECKING:
    from sibyl.models import ExtractOptions


class MarkItDownExtractor(Extractor):
    """Extract content from Office formats and PDFs using MarkItDown.

    Supports DOCX, PPTX, XLSX, HTML, PDF, and other formats that
    MarkItDown can handle.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        ".html",
        ".htm",
        ".csv",
        ".json",
        ".xml",
        ".rtf",
        ".epub",
    }

    def __init__(self):
        """Initialize MarkItDown extractor."""
        self._converter = MarkItDown()

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "markitdown"

    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the file.

        Args:
            file_path: Path to check

        Returns:
            True for supported Office formats
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract(
        self,
        file_path: Path,
        options: "ExtractOptions",
    ) -> ExtractionResult:
        """Extract content using MarkItDown.

        Args:
            file_path: Path to document
            options: Extraction options

        Returns:
            ExtractionResult with extracted content
        """
        # Convert document
        result = self._converter.convert(str(file_path))
        markdown_content = result.text_content

        # MarkItDown provides document-level output
        # We create a single "page" for the content
        pages: list[PageResult] = []
        tables: list[TableResult] = []

        # Extract tables from markdown if requested
        if options.extract_tables:
            tables = self._extract_tables_from_markdown(markdown_content)

        # Create single page result
        pages.append(
            PageResult(
                page_number=1,
                content=markdown_content,
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=tables,
            )
        )

        # Try to extract title from content
        title = self._extract_title(markdown_content)

        return ExtractionResult(
            pages=pages,
            title=title,
            author=None,
        )

    def _extract_title(self, markdown: str) -> str | None:
        """Extract title from markdown content.

        Looks for first H1 heading.
        """
        lines = markdown.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                return line[2:].strip()
        return None

    def _extract_tables_from_markdown(
        self,
        markdown: str,
    ) -> list[TableResult]:
        """Extract table structures from markdown text."""
        tables: list[TableResult] = []

        lines = markdown.split("\n")
        in_table = False
        table_lines: list[str] = []

        for line in lines:
            is_table_row = "|" in line and line.strip().startswith("|")

            if is_table_row:
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            elif in_table:
                if table_lines:
                    table_md = "\n".join(table_lines)
                    rows, cols = self._count_table_dimensions(table_lines)
                    tables.append(
                        TableResult(
                            markdown=table_md,
                            page_number=1,
                            rows=rows,
                            columns=cols,
                        )
                    )
                in_table = False
                table_lines = []

        # Handle table at end of content
        if in_table and table_lines:
            table_md = "\n".join(table_lines)
            rows, cols = self._count_table_dimensions(table_lines)
            tables.append(
                TableResult(
                    markdown=table_md,
                    page_number=1,
                    rows=rows,
                    columns=cols,
                )
            )

        return tables

    def _count_table_dimensions(self, table_lines: list[str]) -> tuple[int, int]:
        """Count rows and columns in a markdown table."""
        data_lines = [
            line for line in table_lines if not all(c in "|-: " for c in line.strip())
        ]

        rows = len(data_lines)
        cols = 0
        if data_lines:
            cols = data_lines[0].count("|") - 1
            cols = max(0, cols)

        return rows, cols
