"""PyMuPDF extractor for simple PDF text extraction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sibyl.extractors.base import ExtractionResult, Extractor
from sibyl.models import ImageResult, PageResult, TableResult

if TYPE_CHECKING:
    from sibyl.models import ExtractOptions


class PyMuPDFExtractor(Extractor):
    """Extract content from PDFs using PyMuPDF (fitz).

    This is a simple, fast extractor that uses PyMuPDF's native
    text extraction. It doesn't do advanced structure detection
    but handles decorative fonts better than some other extractors.
    """

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "pymupdf"

    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the file."""
        return file_path.suffix.lower() == ".pdf"

    def extract(
        self,
        file_path: Path,
        options: "ExtractOptions",
    ) -> ExtractionResult:
        """Extract content from a PDF using PyMuPDF.

        Args:
            file_path: Path to PDF file
            options: Extraction options

        Returns:
            ExtractionResult with extracted pages
        """
        from sibyl.utils.pdf import _get_fitz

        fitz = _get_fitz()
        pages: list[PageResult] = []

        with fitz.open(file_path) as pdf:
            page_numbers = options.pages or list(range(1, len(pdf) + 1))

            for page_num in page_numbers:
                page_idx = page_num - 1  # 0-indexed
                if page_idx < 0 or page_idx >= len(pdf):
                    continue

                page = pdf[page_idx]
                text = page.get_text("text")

                # Convert to basic markdown (add headings heuristics)
                markdown = self._text_to_markdown(text)

                pages.append(
                    PageResult(
                        page_number=page_num,
                        content=markdown,
                        extraction_method="pymupdf",
                        confidence=0.9,
                        images=[],
                        tables=[],
                    )
                )

        return ExtractionResult(
            pages=pages,
            title=None,
            author=None,
        )

    def _text_to_markdown(self, text: str) -> str:
        """Convert plain text to basic markdown.

        Applies simple heuristics to detect headings.
        """
        import re

        lines = text.split("\n")
        result_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                result_lines.append("")
                continue

            # Heuristic: Short lines in all caps might be headings
            if (
                len(stripped) < 60
                and stripped.isupper()
                and not stripped.endswith((".", ",", ";", ":"))
            ):
                # Check if it looks like a chapter/section heading
                if any(
                    word in stripped
                    for word in ["CHAPTER", "PART", "SECTION", "APPENDIX"]
                ):
                    result_lines.append(f"# {stripped.title()}")
                else:
                    result_lines.append(f"## {stripped.title()}")
            else:
                result_lines.append(stripped)

        return "\n".join(result_lines)

