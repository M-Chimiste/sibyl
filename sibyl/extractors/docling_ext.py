"""Docling extractor for native PDF text extraction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from docling.document_converter import DocumentConverter

from sibyl.extractors.base import ExtractionResult, Extractor
from sibyl.models import ImageResult, PageResult, TableResult

if TYPE_CHECKING:
    from sibyl.models import ExtractOptions


class DoclingExtractor(Extractor):
    """Extract content from PDFs using Docling.

    Docling provides high-quality extraction for digitally-born PDFs
    with native text layers, including structure preservation and
    table detection.
    """

    def __init__(self):
        """Initialize Docling extractor."""
        self._converter = DocumentConverter()

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "docling"

    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the file.

        Args:
            file_path: Path to check

        Returns:
            True for PDF files
        """
        return file_path.suffix.lower() == ".pdf"

    def extract(
        self,
        file_path: Path,
        options: "ExtractOptions",
    ) -> ExtractionResult:
        """Extract content from a PDF using Docling.

        Args:
            file_path: Path to PDF file
            options: Extraction options

        Returns:
            ExtractionResult with extracted pages
        """
        # Convert document
        result = self._converter.convert(str(file_path))
        doc = result.document

        # Get full markdown export
        markdown_content = doc.export_to_markdown()

        # Get page count
        page_count = self._get_page_count(file_path)

        # Extract tables
        all_tables: list[TableResult] = []
        if options.extract_tables:
            all_tables = self._extract_tables(doc)

        # Extract images
        all_images: list[ImageResult] = []
        if options.extract_images:
            all_images = self._extract_images(doc)

        # Docling processes the document holistically, so we return
        # the full content. For page-level chunking, users can use
        # the chunker module.
        pages = [
            PageResult(
                page_number=1,
                content=markdown_content,
                extraction_method="docling",
                confidence=1.0,
                images=all_images,
                tables=all_tables,
            )
        ]

        # Get metadata
        title = None
        author = None

        return ExtractionResult(
            pages=pages,
            title=title,
            author=author,
        )

    def _get_page_count(self, file_path: Path) -> int:
        """Get page count from PDF."""
        import fitz

        with fitz.open(file_path) as pdf:
            return len(pdf)

    def _extract_tables(self, doc) -> list[TableResult]:
        """Extract tables from document."""
        tables: list[TableResult] = []

        # Docling stores tables in the document's content items
        try:
            for item, _level in doc.iterate_items():
                if hasattr(item, "export_to_markdown") and "table" in type(item).__name__.lower():
                    try:
                        markdown = item.export_to_markdown(doc=doc)
                    except TypeError:
                        # Fallback for older API
                        markdown = item.export_to_markdown()

                    # Try to get page number from item's provenance
                    page_num = 1
                    if hasattr(item, "prov") and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                page_num = prov.page_no
                                break

                    # Try to get table dimensions
                    rows = 0
                    cols = 0
                    if hasattr(item, "data") and hasattr(item.data, "num_rows"):
                        rows = item.data.num_rows
                        cols = item.data.num_cols

                    tables.append(
                        TableResult(
                            markdown=markdown,
                            page_number=page_num,
                            rows=rows,
                            columns=cols,
                        )
                    )
        except Exception:
            # If iteration fails, tables will be empty
            pass

        return tables

    def _extract_images(self, doc) -> list[ImageResult]:
        """Extract images from document."""
        images: list[ImageResult] = []

        try:
            for item, _level in doc.iterate_items():
                if "picture" in type(item).__name__.lower():
                    page_num = 1
                    if hasattr(item, "prov") and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                page_num = prov.page_no
                                break

                    images.append(
                        ImageResult(
                            path=None,
                            ocr_text=None,
                            page_number=page_num,
                            width=None,
                            height=None,
                        )
                    )
        except Exception:
            pass

        return images
