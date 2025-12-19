"""Vision OCR extractor for scanned documents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sibyl.extractors.base import ExtractionResult, Extractor
from sibyl.models import ImageResult, PageResult, ProgressCallback, TableResult
from sibyl.utils.images import is_image_file, load_image, resize_for_vlm
from sibyl.utils.pdf import get_page_count, get_pdf_metadata, render_page_to_image

if TYPE_CHECKING:
    from sibyl.backends.base import OCRBackend
    from sibyl.models import ExtractOptions


class VisionOCRExtractor(Extractor):
    """Extract content using vision-language models.

    Renders PDF pages or loads images and sends them to a VLM
    for OCR and markdown extraction.
    """

    def __init__(self, ocr_backend: "OCRBackend"):
        """Initialize Vision OCR extractor.

        Args:
            ocr_backend: OCR backend to use for processing
        """
        self._backend = ocr_backend

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "vision_ocr"

    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the file.

        Args:
            file_path: Path to check

        Returns:
            True for PDFs and image files
        """
        suffix = file_path.suffix.lower()
        return suffix == ".pdf" or is_image_file(file_path)

    def extract(
        self,
        file_path: Path,
        options: "ExtractOptions",
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Extract content using vision OCR.

        Args:
            file_path: Path to document
            options: Extraction options
            on_progress: Optional progress callback (stage, current, total)

        Returns:
            ExtractionResult with extracted pages
        """
        if is_image_file(file_path):
            return self._extract_image(file_path, options, on_progress)
        else:
            return self._extract_pdf(file_path, options, on_progress)

    def _extract_image(
        self,
        file_path: Path,
        options: "ExtractOptions",
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Extract content from a single image file."""
        if on_progress:
            on_progress("ocr", 0, 1)

        image = load_image(file_path)
        image = resize_for_vlm(image)

        result = self._backend.ocr_image(image)

        if on_progress:
            on_progress("ocr", 1, 1)

        page = PageResult(
            page_number=1,
            content=result.text,
            extraction_method="vision_ocr",
            confidence=result.confidence,
            images=[
                ImageResult(
                    path=file_path,
                    ocr_text=result.text,
                    page_number=1,
                    width=image.width,
                    height=image.height,
                )
            ],
            tables=[],
        )

        return ExtractionResult(
            pages=[page],
            title=None,
            author=None,
        )

    def _extract_pdf(
        self,
        file_path: Path,
        options: "ExtractOptions",
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Extract content from a PDF by rendering pages to images."""
        page_count = get_page_count(file_path)
        metadata = get_pdf_metadata(file_path)

        # Determine which pages to process
        if options.pages is not None:
            page_numbers = [p for p in options.pages if 1 <= p <= page_count]
        else:
            page_numbers = list(range(1, page_count + 1))

        total_pages = len(page_numbers)
        pages: list[PageResult] = []
        all_tables: list[TableResult] = []

        for i, page_num in enumerate(page_numbers):
            if on_progress:
                on_progress("ocr", i, total_pages)

            # Render page to image (0-indexed for rendering)
            image = render_page_to_image(file_path, page_num - 1)
            image = resize_for_vlm(image)

            # OCR the image
            result = self._backend.ocr_image(image)

            # Parse tables from markdown if present
            page_tables = self._extract_tables_from_markdown(result.text, page_num)
            all_tables.extend(page_tables)

            page = PageResult(
                page_number=page_num,
                content=result.text,
                extraction_method="vision_ocr",
                confidence=result.confidence,
                images=[],
                tables=page_tables,
            )
            pages.append(page)

        if on_progress:
            on_progress("ocr", total_pages, total_pages)

        return ExtractionResult(
            pages=pages,
            title=metadata.get("title"),
            author=metadata.get("author"),
        )

    def _extract_tables_from_markdown(
        self,
        markdown: str,
        page_number: int,
    ) -> list[TableResult]:
        """Extract table structures from markdown text.

        Looks for markdown table syntax and creates TableResult objects.
        """
        tables: list[TableResult] = []

        lines = markdown.split("\n")
        in_table = False
        table_lines: list[str] = []

        for line in lines:
            # Detect table rows (lines with | characters)
            is_table_row = "|" in line and line.strip().startswith("|")

            if is_table_row:
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            elif in_table:
                # End of table
                if table_lines:
                    table_md = "\n".join(table_lines)
                    rows, cols = self._count_table_dimensions(table_lines)
                    tables.append(
                        TableResult(
                            markdown=table_md,
                            page_number=page_number,
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
                    page_number=page_number,
                    rows=rows,
                    columns=cols,
                )
            )

        return tables

    def _count_table_dimensions(self, table_lines: list[str]) -> tuple[int, int]:
        """Count rows and columns in a markdown table."""
        # Filter out separator lines (---|---)
        data_lines = [
            line for line in table_lines if not all(c in "|-: " for c in line.strip())
        ]

        rows = len(data_lines)
        cols = 0
        if data_lines:
            # Count columns from first row
            cols = data_lines[0].count("|") - 1
            cols = max(0, cols)

        return rows, cols
