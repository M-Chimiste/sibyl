"""Content analyzer for document type detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sibyl.utils.images import is_image_file
from sibyl.utils.pdf import get_page_classification, get_page_count


class FileType(Enum):
    """Supported file types."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    CSV = "csv"
    RTF = "rtf"
    EPUB = "epub"
    JSON = "json"
    XML = "xml"
    IMAGE = "image"
    OTHER = "other"


class ContentType(Enum):
    """Content type classification."""

    NATIVE = "native"  # Has extractable text
    SCANNED = "scanned"  # Needs OCR
    HYBRID = "hybrid"  # Mix of both


class ExtractionMethod(Enum):
    """Recommended extraction method."""

    DOCLING = "docling"
    MARKITDOWN = "markitdown"
    VISION_OCR = "vision_ocr"


@dataclass
class PageAnalysis:
    """Analysis result for a single page."""

    page_number: int
    content_type: ContentType
    recommended_method: ExtractionMethod
    text_density: float


@dataclass
class DocumentAnalysis:
    """Complete analysis of a document."""

    file_path: Path
    file_type: FileType
    content_type: ContentType
    page_count: int
    pages: list[PageAnalysis]
    primary_method: ExtractionMethod

    def get_pages_for_method(self, method: ExtractionMethod) -> list[int]:
        """Get page numbers that should use a specific method.

        Args:
            method: Extraction method to filter by

        Returns:
            List of page numbers (1-indexed)
        """
        return [p.page_number for p in self.pages if p.recommended_method == method]


class ContentAnalyzer:
    """Analyze documents to determine optimal extraction strategy."""

    # File extension to type mapping
    EXTENSION_MAP = {
        # PDF
        ".pdf": FileType.PDF,
        # Microsoft Office
        ".docx": FileType.DOCX,
        ".doc": FileType.DOCX,
        ".pptx": FileType.PPTX,
        ".ppt": FileType.PPTX,
        ".xlsx": FileType.XLSX,
        ".xls": FileType.XLSX,
        # Web
        ".html": FileType.HTML,
        ".htm": FileType.HTML,
        # Data formats
        ".csv": FileType.CSV,
        ".json": FileType.JSON,
        ".xml": FileType.XML,
        # Other documents
        ".rtf": FileType.RTF,
        ".epub": FileType.EPUB,
    }

    def __init__(self, ocr_threshold: float = 0.1):
        """Initialize analyzer.

        Args:
            ocr_threshold: Text density below this triggers OCR recommendation
        """
        self.ocr_threshold = ocr_threshold

    def analyze(self, file_path: Path) -> DocumentAnalysis:
        """Analyze a document and determine extraction strategy.

        Args:
            file_path: Path to document

        Returns:
            DocumentAnalysis with recommendations
        """
        file_path = Path(file_path)
        file_type = self._detect_file_type(file_path)

        if file_type == FileType.PDF:
            return self._analyze_pdf(file_path)
        elif file_type == FileType.IMAGE:
            return self._analyze_image(file_path)
        elif file_type in (
            FileType.DOCX,
            FileType.PPTX,
            FileType.XLSX,
            FileType.HTML,
            FileType.CSV,
            FileType.RTF,
            FileType.EPUB,
            FileType.JSON,
            FileType.XML,
        ):
            return self._analyze_office(file_path, file_type)
        else:
            return self._analyze_unknown(file_path)

    def _detect_file_type(self, file_path: Path) -> FileType:
        """Detect file type from extension."""
        suffix = file_path.suffix.lower()

        if suffix in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[suffix]
        elif is_image_file(file_path):
            return FileType.IMAGE
        else:
            return FileType.OTHER

    def _analyze_pdf(self, file_path: Path) -> DocumentAnalysis:
        """Analyze a PDF document."""
        page_count = get_page_count(file_path)
        classifications = get_page_classification(file_path, self.ocr_threshold)

        pages: list[PageAnalysis] = []
        native_count = 0
        scanned_count = 0

        for page_num, classification in enumerate(classifications, start=1):
            if classification == "native":
                content_type = ContentType.NATIVE
                method = ExtractionMethod.DOCLING
                native_count += 1
            else:
                content_type = ContentType.SCANNED
                method = ExtractionMethod.VISION_OCR
                scanned_count += 1

            pages.append(
                PageAnalysis(
                    page_number=page_num,
                    content_type=content_type,
                    recommended_method=method,
                    text_density=0.0,  # Could calculate actual density if needed
                )
            )

        # Determine overall content type
        if scanned_count == 0:
            overall_type = ContentType.NATIVE
            primary_method = ExtractionMethod.DOCLING
        elif native_count == 0:
            overall_type = ContentType.SCANNED
            primary_method = ExtractionMethod.VISION_OCR
        else:
            overall_type = ContentType.HYBRID
            # Use method that covers majority of pages
            primary_method = (
                ExtractionMethod.DOCLING
                if native_count >= scanned_count
                else ExtractionMethod.VISION_OCR
            )

        return DocumentAnalysis(
            file_path=file_path,
            file_type=FileType.PDF,
            content_type=overall_type,
            page_count=page_count,
            pages=pages,
            primary_method=primary_method,
        )

    def _analyze_image(self, file_path: Path) -> DocumentAnalysis:
        """Analyze an image file."""
        return DocumentAnalysis(
            file_path=file_path,
            file_type=FileType.IMAGE,
            content_type=ContentType.SCANNED,
            page_count=1,
            pages=[
                PageAnalysis(
                    page_number=1,
                    content_type=ContentType.SCANNED,
                    recommended_method=ExtractionMethod.VISION_OCR,
                    text_density=0.0,
                )
            ],
            primary_method=ExtractionMethod.VISION_OCR,
        )

    def _analyze_office(
        self, file_path: Path, file_type: FileType
    ) -> DocumentAnalysis:
        """Analyze an Office document."""
        return DocumentAnalysis(
            file_path=file_path,
            file_type=file_type,
            content_type=ContentType.NATIVE,
            page_count=1,  # Office docs treated as single unit
            pages=[
                PageAnalysis(
                    page_number=1,
                    content_type=ContentType.NATIVE,
                    recommended_method=ExtractionMethod.MARKITDOWN,
                    text_density=1.0,
                )
            ],
            primary_method=ExtractionMethod.MARKITDOWN,
        )

    def _analyze_unknown(self, file_path: Path) -> DocumentAnalysis:
        """Handle unknown file types."""
        # Try MarkItDown as fallback for unknown types
        return DocumentAnalysis(
            file_path=file_path,
            file_type=FileType.OTHER,
            content_type=ContentType.NATIVE,
            page_count=1,
            pages=[
                PageAnalysis(
                    page_number=1,
                    content_type=ContentType.NATIVE,
                    recommended_method=ExtractionMethod.MARKITDOWN,
                    text_density=0.0,
                )
            ],
            primary_method=ExtractionMethod.MARKITDOWN,
        )
