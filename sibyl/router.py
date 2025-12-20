"""Extraction router for directing documents to appropriate backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sibyl.analyzer import ContentAnalyzer, ContentType, DocumentAnalysis, ExtractionMethod
from sibyl.extractors.base import ExtractionResult
from sibyl.extractors.docling_ext import DoclingExtractor
from sibyl.extractors.markitdown_ext import MarkItDownExtractor
from sibyl.extractors.vision_ocr import VisionOCRExtractor
from sibyl.models import ExtractOptions, PageResult, PdfEngine, ProgressCallback
from sibyl.utils.text_quality import analyze_text_quality, clean_text

if TYPE_CHECKING:
    from sibyl.backends.base import OCRBackend

logger = logging.getLogger(__name__)


class ExtractionRouter:
    """Routes documents to appropriate extraction backends.

    Analyzes each document to determine the optimal extraction
    strategy, supporting hybrid extraction for documents with
    mixed native and scanned content.
    """

    def __init__(
        self,
        ocr_backend: "OCRBackend | None" = None,
        ocr_threshold: float = 0.1,
        pdf_engine: PdfEngine = "docling",
    ):
        """Initialize the router.

        Args:
            ocr_backend: OCR backend for vision-based extraction
            ocr_threshold: Text density threshold for OCR triggering
            pdf_engine: PDF extraction engine ("docling", "markitdown", or "auto")
        """
        self.ocr_backend = ocr_backend
        self.pdf_engine = pdf_engine
        self.analyzer = ContentAnalyzer(ocr_threshold=ocr_threshold)

        # Initialize extractors
        self._docling = DoclingExtractor()
        self._markitdown = MarkItDownExtractor()
        self._vision_ocr = VisionOCRExtractor(ocr_backend) if ocr_backend else None

    def route(
        self,
        file_path: Path,
        options: ExtractOptions,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Route a document to the appropriate extractor(s).

        Args:
            file_path: Path to document
            options: Extraction options
            on_progress: Optional progress callback (stage, current, total)

        Returns:
            ExtractionResult from the selected extractor(s)
        """
        file_path = Path(file_path)
        analysis = self.analyzer.analyze(file_path)

        # For PDFs, use the configured engine with fallback
        if file_path.suffix.lower() == ".pdf":
            return self._extract_pdf_with_fallback(file_path, analysis, options, on_progress)

        # Simple case: single extraction method for whole document
        if analysis.content_type != ContentType.HYBRID:
            return self._extract_simple(file_path, analysis, options, on_progress)

        # Complex case: hybrid document needs multiple extractors
        return self._extract_hybrid(file_path, analysis, options, on_progress)

    def _extract_pdf_with_fallback(
        self,
        file_path: Path,
        analysis: DocumentAnalysis,
        options: ExtractOptions,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Extract PDF with engine selection and fallback.

        Tries the configured engine first, falls back to alternate engine,
        then falls back to OCR if both fail.
        """
        # For scanned content, go directly to OCR
        if analysis.content_type == ContentType.SCANNED:
            if self._vision_ocr:
                return self._vision_ocr.extract(file_path, options, on_progress)
            raise ValueError("OCR backend required for scanned documents")

        # Determine primary and fallback extractors
        if self.pdf_engine == "markitdown":
            primary, fallback = self._markitdown, self._docling
            primary_name, fallback_name = "markitdown", "docling"
        elif self.pdf_engine == "docling":
            primary, fallback = self._docling, self._markitdown
            primary_name, fallback_name = "docling", "markitdown"
        else:  # auto - try docling first as it has better structure preservation
            primary, fallback = self._docling, self._markitdown
            primary_name, fallback_name = "docling", "markitdown"

        # Try primary extractor
        try:
            if on_progress:
                on_progress("extracting", 0, 1)
            result = primary.extract(file_path, options)
            if on_progress:
                on_progress("extracting", 1, 1)

            # Check if extraction produced meaningful content
            if self._is_valid_extraction(result):
                # Clean text and optionally re-extract poor quality pages
                return self._clean_and_validate_pages(result, file_path, options, on_progress)
            logger.warning(f"{primary_name} produced empty/invalid result, trying {fallback_name}")
        except Exception as e:
            logger.warning(f"{primary_name} extraction failed: {e}, trying {fallback_name}")

        # Try fallback extractor
        try:
            if on_progress:
                on_progress("extracting", 0, 1)
            result = fallback.extract(file_path, options)
            if on_progress:
                on_progress("extracting", 1, 1)

            if self._is_valid_extraction(result):
                # Clean text and optionally re-extract poor quality pages
                return self._clean_and_validate_pages(result, file_path, options, on_progress)
            logger.warning(f"{fallback_name} produced empty/invalid result")
        except Exception as e:
            logger.warning(f"{fallback_name} extraction failed: {e}")

        # Last resort: OCR if available
        if self._vision_ocr:
            logger.info("Falling back to OCR extraction")
            result = self._vision_ocr.extract(file_path, options, on_progress)
            # Clean OCR result too
            return self._clean_and_validate_pages(result, file_path, options, on_progress)

        raise ValueError(
            f"Both {primary_name} and {fallback_name} extraction failed. "
            "Consider providing an OCR backend for fallback."
        )

    def _is_valid_extraction(
        self,
        result: ExtractionResult,
        check_quality: bool = False,
        quality_threshold: float = 0.7,
    ) -> bool:
        """Check if extraction result has meaningful content.

        Args:
            result: Extraction result to validate
            check_quality: Whether to check text quality/coherence
            quality_threshold: Minimum quality score (0.0-1.0)

        Returns:
            True if extraction is valid
        """
        if not result.pages:
            return False

        # Check if there's actual content (not just whitespace)
        total_content = "".join(p.content for p in result.pages).strip()
        if len(total_content) <= 10:
            return False

        # Optionally check text quality
        if check_quality:
            quality = analyze_text_quality(total_content, threshold=quality_threshold)
            if not quality.is_acceptable:
                logger.warning(f"Text quality check failed: {quality}")
                return False

        return True

    def _clean_and_validate_pages(
        self,
        result: ExtractionResult,
        file_path: Path,
        options: ExtractOptions,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Clean extracted text and re-extract poor quality pages with OCR.

        Args:
            result: Initial extraction result
            file_path: Path to source document
            options: Extraction options
            on_progress: Progress callback

        Returns:
            Cleaned and validated extraction result
        """
        if not options.check_quality or not self._vision_ocr:
            # Just clean the text without OCR fallback
            cleaned_pages = []
            for page in result.pages:
                cleaned_content = clean_text(page.content)
                cleaned_pages.append(PageResult(
                    page_number=page.page_number,
                    content=cleaned_content,
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    images=page.images,
                    tables=page.tables,
                ))
            return ExtractionResult(
                pages=cleaned_pages,
                title=result.title,
                author=result.author,
            )

        # Check quality of each page and collect pages needing OCR
        good_pages: list[PageResult] = []
        pages_needing_ocr: list[int] = []

        for page in result.pages:
            # Clean the text first
            cleaned_content = clean_text(page.content)
            quality = analyze_text_quality(
                cleaned_content,
                threshold=options.quality_threshold,
            )

            if quality.is_acceptable:
                good_pages.append(PageResult(
                    page_number=page.page_number,
                    content=cleaned_content,
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    images=page.images,
                    tables=page.tables,
                ))
            else:
                logger.info(
                    f"Page {page.page_number} quality check failed ({quality.score:.2f}), "
                    f"will re-extract with OCR. Issues: {quality.issues}"
                )
                pages_needing_ocr.append(page.page_number)

        # Re-extract poor quality pages with OCR
        if pages_needing_ocr and self._vision_ocr:
            ocr_options = ExtractOptions(
                extract_tables=options.extract_tables,
                extract_images=options.extract_images,
                ocr_images=options.ocr_images,
                ocr_threshold=options.ocr_threshold,
                pages=pages_needing_ocr,
            )
            ocr_result = self._vision_ocr.extract(file_path, ocr_options, on_progress)

            # Clean OCR results too
            for page in ocr_result.pages:
                good_pages.append(PageResult(
                    page_number=page.page_number,
                    content=clean_text(page.content),
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    images=page.images,
                    tables=page.tables,
                ))

        # Sort pages by page number
        good_pages.sort(key=lambda p: p.page_number)

        return ExtractionResult(
            pages=good_pages,
            title=result.title,
            author=result.author,
        )

    def _extract_simple(
        self,
        file_path: Path,
        analysis: DocumentAnalysis,
        options: ExtractOptions,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Extract using a single method for the whole document."""
        method = analysis.primary_method
        extractor = self._get_extractor(method)

        if extractor is None:
            raise ValueError(
                f"No extractor available for method '{method.value}'. "
                "OCR backend may be required."
            )

        # Pass progress callback to vision OCR extractor
        if method == ExtractionMethod.VISION_OCR and self._vision_ocr:
            return self._vision_ocr.extract(file_path, options, on_progress)

        return extractor.extract(file_path, options)

    def _extract_hybrid(
        self,
        file_path: Path,
        analysis: DocumentAnalysis,
        options: ExtractOptions,
        on_progress: ProgressCallback | None = None,
    ) -> ExtractionResult:
        """Extract using multiple methods for hybrid documents.

        For PDFs with mixed native and scanned pages, we extract
        each set of pages with the appropriate method and merge.
        """
        # Get page lists for each method
        docling_pages = analysis.get_pages_for_method(ExtractionMethod.DOCLING)
        ocr_pages = analysis.get_pages_for_method(ExtractionMethod.VISION_OCR)

        all_pages: list[PageResult] = []
        title: str | None = None
        author: str | None = None

        # Extract native pages with Docling
        if docling_pages:
            if on_progress:
                on_progress("extracting", 0, len(docling_pages))

            native_options = ExtractOptions(
                extract_tables=options.extract_tables,
                extract_images=options.extract_images,
                ocr_images=False,  # Don't OCR images in native extraction
                ocr_threshold=options.ocr_threshold,
                pages=docling_pages,
            )
            native_result = self._docling.extract(file_path, native_options)
            all_pages.extend(native_result.pages)
            title = title or native_result.title
            author = author or native_result.author

            if on_progress:
                on_progress("extracting", len(docling_pages), len(docling_pages))

        # Extract scanned pages with Vision OCR
        if ocr_pages and self._vision_ocr:
            ocr_options = ExtractOptions(
                extract_tables=options.extract_tables,
                extract_images=options.extract_images,
                ocr_images=options.ocr_images,
                ocr_threshold=options.ocr_threshold,
                pages=ocr_pages,
            )
            ocr_result = self._vision_ocr.extract(file_path, ocr_options, on_progress)
            all_pages.extend(ocr_result.pages)

        # Sort pages by page number
        all_pages.sort(key=lambda p: p.page_number)

        return ExtractionResult(
            pages=all_pages,
            title=title,
            author=author,
        )

    def _get_extractor(self, method: ExtractionMethod):
        """Get the extractor for a given method."""
        if method == ExtractionMethod.DOCLING:
            return self._docling
        elif method == ExtractionMethod.MARKITDOWN:
            return self._markitdown
        elif method == ExtractionMethod.VISION_OCR:
            return self._vision_ocr
        else:
            return None

    def get_analysis(self, file_path: Path) -> DocumentAnalysis:
        """Get document analysis without extracting.

        Args:
            file_path: Path to document

        Returns:
            DocumentAnalysis with routing recommendations
        """
        return self.analyzer.analyze(file_path)
