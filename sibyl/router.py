"""Extraction router for directing documents to appropriate backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sibyl.analyzer import ContentAnalyzer, ContentType, DocumentAnalysis, ExtractionMethod
from sibyl.extractors.base import ExtractionResult
from sibyl.extractors.docling_ext import DoclingExtractor
from sibyl.extractors.markitdown_ext import MarkItDownExtractor
from sibyl.extractors.pymupdf_ext import PyMuPDFExtractor
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
        self._pymupdf = PyMuPDFExtractor()
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

        # If quality checking is enabled, do a quick per-page pre-scan
        # to identify pages that need OCR before full extraction
        pages_needing_ocr: list[int] = []
        if options.check_quality and self._vision_ocr:
            pages_needing_ocr = self._prescan_page_quality(
                file_path, options.quality_threshold
            )
            if pages_needing_ocr:
                print(f"  [Quality Pre-scan] {len(pages_needing_ocr)} pages need OCR: "
                      f"{pages_needing_ocr[:10]}{'...' if len(pages_needing_ocr) > 10 else ''}")

        # If we found pages needing OCR in pre-scan, extract them directly with OCR
        ocr_pages_result: list[PageResult] = []
        if pages_needing_ocr and self._vision_ocr:
            print(f"  [OCR] Extracting {len(pages_needing_ocr)} pages with quality issues...")
            ocr_options = ExtractOptions(
                extract_tables=options.extract_tables,
                extract_images=options.extract_images,
                ocr_images=options.ocr_images,
                ocr_threshold=options.ocr_threshold,
                pages=pages_needing_ocr,
            )
            ocr_result = self._vision_ocr.extract(file_path, ocr_options, on_progress)
            for page in ocr_result.pages:
                ocr_pages_result.append(PageResult(
                    page_number=page.page_number,
                    content=clean_text(page.content),
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    images=page.images,
                    tables=page.tables,
                ))

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
        native_result = None
        try:
            if on_progress:
                on_progress("extracting", 0, 1)
            native_result = primary.extract(file_path, options)
            if on_progress:
                on_progress("extracting", 1, 1)

            if not self._is_valid_extraction(native_result):
                logger.warning(f"{primary_name} produced empty/invalid result, trying {fallback_name}")
                native_result = None
        except Exception as e:
            logger.warning(f"{primary_name} extraction failed: {e}, trying {fallback_name}")

        # Try fallback extractor if primary failed
        if native_result is None:
            try:
                if on_progress:
                    on_progress("extracting", 0, 1)
                native_result = fallback.extract(file_path, options)
                if on_progress:
                    on_progress("extracting", 1, 1)

                if not self._is_valid_extraction(native_result):
                    logger.warning(f"{fallback_name} produced empty/invalid result")
                    native_result = None
            except Exception as e:
                logger.warning(f"{fallback_name} extraction failed: {e}")

        # If we have OCR pages but no native result, use full OCR
        if native_result is None and self._vision_ocr:
            logger.info("Falling back to full OCR extraction")
            result = self._vision_ocr.extract(file_path, options, on_progress)
            return self._clean_and_validate_pages(result, file_path, options, on_progress)
        elif native_result is None:
            raise ValueError(
                f"Both {primary_name} and {fallback_name} extraction failed. "
                "Consider providing an OCR backend for fallback."
            )

        # If we have OCR pages from pre-scan, we need to merge them with native
        if ocr_pages_result:
            # Clean native result
            cleaned_native = self._clean_and_validate_pages(
                native_result, file_path,
                ExtractOptions(**{**options.model_dump(), 'check_quality': False}),  # Skip re-check
                on_progress
            )

            # Merge: use OCR pages where we have them, native for the rest
            ocr_page_nums = {p.page_number for p in ocr_pages_result}
            merged_pages = list(ocr_pages_result)

            # For native result, we need to handle that Docling returns one big "page"
            # We'll skip adding native content for pages we OCR'd
            # Since Docling doesn't give per-page content, we just add the whole thing
            # and let the user know OCR was used for specific pages
            for page in cleaned_native.pages:
                if page.page_number not in ocr_page_nums:
                    merged_pages.append(page)

            merged_pages.sort(key=lambda p: p.page_number)

            return ExtractionResult(
                pages=merged_pages,
                title=native_result.title,
                author=native_result.author,
            )

        # No pre-scanned OCR pages, just clean and return native
        return self._clean_and_validate_pages(native_result, file_path, options, on_progress)

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

            # For large single-page output (like Docling's full document blob),
            # do surgical text replacement: keep structure, fix garbled text
            if len(cleaned_content) > 5000 and file_path.suffix.lower() == ".pdf":
                # Check quality on ORIGINAL content (before cleaning) to detect private_use_chars
                # Sample multiple sections to catch issues throughout the document
                original_content = page.content or ""
                has_quality_issues = False
                
                # Check in chunks throughout the document
                chunk_size = 10000
                for i in range(0, len(original_content), chunk_size):
                    chunk = original_content[i:i + chunk_size]
                    chunk_quality = analyze_text_quality(chunk, threshold=options.quality_threshold)
                    if not chunk_quality.is_acceptable:
                        print(f"  [Quality] Detected issue at offset {i}: score={chunk_quality.score:.2f}, "
                              f"issues={chunk_quality.issues}")
                        has_quality_issues = True
                        break
                
                if has_quality_issues:
                    # First try: decode private use characters (they're often ASCII + offset)
                    decoded_content, chars_decoded = self._decode_private_use_chars(original_content)
                    
                    if chars_decoded > 0:
                        print(f"  [Quality] Decoded {chars_decoded} decorative font characters")
                        cleaned_content = clean_text(decoded_content)
                    else:
                        # Second try: PyMuPDF text replacement
                        print(f"  [Quality] Attempting fix with PyMuPDF text...")
                        fixed_content = self._fix_garbled_text_with_pymupdf(file_path, original_content)
                        
                        if fixed_content != original_content:
                            print(f"  [Quality] Fixed garbled text while preserving structure")
                            cleaned_content = clean_text(fixed_content)
                        elif self._vision_ocr:
                            # Last resort: surgical OCR for unfixable regions
                            print(f"  [Quality] Attempting surgical OCR for remaining issues...")
                            fixed_content, regions_fixed = self._surgical_ocr_fix(
                                file_path, original_content, options, on_progress
                            )
                            if regions_fixed > 0:
                                print(f"  [Quality] Fixed {regions_fixed} regions via OCR")
                                cleaned_content = clean_text(fixed_content)
                            else:
                                cleaned_content = clean_text(original_content)
                        else:
                            cleaned_content = clean_text(original_content)
                
                good_pages.append(PageResult(
                    page_number=page.page_number,
                    content=cleaned_content,
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    images=page.images,
                    tables=page.tables,
                ))
            elif len(cleaned_content) > 5000:
                # Non-PDF large content, just keep it
                good_pages.append(PageResult(
                    page_number=page.page_number,
                    content=cleaned_content,
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    images=page.images,
                    tables=page.tables,
                ))
            else:
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
                    logger.warning(
                        f"Page {page.page_number} quality failed ({quality.score:.2f}): "
                        f"{quality.issues}"
                    )
                    print(f"  [Quality] Page {page.page_number}: score={quality.score:.2f}, "
                          f"issues={quality.issues}")
                    pages_needing_ocr.append(page.page_number)

        # Re-extract poor quality pages with vision OCR (preserves markdown structure)
        if pages_needing_ocr and file_path.suffix.lower() == ".pdf" and self._vision_ocr:
            # Dedupe and sort
            pages_to_extract = sorted(set(pages_needing_ocr))
            print(f"  [Quality] Re-extracting {len(pages_to_extract)} pages with vision OCR...")

            # Use vision OCR - it produces structured markdown output
            ocr_options = ExtractOptions(
                extract_tables=options.extract_tables,
                extract_images=options.extract_images,
                ocr_images=options.ocr_images,
                ocr_threshold=options.ocr_threshold,
                pages=pages_to_extract,
            )
            ocr_result = self._vision_ocr.extract(file_path, ocr_options, on_progress)

            for page in ocr_result.pages:
                good_pages.append(PageResult(
                    page_number=page.page_number,
                    content=clean_text(page.content),
                    extraction_method=page.extraction_method,
                    confidence=page.confidence,
                    images=page.images,
                    tables=page.tables,
                ))
            print(f"  [Quality] Vision OCR extraction complete for {len(ocr_result.pages)} pages")

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

    def _prescan_page_quality(
        self,
        file_path: Path,
        quality_threshold: float = 0.7,
    ) -> list[int]:
        """Quick per-page quality pre-scan using PyMuPDF.

        Extracts text from each page and checks for quality issues.
        This is faster than full Docling extraction and gives us
        per-page granularity for targeted OCR.

        Args:
            file_path: Path to PDF file
            quality_threshold: Minimum quality score

        Returns:
            List of 1-indexed page numbers that need OCR
        """
        from sibyl.utils.pdf import _get_fitz

        fitz = _get_fitz()
        pages_needing_ocr: list[int] = []

        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text")

                if len(text.strip()) < 50:
                    # Very little text - might be scanned or image-heavy
                    continue

                # Check quality
                quality = analyze_text_quality(text, threshold=quality_threshold)

                if not quality.is_acceptable:
                    page_number = page_num + 1  # 1-indexed
                    logger.info(
                        f"Page {page_number} pre-scan failed "
                        f"({quality.score:.2f}): {quality.issues}"
                    )
                    pages_needing_ocr.append(page_number)

        return pages_needing_ocr

    def _decode_private_use_chars(self, content: str) -> tuple[str, int]:
        """Decode private use characters that are ASCII + 0xF700 offset.
        
        Many decorative fonts map ASCII characters to the private use area
        by adding 0xF700 to each character code. This reverses that mapping.
        
        Args:
            content: Text content with potential private use characters
            
        Returns:
            Tuple of (decoded content, number of characters decoded)
        """
        result = []
        chars_decoded = 0
        
        for char in content:
            code = ord(char)
            # Check if it's in the private use range that maps from ASCII
            # Common pattern: ASCII + 0xF700 (e.g., 'e' (0x65) -> 0xF765)
            if 0xF720 <= code <= 0xF7FF:
                # Decode by subtracting 0xF700
                decoded_code = code - 0xF700
                if 0x20 <= decoded_code <= 0x7E:  # Printable ASCII
                    result.append(chr(decoded_code))
                    chars_decoded += 1
                else:
                    result.append(char)  # Keep as-is if not valid ASCII
            # Also check E000-E0FF range (another common offset pattern)
            elif 0xE020 <= code <= 0xE0FF:
                decoded_code = code - 0xE000
                if 0x20 <= decoded_code <= 0x7E:
                    result.append(chr(decoded_code))
                    chars_decoded += 1
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result), chars_decoded

    def _surgical_ocr_fix(
        self,
        file_path: Path,
        docling_content: str,
        options: ExtractOptions,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[str, int]:
        """Surgically fix garbled text regions using targeted OCR.
        
        Finds each garbled text sequence, locates it on the PDF page,
        crops that region, OCRs it, and replaces the garbled text.
        
        Args:
            file_path: Path to PDF file
            docling_content: Docling output with garbled text
            options: Extraction options
            on_progress: Progress callback
            
        Returns:
            Tuple of (fixed content, number of regions fixed)
        """
        import re
        import tempfile
        from sibyl.utils.pdf import _get_fitz
        
        fitz = _get_fitz()
        
        # Find all garbled sequences
        garbled_pattern = re.compile(r'[\uE000-\uF8FF][\uE000-\uF8FF\s]*[\uE000-\uF8FF]?')
        matches = list(garbled_pattern.finditer(docling_content))
        
        if not matches:
            return docling_content, 0
        
        print(f"  [Quality] Found {len(matches)} garbled text regions to fix")
        
        # Get per-page text and structure from PyMuPDF
        pdf = fitz.open(file_path)
        
        fixed_content = docling_content
        regions_fixed = 0
        
        # Process matches in reverse order to preserve string positions
        for match in reversed(matches):
            garbled = match.group()
            start, end = match.start(), match.end()
            
            # Get context around the garbled text
            context_before = docling_content[max(0, start-150):start]
            context_after = docling_content[end:end+150]
            
            # Find which page and location this corresponds to
            page_num, bbox = self._find_garbled_region_bbox(
                pdf, context_before, context_after
            )
            
            if page_num is None or bbox is None:
                continue
            
            # Crop the region and OCR it
            ocr_text = self._ocr_region(pdf, page_num, bbox, file_path)
            
            if ocr_text:
                # Replace garbled text with OCR result
                fixed_content = fixed_content[:start] + ocr_text + fixed_content[end:]
                regions_fixed += 1
        
        pdf.close()
        return fixed_content, regions_fixed
    
    def _find_garbled_region_bbox(
        self,
        pdf,
        context_before: str,
        context_after: str,
    ) -> tuple[int | None, tuple | None]:
        """Find the bounding box of a garbled region on a PDF page.
        
        Args:
            pdf: Open PyMuPDF document
            context_before: Text before the garbled region
            context_after: Text after the garbled region
            
        Returns:
            Tuple of (page_number (0-indexed), bounding_box) or (None, None)
        """
        import re
        
        # Extract key words from context
        words_before = re.findall(r'\b[a-zA-Z]{3,}\b', context_before)[-3:]
        words_after = re.findall(r'\b[a-zA-Z]{3,}\b', context_after)[:3]
        
        if not words_before and not words_after:
            return None, None
        
        # Search each page for the context
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            
            # Search for words before the garbled text
            before_rects = []
            for word in words_before:
                rects = page.search_for(word, quads=False)
                if rects:
                    before_rects.extend(rects)
            
            # Search for words after the garbled text
            after_rects = []
            for word in words_after:
                rects = page.search_for(word, quads=False)
                if rects:
                    after_rects.extend(rects)
            
            if not before_rects and not after_rects:
                continue
            
            # Estimate the garbled region - between before and after context
            if before_rects and after_rects:
                # Get rightmost "before" rect and leftmost "after" rect
                before_rect = max(before_rects, key=lambda r: r.x1)
                after_rect = min(after_rects, key=lambda r: r.x0)
                
                # Create bounding box for the garbled region
                # Expand slightly to capture full decorative text
                x0 = before_rect.x1
                x1 = after_rect.x0
                y0 = min(before_rect.y0, after_rect.y0) - 5
                y1 = max(before_rect.y1, after_rect.y1) + 5
                
                # Sanity check - region should be reasonable size
                if x1 > x0 and (x1 - x0) < 500 and (y1 - y0) < 100:
                    return page_num, (x0, y0, x1, y1)
            
            elif before_rects:
                # Only have "before" context - look to the right
                before_rect = max(before_rects, key=lambda r: r.x1)
                x0 = before_rect.x1
                x1 = min(x0 + 300, page.rect.width)  # Max 300px wide
                y0 = before_rect.y0 - 5
                y1 = before_rect.y1 + 5
                return page_num, (x0, y0, x1, y1)
            
            elif after_rects:
                # Only have "after" context - look to the left
                after_rect = min(after_rects, key=lambda r: r.x0)
                x1 = after_rect.x0
                x0 = max(x1 - 300, 0)  # Max 300px wide
                y0 = after_rect.y0 - 5
                y1 = after_rect.y1 + 5
                return page_num, (x0, y0, x1, y1)
        
        return None, None
    
    def _ocr_region(
        self,
        pdf,
        page_num: int,
        bbox: tuple,
        file_path: Path,
    ) -> str | None:
        """OCR a specific region of a PDF page.
        
        Args:
            pdf: Open PyMuPDF document
            page_num: Page number (0-indexed)
            bbox: Bounding box (x0, y0, x1, y1)
            file_path: Path to PDF (for context)
            
        Returns:
            OCR'd text or None
        """
        from sibyl.utils.pdf import _get_fitz
        from PIL import Image
        import io
        
        fitz = _get_fitz()
        
        try:
            page = pdf[page_num]
            
            # Expand bbox slightly and create clip rect
            x0, y0, x1, y1 = bbox
            clip = fitz.Rect(x0 - 10, y0 - 10, x1 + 10, y1 + 10)
            
            # Render the region as an image
            mat = fitz.Matrix(3, 3)  # 3x zoom for better OCR
            pix = page.get_pixmap(matrix=mat, clip=clip)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # OCR the image using the vision backend
            if self.ocr_backend:
                prompt = "Extract the exact text shown in this image. Output only the text, nothing else."
                result = self.ocr_backend.ocr_image(pil_image, prompt=prompt)
                
                if result and result.text and result.text.strip():
                    return result.text.strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to OCR region: {e}")
            return None

    def _find_pages_with_garbled_text(
        self,
        file_path: Path,
        docling_content: str | None = None,
    ) -> list[int]:
        """Find which PDF pages have decorative fonts that cause garbled text.
        
        Finds garbled sequences in Docling output and maps them to PDF pages
        by matching surrounding context text.
        
        Args:
            file_path: Path to PDF file
            docling_content: The Docling output with garbled text (optional)
            
        Returns:
            List of 1-indexed page numbers with garbled text
        """
        from sibyl.utils.pdf import _get_fitz
        import re
        
        if not docling_content:
            return []
        
        fitz = _get_fitz()
        problem_pages: set[int] = set()
        
        # Find all garbled sequences (private use chars)
        garbled_pattern = re.compile(r'[\uE000-\uF8FF]+')
        matches = list(garbled_pattern.finditer(docling_content))
        
        if not matches:
            return []
        
        # Get per-page text from PyMuPDF
        page_texts: list[tuple[int, str]] = []
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text").lower()
                page_texts.append((page_num + 1, text))
        
        # For each garbled sequence, find which page it's on
        for match in matches:
            start, end = match.start(), match.end()
            
            # Get surrounding context (text before and after the garbled part)
            context_before = docling_content[max(0, start-200):start]
            context_after = docling_content[end:end+200]
            
            # Extract recognizable words from context
            words_before = re.findall(r'\b[a-zA-Z]{4,}\b', context_before)[-5:]
            words_after = re.findall(r'\b[a-zA-Z]{4,}\b', context_after)[:5]
            
            # Search for these words in each page
            best_page = None
            best_score = 0
            
            for page_num, page_text in page_texts:
                score = 0
                for word in words_before + words_after:
                    if word.lower() in page_text:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_page = page_num
            
            if best_page and best_score >= 3:
                problem_pages.add(best_page)
        
        return sorted(problem_pages)

    def _fix_garbled_text_with_pymupdf(
        self,
        file_path: Path,
        docling_content: str,
    ) -> str:
        """Fix garbled text in Docling output using PyMuPDF's correct extraction.
        
        Finds sequences of private use characters (garbled decorative fonts)
        and replaces them with the corresponding correct text from PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            docling_content: Docling's markdown output with potential garbled text
            
        Returns:
            Fixed content with garbled text replaced
        """
        import re
        from sibyl.utils.pdf import _get_fitz
        
        fitz = _get_fitz()
        
        # Get full text from PyMuPDF (correct extraction)
        pymupdf_text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                pymupdf_text += page.get_text("text") + "\n"
        
        # Find garbled sequences (private use chars: U+E000-U+F8FF)
        # These appear as sequences of chars in the private use area
        garbled_pattern = r'[\uE000-\uF8FF][\uE000-\uF8FF\s]*[\uE000-\uF8FF]'
        
        fixed_content = docling_content
        matches = list(re.finditer(garbled_pattern, docling_content))
        
        if not matches:
            return docling_content
        
        # For each garbled sequence, find surrounding context and match in PyMuPDF
        for match in reversed(matches):  # Reverse to preserve positions during replacement
            garbled = match.group()
            start, end = match.start(), match.end()
            
            # Get context around the garbled text (words before/after)
            context_before = docling_content[max(0, start-100):start]
            context_after = docling_content[end:end+100]
            
            # Find last few words before garbled section
            words_before = re.findall(r'\b\w{3,}\b', context_before)[-3:]
            words_after = re.findall(r'\b\w{3,}\b', context_after)[:3]
            
            if not words_before and not words_after:
                continue
            
            # Search for this context in PyMuPDF text
            replacement = self._find_replacement_text(
                pymupdf_text, words_before, words_after, len(garbled)
            )
            
            if replacement:
                fixed_content = fixed_content[:start] + replacement + fixed_content[end:]
        
        return fixed_content
    
    def _find_replacement_text(
        self,
        pymupdf_text: str,
        words_before: list[str],
        words_after: list[str],
        approx_length: int,
    ) -> str | None:
        """Find the correct text that should replace garbled content.
        
        Args:
            pymupdf_text: Full correct text from PyMuPDF
            words_before: Context words before the garbled section
            words_after: Context words after the garbled section
            approx_length: Approximate length of garbled section
            
        Returns:
            Replacement text if found, None otherwise
        """
        import re
        
        # Build a pattern to find the context
        before_pattern = r'.*'.join(re.escape(w) for w in words_before) if words_before else ''
        after_pattern = r'.*'.join(re.escape(w) for w in words_after) if words_after else ''
        
        # Look for text between the context words
        if before_pattern and after_pattern:
            pattern = f'{before_pattern}(.{{1,{approx_length + 50}}}?){after_pattern}'
        elif before_pattern:
            pattern = f'{before_pattern}(.{{1,{approx_length + 50}}}?)$'
        elif after_pattern:
            pattern = f'^(.{{1,{approx_length + 50}}}?){after_pattern}'
        else:
            return None
        
        try:
            match = re.search(pattern, pymupdf_text, re.IGNORECASE | re.DOTALL)
            if match:
                replacement = match.group(1).strip()
                # Sanity check: replacement should be roughly similar length
                if 0.3 < len(replacement) / max(approx_length, 1) < 3:
                    return replacement
        except re.error:
            pass
        
        return None

    def _find_pages_for_sections(
        self,
        file_path: Path,
        bad_sections: list[str],
    ) -> list[int]:
        """Find which PDF pages contain the bad sections.
        
        Uses PyMuPDF to get per-page text, then searches for unique
        text snippets from each bad section to identify the page.
        
        Args:
            file_path: Path to PDF file
            bad_sections: List of sections with quality issues
            
        Returns:
            List of 1-indexed page numbers containing bad sections
        """
        from sibyl.utils.pdf import _get_fitz
        import re
        
        fitz = _get_fitz()
        bad_pages: set[int] = set()
        
        # Get per-page text from PyMuPDF
        page_texts: list[tuple[int, str]] = []
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text")
                page_texts.append((page_num + 1, text))  # 1-indexed
        
        for section in bad_sections:
            # Extract searchable text snippets from the section
            # Look for words that aren't garbled (no private use chars)
            words = re.findall(r'\b[A-Za-z]{4,}\b', section)
            
            if not words:
                continue
                
            # Search for these words in each page
            # The page with most matches is likely the source
            best_page = None
            best_score = 0
            
            for page_num, page_text in page_texts:
                score = 0
                for word in words[:20]:  # Check first 20 words
                    if word.lower() in page_text.lower():
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_page = page_num
            
            if best_page and best_score >= 3:  # At least 3 word matches
                bad_pages.add(best_page)
        
        return sorted(bad_pages)

    def _split_into_sections(self, content: str, max_section_size: int = 3000) -> list[str]:
        """Split content into sections for quality checking.

        Splits on markdown headings first, then by size if sections are too large.

        Args:
            content: Text content to split
            max_section_size: Maximum characters per section

        Returns:
            List of content sections
        """
        import re

        # Split on markdown headings (## or ###)
        heading_pattern = r"(?=^#{2,3}\s+)"
        sections = re.split(heading_pattern, content, flags=re.MULTILINE)

        # Remove empty sections and trim
        sections = [s.strip() for s in sections if s.strip()]

        # If sections are too large, split further by paragraphs
        final_sections = []
        for section in sections:
            if len(section) <= max_section_size:
                final_sections.append(section)
            else:
                # Split by double newlines (paragraphs)
                paragraphs = section.split("\n\n")
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) > max_section_size and current_chunk:
                        final_sections.append(current_chunk.strip())
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para
                if current_chunk.strip():
                    final_sections.append(current_chunk.strip())

        return final_sections if final_sections else [content]

    def get_analysis(self, file_path: Path) -> DocumentAnalysis:
        """Get document analysis without extracting.

        Args:
            file_path: Path to document

        Returns:
            DocumentAnalysis with routing recommendations
        """
        return self.analyzer.analyze(file_path)
