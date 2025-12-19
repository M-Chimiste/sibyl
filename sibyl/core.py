"""Core Sibyl class - main entry point for document processing."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from sibyl.chunker import chunk_by_page, chunk_by_section, chunk_by_size
from sibyl.models import (
    Chunk,
    DocumentMetadata,
    ExtractOptions,
    PdfEngine,
    ProcessingResult,
    ProcessingStats,
    ProgressCallback,
)
from sibyl.router import ExtractionRouter
from sibyl.utils.pdf import get_page_count, get_pdf_metadata

if TYPE_CHECKING:
    from PIL import Image

    from sibyl.backends.base import OCRBackend


class Sibyl:
    """Main class for document-to-markdown conversion.

    Sibyl provides a simple API for extracting structured markdown
    from various document formats, with automatic routing to the
    optimal extraction backend.

    Example:
        >>> from sibyl import Sibyl
        >>> from sibyl.backends import OllamaBackend
        >>>
        >>> sb = Sibyl(ocr_backend=OllamaBackend())
        >>> result = sb.process("document.pdf")
        >>> print(result.markdown)
    """

    # Default prompt for image description
    IMAGE_DESCRIPTION_PROMPT = """Describe this image concisely in 1-2 sentences.
Focus on what the image depicts and any important visual elements.
Do not include any preamble, just provide the description."""

    def __init__(
        self,
        ocr_backend: "OCRBackend | None" = None,
        ocr_threshold: float = 0.1,
        pdf_engine: PdfEngine = "docling",
    ):
        """Initialize Sibyl.

        Args:
            ocr_backend: Backend for vision-based OCR (required for scanned docs)
            ocr_threshold: Text density threshold below which OCR is triggered
            pdf_engine: PDF extraction engine ("docling", "markitdown", or "auto").
                - "docling": Use Docling for better structure preservation (default)
                - "markitdown": Use MarkItDown for faster extraction
                - "auto": Try docling first, fall back to markitdown
        """
        self.ocr_backend = ocr_backend
        self.ocr_threshold = ocr_threshold
        self.pdf_engine = pdf_engine
        self._router = ExtractionRouter(
            ocr_backend=ocr_backend,
            ocr_threshold=ocr_threshold,
            pdf_engine=pdf_engine,
        )

    def process(
        self,
        file_path: str | Path,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_images: bool = True,
        describe_images: bool = False,
        ocr_threshold: float | None = None,
        pages: list[int] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Process a document and extract markdown content.

        Args:
            file_path: Path to the document file
            extract_tables: Whether to extract tables
            extract_images: Whether to extract embedded images
            ocr_images: Whether to OCR text in images
            describe_images: Use VLM to describe images (replaces <!-- image --> tags)
            ocr_threshold: Override default OCR threshold
            pages: Specific pages to process (None = all)
            on_progress: Optional progress callback (stage, current, total).
                Stages: "analyzing", "extracting", "ocr", "describing_images"

        Returns:
            ProcessingResult with extracted content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        start_time = time.time()

        # Build options
        options = ExtractOptions(
            extract_tables=extract_tables,
            extract_images=extract_images,
            ocr_images=ocr_images,
            describe_images=describe_images,
            ocr_threshold=ocr_threshold or self.ocr_threshold,
            pages=pages,
        )

        # Route and extract
        extraction_result = self._router.route(file_path, options, on_progress)

        # Build metadata
        metadata = self._build_metadata(file_path, extraction_result)

        # Collect all tables and images from pages
        all_tables = []
        all_images = []
        methods_used = set()

        for page in extraction_result.pages:
            all_tables.extend(page.tables)
            all_images.extend(page.images)
            methods_used.add(page.extraction_method)

        # Build stats
        ocr_count = sum(
            1 for p in extraction_result.pages if p.extraction_method == "vision_ocr"
        )
        native_count = len(extraction_result.pages) - ocr_count

        stats = ProcessingStats(
            total_time_seconds=time.time() - start_time,
            methods_used=list(methods_used),
            pages_processed=len(extraction_result.pages),
            ocr_pages=ocr_count,
            native_pages=native_count,
        )

        # Combine all page content into full markdown
        markdown = "\n\n".join(page.content for page in extraction_result.pages)

        # Describe images if requested
        if describe_images and self.ocr_backend is not None:
            markdown = self._describe_images_in_markdown(file_path, markdown, on_progress)

        return ProcessingResult(
            markdown=markdown,
            pages=extraction_result.pages,
            tables=all_tables,
            images=all_images,
            metadata=metadata,
            stats=stats,
        )

    def _describe_images_in_markdown(
        self,
        file_path: Path,
        markdown: str,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        """Replace <!-- image --> placeholders with image descriptions.

        Args:
            file_path: Path to the source document
            markdown: Markdown content with image placeholders
            on_progress: Optional progress callback

        Returns:
            Markdown with image descriptions
        """
        if self.ocr_backend is None:
            return markdown

        # Find all image placeholders
        pattern = r"<!-- image -->"
        matches = list(re.finditer(pattern, markdown))

        if not matches:
            return markdown

        # Extract images from the document
        images = self._extract_images_for_description(file_path)

        # If we have fewer images than placeholders, we'll describe what we can
        # and leave remaining placeholders as-is
        total_images = min(len(images), len(matches))
        descriptions = []
        for i, img in enumerate(images[: len(matches)]):
            if on_progress:
                on_progress("describing_images", i, total_images)

            try:
                result = self.ocr_backend.ocr_image(
                    img, prompt=self.IMAGE_DESCRIPTION_PROMPT
                )
                description = result.text.strip()
                # Clean up the description
                description = description.replace("\n", " ").strip()
                descriptions.append(f"<!-- image: {description} -->")
            except Exception:
                descriptions.append("<!-- image: [description unavailable] -->")

        # Pad with original placeholders if we have fewer images than matches
        while len(descriptions) < len(matches):
            descriptions.append("<!-- image -->")

        # Replace placeholders in reverse order to preserve positions
        result_markdown = markdown
        for i, match in enumerate(reversed(matches)):
            desc_idx = len(matches) - 1 - i
            result_markdown = (
                result_markdown[: match.start()]
                + descriptions[desc_idx]
                + result_markdown[match.end() :]
            )

        return result_markdown

    def _extract_images_for_description(
        self,
        file_path: Path,
    ) -> list["Image.Image"]:
        """Extract images from a document for description.

        Args:
            file_path: Path to document

        Returns:
            List of PIL Images
        """
        from PIL import Image as PILImage

        images: list[PILImage.Image] = []
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            images = self._extract_images_from_pdf(file_path)
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}:
            # Single image file
            images = [PILImage.open(file_path)]

        return images

    def _extract_images_from_pdf(
        self,
        file_path: Path,
    ) -> list["Image.Image"]:
        """Extract all images from a PDF in document order.

        Args:
            file_path: Path to PDF

        Returns:
            List of PIL Images
        """
        import io

        import fitz
        from PIL import Image as PILImage

        images: list[PILImage.Image] = []

        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                image_list = page.get_images(full=True)

                for img_info in image_list:
                    xref = img_info[0]
                    try:
                        base_image = pdf.extract_image(xref)
                        if base_image:
                            image_bytes = base_image["image"]
                            pil_image = PILImage.open(io.BytesIO(image_bytes))
                            # Convert to RGB if necessary
                            if pil_image.mode not in ("RGB", "RGBA"):
                                pil_image = pil_image.convert("RGB")
                            images.append(pil_image)
                    except Exception:
                        # Skip images that can't be extracted
                        pass

        return images

    def process_batch(
        self,
        file_paths: list[str | Path],
        workers: int = 4,
        on_progress: Callable[[int, int, Path], None] | None = None,
        **kwargs,
    ) -> list[ProcessingResult]:
        """Process multiple documents in parallel.

        Args:
            file_paths: List of paths to process
            workers: Number of parallel workers
            on_progress: Callback(completed, total, current_file)
            **kwargs: Additional arguments passed to process()

        Returns:
            List of ProcessingResult objects in input order
        """
        file_paths = [Path(p) for p in file_paths]
        total = len(file_paths)
        results: dict[int, ProcessingResult] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process, path, **kwargs): i
                for i, path in enumerate(file_paths)
            }

            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    # Create error result
                    results[index] = self._create_error_result(
                        file_paths[index], str(e)
                    )

                completed += 1
                if on_progress:
                    on_progress(completed, total, file_paths[index])

        # Return in original order
        return [results[i] for i in range(total)]

    def chunk(
        self,
        result: ProcessingResult,
        method: str = "page",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[Chunk]:
        """Generate chunks from a processing result.

        Args:
            result: ProcessingResult to chunk
            method: Chunking method ("page", "size", or "section")
            chunk_size: Target chunk size for size-based chunking
            chunk_overlap: Overlap for size-based chunking

        Returns:
            List of Chunk objects
        """
        if method == "page":
            return chunk_by_page(result)
        elif method == "size":
            return chunk_by_size(result, chunk_size, chunk_overlap)
        elif method == "section":
            return chunk_by_section(result)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

    def analyze(self, file_path: str | Path):
        """Analyze a document without extracting content.

        Args:
            file_path: Path to document

        Returns:
            DocumentAnalysis with routing recommendations
        """
        return self._router.get_analysis(Path(file_path))

    def _build_metadata(self, file_path: Path, extraction_result) -> DocumentMetadata:
        """Build document metadata."""
        file_type = file_path.suffix.lower().lstrip(".")

        # Get page count
        if file_type == "pdf":
            page_count = get_page_count(file_path)
            pdf_meta = get_pdf_metadata(file_path)
            title = extraction_result.title or pdf_meta.get("title")
            author = extraction_result.author or pdf_meta.get("author")
        else:
            page_count = len(extraction_result.pages)
            title = extraction_result.title
            author = extraction_result.author

        return DocumentMetadata(
            title=title,
            author=author,
            page_count=page_count,
            file_type=file_type,
            file_size_bytes=file_path.stat().st_size,
        )

    def _create_error_result(self, file_path: Path, error: str) -> ProcessingResult:
        """Create a result object for failed processing."""
        return ProcessingResult(
            markdown=f"# Error Processing Document\n\nFailed to process: {error}",
            pages=[],
            tables=[],
            images=[],
            metadata=DocumentMetadata(
                title=None,
                author=None,
                page_count=0,
                file_type=file_path.suffix.lower().lstrip("."),
                file_size_bytes=file_path.stat().st_size if file_path.exists() else None,
            ),
            stats=ProcessingStats(
                total_time_seconds=0.0,
                methods_used=[],
                pages_processed=0,
                ocr_pages=0,
                native_pages=0,
            ),
        )
