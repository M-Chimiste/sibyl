"""Pydantic models for Sibyl document processing results."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

# Progress callback type: (stage, current, total) -> None
# Stages: "analyzing", "extracting", "ocr", "describing_images"
ProgressCallback = Callable[[str, int, int], None]

# PDF extraction engine type
PdfEngine = Literal["docling", "markitdown", "auto"]


class TableResult(BaseModel):
    """Represents an extracted table from a document."""

    markdown: str = Field(description="Table content as markdown")
    page_number: int = Field(ge=1, description="Page number where table was found")
    rows: int = Field(ge=0, description="Number of rows in the table")
    columns: int = Field(ge=0, description="Number of columns in the table")


class ImageResult(BaseModel):
    """Represents an extracted or processed image from a document."""

    path: Path | None = Field(default=None, description="Path to extracted image file")
    ocr_text: str | None = Field(default=None, description="OCR text extracted from image")
    page_number: int = Field(ge=1, description="Page number where image was found")
    width: int | None = Field(default=None, description="Image width in pixels")
    height: int | None = Field(default=None, description="Image height in pixels")


class PageResult(BaseModel):
    """Represents extraction results for a single page."""

    page_number: int = Field(ge=1, description="Page number (1-indexed)")
    content: str = Field(description="Markdown content extracted from page")
    extraction_method: Literal["docling", "vision_ocr", "markitdown"] = Field(
        description="Method used to extract this page"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Extraction confidence score"
    )
    images: list[ImageResult] = Field(default_factory=list)
    tables: list[TableResult] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata extracted from the document."""

    title: str | None = Field(default=None, description="Document title")
    author: str | None = Field(default=None, description="Document author")
    page_count: int = Field(ge=0, description="Total number of pages")
    file_type: str = Field(description="File type (pdf, docx, etc.)")
    file_size_bytes: int | None = Field(default=None, description="File size in bytes")


class ProcessingStats(BaseModel):
    """Statistics about the document processing."""

    total_time_seconds: float = Field(ge=0.0, description="Total processing time")
    methods_used: list[str] = Field(default_factory=list, description="Extraction methods used")
    pages_processed: int = Field(ge=0, description="Number of pages processed")
    ocr_pages: int = Field(default=0, description="Number of pages processed with OCR")
    native_pages: int = Field(default=0, description="Number of pages with native text extraction")
    tables_merged: int = Field(default=0, description="Number of horizontally-split tables merged")
    quality_reextracted_pages: int = Field(
        default=0, description="Pages re-extracted due to quality issues"
    )


class Chunk(BaseModel):
    """A chunk of text for RAG pipelines."""

    text: str = Field(description="Chunk text content")
    page_number: int = Field(ge=1, description="Source page number")
    chunk_index: int = Field(ge=0, description="Index of this chunk within the document")
    metadata: dict = Field(default_factory=dict, description="Additional chunk metadata")


class ProcessingResult(BaseModel):
    """Complete result of document processing."""

    markdown: str = Field(description="Complete document as markdown")
    pages: list[PageResult] = Field(default_factory=list, description="Per-page results")
    tables: list[TableResult] = Field(default_factory=list, description="All extracted tables")
    images: list[ImageResult] = Field(default_factory=list, description="All extracted images")
    metadata: DocumentMetadata = Field(description="Document metadata")
    stats: ProcessingStats = Field(description="Processing statistics")

    def chunk(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        by_page: bool = True,
    ) -> list[Chunk]:
        """Generate chunks for RAG pipelines.

        Args:
            chunk_size: Target size per chunk in characters (used when splitting large pages)
            chunk_overlap: Overlap between chunks in characters
            by_page: If True, each page becomes a chunk (default). If False, split by size.

        Returns:
            List of Chunk objects
        """
        chunks: list[Chunk] = []

        if by_page:
            # Page-based chunking: each page is a chunk
            for i, page in enumerate(self.pages):
                chunks.append(
                    Chunk(
                        text=page.content,
                        page_number=page.page_number,
                        chunk_index=i,
                        metadata={
                            "extraction_method": page.extraction_method,
                            "confidence": page.confidence,
                            "has_tables": len(page.tables) > 0,
                            "has_images": len(page.images) > 0,
                        },
                    )
                )
        else:
            # Size-based chunking with overlap
            chunk_index = 0
            for page in self.pages:
                content = page.content
                if len(content) <= chunk_size:
                    chunks.append(
                        Chunk(
                            text=content,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            metadata={"extraction_method": page.extraction_method},
                        )
                    )
                    chunk_index += 1
                else:
                    # Split large pages
                    start = 0
                    while start < len(content):
                        end = min(start + chunk_size, len(content))
                        chunk_text = content[start:end]
                        chunks.append(
                            Chunk(
                                text=chunk_text,
                                page_number=page.page_number,
                                chunk_index=chunk_index,
                                metadata={
                                    "extraction_method": page.extraction_method,
                                    "is_partial": True,
                                },
                            )
                        )
                        chunk_index += 1
                        # Calculate next start with overlap, prevent infinite loop
                        new_start = end - chunk_overlap
                        if new_start <= start or new_start >= len(content):
                            break
                        start = new_start

        return chunks


class ExtractOptions(BaseModel):
    """Options for document extraction."""

    extract_tables: bool = Field(default=True, description="Extract tables from document")
    extract_images: bool = Field(default=True, description="Extract embedded images")
    ocr_images: bool = Field(default=True, description="OCR text in extracted images")
    describe_images: bool = Field(
        default=False,
        description="Use VLM to describe images and replace <!-- image --> placeholders",
    )
    merge_split_tables: bool = Field(
        default=False,
        description="Merge horizontally-split tables (e.g., 4-column tables with repeated headers into 2-column)",
    )
    check_quality: bool = Field(
        default=False,
        description="Check text quality and re-extract poor pages with OCR",
    )
    quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for text (below triggers OCR re-extraction)",
    )
    ocr_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum text density for native extraction (below triggers OCR)",
    )
    pages: list[int] | None = Field(
        default=None, description="Specific pages to process (None = all)"
    )
