"""Page-based chunking for RAG pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sibyl.models import Chunk

if TYPE_CHECKING:
    from sibyl.models import ProcessingResult


def chunk_by_page(result: "ProcessingResult") -> list[Chunk]:
    """Create one chunk per page.

    This is the primary chunking strategy for Sibyl, as each page
    (especially from OCR) represents a natural content boundary.

    Args:
        result: ProcessingResult to chunk

    Returns:
        List of Chunk objects, one per page
    """
    chunks: list[Chunk] = []

    for i, page in enumerate(result.pages):
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
                    "table_count": len(page.tables),
                    "image_count": len(page.images),
                },
            )
        )

    return chunks


def chunk_by_size(
    result: "ProcessingResult",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Create chunks of approximately equal size.

    Splits content across page boundaries if needed.

    Args:
        result: ProcessingResult to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of Chunk objects
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in result.pages:
        content = page.content

        if len(content) <= chunk_size:
            # Page fits in single chunk
            chunks.append(
                Chunk(
                    text=content,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    metadata={
                        "extraction_method": page.extraction_method,
                        "is_complete_page": True,
                    },
                )
            )
            chunk_index += 1
        else:
            # Split page into multiple chunks
            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))

                # Try to break at sentence boundary
                if end < len(content):
                    # Look for sentence endings in last 20% of chunk
                    search_start = start + int(chunk_size * 0.8)
                    search_region = content[search_start:end]

                    for ending in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                        last_ending = search_region.rfind(ending)
                        if last_ending != -1:
                            end = search_start + last_ending + len(ending)
                            break

                chunk_text = content[start:end].strip()

                if chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            metadata={
                                "extraction_method": page.extraction_method,
                                "is_complete_page": False,
                                "chunk_start": start,
                                "chunk_end": end,
                            },
                        )
                    )
                    chunk_index += 1

                # Move start with overlap
                new_start = end - chunk_overlap
                if new_start < 0:
                    new_start = 0
                # Prevent infinite loop: ensure we make progress
                if new_start <= start or new_start >= len(content):
                    break
                start = new_start

    return chunks


def chunk_by_section(result: "ProcessingResult") -> list[Chunk]:
    """Create chunks based on markdown headings.

    Splits content at heading boundaries (# and ##).

    Args:
        result: ProcessingResult to chunk

    Returns:
        List of Chunk objects, one per section
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in result.pages:
        lines = page.content.split("\n")
        current_section: list[str] = []
        current_heading: str | None = None

        for line in lines:
            # Check if line is a heading
            is_heading = line.startswith("# ") or line.startswith("## ")

            if is_heading and current_section:
                # Save previous section
                section_text = "\n".join(current_section).strip()
                if section_text:
                    chunks.append(
                        Chunk(
                            text=section_text,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            metadata={
                                "extraction_method": page.extraction_method,
                                "section_heading": current_heading,
                            },
                        )
                    )
                    chunk_index += 1

                current_section = [line]
                current_heading = line.lstrip("#").strip()
            else:
                current_section.append(line)
                if is_heading:
                    current_heading = line.lstrip("#").strip()

        # Don't forget the last section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                chunks.append(
                    Chunk(
                        text=section_text,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        metadata={
                            "extraction_method": page.extraction_method,
                            "section_heading": current_heading,
                        },
                    )
                )
                chunk_index += 1

    return chunks
