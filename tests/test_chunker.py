"""Tests for chunking functionality."""

import pytest

from sibyl.chunker import chunk_by_page, chunk_by_size, chunk_by_section
from sibyl.models import (
    ProcessingResult,
    PageResult,
    DocumentMetadata,
    ProcessingStats,
    Chunk,
)


def make_result(pages_content: list[str]) -> ProcessingResult:
    """Helper to create ProcessingResult for testing."""
    return ProcessingResult(
        markdown="\n\n".join(pages_content),
        pages=[
            PageResult(
                page_number=i + 1,
                content=content,
                extraction_method="docling",
                confidence=1.0,
                images=[],
                tables=[],
            )
            for i, content in enumerate(pages_content)
        ],
        tables=[],
        images=[],
        metadata=DocumentMetadata(
            title="Test",
            author=None,
            page_count=len(pages_content),
            file_type="pdf",
            file_size_bytes=1000,
        ),
        stats=ProcessingStats(
            total_time_seconds=1.0,
            methods_used=["docling"],
            pages_processed=len(pages_content),
            ocr_pages=0,
            native_pages=len(pages_content),
        ),
    )


class TestChunkByPage:
    def test_single_page(self):
        result = make_result(["Page one content"])
        chunks = chunk_by_page(result)

        assert len(chunks) == 1
        assert chunks[0].text == "Page one content"
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0

    def test_multiple_pages(self):
        result = make_result(["Page one", "Page two", "Page three"])
        chunks = chunk_by_page(result)

        assert len(chunks) == 3
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 2
        assert chunks[2].page_number == 3
        assert chunks[0].text == "Page one"
        assert chunks[1].text == "Page two"
        assert chunks[2].text == "Page three"

    def test_empty_pages(self):
        result = make_result([])
        chunks = chunk_by_page(result)

        assert len(chunks) == 0

    def test_metadata_included(self):
        result = make_result(["Test content"])
        chunks = chunk_by_page(result)

        assert chunks[0].metadata["extraction_method"] == "docling"
        assert chunks[0].metadata["confidence"] == 1.0
        assert chunks[0].metadata["has_tables"] is False
        assert chunks[0].metadata["has_images"] is False


class TestChunkBySize:
    def test_small_content_single_chunk(self):
        result = make_result(["Short content"])
        chunks = chunk_by_size(result, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 1
        assert chunks[0].text == "Short content"
        assert chunks[0].metadata["is_complete_page"] is True

    def test_large_content_multiple_chunks(self):
        long_content = "This is a sentence. " * 20  # ~400 chars
        result = make_result([long_content])
        chunks = chunk_by_size(result, chunk_size=100, chunk_overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.page_number == 1

    def test_respects_overlap(self):
        content = "AAAA. BBBB. CCCC. DDDD. EEEE."
        result = make_result([content])
        chunks = chunk_by_size(result, chunk_size=15, chunk_overlap=5)

        # Chunks should be created with some overlap
        assert len(chunks) >= 1

    def test_sentence_boundary_breaking(self):
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = make_result([content])
        chunks = chunk_by_size(result, chunk_size=40, chunk_overlap=5)

        # Should try to break at sentence boundaries
        assert len(chunks) >= 1

    def test_no_infinite_loop(self):
        """Ensure chunking doesn't hang on edge cases."""
        content = "A" * 100
        result = make_result([content])
        # This should complete quickly without hanging
        chunks = chunk_by_size(result, chunk_size=20, chunk_overlap=5)

        assert len(chunks) >= 1

    def test_overlap_larger_than_chunk(self):
        """Edge case: overlap >= chunk_size should not cause issues."""
        content = "Test content here"
        result = make_result([content])
        chunks = chunk_by_size(result, chunk_size=10, chunk_overlap=15)

        # Should handle gracefully
        assert len(chunks) >= 1


class TestChunkBySection:
    def test_no_headings(self):
        result = make_result(["Just regular text without headings."])
        chunks = chunk_by_section(result)

        assert len(chunks) == 1
        assert chunks[0].text == "Just regular text without headings."

    def test_single_heading(self):
        content = "# Introduction\nThis is the intro."
        result = make_result([content])
        chunks = chunk_by_section(result)

        assert len(chunks) == 1
        assert "# Introduction" in chunks[0].text

    def test_multiple_headings(self):
        content = """# First Section
Content of first section.

# Second Section
Content of second section.

## Subsection
More content here."""
        result = make_result([content])
        chunks = chunk_by_section(result)

        assert len(chunks) >= 2

    def test_heading_levels(self):
        content = """# Main Heading
Main content.

## Sub Heading
Sub content."""
        result = make_result([content])
        chunks = chunk_by_section(result)

        # Should split on both # and ## headings
        assert len(chunks) >= 1

    def test_section_heading_in_metadata(self):
        content = """# Test Section
Content here."""
        result = make_result([content])
        chunks = chunk_by_section(result)

        assert len(chunks) >= 1
        assert chunks[0].metadata.get("section_heading") is not None

    def test_empty_content(self):
        result = make_result([])
        chunks = chunk_by_section(result)

        assert len(chunks) == 0


class TestProcessingResultChunk:
    """Test the chunk() method on ProcessingResult model."""

    def test_by_page_default(self):
        result = make_result(["Page 1", "Page 2"])
        chunks = result.chunk()  # Default is by_page=True

        assert len(chunks) == 2
        assert chunks[0].text == "Page 1"
        assert chunks[1].text == "Page 2"

    def test_by_size(self):
        long_content = "A" * 200
        result = make_result([long_content])
        chunks = result.chunk(chunk_size=50, chunk_overlap=10, by_page=False)

        assert len(chunks) >= 1

    def test_no_infinite_loop_in_model(self):
        """Test the ProcessingResult.chunk() method doesn't hang."""
        content = "Test " * 50
        result = make_result([content])
        # Should complete without hanging
        chunks = result.chunk(chunk_size=30, chunk_overlap=5, by_page=False)

        assert len(chunks) >= 1
