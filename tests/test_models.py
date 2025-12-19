"""Tests for Sibyl Pydantic models."""

import pytest

from sibyl.models import (
    Chunk,
    DocumentMetadata,
    ExtractOptions,
    ImageResult,
    PageResult,
    ProcessingResult,
    ProcessingStats,
    TableResult,
)


class TestTableResult:
    def test_create_table(self):
        table = TableResult(
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            page_number=1,
            rows=2,
            columns=2,
        )
        assert table.page_number == 1
        assert table.rows == 2
        assert table.columns == 2

    def test_page_number_must_be_positive(self):
        with pytest.raises(ValueError):
            TableResult(markdown="test", page_number=0, rows=1, columns=1)


class TestImageResult:
    def test_create_image(self):
        image = ImageResult(
            path=None,
            ocr_text="Hello world",
            page_number=1,
            width=100,
            height=200,
        )
        assert image.ocr_text == "Hello world"
        assert image.width == 100


class TestPageResult:
    def test_create_page(self):
        page = PageResult(
            page_number=1,
            content="# Hello\n\nThis is content.",
            extraction_method="docling",
            confidence=0.95,
        )
        assert page.page_number == 1
        assert page.extraction_method == "docling"
        assert page.confidence == 0.95

    def test_extraction_method_validation(self):
        with pytest.raises(ValueError):
            PageResult(
                page_number=1,
                content="test",
                extraction_method="invalid_method",
            )


class TestDocumentMetadata:
    def test_create_metadata(self):
        meta = DocumentMetadata(
            title="Test Doc",
            author="Test Author",
            page_count=10,
            file_type="pdf",
        )
        assert meta.title == "Test Doc"
        assert meta.page_count == 10


class TestProcessingStats:
    def test_create_stats(self):
        stats = ProcessingStats(
            total_time_seconds=1.5,
            methods_used=["docling", "vision_ocr"],
            pages_processed=5,
            ocr_pages=2,
            native_pages=3,
        )
        assert stats.total_time_seconds == 1.5
        assert len(stats.methods_used) == 2


class TestChunk:
    def test_create_chunk(self):
        chunk = Chunk(
            text="This is chunk content.",
            page_number=1,
            chunk_index=0,
            metadata={"section": "intro"},
        )
        assert chunk.text == "This is chunk content."
        assert chunk.metadata["section"] == "intro"


class TestProcessingResult:
    def test_create_result(self):
        result = ProcessingResult(
            markdown="# Title\n\nContent here.",
            pages=[
                PageResult(
                    page_number=1,
                    content="# Title\n\nContent here.",
                    extraction_method="docling",
                )
            ],
            tables=[],
            images=[],
            metadata=DocumentMetadata(
                title="Test",
                page_count=1,
                file_type="pdf",
            ),
            stats=ProcessingStats(
                total_time_seconds=0.5,
                methods_used=["docling"],
                pages_processed=1,
            ),
        )
        assert result.markdown == "# Title\n\nContent here."
        assert len(result.pages) == 1

    def test_chunk_by_page(self):
        result = ProcessingResult(
            markdown="Page 1\n\nPage 2",
            pages=[
                PageResult(page_number=1, content="Page 1", extraction_method="docling"),
                PageResult(page_number=2, content="Page 2", extraction_method="docling"),
            ],
            tables=[],
            images=[],
            metadata=DocumentMetadata(page_count=2, file_type="pdf"),
            stats=ProcessingStats(
                total_time_seconds=0.5,
                methods_used=["docling"],
                pages_processed=2,
            ),
        )

        chunks = result.chunk(by_page=True)
        assert len(chunks) == 2
        assert chunks[0].text == "Page 1"
        assert chunks[0].page_number == 1
        assert chunks[1].text == "Page 2"
        assert chunks[1].page_number == 2


class TestExtractOptions:
    def test_default_options(self):
        opts = ExtractOptions()
        assert opts.extract_tables is True
        assert opts.extract_images is True
        assert opts.ocr_threshold == 0.8

    def test_custom_options(self):
        opts = ExtractOptions(
            extract_tables=False,
            ocr_threshold=0.5,
            pages=[1, 2, 3],
        )
        assert opts.extract_tables is False
        assert opts.ocr_threshold == 0.5
        assert opts.pages == [1, 2, 3]
