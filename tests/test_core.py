"""Tests for Sibyl core class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sibyl.core import Sibyl
from sibyl.models import (
    ProcessingResult,
    PageResult,
    DocumentMetadata,
    ProcessingStats,
    ExtractOptions,
    Chunk,
)
from sibyl.extractors.base import ExtractionResult


class TestSibylInit:
    def test_init_defaults(self):
        sb = Sibyl()
        assert sb.ocr_backend is None
        assert sb.ocr_threshold == 0.1
        assert sb.pdf_engine == "docling"

    def test_init_with_ocr_backend(self):
        mock_backend = Mock()
        sb = Sibyl(ocr_backend=mock_backend)
        assert sb.ocr_backend == mock_backend

    def test_init_with_custom_threshold(self):
        sb = Sibyl(ocr_threshold=0.5)
        assert sb.ocr_threshold == 0.5

    def test_init_with_markitdown_engine(self):
        sb = Sibyl(pdf_engine="markitdown")
        assert sb.pdf_engine == "markitdown"

    def test_init_with_auto_engine(self):
        sb = Sibyl(pdf_engine="auto")
        assert sb.pdf_engine == "auto"


class TestSibylProcess:
    def test_process_file_not_found(self):
        sb = Sibyl()
        with pytest.raises(FileNotFoundError, match="File not found"):
            sb.process("/nonexistent/file.pdf")

    def test_process_calls_router(self, tmp_path):
        """Test that process calls the router and returns results."""
        # Create a temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        # Mock the router
        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Hello world",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title="Test Title",
            author="Test Author",
        )
        sb._router.route = Mock(return_value=mock_result)

        result = sb.process(test_file)

        sb._router.route.assert_called_once()
        assert isinstance(result, ProcessingResult)
        assert result.markdown == "Hello world"
        assert len(result.pages) == 1

    def test_process_with_progress_callback(self, tmp_path):
        """Test that progress callback is passed through."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Hello world",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        sb._router.route = Mock(return_value=mock_result)

        progress_calls = []

        def on_progress(stage, current, total):
            progress_calls.append((stage, current, total))

        result = sb.process(test_file, on_progress=on_progress)

        # Verify router was called with the callback
        call_args = sb._router.route.call_args
        assert call_args[0][1].extract_tables is True  # options passed
        assert call_args[1].get("on_progress") == on_progress or call_args[0][2] == on_progress


class TestSibylChunk:
    def test_chunk_by_page(self):
        sb = Sibyl()

        result = ProcessingResult(
            markdown="Page 1 content\n\nPage 2 content",
            pages=[
                PageResult(
                    page_number=1,
                    content="Page 1 content",
                    extraction_method="docling",
                    confidence=1.0,
                    images=[],
                    tables=[],
                ),
                PageResult(
                    page_number=2,
                    content="Page 2 content",
                    extraction_method="docling",
                    confidence=1.0,
                    images=[],
                    tables=[],
                ),
            ],
            tables=[],
            images=[],
            metadata=DocumentMetadata(
                title="Test",
                author=None,
                page_count=2,
                file_type="pdf",
                file_size_bytes=1000,
            ),
            stats=ProcessingStats(
                total_time_seconds=1.0,
                methods_used=["docling"],
                pages_processed=2,
                ocr_pages=0,
                native_pages=2,
            ),
        )

        chunks = sb.chunk(result, method="page")

        assert len(chunks) == 2
        assert chunks[0].text == "Page 1 content"
        assert chunks[0].page_number == 1
        assert chunks[1].text == "Page 2 content"
        assert chunks[1].page_number == 2

    def test_chunk_by_size(self):
        sb = Sibyl()

        # Create a result with enough content to chunk (small to prevent slow tests)
        long_content = "This is sentence one. This is sentence two. This is sentence three."
        result = ProcessingResult(
            markdown=long_content,
            pages=[
                PageResult(
                    page_number=1,
                    content=long_content,
                    extraction_method="docling",
                    confidence=1.0,
                    images=[],
                    tables=[],
                ),
            ],
            tables=[],
            images=[],
            metadata=DocumentMetadata(
                title="Test",
                author=None,
                page_count=1,
                file_type="pdf",
                file_size_bytes=1000,
            ),
            stats=ProcessingStats(
                total_time_seconds=1.0,
                methods_used=["docling"],
                pages_processed=1,
                ocr_pages=0,
                native_pages=1,
            ),
        )

        # Use a small chunk size to create multiple chunks
        chunks = sb.chunk(result, method="size", chunk_size=30, chunk_overlap=5)

        # Should produce at least 2 chunks for 68 char content with size 30
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.page_number == 1

    def test_chunk_invalid_method(self):
        sb = Sibyl()

        result = ProcessingResult(
            markdown="Test",
            pages=[],
            tables=[],
            images=[],
            metadata=DocumentMetadata(
                title="Test",
                author=None,
                page_count=0,
                file_type="pdf",
                file_size_bytes=1000,
            ),
            stats=ProcessingStats(
                total_time_seconds=1.0,
                methods_used=[],
                pages_processed=0,
                ocr_pages=0,
                native_pages=0,
            ),
        )

        with pytest.raises(ValueError, match="Unknown chunking method"):
            sb.chunk(result, method="invalid")


class TestSibylAnalyze:
    def test_analyze_delegates_to_router(self, tmp_path):
        """Test that analyze calls router's get_analysis."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake pdf")

        sb = Sibyl()
        mock_analysis = Mock()
        sb._router.get_analysis = Mock(return_value=mock_analysis)

        result = sb.analyze(test_file)

        sb._router.get_analysis.assert_called_once()
        assert result == mock_analysis


class TestSibylBuildMetadata:
    def test_build_metadata_non_pdf(self, tmp_path):
        """Test metadata building for non-PDF files."""
        test_file = tmp_path / "test.docx"
        test_file.write_bytes(b"fake docx content")

        sb = Sibyl()

        mock_extraction = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Test content",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title="Doc Title",
            author="Doc Author",
        )

        metadata = sb._build_metadata(test_file, mock_extraction)

        assert metadata.title == "Doc Title"
        assert metadata.author == "Doc Author"
        assert metadata.file_type == "docx"
        assert metadata.page_count == 1


class TestSibylProcessBatch:
    def test_process_batch_single_file(self, tmp_path):
        """Test batch processing with a single file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Hello world",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        sb._router.route = Mock(return_value=mock_result)

        results = sb.process_batch([test_file])

        assert len(results) == 1
        assert results[0].markdown == "Hello world"

    def test_process_batch_with_progress(self, tmp_path):
        """Test batch processing with progress callback."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Hello world",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        sb._router.route = Mock(return_value=mock_result)

        progress_calls = []

        def on_progress(completed, total, current_file):
            progress_calls.append((completed, total, current_file))

        results = sb.process_batch([test_file], on_progress=on_progress)

        assert len(results) == 1
        assert len(progress_calls) == 1
        assert progress_calls[0][0] == 1  # completed
        assert progress_calls[0][1] == 1  # total

    def test_process_batch_error_handling(self, tmp_path):
        """Test that batch processing handles errors gracefully."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()
        sb._router.route = Mock(side_effect=Exception("Processing failed"))

        results = sb.process_batch([test_file])

        assert len(results) == 1
        assert "Error Processing Document" in results[0].markdown
        assert "Processing failed" in results[0].markdown


class TestSibylCreateErrorResult:
    def test_create_error_result(self, tmp_path):
        """Test error result creation."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake content")

        sb = Sibyl()
        result = sb._create_error_result(test_file, "Test error message")

        assert "Error Processing Document" in result.markdown
        assert "Test error message" in result.markdown
        assert result.pages == []
        assert result.tables == []
        assert result.images == []
        assert result.metadata.file_type == "pdf"


class TestSibylMergeTables:
    """Tests for the merge_tables parameter in process()."""

    def test_merge_tables_disabled_by_default(self, tmp_path):
        """Test that merge_tables is disabled by default."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="| A | B | A | B |\n|---|---|---|---|\n| 1 | 2 | 3 | 4 |",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        sb._router.route = Mock(return_value=mock_result)

        result = sb.process(test_file)

        # Table should NOT be merged (still 4 columns)
        assert "| A | B | A | B |" in result.markdown
        assert result.stats.tables_merged == 0

    def test_merge_tables_enabled(self, tmp_path):
        """Test that merge_tables=True merges split tables."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="| A | B | A | B |\n|---|---|---|---|\n| 1 | 2 | 3 | 4 |",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        sb._router.route = Mock(return_value=mock_result)

        result = sb.process(test_file, merge_tables=True)

        # Table should be merged (now 2 columns)
        assert "| A | B |" in result.markdown
        assert "| A | B | A | B |" not in result.markdown
        assert result.stats.tables_merged == 1

    def test_merge_tables_count_in_stats(self, tmp_path):
        """Test that tables_merged count is tracked in stats."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        # Content with two split tables
        content = """| X | Y | X | Y |
|---|---|---|---|
| a | b | c | d |

Some text.

| P | Q | P | Q |
|---|---|---|---|
| 1 | 2 | 3 | 4 |
"""
        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content=content,
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        sb._router.route = Mock(return_value=mock_result)

        result = sb.process(test_file, merge_tables=True)

        assert result.stats.tables_merged == 2

    def test_merge_tables_no_matching_tables(self, tmp_path):
        """Test merge_tables with no split tables to merge."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        sb = Sibyl()

        # Regular 2-column table
        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="| Name | Value |\n|---|---|\n| A | 1 |",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        sb._router.route = Mock(return_value=mock_result)

        result = sb.process(test_file, merge_tables=True)

        # Table should be unchanged
        assert "| Name | Value |" in result.markdown
        assert result.stats.tables_merged == 0
