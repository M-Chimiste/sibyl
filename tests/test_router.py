"""Tests for extraction router."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from sibyl.router import ExtractionRouter
from sibyl.analyzer import ContentType, ExtractionMethod, FileType, DocumentAnalysis, PageAnalysis
from sibyl.extractors.base import ExtractionResult
from sibyl.models import ExtractOptions, PageResult


class TestExtractionRouter:
    def test_init_without_ocr_backend(self):
        router = ExtractionRouter()
        assert router.ocr_backend is None
        assert router.pdf_engine == "docling"
        assert router._vision_ocr is None

    def test_init_with_ocr_backend(self):
        mock_backend = Mock()
        router = ExtractionRouter(ocr_backend=mock_backend)
        assert router.ocr_backend == mock_backend
        assert router._vision_ocr is not None

    def test_init_with_markitdown_engine(self):
        router = ExtractionRouter(pdf_engine="markitdown")
        assert router.pdf_engine == "markitdown"

    def test_init_with_auto_engine(self):
        router = ExtractionRouter(pdf_engine="auto")
        assert router.pdf_engine == "auto"


class TestRouterPdfEngineSelection:
    def test_docling_engine_uses_docling_first(self):
        """When pdf_engine='docling', docling should be tried first."""
        router = ExtractionRouter(pdf_engine="docling")

        # Mock a valid extraction result
        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Test content here",
                extraction_method="docling",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        router._docling = Mock()
        router._docling.extract = Mock(return_value=mock_result)

        # Mock analyzer
        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.NATIVE,
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.DOCLING,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()
        result = router.route(Path("test.pdf"), options)

        router._docling.extract.assert_called_once()
        assert result == mock_result

    def test_markitdown_engine_uses_markitdown_first(self):
        """When pdf_engine='markitdown', markitdown should be tried first."""
        router = ExtractionRouter(pdf_engine="markitdown")

        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Test content here",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        router._markitdown = Mock()
        router._markitdown.extract = Mock(return_value=mock_result)

        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.NATIVE,
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.DOCLING,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()
        result = router.route(Path("test.pdf"), options)

        router._markitdown.extract.assert_called_once()


class TestRouterFallback:
    def test_fallback_to_alternate_engine_on_failure(self):
        """If primary engine fails, fallback should be tried."""
        router = ExtractionRouter(pdf_engine="docling")

        # Make docling raise an exception
        router._docling.extract = Mock(side_effect=Exception("Docling failed"))

        # Make markitdown succeed
        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Fallback content",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        router._markitdown.extract = Mock(return_value=mock_result)

        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.NATIVE,
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.DOCLING,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()
        result = router.route(Path("test.pdf"), options)

        # Both should have been called
        router._docling.extract.assert_called_once()
        router._markitdown.extract.assert_called_once()
        assert result == mock_result

    def test_fallback_to_alternate_on_empty_result(self):
        """If primary engine returns empty result, fallback should be tried."""
        router = ExtractionRouter(pdf_engine="docling")

        # Make docling return empty result
        empty_result = ExtractionResult(pages=[], title=None, author=None)
        router._docling.extract = Mock(return_value=empty_result)

        # Make markitdown succeed
        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="Fallback content",
                extraction_method="markitdown",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        router._markitdown.extract = Mock(return_value=mock_result)

        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.NATIVE,
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.DOCLING,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()
        result = router.route(Path("test.pdf"), options)

        assert result == mock_result

    def test_fallback_to_ocr_when_both_engines_fail(self):
        """If both engines fail and OCR backend is available, use OCR."""
        mock_backend = Mock()
        router = ExtractionRouter(ocr_backend=mock_backend, pdf_engine="docling")

        # Both extractors fail
        router._docling.extract = Mock(side_effect=Exception("Docling failed"))
        router._markitdown.extract = Mock(side_effect=Exception("MarkItDown failed"))

        # OCR succeeds
        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="OCR content",
                extraction_method="vision_ocr",
                confidence=0.9,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        router._vision_ocr.extract = Mock(return_value=mock_result)

        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.NATIVE,
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.DOCLING,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()
        result = router.route(Path("test.pdf"), options)

        router._vision_ocr.extract.assert_called_once()
        assert result == mock_result

    def test_raises_when_all_methods_fail_no_ocr(self):
        """If all methods fail and no OCR backend, raise error."""
        router = ExtractionRouter(pdf_engine="docling")  # No OCR backend

        router._docling.extract = Mock(side_effect=Exception("Docling failed"))
        router._markitdown.extract = Mock(side_effect=Exception("MarkItDown failed"))

        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.NATIVE,
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.DOCLING,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()

        with pytest.raises(ValueError, match="extraction failed"):
            router.route(Path("test.pdf"), options)


class TestRouterScannedContent:
    def test_scanned_pdf_goes_directly_to_ocr(self):
        """Scanned PDFs should bypass native extractors and go to OCR."""
        mock_backend = Mock()
        router = ExtractionRouter(ocr_backend=mock_backend, pdf_engine="docling")

        mock_result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="OCR content",
                extraction_method="vision_ocr",
                confidence=0.9,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )
        router._vision_ocr.extract = Mock(return_value=mock_result)

        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.SCANNED,  # Scanned content
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.VISION_OCR,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()
        result = router.route(Path("test.pdf"), options)

        # Should go directly to OCR, not try docling/markitdown
        router._vision_ocr.extract.assert_called_once()

    def test_scanned_pdf_raises_without_ocr_backend(self):
        """Scanned PDFs without OCR backend should raise error."""
        router = ExtractionRouter(pdf_engine="docling")  # No OCR backend

        mock_analysis = DocumentAnalysis(
            file_path=Path("test.pdf"),
            file_type=FileType.PDF,
            content_type=ContentType.SCANNED,
            page_count=1,
            pages=[],
            primary_method=ExtractionMethod.VISION_OCR,
        )
        router.analyzer.analyze = Mock(return_value=mock_analysis)

        options = ExtractOptions()

        with pytest.raises(ValueError, match="OCR backend required"):
            router.route(Path("test.pdf"), options)


class TestDecodePrivateUseChars:
    """Tests for private use character decoding."""

    def test_decode_f700_offset_chars(self):
        """Test decoding ASCII + 0xF700 pattern."""
        router = ExtractionRouter()
        
        # 'Yer late, elf!' encoded as ASCII + 0xF700
        encoded = "'Y\uf765\uf772 \uf76c\uf761\uf774\uf765, \uf765\uf76c\uf766!'"
        decoded, count = router._decode_private_use_chars(encoded)
        
        assert decoded == "'Yer late, elf!'"
        assert count == 9  # 9 private use chars decoded

    def test_decode_longer_text(self):
        """Test decoding longer decorative font text."""
        router = ExtractionRouter()
        
        # 'I have never imagined' encoded
        encoded = "'\uf749 \uf768\uf761\uf776\uf765 \uf76e\uf765\uf776\uf765\uf772 \uf769\uf76d\uf761\uf767\uf769\uf76e\uf765\uf764'"
        decoded, count = router._decode_private_use_chars(encoded)
        
        assert "I have never imagined" in decoded
        assert count > 0

    def test_decode_preserves_normal_text(self):
        """Test that normal ASCII text is preserved."""
        router = ExtractionRouter()
        
        normal = "This is normal text without any special characters."
        decoded, count = router._decode_private_use_chars(normal)
        
        assert decoded == normal
        assert count == 0

    def test_decode_mixed_content(self):
        """Test decoding mixed normal and encoded content."""
        router = ExtractionRouter()
        
        # Mix of normal text and encoded chars
        mixed = "Normal \uf774\uf765\uf778\uf774 here"  # "text" encoded
        decoded, count = router._decode_private_use_chars(mixed)
        
        assert decoded == "Normal text here"
        assert count == 4

    def test_decode_e000_offset_chars(self):
        """Test decoding ASCII + 0xE000 pattern."""
        router = ExtractionRouter()
        
        # Test E0xx range (another common offset)
        encoded = "\ue054\ue065\ue073\ue074"  # "Test" with 0xE000 offset
        decoded, count = router._decode_private_use_chars(encoded)
        
        assert decoded == "Test"
        assert count == 4

    def test_decode_invalid_range_preserved(self):
        """Test that chars outside valid ASCII range are preserved."""
        router = ExtractionRouter()
        
        # Char that would decode to non-printable ASCII
        text = "Test\uf710 text"  # Would decode to 0x10 (control char)
        decoded, count = router._decode_private_use_chars(text)
        
        # The invalid char should be preserved, not decoded
        assert "\uf710" in decoded or count == 0


class TestFindPagesWithGarbledText:
    """Tests for finding pages containing garbled text."""

    def test_no_garbled_text_returns_empty(self):
        """Test that clean content returns no problem pages."""
        router = ExtractionRouter()
        
        clean_content = "This is perfectly normal text without any issues."
        # Note: We can't easily test without a real PDF, so we test the pattern matching
        
        import re
        garbled_pattern = re.compile(r'[\uE000-\uF8FF]+')
        assert not garbled_pattern.search(clean_content)

    def test_garbled_pattern_detected(self):
        """Test that garbled text pattern is correctly identified."""
        import re
        
        # Content with private use chars
        content = "Normal text \uf765\uf772 more text"
        garbled_pattern = re.compile(r'[\uE000-\uF8FF]+')
        
        matches = garbled_pattern.findall(content)
        assert len(matches) == 1
        assert matches[0] == "\uf765\uf772"


class TestRouterValidExtraction:
    def test_is_valid_extraction_with_content(self):
        router = ExtractionRouter()

        result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="This is valid content",
                extraction_method="docling",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )

        assert router._is_valid_extraction(result) is True

    def test_is_valid_extraction_empty_pages(self):
        router = ExtractionRouter()
        result = ExtractionResult(pages=[], title=None, author=None)
        assert router._is_valid_extraction(result) is False

    def test_is_valid_extraction_whitespace_only(self):
        router = ExtractionRouter()

        result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="   \n\t  ",
                extraction_method="docling",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )

        assert router._is_valid_extraction(result) is False

    def test_is_valid_extraction_too_short(self):
        router = ExtractionRouter()

        result = ExtractionResult(
            pages=[PageResult(
                page_number=1,
                content="short",  # Less than 10 chars
                extraction_method="docling",
                confidence=1.0,
                images=[],
                tables=[],
            )],
            title=None,
            author=None,
        )

        assert router._is_valid_extraction(result) is False
