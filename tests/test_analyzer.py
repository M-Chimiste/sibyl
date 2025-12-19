"""Tests for content analyzer."""

import pytest
from pathlib import Path

from sibyl.analyzer import (
    ContentAnalyzer,
    ContentType,
    ExtractionMethod,
    FileType,
)


class TestContentAnalyzer:
    def test_detect_pdf_file_type(self, tmp_path):
        # Create a dummy file path (we're just testing extension detection)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        analyzer = ContentAnalyzer()
        file_type = analyzer._detect_file_type(pdf_path)
        assert file_type == FileType.PDF

    def test_detect_docx_file_type(self, tmp_path):
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        analyzer = ContentAnalyzer()
        file_type = analyzer._detect_file_type(docx_path)
        assert file_type == FileType.DOCX

    def test_detect_image_file_type(self, tmp_path):
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            img_path = tmp_path / f"test{ext}"
            img_path.touch()

            analyzer = ContentAnalyzer()
            file_type = analyzer._detect_file_type(img_path)
            assert file_type == FileType.IMAGE

    def test_detect_unknown_file_type(self, tmp_path):
        unknown_path = tmp_path / "test.xyz"
        unknown_path.touch()

        analyzer = ContentAnalyzer()
        file_type = analyzer._detect_file_type(unknown_path)
        assert file_type == FileType.OTHER

    def test_analyze_image_returns_vision_ocr(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.touch()

        analyzer = ContentAnalyzer()
        analysis = analyzer._analyze_image(img_path)

        assert analysis.file_type == FileType.IMAGE
        assert analysis.content_type == ContentType.SCANNED
        assert analysis.primary_method == ExtractionMethod.VISION_OCR
        assert analysis.page_count == 1

    def test_analyze_office_returns_markitdown(self, tmp_path):
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        analyzer = ContentAnalyzer()
        analysis = analyzer._analyze_office(docx_path, FileType.DOCX)

        assert analysis.file_type == FileType.DOCX
        assert analysis.content_type == ContentType.NATIVE
        assert analysis.primary_method == ExtractionMethod.MARKITDOWN


class TestExtractionMethod:
    def test_extraction_methods_exist(self):
        assert ExtractionMethod.DOCLING.value == "docling"
        assert ExtractionMethod.MARKITDOWN.value == "markitdown"
        assert ExtractionMethod.VISION_OCR.value == "vision_ocr"


class TestContentType:
    def test_content_types_exist(self):
        assert ContentType.NATIVE.value == "native"
        assert ContentType.SCANNED.value == "scanned"
        assert ContentType.HYBRID.value == "hybrid"
