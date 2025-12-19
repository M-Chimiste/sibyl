"""Document extractors for Sibyl.

Available extractors:
    - DoclingExtractor: Native PDF text extraction
    - MarkItDownExtractor: Office format conversion
    - VisionOCRExtractor: Vision-based OCR for scanned documents
"""

from sibyl.extractors.base import ExtractionResult, Extractor
from sibyl.extractors.docling_ext import DoclingExtractor
from sibyl.extractors.markitdown_ext import MarkItDownExtractor
from sibyl.extractors.vision_ocr import VisionOCRExtractor

__all__ = [
    "Extractor",
    "ExtractionResult",
    "DoclingExtractor",
    "MarkItDownExtractor",
    "VisionOCRExtractor",
]
