"""Sibyl: Hybrid Document-to-Markdown Python Library.

Sibyl converts diverse document formats into structured, LLM-ready markdown
using intelligent extraction routing.

Example:
    >>> from sibyl import Sibyl
    >>> from sibyl.backends import OllamaBackend
    >>>
    >>> sb = Sibyl(ocr_backend=OllamaBackend())  # uses deepseek-ocr by default
    >>> result = sb.process("document.pdf")
    >>> print(result.markdown)
"""

from sibyl.core import Sibyl
from sibyl.models import (
    Chunk,
    DocumentMetadata,
    ExtractOptions,
    ImageResult,
    PageResult,
    PdfEngine,
    ProcessingResult,
    ProcessingStats,
    ProgressCallback,
    TableResult,
)

__version__ = "0.1.0"

__all__ = [
    "Sibyl",
    "ProcessingResult",
    "PageResult",
    "TableResult",
    "ImageResult",
    "DocumentMetadata",
    "ProcessingStats",
    "Chunk",
    "ExtractOptions",
    "PdfEngine",
    "ProgressCallback",
]
