"""Abstract base class for document extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from sibyl.models import PageResult

if TYPE_CHECKING:
    from sibyl.models import ExtractOptions


class ExtractionResult(BaseModel):
    """Result from a document extractor."""

    pages: list[PageResult] = Field(default_factory=list, description="Extracted pages")
    title: str | None = Field(default=None, description="Document title if detected")
    author: str | None = Field(default=None, description="Document author if detected")


class Extractor(ABC):
    """Abstract base class for document extractors.

    Implementations wrap specific extraction libraries (Docling, MarkItDown, etc.)
    and convert their output to a common format.
    """

    @abstractmethod
    def extract(
        self,
        file_path: Path,
        options: "ExtractOptions",
    ) -> ExtractionResult:
        """Extract content from a document.

        Args:
            file_path: Path to the document file
            options: Extraction options

        Returns:
            ExtractionResult with extracted pages
        """
        ...

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Check if this extractor supports the given file.

        Args:
            file_path: Path to check

        Returns:
            True if this extractor can handle the file
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this extractor.

        Returns:
            Extractor name (e.g., 'docling', 'markitdown', 'vision_ocr')
        """
        ...
