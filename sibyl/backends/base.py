"""Abstract base class for OCR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from PIL import Image


class OCRResult(BaseModel):
    """Result from OCR processing."""

    text: str = Field(description="Extracted text content")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score of OCR"
    )


class OCRBackend(ABC):
    """Abstract base class for OCR backends.

    Implementations should connect to vision-language models
    (e.g., via Ollama or LMStudio) to perform OCR on images.
    """

    @abstractmethod
    def ocr_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> OCRResult:
        """Perform OCR on an image.

        Args:
            image: PIL Image to process
            prompt: Optional custom prompt for the VLM

        Returns:
            OCRResult with extracted text and confidence
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and ready.

        Returns:
            True if the backend can process requests
        """
        ...

    def get_default_prompt(self) -> str:
        """Get the default OCR prompt for markdown extraction.

        Returns:
            Prompt string for the VLM
        """
        return "Convert this image into Markdown."
