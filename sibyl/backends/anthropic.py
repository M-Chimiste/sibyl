"""Anthropic Claude OCR backend implementation."""

from __future__ import annotations

import base64
import io
import os
from typing import TYPE_CHECKING

import httpx

from sibyl.backends.base import OCRBackend, OCRResult

if TYPE_CHECKING:
    from PIL import Image


class AnthropicBackend(OCRBackend):
    """OCR backend using Anthropic Claude vision models.

    Supports Claude 3 and 3.5 models with vision capabilities.
    """

    API_BASE = "https://api.anthropic.com/v1"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ):
        """Initialize Anthropic backend.

        Args:
            model: Model name (claude-sonnet-4-20250514, claude-3-5-sonnet-20241022, etc.)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout)

    def ocr_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> OCRResult:
        """Perform OCR on an image using Claude.

        Args:
            image: PIL Image to process
            prompt: Optional custom prompt (uses default if None)

        Returns:
            OCRResult with extracted text

        Raises:
            ValueError: If API key is not set
            httpx.HTTPError: If the request fails
        """
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not set. Pass api_key or set ANTHROPIC_API_KEY env var."
            )

        # Convert image to base64
        buffer = io.BytesIO()
        image_format = "PNG" if image.mode == "RGBA" else "JPEG"
        media_type = "image/png" if image_format == "PNG" else "image/jpeg"
        image.save(buffer, format=image_format)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Use default prompt if none provided
        if prompt is None:
            prompt = self.get_default_prompt()

        # Build request payload
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        # Make request to Anthropic API
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = self._client.post(
            f"{self.API_BASE}/messages",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()

        # Extract text from response
        text = ""
        if "content" in result:
            for block in result["content"]:
                if block.get("type") == "text":
                    text += block.get("text", "")

        return OCRResult(text=text, confidence=None)

    def is_available(self) -> bool:
        """Check if Anthropic API is available.

        Returns:
            True if API key is set
        """
        # Anthropic doesn't have a simple health check endpoint
        # Just verify the API key is set
        return bool(self.api_key)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
