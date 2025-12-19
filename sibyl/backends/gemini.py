"""Google Gemini OCR backend implementation."""

from __future__ import annotations

import base64
import io
import os
from typing import TYPE_CHECKING

import httpx

from sibyl.backends.base import OCRBackend, OCRResult

if TYPE_CHECKING:
    from PIL import Image


class GeminiBackend(OCRBackend):
    """OCR backend using Google Gemini vision models.

    Supports Gemini 1.5 and 2.0 models with vision capabilities.
    """

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize Gemini backend.

        Args:
            model: Model name (gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash)
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def ocr_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> OCRResult:
        """Perform OCR on an image using Gemini.

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
                "Gemini API key not set. Pass api_key or set GOOGLE_API_KEY env var."
            )

        # Convert image to base64
        buffer = io.BytesIO()
        image_format = "PNG" if image.mode == "RGBA" else "JPEG"
        mime_type = "image/png" if image_format == "PNG" else "image/jpeg"
        image.save(buffer, format=image_format)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Use default prompt if none provided
        if prompt is None:
            prompt = self.get_default_prompt()

        # Build request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64,
                            }
                        },
                    ]
                }
            ]
        }

        # Make request to Gemini API
        url = f"{self.API_BASE}/models/{self.model}:generateContent?key={self.api_key}"
        response = self._client.post(url, json=payload)
        response.raise_for_status()

        result = response.json()

        # Extract text from response
        text = ""
        if "candidates" in result and result["candidates"]:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        text += part["text"]

        return OCRResult(text=text, confidence=None)

    def is_available(self) -> bool:
        """Check if Gemini API is available.

        Returns:
            True if API key is set and API is reachable
        """
        if not self.api_key:
            return False

        try:
            url = f"{self.API_BASE}/models?key={self.api_key}"
            response = self._client.get(url)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
