"""OpenAI GPT-4 Vision OCR backend implementation."""

from __future__ import annotations

import base64
import io
import os
from typing import TYPE_CHECKING

import httpx

from sibyl.backends.base import OCRBackend, OCRResult

if TYPE_CHECKING:
    from PIL import Image


class OpenAIBackend(OCRBackend):
    """OCR backend using OpenAI GPT-4 Vision models.

    Supports GPT-4o and GPT-4 Turbo with vision capabilities.
    """

    API_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 4096,
        detail: str = "high",
    ):
        """Initialize OpenAI backend.

        Args:
            model: Model name (gpt-4o, gpt-4o-mini, gpt-4-turbo)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            detail: Image detail level ("low", "high", or "auto")
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.detail = detail
        self._client = httpx.Client(timeout=timeout)

    def ocr_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> OCRResult:
        """Perform OCR on an image using GPT-4 Vision.

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
                "OpenAI API key not set. Pass api_key or set OPENAI_API_KEY env var."
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
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": self.detail,
                            },
                        },
                    ],
                }
            ],
        }

        # Make request to OpenAI API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._client.post(
            f"{self.API_BASE}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()

        # Extract text from response
        text = ""
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            text = message.get("content", "")

        return OCRResult(text=text, confidence=None)

    def is_available(self) -> bool:
        """Check if OpenAI API is available.

        Returns:
            True if API key is set and API is reachable
        """
        if not self.api_key:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self._client.get(f"{self.API_BASE}/models", headers=headers)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
