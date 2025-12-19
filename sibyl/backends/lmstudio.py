"""LMStudio OCR backend implementation."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import httpx

from sibyl.backends.base import OCRBackend, OCRResult

if TYPE_CHECKING:
    from PIL import Image


class LMStudioBackend(OCRBackend):
    """OCR backend using LMStudio with vision-language models.

    LMStudio exposes an OpenAI-compatible API, so this backend
    uses the chat completions endpoint with vision support.
    """

    def __init__(
        self,
        model: str = "deepseek-ocr",
        base_url: str = "http://localhost:1234",
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ):
        """Initialize LMStudio backend.

        Args:
            model: Model name/identifier loaded in LMStudio
            base_url: Full LMStudio server URL (e.g., "http://localhost:1234" or
                     "http://192.168.1.100:1234")
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client = httpx.Client(timeout=timeout)

    def ocr_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> OCRResult:
        """Perform OCR on an image using LMStudio.

        Args:
            image: PIL Image to process
            prompt: Optional custom prompt (uses default if None)

        Returns:
            OCRResult with extracted text

        Raises:
            httpx.HTTPError: If the request fails
        """
        # Convert image to base64
        buffer = io.BytesIO()
        image_format = "PNG" if image.mode == "RGBA" else "JPEG"
        mime_type = "image/png" if image_format == "PNG" else "image/jpeg"
        image.save(buffer, format=image_format)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Use default prompt if none provided
        if prompt is None:
            prompt = self.get_default_prompt()

        # Build request payload (OpenAI-compatible format)
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
                            },
                        },
                    ],
                }
            ],
        }

        # Make request to LMStudio (OpenAI-compatible endpoint)
        response = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
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
        """Check if LMStudio server is available.

        Returns:
            True if LMStudio is reachable
        """
        try:
            # LMStudio exposes /v1/models endpoint
            response = self._client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
