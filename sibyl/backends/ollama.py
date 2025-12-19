"""Ollama OCR backend implementation."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import httpx

from sibyl.backends.base import OCRBackend, OCRResult

if TYPE_CHECKING:
    from PIL import Image


class OllamaBackend(OCRBackend):
    """OCR backend using Ollama with vision-language models.

    Connects to an Ollama instance to perform OCR using
    models like deepseek-ocr, llava, or other vision models.
    """

    def __init__(
        self,
        model: str = "deepseek-ocr",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        """Initialize Ollama backend.

        Args:
            model: Model name to use for OCR
            base_url: Full Ollama server URL (e.g., "http://localhost:11434" or
                     "http://192.168.1.100:11434" or "https://ollama.example.com")
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def ocr_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> OCRResult:
        """Perform OCR on an image using Ollama.

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
        image.save(buffer, format=image_format)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Use default prompt if none provided
        if prompt is None:
            prompt = self.get_default_prompt()

        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
        }

        # Make request to Ollama
        response = self._client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("response", "")

        return OCRResult(text=text, confidence=None)

    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded.

        Returns:
            True if Ollama is reachable and model exists
        """
        try:
            # Check if Ollama is running
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return self.model.split(":")[0] in model_names

        except httpx.HTTPError:
            return False

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
