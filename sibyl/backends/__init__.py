"""OCR backends for Sibyl.

Available backends:
    Local:
    - OllamaBackend: Ollama server with vision models (deepseek-ocr, llava, etc.)
    - LMStudioBackend: LMStudio server with vision models

    Cloud:
    - GeminiBackend: Google Gemini API (gemini-2.0-flash, gemini-1.5-pro)
    - AnthropicBackend: Anthropic Claude API (claude-sonnet-4-20250514, claude-3-5-sonnet)
    - OpenAIBackend: OpenAI API (gpt-4o, gpt-4-turbo)
"""

from sibyl.backends.anthropic import AnthropicBackend
from sibyl.backends.base import OCRBackend, OCRResult
from sibyl.backends.gemini import GeminiBackend
from sibyl.backends.lmstudio import LMStudioBackend
from sibyl.backends.ollama import OllamaBackend
from sibyl.backends.openai import OpenAIBackend

__all__ = [
    "OCRBackend",
    "OCRResult",
    # Local backends
    "OllamaBackend",
    "LMStudioBackend",
    # Cloud backends
    "GeminiBackend",
    "AnthropicBackend",
    "OpenAIBackend",
]
