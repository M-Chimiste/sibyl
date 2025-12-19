"""Tests for OCR backends."""

import pytest

from sibyl.backends import (
    AnthropicBackend,
    GeminiBackend,
    LMStudioBackend,
    OCRBackend,
    OllamaBackend,
    OpenAIBackend,
)
from sibyl.backends.base import OCRResult


class TestOCRResult:
    def test_create_result(self):
        result = OCRResult(text="Hello world", confidence=0.95)
        assert result.text == "Hello world"
        assert result.confidence == 0.95

    def test_result_without_confidence(self):
        result = OCRResult(text="Hello world")
        assert result.confidence is None


class TestOllamaBackend:
    def test_init_defaults(self):
        backend = OllamaBackend()
        assert backend.model == "deepseek-ocr"
        assert backend.base_url == "http://localhost:11434"

    def test_init_custom_url(self):
        backend = OllamaBackend(
            model="llava",
            base_url="http://192.168.1.100:11434",
        )
        assert backend.model == "llava"
        assert backend.base_url == "http://192.168.1.100:11434"

    def test_url_trailing_slash_stripped(self):
        backend = OllamaBackend(base_url="http://localhost:11434/")
        assert backend.base_url == "http://localhost:11434"

    def test_is_subclass_of_ocr_backend(self):
        assert issubclass(OllamaBackend, OCRBackend)


class TestLMStudioBackend:
    def test_init_defaults(self):
        backend = LMStudioBackend()
        assert backend.model == "deepseek-ocr"
        assert backend.base_url == "http://localhost:1234"

    def test_init_custom_url(self):
        backend = LMStudioBackend(
            model="custom-model",
            base_url="http://my-server:5000",
        )
        assert backend.model == "custom-model"
        assert backend.base_url == "http://my-server:5000"


class TestGeminiBackend:
    def test_init_defaults(self):
        backend = GeminiBackend()
        assert backend.model == "gemini-2.0-flash"
        assert backend.api_key is None  # No env var set in test

    def test_init_with_api_key(self):
        backend = GeminiBackend(api_key="test-key")
        assert backend.api_key == "test-key"

    def test_is_available_without_key(self):
        backend = GeminiBackend()
        assert backend.is_available() is False


class TestAnthropicBackend:
    def test_init_defaults(self):
        backend = AnthropicBackend()
        assert backend.model == "claude-sonnet-4-20250514"
        assert backend.api_key is None

    def test_init_with_api_key(self):
        backend = AnthropicBackend(api_key="test-key")
        assert backend.api_key == "test-key"

    def test_is_available_without_key(self):
        backend = AnthropicBackend()
        assert backend.is_available() is False


class TestOpenAIBackend:
    def test_init_defaults(self):
        backend = OpenAIBackend()
        assert backend.model == "gpt-4o"
        assert backend.api_key is None
        assert backend.detail == "high"

    def test_init_with_options(self):
        backend = OpenAIBackend(
            model="gpt-4-turbo",
            api_key="test-key",
            detail="low",
        )
        assert backend.model == "gpt-4-turbo"
        assert backend.api_key == "test-key"
        assert backend.detail == "low"

    def test_is_available_without_key(self):
        backend = OpenAIBackend()
        assert backend.is_available() is False
