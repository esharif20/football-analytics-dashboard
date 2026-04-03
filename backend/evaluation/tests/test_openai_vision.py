"""Unit tests for OpenAIVisionProvider."""

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def mock_openai_response():
    """Create a mock OpenAI chat completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "test vision response"
    return response


class TestOpenAIVisionProvider:

    def test_is_available_returns_true_when_key_set(self):
        """is_available() returns True when OPENAI_API_KEY is set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from evaluation.vlm_comparison import OpenAIVisionProvider
            provider = OpenAIVisionProvider()
            assert provider.is_available() is True

    def test_is_available_returns_false_when_key_empty(self):
        """is_available() returns False when OPENAI_API_KEY is empty or missing."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            from evaluation.vlm_comparison import OpenAIVisionProvider
            provider = OpenAIVisionProvider()
            assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_without_images(self, mock_openai_response):
        """generate() without images sends a single text content part."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from evaluation.vlm_comparison import OpenAIVisionProvider

            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

            provider = OpenAIVisionProvider()
            with patch("openai.AsyncOpenAI", return_value=mock_client):
                result = await provider.generate("system prompt", "user prompt")

            assert result == "test vision response"
            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["messages"]
            user_message = next(m for m in messages if m["role"] == "user")
            content = user_message["content"]
            # Should be a list with a single text part
            assert isinstance(content, list)
            assert len(content) == 1
            assert content[0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_generate_with_images(self, mock_openai_response):
        """generate() with images sends text + image_url content parts."""
        img_bytes = b"\xff\xd8\xff\xe0test"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from evaluation.vlm_comparison import OpenAIVisionProvider

            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

            provider = OpenAIVisionProvider()
            with patch("openai.AsyncOpenAI", return_value=mock_client):
                result = await provider.generate("system prompt", "user prompt", images=[img_bytes])

            assert result == "test vision response"
            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.kwargs["messages"]
            user_message = next(m for m in messages if m["role"] == "user")
            content = user_message["content"]
            # Should have text part + image_url part
            assert isinstance(content, list)
            assert len(content) == 2
            text_part = content[0]
            image_part = content[1]
            assert text_part["type"] == "text"
            assert image_part["type"] == "image_url"
            expected_b64 = base64.b64encode(img_bytes).decode("utf-8")
            assert image_part["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"

    def test_default_model_is_gpt4o_mini(self):
        """OpenAIVisionProvider uses model 'gpt-4o-mini' by default."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from evaluation.vlm_comparison import OpenAIVisionProvider
            provider = OpenAIVisionProvider()
            assert provider.model_name == "gpt-4o-mini"
