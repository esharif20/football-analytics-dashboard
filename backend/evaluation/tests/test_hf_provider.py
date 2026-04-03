"""Unit tests for HuggingFaceProvider.

Tests:
  1. is_available() returns True when api_key is set, False when empty
  2. generate() calls AsyncInferenceClient.chat.completions.create() correctly
  3. get_provider("huggingface") returns HuggingFaceProvider instance
  4. Default model is "mistralai/Mistral-7B-Instruct-v0.3"
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.llm_providers import HuggingFaceProvider, get_provider


class TestHuggingFaceProviderIsAvailable:
    def test_is_available_returns_true_when_key_set(self):
        provider = HuggingFaceProvider(api_key="test-key")
        assert provider.is_available() is True

    def test_is_available_returns_false_when_empty(self):
        provider = HuggingFaceProvider(api_key="")
        assert provider.is_available() is False

    def test_is_available_reads_env_var(self):
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "env-key"}):
            provider = HuggingFaceProvider()
            assert provider.is_available() is True

    def test_is_available_false_when_env_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "HUGGINGFACE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            provider = HuggingFaceProvider()
            assert provider.is_available() is False


class TestHuggingFaceProviderGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self, mock_hf_client):
        """generate() returns choices[0].message.content from AsyncInferenceClient."""
        provider = HuggingFaceProvider(api_key="test-key")
        result = await provider.generate("system prompt", "user prompt")
        assert result == "test response"

    @pytest.mark.asyncio
    async def test_generate_calls_create_with_correct_args(self, mock_hf_client):
        """generate() passes correct model, messages, temperature, max_tokens."""
        provider = HuggingFaceProvider(api_key="test-key")
        await provider.generate("sys", "usr")

        mock_hf_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_hf_client.chat.completions.create.call_args

        # Check keyword arguments
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert kwargs.get("model") == "mistralai/Mistral-7B-Instruct-v0.3"
        assert kwargs.get("temperature") == 0.7
        assert kwargs.get("max_tokens") == 4096

        messages = kwargs.get("messages", [])
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1] == {"role": "user", "content": "usr"}


class TestHuggingFaceProviderDefaultModel:
    def test_default_model_is_mistral(self):
        """Default model must be mistralai/Mistral-7B-Instruct-v0.3."""
        provider = HuggingFaceProvider(api_key="key")
        assert provider.model_name == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_custom_model_can_be_set(self):
        provider = HuggingFaceProvider(api_key="key", model="mistralai/Mistral-7B-v0.1")
        assert provider.model_name == "mistralai/Mistral-7B-v0.1"


class TestGetProviderHuggingFace:
    def test_get_provider_returns_hf_instance(self):
        """get_provider('huggingface') returns HuggingFaceProvider when key is set."""
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test-key"}):
            provider = get_provider("huggingface")
        assert isinstance(provider, HuggingFaceProvider)
