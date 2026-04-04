"""LLM provider abstraction layer.

Supports Gemini (default) and OpenAI as providers for tactical analysis.
Each provider implements a simple async generate() interface.
"""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion from the LLM.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: User message containing the grounded analytics data.

        Returns:
            Generated text response.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider has valid credentials."""


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the google-generativeai SDK."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 4096,
                },
            )
        return self._client

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        import asyncio

        client = self._get_client()

        # Gemini uses a combined prompt with system instruction prefix
        combined = f"{system_prompt}\n\n---\n\n{user_prompt}"

        # Run synchronous SDK call in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.generate_content(combined))
        return response.text


class OpenAIProvider(LLMProvider):
    """OpenAI provider using the openai SDK."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider using huggingface_hub SDK.

    Default model: mistralai/Mistral-7B-Instruct-v0.3
    """

    def __init__(self, api_key: str | None = None, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.model_name = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import AsyncInferenceClient

            self._client = AsyncInferenceClient(api_key=self.api_key)
        return self._client

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""


class StubProvider(LLMProvider):
    """Deterministic stub provider for tests (no external calls)."""

    def __init__(self, response_text: str = "STUB-COMMENTARY"):
        self.response_text = response_text

    def is_available(self) -> bool:  # pragma: no cover - trivial
        return True

    async def generate(
        self, system_prompt: str, user_prompt: str
    ) -> str:  # pragma: no cover - trivial
        return self.response_text


def get_provider(provider_name: str | None = None) -> LLMProvider:
    """Get the configured LLM provider.

    Tries the requested provider first, then falls back to any available one.
    StubProvider is only selected when explicitly requested by name.

    Args:
        provider_name: "gemini", "openai", or "stub". If None, uses LLM_PROVIDER env var.

    Returns:
        An initialized LLMProvider.

    Raises:
        RuntimeError: If no provider has valid credentials.
    """
    name = (provider_name or os.getenv("LLM_PROVIDER", "gemini")).lower()

    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
        "stub": StubProvider,
    }

    # Try requested provider first
    if name in providers:
        provider = providers[name]()
        if provider.is_available():
            logger.info("Using LLM provider: %s", name)
            return provider
        logger.warning("Provider '%s' not available (no API key), trying fallbacks", name)

    # Fallback to any available real provider (exclude stub from auto-fallback)
    for fallback_name, cls in providers.items():
        if fallback_name in (name, "stub"):
            continue
        provider = cls()
        if provider.is_available():
            logger.info("Falling back to LLM provider: %s", fallback_name)
            return provider

    raise RuntimeError(
        "No LLM provider available. Set GEMINI_API_KEY, OPENAI_API_KEY, or HUGGINGFACE_API_KEY in your .env file."
    )
