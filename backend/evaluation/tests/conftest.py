"""pytest configuration for evaluation tests.

Adds backend/api to sys.path so `from services.llm_providers import ...` works.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

API_ROOT = Path(__file__).resolve().parents[2] / "api"  # backend/api/
BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/

for p in (str(API_ROOT), str(BACKEND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def mock_hf_client(monkeypatch):
    """Patch AsyncInferenceClient so no real API calls are made.

    Returns a mock whose chat.completions.create is an AsyncMock that
    returns a response object with choices[0].message.content = "test response".
    """
    message = MagicMock()
    message.content = "test response"

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    mock_cls = MagicMock(return_value=mock_client)

    monkeypatch.setattr("huggingface_hub.AsyncInferenceClient", mock_cls)

    return mock_client
