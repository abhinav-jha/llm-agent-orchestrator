import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.services.orchestrator as orchestrator
from app.services.orchestrator import _execute_tool


def test_chat_client_uses_gemini_compatibility_endpoint(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(orchestrator.settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(orchestrator, "AsyncOpenAI", FakeClient)

    orchestrator._client()

    assert captured == {
        "api_key": "test-gemini-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }


class MemorySession:
    async def flush(self) -> None:
        return None


@pytest.mark.asyncio
async def test_unknown_tool_returns_explicit_error() -> None:
    result = await _execute_tool("unexpected_tool", "{}", MemorySession(), user_id=None)

    assert json.loads(result) == {"error": "Unknown tool: unexpected_tool"}


@pytest.mark.asyncio
async def test_invalid_arguments_return_explicit_error() -> None:
    result = await _execute_tool("search_memory", "[]", MemorySession(), user_id=None)

    assert json.loads(result) == {"error": "Tool arguments must be a JSON object"}


@pytest.mark.asyncio
async def test_calculator_evaluates_safe_expression() -> None:
    result = await _execute_tool(
        "calculator",
        '{"expression": "(2 + 3) * 4"}',
        MemorySession(),
        user_id=None,
    )

    assert json.loads(result) == {"result": 20.0}


@pytest.mark.asyncio
async def test_current_datetime_returns_iso_timestamp() -> None:
    result = await _execute_tool("get_current_datetime", "{}", MemorySession(), user_id=None)

    payload = json.loads(result)
    assert "utc" in payload
    assert payload["utc"].endswith("Z") or "+" in payload["utc"]


@pytest.mark.asyncio
async def test_remember_fact_saves_memory(monkeypatch) -> None:
    user_id = uuid4()

    async def fake_save_memory(_session, _user_id, content):
        assert content == "Prefers concise updates"
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(orchestrator, "save_memory", fake_save_memory)

    result = await _execute_tool(
        "remember_fact",
        '{"fact": "Prefers concise updates"}',
        MemorySession(),
        user_id=user_id,
    )

    payload = json.loads(result)
    assert payload["saved"] is True
    assert "memory_id" in payload
