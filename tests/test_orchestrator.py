import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.services.orchestrator as orchestrator
from app.services.orchestrator import stream_agent_response
from app.services.orchestrator import _execute_tool


def test_chat_client_uses_gemini_compatible_endpoint(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(orchestrator.settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(
        orchestrator.settings,
        "gemini_base_url",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    monkeypatch.setattr(orchestrator, "AsyncOpenAI", FakeClient)

    orchestrator._client()

    assert captured == {
        "api_key": "test-gemini-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "timeout": orchestrator.settings.gemini_request_timeout_seconds,
    }


class MemorySession:
    async def flush(self) -> None:
        return None

    async def scalar(self, _statement):
        return 0

    async def scalars(self, _statement):
        return SimpleNamespace(
            all=lambda: [
                SimpleNamespace(
                    role="user",
                    content="Hello",
                    tool_calls=None,
                    tool_call_id=None,
                )
            ]
        )

    def add(self, _item) -> None:
        return None

    async def commit(self) -> None:
        return None


@pytest.mark.asyncio
async def test_interactive_chat_stream_uses_gemini_compatible_streaming(monkeypatch) -> None:
    captured = {}

    class FakeStream:
        def __aiter__(self):
            async def chunks():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Hello back", tool_calls=None)
                        )
                    ]
                )

            return chunks()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            captured["request"] = kwargs
            return FakeStream()

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr(orchestrator.settings, "gemini_api_key", "gemini-key")
    monkeypatch.setattr(orchestrator, "AsyncOpenAI", FakeClient)
    session = MemorySession()
    user = SimpleNamespace(id=uuid4())
    conversation = SimpleNamespace(id=uuid4())

    events = [
        event
        async for event in stream_agent_response(session, user, conversation, "Hello")
    ]

    assert events == [
        {"type": "token", "content": "Hello back"},
        {"type": "done", "content": "Hello back"},
    ]
    assert captured["request"]["model"] == orchestrator.settings.gemini_model
    assert captured["request"]["stream"] is True
    assert captured["request"]["tools"] == orchestrator.TOOL_DEFINITIONS
    assert captured["client"]["base_url"] == orchestrator.settings.gemini_base_url
    assert captured["closed"]


@pytest.mark.asyncio
async def test_interactive_stream_assembles_gemini_tool_call_arguments(monkeypatch) -> None:
    captured_requests = []

    class FakeStream:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def chunks():
                for chunk in self.chunks:
                    yield chunk

            return chunks()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            captured_requests.append(kwargs)
            if len(captured_requests) == 1:
                return FakeStream(
                    [
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(
                                        content=None,
                                        tool_calls=[
                                            SimpleNamespace(
                                                index=0,
                                                id="call-id",
                                                function=SimpleNamespace(
                                                    name="calculator",
                                                    arguments='{"expression":',
                                                ),
                                            )
                                        ],
                                    )
                                )
                            ]
                        ),
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(
                                        content=None,
                                        tool_calls=[
                                            SimpleNamespace(
                                                index=0,
                                                id=None,
                                                function=SimpleNamespace(
                                                    name=None,
                                                    arguments='"2 + 2"}',
                                                ),
                                            )
                                        ],
                                    )
                                )
                            ]
                        ),
                    ]
                )
            return FakeStream(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="Four", tool_calls=None)
                            )
                        ]
                    )
                ]
            )

        async def close(self):
            return None

    monkeypatch.setattr(orchestrator.settings, "gemini_api_key", "gemini-key")
    monkeypatch.setattr(orchestrator, "AsyncOpenAI", FakeClient)
    events = [
        event
        async for event in stream_agent_response(
            MemorySession(),
            SimpleNamespace(id=uuid4()),
            SimpleNamespace(id=uuid4()),
            "What is 2 + 2?",
        )
    ]

    assert events == [
        {"type": "tool_start", "name": "calculator"},
        {"type": "tool_complete", "name": "calculator"},
        {"type": "token", "content": "Four"},
        {"type": "done", "content": "Four"},
    ]
    assert captured_requests[0]["tools"] == orchestrator.TOOL_DEFINITIONS
    assert captured_requests[1]["messages"][-1]["content"] == '{"result": 4.0}'


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
