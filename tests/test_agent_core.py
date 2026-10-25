import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.agent.runner as runner
import app.agent.tools as agent_tools
from app.core.config import settings
from app.agent.memory_manager import history_key, load_short_term_history
from app.agent.tools import dispatch_tool, tool_definitions
from app.models import RunStatus


def test_tool_definitions_are_closed_json_schemas() -> None:
    definitions = tool_definitions(
        [
            "web_search",
            "calculator",
            "get_current_datetime",
            "summarise_text",
            "remember_fact",
        ]
    )
    assert [item["function"]["name"] for item in definitions] == [
        "web_search",
        "calculator",
        "get_current_datetime",
        "summarise_text",
        "remember_fact",
    ]
    calculator_definition = definitions[1]["function"]

    assert calculator_definition["parameters"]["required"] == ["expression"]
    assert calculator_definition["parameters"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_short_term_context_is_limited_to_ten_messages() -> None:
    class FakeRedis:
        async def lrange(self, key, start, end):
            assert key == history_key("session-1")
            assert (start, end) == (0, 4)
            turns = [
                json.dumps({"user": f"user-{index}", "assistant": f"assistant-{index}"})
                for index in range(9, -1, -1)
            ]
            return turns[start : end + 1]

    messages = await load_short_term_history(FakeRedis(), "session-1")

    assert messages == [
        item
        for index in range(5, 10)
        for item in (
            {"role": "user", "content": f"user-{index}"},
            {"role": "assistant", "content": f"assistant-{index}"},
        )
    ]


@pytest.mark.asyncio
async def test_run_agent_persists_final_answer_and_token_usage(monkeypatch) -> None:
    user_id = uuid4()
    run = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        user_message="hello",
        prompt="hello",
        status=RunStatus.RUNNING.value,
        tokens_used=0,
        final_answer=None,
        result=None,
    )
    agent_session = SimpleNamespace(
        id="session-1", system_prompt="", tools_enabled=[]
    )
    recorded_steps = []
    published = []
    client_configuration = {}

    class FakeSession:
        def add(self, item):
            recorded_steps.append(item)

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, _item):
            return None

    class FakeRedis:
        async def publish(self, channel, data):
            published.append((channel, json.loads(data)))

    class FakeClient:
        def __init__(self, **kwargs):
            client_configuration.update(kwargs)
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            assert kwargs["model"] == runner.settings.gemini_model
            assert kwargs["tools"] is None
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="hello back", tool_calls=None)
                    )
                ],
                usage=SimpleNamespace(total_tokens=7),
            )

        async def close(self):
            return None

    async def no_history(*_args):
        return []

    async def no_memories(*_args):
        return []

    async def no_append(*_args):
        return None

    monkeypatch.setattr(runner.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(runner, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(runner, "load_short_term_history", no_history)
    monkeypatch.setattr(runner, "retrieve_long_term_memories", no_memories)
    monkeypatch.setattr(runner, "append_completed_turn", no_append)

    answer = await runner.run_agent(FakeSession(), FakeRedis(), run, agent_session)

    assert answer == "hello back"
    assert run.final_answer == "hello back"
    assert run.result == "hello back"
    assert run.tokens_used == 7
    assert [step.step_type for step in recorded_steps] == ["llm_call", "final_answer"]
    assert len(published) == 2
    assert client_configuration["api_key"] == "test-key"
    assert client_configuration["base_url"] == runner.settings.gemini_base_url


@pytest.mark.asyncio
async def test_run_agent_rejects_disabled_tools_and_continues_with_error(monkeypatch) -> None:
    run = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        user_message="hello",
        prompt="hello",
        status=RunStatus.RUNNING.value,
        tokens_used=0,
        final_answer=None,
        result=None,
    )
    agent_session = SimpleNamespace(
        id="session-1", system_prompt="", tools_enabled=["get_current_datetime"]
    )
    responses = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="calculator", arguments='{"expression":"1+1"}'
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="disabled", tool_calls=None)
                    )
                ],
                usage=None,
            ),
        ]
    )
    recorded_steps = []

    class FakeSession:
        def add(self, item):
            recorded_steps.append(item)

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, _item):
            return None

    class FakeRedis:
        async def publish(self, _channel, _data):
            return None

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            assert [item["function"]["name"] for item in kwargs["tools"]] == [
                "get_current_datetime"
            ]
            return next(responses)

        async def close(self):
            return None

    async def no_history(*_args):
        return []

    async def no_memories(*_args):
        return []

    async def no_append(*_args):
        return None

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(runner, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(runner, "load_short_term_history", no_history)
    monkeypatch.setattr(runner, "retrieve_long_term_memories", no_memories)
    monkeypatch.setattr(runner, "append_completed_turn", no_append)

    answer = await runner.run_agent(FakeSession(), FakeRedis(), run, agent_session)

    assert answer == "disabled"
    result_step = next(step for step in recorded_steps if step.step_type == "tool_result")
    assert "Unknown or disabled tool: calculator" in result_step.payload["result"]


@pytest.mark.asyncio
async def test_run_agent_stops_at_iteration_limit(monkeypatch) -> None:
    run = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        user_message="repeat",
        prompt="repeat",
        status=RunStatus.RUNNING.value,
        tokens_used=0,
        final_answer=None,
        result=None,
    )
    agent_session = SimpleNamespace(
        id="session-1", system_prompt="", tools_enabled=["get_current_datetime"]
    )
    calls = []
    closed = []
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="get_current_datetime", arguments="{}"),
    )

    class FakeSession:
        async def refresh(self, _item):
            return None

        def add(self, _item):
            return None

        async def flush(self):
            return None

        async def commit(self):
            return None

    class FakeRedis:
        async def publish(self, _channel, _data):
            return None

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **_kwargs):
            calls.append(1)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[tool_call])
                    )
                ],
                usage=None,
            )

        async def close(self):
            closed.append(True)

    async def no_history(*_args):
        return []

    async def no_memories(*_args):
        return []

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(runner, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(runner, "load_short_term_history", no_history)
    monkeypatch.setattr(runner, "retrieve_long_term_memories", no_memories)

    with pytest.raises(RuntimeError, match=f"exceeded {runner.MAX_ITERATIONS}"):
        await runner.run_agent(FakeSession(), FakeRedis(), run, agent_session)

    assert len(calls) == runner.MAX_ITERATIONS
    assert closed == [True]


@pytest.mark.asyncio
async def test_run_agent_honors_cancellation_before_executing_tool(monkeypatch) -> None:
    run = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        user_message="cancel",
        prompt="cancel",
        status=RunStatus.RUNNING.value,
        tokens_used=0,
    )
    agent_session = SimpleNamespace(
        id="session-1", system_prompt="", tools_enabled=["calculator"]
    )
    tool_execution = []
    close_calls = []
    refresh_calls = 0

    class FakeSession:
        def add(self, _item):
            return None

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, current):
            nonlocal refresh_calls
            refresh_calls += 1
            if refresh_calls == 2:
                current.status = RunStatus.CANCELLED.value

    class FakeRedis:
        async def publish(self, _channel, _data):
            return None

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="calculator", arguments='{"expression":"1+1"}'
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=None,
            )

        async def close(self):
            close_calls.append(True)

    async def no_history(*_args):
        return []

    async def no_memories(*_args):
        return []

    async def dispatch(*_args):
        tool_execution.append(True)
        return "2"

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(runner, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(runner, "load_short_term_history", no_history)
    monkeypatch.setattr(runner, "retrieve_long_term_memories", no_memories)
    monkeypatch.setattr(runner, "dispatch_tool", dispatch)

    result = await runner.run_agent(FakeSession(), FakeRedis(), run, agent_session)

    assert result == ""
    assert run.status == RunStatus.CANCELLED.value
    assert tool_execution == []
    assert close_calls == [True]


@pytest.mark.asyncio
async def test_agent_tool_dispatch_reports_invalid_input() -> None:
    result = await dispatch_tool(
        "calculator",
        json.dumps({"unexpected": "argument"}),
        object(),
        uuid4(),
        uuid4(),
        ["calculator"],
    )

    assert result.startswith("ERROR:")


@pytest.mark.asyncio
async def test_session_cannot_invoke_disabled_registered_tool() -> None:
    result = await dispatch_tool("calculator", '{"expression":"1 + 1"}', object(), uuid4(), uuid4(), [])

    assert "disabled tool" in result


@pytest.mark.asyncio
async def test_agent_tools_calculate_time_and_remember_fact(monkeypatch) -> None:
    saved = {}

    class FakeSession:
        def add(self, memory):
            saved["memory"] = memory

        async def flush(self):
            return None

    async def fake_embed(text, task_type):
        saved.update(text=text, task_type=task_type)
        return [0.1] * 1536

    monkeypatch.setattr(agent_tools, "embed_text", fake_embed)
    monkeypatch.setattr(agent_tools.settings, "gemini_embedding_model", "gemini-embedding-001")

    calculated = await dispatch_tool(
        "calculator", '{"expression":"(2 + 3) * 4"}', FakeSession(), uuid4(), uuid4(),
        ["calculator"],
    )
    timestamp = await dispatch_tool(
        "get_current_datetime", "{}", FakeSession(), uuid4(), uuid4(),
        ["get_current_datetime"],
    )
    remembered = await dispatch_tool(
        "remember_fact", '{"fact":"Keep replies concise"}', FakeSession(), uuid4(), uuid4(),
        ["remember_fact"],
    )

    assert calculated == "20"
    assert "T" in timestamp
    assert remembered == "Remembered: Keep replies concise"
    assert saved["task_type"] == "RETRIEVAL_DOCUMENT"
    assert saved["memory"].embedding_model == "gemini-embedding-001"


@pytest.mark.asyncio
async def test_summarise_text_uses_configured_gemini_compatible_client(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
            )

    monkeypatch.setattr(agent_tools.settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(agent_tools, "OpenAI", FakeClient)

    result = agent_tools.summarise_text("A long input")

    assert result == "summary"
    assert captured["api_key"] == "test-gemini-key"
    assert captured["base_url"] == agent_tools.settings.gemini_base_url
    assert captured["request"]["model"] == agent_tools.settings.gemini_model


def test_web_search_returns_real_provider_results(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status_code = 200
        is_error = False

        def json(self):
            return {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.test/page",
                        "content": "Provider result snippet",
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, url, headers, json):
            captured.update(url=url, headers=headers, json=json)
            return FakeResponse()

    monkeypatch.setattr(agent_tools.settings, "tavily_api_key", "test-tavily-key")
    monkeypatch.setattr(agent_tools.httpx, "Client", FakeClient)

    result = json.loads(agent_tools.web_search("  Gemini news  "))

    assert result == [
        {
            "title": "Result",
            "url": "https://example.test/page",
            "snippet": "Provider result snippet",
        }
    ]
    assert captured["json"]["query"] == "Gemini news"
    assert captured["headers"]["Authorization"] == "Bearer test-tavily-key"
    assert captured["json"]["max_results"] == 5


def test_web_search_reports_rate_limit_without_fabricating(monkeypatch) -> None:
    class FakeResponse:
        status_code = 429
        is_error = True

    class FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(agent_tools.settings, "tavily_api_key", "test-tavily-key")
    monkeypatch.setattr(agent_tools.httpx, "Client", FakeClient)

    with pytest.raises(RuntimeError, match="rate limit"):
        agent_tools.web_search("Gemini news")


def test_web_search_requires_provider_key(monkeypatch) -> None:
    monkeypatch.setattr(agent_tools.settings, "tavily_api_key", "")

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        agent_tools.web_search("Gemini news")


def test_web_search_reports_provider_timeout(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            raise agent_tools.httpx.ReadTimeout("search timed out")

    monkeypatch.setattr(agent_tools.settings, "tavily_api_key", "test-tavily-key")
    monkeypatch.setattr(agent_tools.httpx, "Client", FakeClient)

    with pytest.raises(RuntimeError, match="timed out"):
        agent_tools.web_search("Gemini news")


def test_web_search_validates_query_before_request() -> None:
    with pytest.raises(ValueError, match="1 to 400"):
        agent_tools.web_search(" ")
