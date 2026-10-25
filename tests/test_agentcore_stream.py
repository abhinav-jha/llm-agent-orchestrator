import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

import app.api.agentcore as agentcore
from app.models import AgentRun, RunStatus, RunStep


@pytest.mark.asyncio
async def test_run_stream_delivers_terminal_event_and_closes_redis_resources(monkeypatch):
    run_id = uuid4()
    state = {"unsubscribed": False, "pubsub_closed": False, "redis_closed": False}

    async def owned_run(*_args):
        return SimpleNamespace(id=run_id)

    class FakePubSub:
        async def subscribe(self, _channel):
            return None

        async def get_message(self, **_kwargs):
            return {
                "data": json.dumps(
                    {"step_type": "llm_call", "payload": {"content": "thinking"}}
                )
            }

        async def unsubscribe(self, _channel):
            state["unsubscribed"] = True

        async def aclose(self):
            state["pubsub_closed"] = True

    class FakeRedis:
        def pubsub(self):
            return FakePubSub()

        async def aclose(self):
            state["redis_closed"] = True

    class FakeDBSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args):
            return SimpleNamespace(status=RunStatus.RUNNING.value)

        async def refresh(self, run):
            run.status = RunStatus.COMPLETED.value

    monkeypatch.setattr(agentcore, "_owned_run", owned_run)
    monkeypatch.setattr(agentcore, "from_url", lambda *_args, **_kwargs: FakeRedis())
    monkeypatch.setattr(agentcore, "AsyncSessionLocal", FakeDBSession)

    response = await agentcore.stream_run(
        run_id, object(), SimpleNamespace(id=uuid4())
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert any(b'"step_type": "llm_call"' in chunk for chunk in chunks)
    assert any(b'"status": "completed"' in chunk for chunk in chunks)
    assert state == {
        "unsubscribed": True,
        "pubsub_closed": True,
        "redis_closed": True,
    }


@pytest.mark.asyncio
async def test_run_stream_cleans_up_after_client_disconnect(monkeypatch):
    run_id = uuid4()
    state = {"unsubscribed": False, "pubsub_closed": False, "redis_closed": False}

    async def owned_run(*_args):
        return SimpleNamespace(id=run_id)

    class FakePubSub:
        async def subscribe(self, _channel):
            return None

        async def get_message(self, **_kwargs):
            raise asyncio.CancelledError

        async def unsubscribe(self, _channel):
            state["unsubscribed"] = True

        async def aclose(self):
            state["pubsub_closed"] = True

    class FakeRedis:
        def pubsub(self):
            return FakePubSub()

        async def aclose(self):
            state["redis_closed"] = True

    class FakeDBSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args):
            return SimpleNamespace(status=RunStatus.RUNNING.value)

        async def refresh(self, _run):
            return None

    monkeypatch.setattr(agentcore, "_owned_run", owned_run)
    monkeypatch.setattr(agentcore, "from_url", lambda *_args, **_kwargs: FakeRedis())
    monkeypatch.setattr(agentcore, "AsyncSessionLocal", FakeDBSession)

    response = await agentcore.stream_run(
        run_id, object(), SimpleNamespace(id=uuid4())
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == []
    assert state == {
        "unsubscribed": True,
        "pubsub_closed": True,
        "redis_closed": True,
    }


@pytest.mark.asyncio
async def test_run_lookup_is_scoped_to_owner():
    captured = {}

    class FakeSession:
        async def scalar(self, statement):
            captured["sql"] = str(statement)
            return None

    with pytest.raises(HTTPException) as error:
        await agentcore._owned_run(FakeSession(), uuid4(), uuid4())

    assert error.value.status_code == 404
    assert "agent_runs.user_id" in captured["sql"]


@pytest.mark.asyncio
async def test_agent_session_lookup_is_scoped_to_owner():
    captured = {}

    class FakeSession:
        async def scalar(self, statement):
            captured["sql"] = str(statement)
            return None

    with pytest.raises(HTTPException) as error:
        await agentcore.get_agent_session(
            "another-users-session", FakeSession(), SimpleNamespace(id=uuid4())
        )

    assert error.value.status_code == 404
    assert "agent_sessions.user_id" in captured["sql"]


@pytest.mark.asyncio
async def test_memory_listing_is_scoped_to_authenticated_user():
    captured = {}

    class FakeRows:
        def all(self):
            return []

    class FakeSession:
        async def scalars(self, statement):
            captured["sql"] = str(statement)
            return FakeRows()

    result = await agentcore.list_memory(
        FakeSession(), SimpleNamespace(id=uuid4()), limit=100
    )

    assert result == []
    assert "long_term_memories.user_id" in captured["sql"]


@pytest.mark.asyncio
async def test_cancel_run_persists_cancelled_status_and_revokes_task(monkeypatch):
    run = SimpleNamespace(
        id=uuid4(),
        status=RunStatus.QUEUED.value,
        task_id="celery-task-id",
    )
    captured = {"steps": [], "committed": False, "revoked": None, "published": []}

    async def owned_run(*_args):
        return run

    class FakeSession:
        def add(self, step):
            captured["steps"].append(step)

        async def commit(self):
            captured["committed"] = True

    class FakeRedis:
        async def publish(self, channel, data):
            captured["published"].append((channel, json.loads(data)))

    def revoke(task_id, **kwargs):
        captured["revoked"] = (task_id, kwargs)

    monkeypatch.setattr(agentcore, "_owned_run", owned_run)
    monkeypatch.setattr(agentcore.celery_app.control, "revoke", revoke)

    response = await agentcore.cancel_run(
        run.id, FakeSession(), SimpleNamespace(id=uuid4()), FakeRedis()
    )

    assert response.status_code == 204
    assert run.status == RunStatus.CANCELLED.value
    assert captured["committed"]
    assert isinstance(captured["steps"][0], RunStep)
    assert captured["revoked"] == (
        "celery-task-id",
        {"terminate": True, "signal": "SIGTERM"},
    )
    assert captured["published"][0][1]["status"] == RunStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_run_dispatch_failure_persists_failed_run(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    agent_session = SimpleNamespace(id="owned-session")
    captured = {"items": [], "commits": 0}

    class FakeSession:
        async def scalar(self, _statement):
            return agent_session

        def add(self, item):
            captured["items"].append(item)
            if isinstance(item, AgentRun):
                item.id = uuid4()

        async def commit(self):
            captured["commits"] += 1

    class FakeTask:
        def delay(self, _run_id):
            raise ConnectionError("broker unavailable")

    monkeypatch.setattr(agentcore, "execute_agent_run", FakeTask())

    with pytest.raises(HTTPException) as error:
        await agentcore.create_agent_run(
            "owned-session",
            SimpleNamespace(user_message="run this"),
            FakeSession(),
            user,
        )

    run = next(item for item in captured["items"] if isinstance(item, AgentRun))
    trace = next(item for item in captured["items"] if isinstance(item, RunStep))
    assert error.value.status_code == 503
    assert run.status == RunStatus.FAILED.value
    assert run.error == "Could not enqueue agent run"
    assert trace.step_type == "error"
    assert captured["commits"] == 2
