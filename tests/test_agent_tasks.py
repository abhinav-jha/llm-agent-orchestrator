from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.tasks.agent_tasks as tasks
from app.models import AgentRun, AgentSession, RunStatus, RunStep


def task_fixture(monkeypatch, run):
    agent_session = SimpleNamespace(id=run.session_id, tools_enabled=[])
    steps = []
    events = []
    closed = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _query):
            return run

        async def get(self, model, _id):
            if model is AgentSession:
                return agent_session
            if model is AgentRun:
                return run
            return None

        def add(self, item):
            steps.append(item)

        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def refresh(self, _item):
            return None

    class FakeRedis:
        async def publish(self, channel, data):
            events.append((channel, data))

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(tasks, "AsyncSessionLocal", FakeSession)
    monkeypatch.setattr(tasks, "from_url", lambda *args, **kwargs: FakeRedis())
    return agent_session, steps, events, closed


@pytest.mark.asyncio
async def test_celery_task_persists_success_status_and_terminal_event(monkeypatch):
    run = SimpleNamespace(
        id=uuid4(),
        session_id="session-1",
        status=RunStatus.QUEUED.value,
        final_answer=None,
        result=None,
        error=None,
    )
    _agent_session, _steps, events, closed = task_fixture(monkeypatch, run)

    async def fake_run_agent(_session, _redis, current_run, _agent_session):
        assert current_run.status == RunStatus.RUNNING.value
        return "complete"

    monkeypatch.setattr(tasks, "run_agent", fake_run_agent)

    result = await tasks._execute_agentcore_run(run.id)

    assert result == "complete"
    assert run.status == RunStatus.COMPLETED.value
    assert run.final_answer == "complete"
    assert run.result == "complete"
    assert '"status": "completed"' in events[-1][1]
    assert closed == [True]


@pytest.mark.asyncio
async def test_celery_task_persists_failure_and_error_trace(monkeypatch):
    run = SimpleNamespace(
        id=uuid4(),
        session_id="session-1",
        status=RunStatus.QUEUED.value,
        final_answer=None,
        result=None,
        error=None,
    )
    _agent_session, steps, events, _closed = task_fixture(monkeypatch, run)

    async def fail_run_agent(*_args):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(tasks, "run_agent", fail_run_agent)

    with pytest.raises(TimeoutError):
        await tasks._execute_agentcore_run(run.id)

    assert run.status == RunStatus.FAILED.value
    assert run.error == "Agent execution failed: TimeoutError"
    assert any(isinstance(step, RunStep) and step.step_type == "error" for step in steps)
    assert '"status": "failed"' in events[-1][1]


@pytest.mark.asyncio
async def test_celery_task_does_not_mark_cancelled_run_completed(monkeypatch):
    run = SimpleNamespace(
        id=uuid4(),
        session_id="session-1",
        status=RunStatus.QUEUED.value,
        final_answer=None,
        result=None,
        error=None,
    )
    _agent_session, _steps, events, _closed = task_fixture(monkeypatch, run)

    async def cancelled_run(_session, _redis, current_run, _agent_session):
        current_run.status = RunStatus.CANCELLED.value
        return ""

    monkeypatch.setattr(tasks, "run_agent", cancelled_run)

    await tasks._execute_agentcore_run(run.id)

    assert run.status == RunStatus.CANCELLED.value
    assert run.final_answer is None
    assert '"status": "cancelled"' in events[-1][1]


@pytest.mark.asyncio
async def test_celery_duplicate_task_does_not_execute_again(monkeypatch):
    run = SimpleNamespace(
        id=uuid4(),
        session_id="session-1",
        status=RunStatus.RUNNING.value,
        final_answer=None,
        result=None,
        error=None,
    )
    _agent_session, _steps, events, closed = task_fixture(monkeypatch, run)
    invoked = []

    async def should_not_run(*_args):
        invoked.append(True)
        return "unexpected"

    monkeypatch.setattr(tasks, "run_agent", should_not_run)

    result = await tasks._execute_agentcore_run(run.id)

    assert result == ""
    assert invoked == []
    assert events == []
    assert closed == [True]
