from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from celery.exceptions import CeleryError
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis, from_url
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.agent.memory_manager import run_channel
from app.agent.tools import TOOL_REGISTRY
from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.core.dependencies import get_redis
from app.db.session import AsyncSessionLocal
from app.models import AgentRun, AgentSession, LongTermMemory, RunStatus, RunStep, User
from app.schemas import (
    AgentRunAccepted,
    AgentRunCreate,
    AgentRunStatus,
    AgentSessionCreate,
    AgentSessionDetail,
    AgentSessionRead,
    AgentRunSummary,
    LongTermMemoryRead,
    RunStepRead,
)
from app.services.memory import search_memories
from app.tasks.agent_tasks import execute_agent_run
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent sessions", "runs", "memory"])
RedisDep = Annotated[Redis, Depends(get_redis)]


def _session_read(agent_session: AgentSession) -> AgentSessionRead:
    return AgentSessionRead(
        id=agent_session.id,
        name=agent_session.name,
        system_prompt=agent_session.system_prompt,
        tools_enabled=agent_session.tools_enabled,
        created_at=agent_session.created_at,
    )


@router.get("/sessions", response_model=list[AgentSessionRead], summary="List agent sessions")
async def list_sessions(session: SessionDep, user: CurrentUser) -> list[AgentSessionRead]:
    rows = await session.scalars(
        select(AgentSession)
        .where(AgentSession.user_id == user.id)
        .order_by(AgentSession.created_at.desc())
    )
    return [_session_read(item) for item in rows.all()]


@router.post(
    "/sessions",
    response_model=AgentSessionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create an agent session",
)
async def create_session(
    payload: AgentSessionCreate, session: SessionDep, user: CurrentUser
) -> AgentSessionRead:
    unknown = sorted(set(payload.tools_enabled) - TOOL_REGISTRY.keys())
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Unknown tools", "tools": unknown},
        )
    agent_session = AgentSession(
        user_id=user.id,
        name=payload.name,
        system_prompt=payload.system_prompt,
        tools_enabled=list(dict.fromkeys(payload.tools_enabled)),
    )
    session.add(agent_session)
    await session.commit()
    await session.refresh(agent_session)
    return _session_read(agent_session)


@router.get(
    "/sessions/{session_id}", response_model=AgentSessionDetail, summary="Get agent session"
)
async def get_agent_session(
    session_id: str, session: SessionDep, user: CurrentUser
) -> AgentSessionDetail:
    agent_session = await session.scalar(
        select(AgentSession)
        .options(selectinload(AgentSession.runs))
        .where(AgentSession.id == session_id, AgentSession.user_id == user.id)
    )
    if agent_session is None:
        raise HTTPException(status_code=404, detail="Agent session not found")
    return AgentSessionDetail(
        **_session_read(agent_session).model_dump(),
        runs=[
            AgentRunSummary(
                id=run.id,
                user_message=run.user_message or run.prompt,
                final_answer=run.final_answer or run.result,
                status=run.status,
                tokens_used=run.tokens_used,
                created_at=run.created_at,
            )
            for run in sorted(agent_session.runs, key=lambda item: item.created_at, reverse=True)
        ],
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete agent session and its runs",
)
async def delete_agent_session(
    session_id: str, session: SessionDep, user: CurrentUser
) -> Response:
    agent_session = await session.scalar(
        select(AgentSession).where(
            AgentSession.id == session_id, AgentSession.user_id == user.id
        )
    )
    if agent_session is None:
        raise HTTPException(status_code=404, detail="Agent session not found")
    await session.delete(agent_session)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/sessions/{session_id}/run",
    response_model=AgentRunAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an agent run",
)
async def create_agent_run(
    session_id: str,
    payload: AgentRunCreate,
    session: SessionDep,
    user: CurrentUser,
) -> AgentRunAccepted:
    agent_session = await session.scalar(
        select(AgentSession).where(
            AgentSession.id == session_id, AgentSession.user_id == user.id
        )
    )
    if agent_session is None:
        raise HTTPException(status_code=404, detail="Agent session not found")
    run = AgentRun(
        user_id=user.id,
        session_id=agent_session.id,
        user_message=payload.user_message,
        prompt=payload.user_message,
        status=RunStatus.QUEUED.value,
        tokens_used=0,
    )
    session.add(run)
    await session.commit()
    try:
        task = execute_agent_run.delay(str(run.id))
    except CeleryError as exc:
        run.status = RunStatus.FAILED.value
        run.error = "Could not enqueue agent run"
        session.add(
            RunStep(
                run_id=run.id,
                step_type="error",
                payload={"error": "Could not enqueue agent run"},
            )
        )
        await session.commit()
        logger.exception("Could not enqueue agent run %s", run.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background worker is unavailable",
        ) from exc
    except Exception as exc:
        run.status = RunStatus.FAILED.value
        run.error = "Could not enqueue agent run"
        session.add(
            RunStep(
                run_id=run.id,
                step_type="error",
                payload={"error": "Could not enqueue agent run"},
            )
        )
        await session.commit()
        logger.exception("Could not enqueue agent run %s", run.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background worker is unavailable",
        ) from exc
    run.task_id = task.id
    await session.commit()
    return AgentRunAccepted(run_id=run.id)


async def _owned_run(session: SessionDep, run_id: UUID, user_id: UUID) -> AgentRun:
    run = await session.scalar(
        select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id)
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


@router.get("/runs/{run_id}/status", response_model=AgentRunStatus, summary="Get run status")
async def get_run_status(
    run_id: UUID, session: SessionDep, user: CurrentUser
) -> AgentRunStatus:
    run = await _owned_run(session, run_id, user.id)
    count = await session.scalar(select(func.count(RunStep.id)).where(RunStep.run_id == run.id))
    return AgentRunStatus(
        id=run.id,
        status=run.status,
        tokens_used=run.tokens_used,
        step_count=count or 0,
        created_at=run.created_at,
    )


@router.get(
    "/runs/{run_id}/steps", response_model=list[RunStepRead], summary="Get run execution trace"
)
async def get_run_steps(
    run_id: UUID, session: SessionDep, user: CurrentUser
) -> list[RunStepRead]:
    await _owned_run(session, run_id, user.id)
    rows = await session.scalars(
        select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.occurred_at, RunStep.id)
    )
    return [
        RunStepRead(
            id=step.id,
            step_type=step.step_type,
            payload=step.payload,
            occurred_at=step.occurred_at,
        )
        for step in rows.all()
    ]


@router.get("/runs/{run_id}/stream", summary="Stream run execution events")
async def stream_run(
    run_id: UUID, session: SessionDep, user: CurrentUser
) -> StreamingResponse:
    await _owned_run(session, run_id, user.id)

    async def events() -> AsyncIterator[bytes]:
        redis = None
        pubsub = None
        try:
            redis = from_url(settings.redis_url, decode_responses=True)
            pubsub = redis.pubsub()
            await pubsub.subscribe(run_channel(run_id))
            async with AsyncSessionLocal() as stream_session:
                current = await stream_session.get(AgentRun, run_id)
                if current is None:
                    return
                if current.status in {
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                }:
                    yield _sse_done(current.status)
                    return
                while True:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
                    if message and message.get("data"):
                        data = str(message["data"])
                        yield f"data: {data}\n\n".encode("utf-8")
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            event = {}
                        if event.get("step_type") == "done":
                            return
                    await stream_session.refresh(current)
                    if current.status in {
                        RunStatus.COMPLETED.value,
                        RunStatus.FAILED.value,
                        RunStatus.CANCELLED.value,
                    }:
                        yield _sse_done(current.status)
                        return
        except asyncio.CancelledError:
            return
        finally:
            try:
                if pubsub is not None:
                    try:
                        await pubsub.unsubscribe(run_channel(run_id))
                    finally:
                        await pubsub.aclose()
            finally:
                if redis is not None:
                    await redis.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_done(run_status: str) -> bytes:
    return (
        f"data: {json.dumps({'step_type': 'done', 'status': run_status})}\n\n".encode("utf-8")
    )


@router.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Cancel an agent run",
)
async def cancel_run(
    run_id: UUID, session: SessionDep, user: CurrentUser, redis: RedisDep
) -> Response:
    run = await _owned_run(session, run_id, user.id)
    if run.status not in {RunStatus.QUEUED.value, RunStatus.RUNNING.value}:
        raise HTTPException(status_code=409, detail="Only queued or running runs can be cancelled")
    run.status = RunStatus.CANCELLED.value
    session.add(
        RunStep(run_id=run.id, step_type="cancelled", payload={"status": run.status})
    )
    await session.commit()
    if run.task_id:
        celery_app.control.revoke(run.task_id, terminate=True, signal="SIGTERM")
    await redis.publish(
        run_channel(run.id),
        json.dumps({"step_type": "done", "status": RunStatus.CANCELLED.value}),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/memory", response_model=list[LongTermMemoryRead], summary="List long-term memory")
async def list_memory(
    session: SessionDep, user: CurrentUser, limit: int = Query(default=100, ge=1, le=500)
) -> list[LongTermMemoryRead]:
    rows = await session.scalars(
        select(LongTermMemory)
        .where(LongTermMemory.user_id == user.id)
        .order_by(LongTermMemory.created_at.desc())
        .limit(limit)
    )
    return [
        LongTermMemoryRead(
            id=item.id,
            content=item.content,
            source_run_id=item.source_run_id,
            created_at=item.created_at,
        )
        for item in rows.all()
    ]


@router.get(
    "/memory/search", response_model=list[dict[str, object]], summary="Search long-term memory"
)
async def search_memory(
    session: SessionDep,
    user: CurrentUser,
    q: str = Query(min_length=1, max_length=20_000),
) -> list[dict[str, object]]:
    results = await search_memories(session, user.id, q.strip(), limit=5)
    return [{"content": content, "distance": distance} for content, distance in results]


@router.delete(
    "/memory/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete long-term memory",
)
async def delete_memory(
    memory_id: str, session: SessionDep, user: CurrentUser
) -> Response:
    memory = await session.scalar(
        select(LongTermMemory).where(
            LongTermMemory.id == memory_id, LongTermMemory.user_id == user.id
        )
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    await session.delete(memory)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
