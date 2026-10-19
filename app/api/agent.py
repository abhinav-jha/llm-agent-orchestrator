import json
import logging
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, SessionDep
from app.db.session import AsyncSessionLocal
from app.models import AgentRun, Conversation, User
from app.schemas import (
    ChatRequest,
    ConversationDetail,
    ConversationRead,
    MessageRead,
    RunRequest,
    RunRead,
)
from app.services.orchestrator import stream_agent_response
from app.tasks.agent_tasks import process_agent_run

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


async def _owned_conversation(
    session: SessionDep, user_id: UUID, conversation_id: UUID | None
) -> Conversation:
    if conversation_id is None:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        await session.flush()
        return conversation
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        ).with_for_update()
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _sse(event: dict[str, object]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest, session: SessionDep, user: CurrentUser
) -> StreamingResponse:
    conversation = await _owned_conversation(session, user.id, payload.conversation_id)
    await session.commit()
    conversation_id = conversation.id
    user_id = user.id

    async def events() -> AsyncIterator[str]:
        async with AsyncSessionLocal() as stream_session:
            try:
                stream_user = await stream_session.get(User, user_id)
                stream_conversation = await stream_session.get(Conversation, conversation_id)
                if stream_user is None or stream_conversation is None:
                    raise LookupError("Conversation or user no longer exists")
                if stream_conversation.user_id != user_id:
                    raise LookupError("Conversation is not owned by the authenticated user")
                async for event in stream_agent_response(
                    stream_session, stream_user, stream_conversation, payload.prompt
                ):
                    yield _sse(event)
            except Exception:
                await stream_session.rollback()
                logger.exception("Streaming agent request failed")
                yield _sse({"type": "error", "message": "Agent request failed"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations", response_model=list[ConversationRead])
async def list_conversations(session: SessionDep, user: CurrentUser) -> list[Conversation]:
    rows = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
    )
    return list(rows.all())


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID, session: SessionDep, user: CurrentUser
) -> ConversationDetail:
    conversation = await session.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[
            MessageRead.model_validate(message)
            for message in conversation.messages
            if message.role != "tool"
        ],
    )


@router.post("/runs", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: RunRequest, session: SessionDep, user: CurrentUser) -> AgentRun:
    conversation = await _owned_conversation(session, user.id, payload.conversation_id)
    run = AgentRun(
        user_id=user.id,
        conversation_id=conversation.id,
        prompt=payload.prompt,
        status="queued",
    )
    session.add(run)
    await session.commit()
    try:
        task = process_agent_run.delay(str(run.id))
    except Exception as exc:
        run.status = "failed"
        run.error = "Could not enqueue agent run"
        await session.commit()
        logger.exception("Failed to enqueue agent run")
        raise HTTPException(status_code=503, detail="Background worker is unavailable") from exc
    run.task_id = task.id
    await session.commit()
    await session.refresh(run)
    return run


@router.get("/runs/{run_id}", response_model=RunRead)
async def get_run(run_id: UUID, session: SessionDep, user: CurrentUser) -> AgentRun:
    run = await session.scalar(
        select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user.id)
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run
