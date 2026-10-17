import asyncio
import logging
from uuid import UUID

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models import AgentRun, Conversation, User
from app.services.orchestrator import stream_agent_response
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _execute_run(run_id: UUID) -> str:
    async with AsyncSessionLocal() as session:
        run = await session.get(AgentRun, run_id)
        if run is None:
            raise LookupError(f"Agent run {run_id} does not exist")

        user = await session.get(User, run.user_id)
        if user is None:
            run.status = "failed"
            run.error = "Run user no longer exists"
            await session.commit()
            raise LookupError(run.error)
        run.status = "running"
        await session.commit()

        conversation = await session.scalar(
            select(Conversation)
            .where(Conversation.id == run.conversation_id)
            .with_for_update()
        )
        if conversation is None:
            run = await session.get(AgentRun, run_id)
            if run is not None:
                run.status = "failed"
                run.error = "Run conversation no longer exists"
                await session.commit()
            raise LookupError(f"Conversation for agent run {run_id} no longer exists")

        try:
            answer = ""
            async for event in stream_agent_response(session, user, conversation, run.prompt):
                if event["type"] == "token":
                    answer += str(event["content"])
            run.status = "completed"
            run.result = answer
            await session.commit()
            return answer
        except Exception as exc:
            await session.rollback()
            run = await session.get(AgentRun, run_id)
            if run is not None:
                run.status = "failed"
                run.error = "Agent execution failed"
                await session.commit()
            logger.exception("Background agent run %s failed", run_id)
            raise exc


@celery_app.task(name="agent.process_run")
def process_agent_run(run_id: str) -> str:
    return asyncio.run(_execute_run(UUID(run_id)))
