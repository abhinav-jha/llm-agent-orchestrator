import asyncio
import json
import logging
from uuid import UUID

from sqlalchemy import select
from redis.asyncio import from_url

from app.agent.memory_manager import run_channel
from app.agent.runner import run_agent
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models import AgentRun, AgentSession, Conversation, RunStatus, RunStep, User
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


async def _execute_agentcore_run(run_id: UUID) -> str:
    redis = from_url(settings.redis_url, decode_responses=True)
    try:
        async with AsyncSessionLocal() as session:
            run = await session.scalar(
                select(AgentRun).where(AgentRun.id == run_id).with_for_update()
            )
            if run is None:
                raise LookupError(f"Agent run {run_id} does not exist")
            if run.status != RunStatus.QUEUED.value:
                return run.final_answer or run.result or ""

            agent_session = await session.get(AgentSession, run.session_id)
            if agent_session is None:
                run.status = RunStatus.FAILED.value
                run.error = "Agent session no longer exists"
                session.add(
                    RunStep(
                        run_id=run.id,
                        step_type="error",
                        payload={"error": run.error},
                    )
                )
                await session.commit()
                await redis.publish(
                    run_channel(run.id),
                    json.dumps(
                        {"step_type": "error", "payload": {"error": run.error}}
                    ),
                )
                await redis.publish(
                    run_channel(run.id),
                    json.dumps({"step_type": "done", "status": run.status}),
                )
                return ""
            run.status = RunStatus.RUNNING.value
            await session.commit()

            try:
                answer = await run_agent(session, redis, run, agent_session)
                await session.refresh(run)
                if run.status != RunStatus.CANCELLED.value:
                    run.status = RunStatus.COMPLETED.value
                    run.final_answer = answer
                    run.result = answer
                await session.commit()
                await redis.publish(
                    run_channel(run.id),
                    json.dumps({"step_type": "done", "status": run.status}),
                )
                return answer
            except Exception as exc:
                await session.rollback()
                run = await session.get(AgentRun, run_id)
                if run is not None and run.status != RunStatus.CANCELLED.value:
                    run.status = RunStatus.FAILED.value
                    run.error = f"Agent execution failed: {type(exc).__name__}"
                    session.add(
                        RunStep(
                            run_id=run.id,
                            step_type="error",
                            payload={"error": run.error},
                        )
                    )
                    await session.commit()
                    try:
                        await redis.publish(
                            run_channel(run.id),
                            json.dumps(
                                {
                                    "step_type": "error",
                                    "payload": {"error": run.error},
                                }
                            ),
                        )
                        await redis.publish(
                            run_channel(run.id),
                            json.dumps({"step_type": "done", "status": run.status}),
                        )
                    except Exception:
                        logger.exception("Failed to publish terminal event for run %s", run_id)
                logger.exception("Agent run %s failed", run_id)
                raise
    finally:
        await redis.aclose()


@celery_app.task(name="execute_agent_run")
def execute_agent_run(run_id: str) -> str:
    return asyncio.run(_execute_agentcore_run(UUID(run_id)))
