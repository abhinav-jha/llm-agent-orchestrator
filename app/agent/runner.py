from __future__ import annotations

import json
import logging
from typing import Any
from openai import AsyncOpenAI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory_manager import (
    append_completed_turn,
    load_short_term_history,
    retrieve_long_term_memories,
    run_channel,
)
from app.agent.tools import dispatch_tool, tool_definitions
from app.core.config import settings
from app.models import AgentRun, AgentSession, RunStatus, RunStep

logger = logging.getLogger(__name__)
MAX_ITERATIONS = 10


async def _record_step(
    session: AsyncSession, redis: Redis, run: AgentRun, step_type: str, payload: dict[str, Any]
) -> None:
    step = RunStep(run_id=run.id, step_type=step_type, payload=payload)
    session.add(step)
    await session.flush()
    await session.commit()
    await redis.publish(
        run_channel(run.id),
        json.dumps({"step_type": step_type, "payload": payload}, ensure_ascii=False),
    )


async def run_agent(
    session: AsyncSession,
    redis: Redis,
    run: AgentRun,
    agent_session: AgentSession,
) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY must be configured to run the agent")
    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        timeout=settings.gemini_request_timeout_seconds,
    )
    try:
        history = await load_short_term_history(redis, agent_session.id)
        memories = await retrieve_long_term_memories(session, run.user_id, run.user_message or "")
        system_prompt = agent_session.system_prompt or "You are a helpful assistant."
        if memories:
            system_prompt += "\n\nRelevant saved user facts:\n" + "\n".join(
                f"- {content}" for content, _distance in memories
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": run.user_message or run.prompt},
        ]
        tools = tool_definitions(agent_session.tools_enabled)
        answer = ""

        for iteration in range(MAX_ITERATIONS):
            await session.refresh(run)
            if run.status == RunStatus.CANCELLED.value:
                return ""

            response = await client.chat.completions.create(
                model=settings.gemini_model,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
            )
            await session.refresh(run)
            if run.status == RunStatus.CANCELLED.value:
                return ""
            if not response.choices:
                raise RuntimeError("Gemini returned no candidate response")
            choice = response.choices[0]
            message = choice.message
            usage = response.usage.total_tokens if response.usage else 0
            run.tokens_used += usage
            call_data = [
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
                for tool_call in (message.tool_calls or [])
            ]
            await _record_step(
                session,
                redis,
                run,
                "llm_call",
                {
                    "iteration": iteration + 1,
                    "model": settings.gemini_model,
                    "content": message.content or "",
                    "tool_calls": call_data,
                    "tokens_used": usage,
                },
            )

            if not message.tool_calls:
                answer = message.content or ""
                if not answer:
                    raise RuntimeError("Gemini returned no final answer")
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": item.id,
                            "type": "function",
                            "function": {
                                "name": item.function.name,
                                "arguments": item.function.arguments,
                            },
                        }
                        for item in message.tool_calls
                    ],
                }
            )
            for tool_call in message.tool_calls:
                await session.refresh(run)
                if run.status == RunStatus.CANCELLED.value:
                    return ""
                tool_name = tool_call.function.name
                await _record_step(
                    session,
                    redis,
                    run,
                    "tool_call",
                    {"name": tool_name, "arguments": tool_call.function.arguments},
                )
                result = await dispatch_tool(
                    tool_name,
                    tool_call.function.arguments,
                    session,
                    run.user_id,
                    run.id,
                    agent_session.tools_enabled,
                )
                await session.refresh(run)
                if run.status == RunStatus.CANCELLED.value:
                    return ""
                await _record_step(
                    session, redis, run, "tool_result", {"name": tool_name, "result": result}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                )

        else:
            raise RuntimeError(f"Agent exceeded {MAX_ITERATIONS} model iterations")
        run.final_answer = answer
        run.result = answer
        await _record_step(session, redis, run, "final_answer", {"answer": answer})
        await session.flush()
        await append_completed_turn(redis, agent_session.id, run.user_message or run.prompt, answer)
        return answer
    finally:
        await client.close()
