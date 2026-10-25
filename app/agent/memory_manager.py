from __future__ import annotations

import json
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.memory import search_memories


def history_key(session_id: str) -> str:
    return f"agent:session:{session_id}:history"


def run_channel(run_id: UUID | str) -> str:
    return f"agent:run:{run_id}"


async def load_short_term_history(redis: Redis, session_id: str) -> list[dict[str, str]]:
    stored_turns = await redis.lrange(history_key(session_id), 0, 4)
    turns = [json.loads(turn) for turn in reversed(stored_turns)]
    messages = []
    for turn in turns:
        if isinstance(turn.get("user"), str):
            messages.append({"role": "user", "content": turn["user"]})
        if isinstance(turn.get("assistant"), str):
            messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


async def retrieve_long_term_memories(
    session: AsyncSession, user_id: UUID, query: str
) -> list[tuple[str, float]]:
    return await search_memories(session, user_id, query, limit=3)


async def append_completed_turn(
    redis: Redis, session_id: str, user_message: str, final_answer: str
) -> None:
    key = history_key(session_id)
    await redis.lpush(
        key, json.dumps({"user": user_message, "assistant": final_answer}, ensure_ascii=False)
    )
    await redis.ltrim(key, 0, 19)
