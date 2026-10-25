from typing import Optional
from uuid import UUID

import math

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import EMBEDDING_DIMENSIONS, LongTermMemory, Memory


LEGACY_EMBEDDING_MODELS = {
    "legacy-openai-text-embedding-3-small",
    "text-embedding-3-small",
    "legacy-openai",
}
MIGRATED_EMBEDDING_MODEL_PREFIX = "migrated-to-"


async def embed_text(text: str, task_type: str) -> list[float]:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY must be configured to use semantic memory")
    if task_type not in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"}:
        raise ValueError("Unsupported Gemini embedding task type")
    if not text.strip():
        raise ValueError("Embedding input must not be empty")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_embedding_model}:embedContent"
    )
    payload = {
        "model": f"models/{settings.gemini_embedding_model}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": EMBEDDING_DIMENSIONS,
    }
    timeout = httpx.Timeout(settings.gemini_request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url, headers={"x-goog-api-key": settings.gemini_api_key}, json=payload
        )
        response.raise_for_status()
    try:
        embedding = response.json()["embedding"]["values"]
        vector = [float(value) for value in embedding]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Gemini returned an invalid embedding response") from exc
    if len(vector) != EMBEDDING_DIMENSIONS or not all(math.isfinite(value) for value in vector):
        raise RuntimeError(
            f"Gemini returned an invalid embedding: expected "
            f"{EMBEDDING_DIMENSIONS}, got {len(vector)}"
        )
    return vector


async def save_memory(
    session: AsyncSession,
    user_id: UUID,
    content: str,
    source_run_id: Optional[UUID] = None,
) -> LongTermMemory:
    vector = await embed_text(content, "RETRIEVAL_DOCUMENT")
    memory = LongTermMemory(
        user_id=user_id,
        content=content,
        embedding=vector,
        embedding_model=settings.gemini_embedding_model,
        source_run_id=source_run_id,
    )
    session.add(memory)
    await session.flush()
    return memory


async def search_memories(
    session: AsyncSession, user_id: UUID, query: str, limit: int = 5
) -> list[tuple[str, float]]:
    vector = await embed_text(query, "RETRIEVAL_QUERY")
    distance = LongTermMemory.embedding.cosine_distance(vector)
    rows = await session.execute(
        select(LongTermMemory.content, distance.label("distance"))
        .where(LongTermMemory.user_id == user_id)
        .where(LongTermMemory.embedding_model == settings.gemini_embedding_model)
        .order_by(distance)
        .limit(min(max(limit, 1), 3))
    )
    return [(content, float(distance)) for content, distance in rows.all()]


async def reembed_legacy_memories(
    session: AsyncSession, user_id: UUID, batch_size: int = 50
) -> int:
    if batch_size < 1 or batch_size > 500:
        raise ValueError("batch_size must be between 1 and 500")
    rows = await session.scalars(
        select(LongTermMemory)
        .where(LongTermMemory.user_id == user_id)
        .where(LongTermMemory.embedding_model.in_(LEGACY_EMBEDDING_MODELS))
        .order_by(LongTermMemory.created_at, LongTermMemory.id)
        .limit(batch_size)
    )
    count = 0
    for memory in rows.all():
        memory.embedding = await embed_text(memory.content, "RETRIEVAL_DOCUMENT")
        memory.embedding_model = settings.gemini_embedding_model
        await session.commit()
        count += 1

    legacy_rows = await session.scalars(
        select(Memory)
        .where(Memory.user_id == user_id)
        .where(
            Memory.embedding_model.in_(
                LEGACY_EMBEDDING_MODELS | {settings.gemini_embedding_model}
            )
        )
        .order_by(Memory.created_at, Memory.id)
        .limit(max(batch_size - count, 0))
    )
    for legacy_memory in legacy_rows.all():
        if legacy_memory.embedding_model == settings.gemini_embedding_model:
            vector = legacy_memory.embedding
        else:
            vector = await embed_text(legacy_memory.content, "RETRIEVAL_DOCUMENT")
        session.add(
            LongTermMemory(
                user_id=legacy_memory.user_id,
                content=legacy_memory.content,
                embedding=vector,
                embedding_model=settings.gemini_embedding_model,
            )
        )
        legacy_memory.embedding_model = (
            f"{MIGRATED_EMBEDDING_MODEL_PREFIX}{settings.gemini_embedding_model}"
        )
        await session.commit()
        count += 1
    return count
