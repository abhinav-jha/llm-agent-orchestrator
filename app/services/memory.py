from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import EMBEDDING_DIMENSIONS, Memory


GEMINI_EMBEDDING_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)


async def embed_text(text: str, task_type: str) -> list[float]:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY must be configured to use semantic memory")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            GEMINI_EMBEDDING_URL.format(model=settings.gemini_embedding_model),
            headers={"x-goog-api-key": settings.gemini_api_key},
            json={
                "model": f"models/{settings.gemini_embedding_model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                "outputDimensionality": EMBEDDING_DIMENSIONS,
            },
        )
    response.raise_for_status()
    payload = response.json()
    values = payload.get("embedding", {}).get("values")
    if not isinstance(values, list) or len(values) != EMBEDDING_DIMENSIONS:
        raise RuntimeError("Gemini returned an embedding with an unexpected shape")
    if any(not isinstance(value, (int, float)) for value in values):
        raise RuntimeError("Gemini returned a non-numeric embedding")
    return values


async def save_memory(session: AsyncSession, user_id: UUID, content: str) -> Memory:
    vector = await embed_text(content, "RETRIEVAL_DOCUMENT")
    memory = Memory(
        user_id=user_id,
        content=content,
        embedding=vector,
        embedding_model=settings.gemini_embedding_model,
    )
    session.add(memory)
    await session.flush()
    return memory


async def search_memories(
    session: AsyncSession, user_id: UUID, query: str, limit: int = 5
) -> list[tuple[str, float]]:
    vector = await embed_text(query, "RETRIEVAL_QUERY")
    distance = Memory.embedding.cosine_distance(vector)
    rows = await session.execute(
        select(Memory.content, distance.label("distance"))
        .where(
            Memory.user_id == user_id,
            Memory.embedding_model == settings.gemini_embedding_model,
        )
        .order_by(distance)
        .limit(limit)
    )
    return [(content, float(distance)) for content, distance in rows.all()]
