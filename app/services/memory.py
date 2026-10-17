from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import EMBEDDING_DIMENSIONS, Memory


def _client() -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be configured to use semantic memory")
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_text(text: str) -> list[float]:
    response = await _client().embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


async def save_memory(session: AsyncSession, user_id: UUID, content: str) -> Memory:
    vector = await embed_text(content)
    memory = Memory(user_id=user_id, content=content, embedding=vector)
    session.add(memory)
    await session.flush()
    return memory


async def search_memories(
    session: AsyncSession, user_id: UUID, query: str, limit: int = 5
) -> list[tuple[str, float]]:
    vector = await embed_text(query)
    distance = Memory.embedding.cosine_distance(vector)
    rows = await session.execute(
        select(Memory.content, distance.label("distance"))
        .where(Memory.user_id == user_id)
        .order_by(distance)
        .limit(limit)
    )
    return [(content, float(distance)) for content, distance in rows.all()]

