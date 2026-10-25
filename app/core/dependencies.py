from collections.abc import AsyncIterator

from redis.asyncio import Redis, from_url

from app.core.config import settings


async def get_redis() -> AsyncIterator[Redis]:
    client = from_url(settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()
