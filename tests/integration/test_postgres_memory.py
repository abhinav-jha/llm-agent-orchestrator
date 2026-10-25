import os
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.sql import text

from app.db.base import Base
from app.models import User

pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Set TEST_DATABASE_URL to run PostgreSQL/pgvector integration tests",
)


@pytest.fixture
async def postgres_session_factory():
    database_url = os.environ["TEST_DATABASE_URL"]
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
    schema_name = f"agent_test_{uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    scoped_engine = None
    try:
        async with admin_engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
        scoped_engine = create_async_engine(
            database_url,
            connect_args={"server_settings": {"search_path": f"{schema_name},public"}},
        )
        async with scoped_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        yield async_sessionmaker(scoped_engine, expire_on_commit=False)
    finally:
        if scoped_engine is not None:
            await scoped_engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
        await admin_engine.dispose()


@pytest.mark.asyncio
async def test_postgres_pgvector_memory_persists_and_filters_by_user(
    postgres_session_factory, monkeypatch
):
    import app.services.memory as memory

    user_one_id = uuid4()
    user_two_id = uuid4()
    vector = [0.125] * memory.EMBEDDING_DIMENSIONS

    async def fake_embed(_text, _task_type):
        return vector

    monkeypatch.setattr(memory, "embed_text", fake_embed)
    monkeypatch.setattr(memory.settings, "gemini_embedding_model", "gemini-embedding-001")

    async with postgres_session_factory() as session:
        session.add_all(
            [
                User(id=user_one_id, email=f"{user_one_id}@example.test", password_hash="x"),
                User(id=user_two_id, email=f"{user_two_id}@example.test", password_hash="x"),
            ]
        )
        await session.commit()
        await memory.save_memory(session, user_one_id, "first user's fact")
        await memory.save_memory(session, user_two_id, "second user's fact")
        await session.commit()

        results = await memory.search_memories(
            session, user_one_id, "query for the first user", limit=3
        )

    assert [content for content, _distance in results] == ["first user's fact"]
