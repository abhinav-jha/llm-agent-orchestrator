from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

import app.services.memory as memory


@pytest.mark.asyncio
async def test_embed_text_uses_gemini_retrieval_configuration(monkeypatch) -> None:
    captured = {}
    values = [0.1] * memory.EMBEDDING_DIMENSIONS

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embedding": {"values": values}}

    class FakeClient:
        def __init__(self, timeout) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, url, headers, json) -> FakeResponse:
            captured.update(url=url, headers=headers, json=json)
            return FakeResponse()

    monkeypatch.setattr(memory.settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(memory.settings, "gemini_embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(memory, "EMBEDDING_DIMENSIONS", 1536)
    monkeypatch.setattr(memory.httpx, "AsyncClient", FakeClient)

    result = await memory.embed_text("A saved preference", "RETRIEVAL_DOCUMENT")

    assert result == values
    assert captured["url"].endswith("/models/gemini-embedding-001:embedContent")
    assert captured["headers"] == {"x-goog-api-key": "test-gemini-key"}
    assert captured["json"]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert captured["json"]["outputDimensionality"] == 1536


@pytest.mark.asyncio
async def test_embed_text_rejects_wrong_embedding_dimensions(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embedding": {"values": [0.1]}}

    class FakeClient:
        def __init__(self, timeout) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(memory.settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(memory, "EMBEDDING_DIMENSIONS", 1536)
    monkeypatch.setattr(memory.httpx, "AsyncClient", FakeClient)

    with pytest.raises(RuntimeError, match="invalid embedding"):
        await memory.embed_text("A saved preference", "RETRIEVAL_DOCUMENT")


@pytest.mark.asyncio
async def test_search_memories_uses_query_task_model_and_user_filter(monkeypatch) -> None:
    user_id = uuid4()
    captured = {}
    vector = [0.2] * memory.EMBEDDING_DIMENSIONS

    async def fake_embed(text, task_type):
        captured.update(text=text, task_type=task_type)
        return vector

    class FakeResult:
        def all(self):
            return [("owned fact", 0.15)]

    class FakeSession:
        async def execute(self, statement):
            captured["sql"] = str(statement.compile(dialect=postgresql.dialect()))
            captured["params"] = statement.compile(dialect=postgresql.dialect()).params
            return FakeResult()

    monkeypatch.setattr(memory, "embed_text", fake_embed)
    monkeypatch.setattr(memory.settings, "gemini_embedding_model", "gemini-embedding-001")

    results = await memory.search_memories(FakeSession(), user_id, "query", limit=20)

    assert captured["task_type"] == "RETRIEVAL_QUERY"
    assert captured["text"] == "query"
    assert "long_term_memories.user_id" in captured["sql"]
    assert "long_term_memories.embedding_model" in captured["sql"]
    assert "LIMIT" in captured["sql"]
    assert results == [("owned fact", 0.15)]


@pytest.mark.asyncio
async def test_save_memory_records_gemini_model_and_document_vector(monkeypatch) -> None:
    user_id = uuid4()
    vector = [0.4] * memory.EMBEDDING_DIMENSIONS
    captured = {}

    async def fake_embed(text, task_type):
        captured.update(text=text, task_type=task_type)
        return vector

    class FakeSession:
        def add(self, item):
            captured["memory"] = item

        async def flush(self):
            return None

    monkeypatch.setattr(memory, "embed_text", fake_embed)
    monkeypatch.setattr(memory.settings, "gemini_embedding_model", "gemini-embedding-001")

    saved = await memory.save_memory(FakeSession(), user_id, "fact")

    assert captured["task_type"] == "RETRIEVAL_DOCUMENT"
    assert saved.embedding_model == "gemini-embedding-001"
    assert saved.user_id == user_id


@pytest.mark.asyncio
async def test_reembed_legacy_memories_updates_provider_tag_per_record(monkeypatch) -> None:
    user_id = uuid4()
    old = SimpleNamespace(
        content="old saved fact",
        embedding=[0.2] * 1536,
        embedding_model="legacy-openai-text-embedding-3-small",
    )
    captured = {"queries": []}

    class FakeRows:
        def __init__(self, items):
            self.items = items

        def all(self):
            return self.items

    class FakeSession:
        async def scalars(self, statement):
            query_count = len(captured["queries"])
            captured["queries"].append(
                str(statement.compile(dialect=postgresql.dialect()))
            )
            return FakeRows([old] if query_count == 0 else [])

        async def commit(self):
            captured["commits"] = captured.get("commits", 0) + 1

    async def fake_embed(text, task_type):
        captured.update(text=text, task_type=task_type)
        return [0.3] * 1536

    def add(item):
        captured["added"] = item

    monkeypatch.setattr(memory, "embed_text", fake_embed)
    monkeypatch.setattr(memory.settings, "gemini_embedding_model", "gemini-embedding-001")

    session = FakeSession()
    session.add = add
    count = await memory.reembed_legacy_memories(session, user_id, batch_size=10)

    assert count == 1
    assert old.embedding_model == "gemini-embedding-001"
    assert old.embedding == [0.3] * 1536
    assert "long_term_memories.user_id" in captured["queries"][0]
    assert "long_term_memories.embedding_model" in captured["queries"][0]
    assert captured["commits"] == 1
    assert captured["task_type"] == "RETRIEVAL_DOCUMENT"


@pytest.mark.asyncio
async def test_reembed_legacy_memories_copies_old_table_rows_without_deleting(monkeypatch) -> None:
    old = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        content="old memory table fact",
        embedding=[0.2] * 1536,
        embedding_model="legacy-openai-text-embedding-3-small",
    )
    captured = {"added": [], "commits": 0, "queries": 0}

    class FakeRows:
        def __init__(self, items):
            self.items = items

        def all(self):
            return self.items

    class FakeSession:
        async def scalars(self, _statement):
            captured["queries"] += 1
            return FakeRows([] if captured["queries"] == 1 else [old])

        def add(self, item):
            captured["added"].append(item)

        async def commit(self):
            captured["commits"] += 1

    async def fake_embed(text, task_type):
        assert text == old.content
        assert task_type == "RETRIEVAL_DOCUMENT"
        return [0.3] * 1536

    monkeypatch.setattr(memory, "embed_text", fake_embed)
    monkeypatch.setattr(memory.settings, "gemini_embedding_model", "gemini-embedding-001")

    count = await memory.reembed_legacy_memories(FakeSession(), old.user_id, batch_size=5)

    assert count == 1
    assert len(captured["added"]) == 1
    migrated = captured["added"][0]
    assert migrated.content == old.content
    assert migrated.embedding == [0.3] * 1536
    assert migrated.embedding_model == "gemini-embedding-001"
    assert old.embedding == [0.2] * 1536
    assert old.embedding_model == "migrated-to-gemini-embedding-001"
    assert captured["commits"] == 1


@pytest.mark.asyncio
async def test_reembed_legacy_memories_reuses_existing_gemini_vector(monkeypatch) -> None:
    original_vector = [0.2] * 1536
    old = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        content="existing Gemini fact",
        embedding=original_vector,
        embedding_model="gemini-embedding-001",
    )
    added = []
    query_count = 0

    class FakeRows:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class FakeSession:
        async def scalars(self, _statement):
            nonlocal query_count
            query_count += 1
            return FakeRows([]) if query_count == 1 else FakeRows([old])

        def add(self, item):
            added.append(item)

        async def commit(self):
            return None

    async def fail_if_embedded(*_args):
        raise AssertionError("already-compatible Gemini embedding should be reused")

    monkeypatch.setattr(memory, "embed_text", fail_if_embedded)
    monkeypatch.setattr(memory.settings, "gemini_embedding_model", "gemini-embedding-001")

    count = await memory.reembed_legacy_memories(FakeSession(), old.user_id, batch_size=5)

    assert count == 1
    assert added[0].embedding == original_vector
    assert old.embedding_model == "migrated-to-gemini-embedding-001"
