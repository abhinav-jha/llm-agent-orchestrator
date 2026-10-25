import pytest

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
    monkeypatch.setattr(memory.httpx, "AsyncClient", FakeClient)

    result = await memory.embed_text("A saved preference", "RETRIEVAL_DOCUMENT")

    assert result == values
    assert captured["url"].endswith("/models/gemini-embedding-001:embedContent")
    assert captured["headers"] == {"x-goog-api-key": "test-gemini-key"}
    assert captured["json"]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert captured["json"]["outputDimensionality"] == memory.EMBEDDING_DIMENSIONS


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
    monkeypatch.setattr(memory.httpx, "AsyncClient", FakeClient)

    with pytest.raises(RuntimeError, match="unexpected shape"):
        await memory.embed_text("A saved preference", "RETRIEVAL_DOCUMENT")
