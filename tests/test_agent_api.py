from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.api.agent as agent_api
from app.models import Conversation, User
from app.schemas import ChatRequest


@pytest.mark.asyncio
async def test_chat_stream_commits_conversation_and_uses_stream_session(monkeypatch) -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    user = SimpleNamespace(id=user_id)
    conversation = SimpleNamespace(id=conversation_id, user_id=user_id)

    class FakeSession:
        def __init__(self) -> None:
            self.committed = False
            self.rolled_back = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def scalar(self, _query):
            return conversation

        async def get(self, model, _object_id):
            return user if model is User else conversation

        async def commit(self) -> None:
            self.committed = True

        async def rollback(self) -> None:
            self.rolled_back = True

    request_session = FakeSession()
    stream_session = FakeSession()
    monkeypatch.setattr(agent_api, "AsyncSessionLocal", lambda: stream_session)

    async def fake_stream_agent_response(session, stream_user, stream_conversation, prompt):
        assert session is stream_session
        assert stream_user is user
        assert stream_conversation is conversation
        assert prompt == "Hello"
        yield {"type": "done", "content": "Hi"}

    monkeypatch.setattr(agent_api, "stream_agent_response", fake_stream_agent_response)

    response = await agent_api.chat_stream(
        ChatRequest(prompt="Hello", conversation_id=conversation_id),
        request_session,
        user,
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert request_session.committed
    assert not stream_session.rolled_back
    assert response.headers["X-Conversation-ID"] == str(conversation_id)
    assert b'data: {"type": "done", "content": "Hi"}' in chunks[0]
