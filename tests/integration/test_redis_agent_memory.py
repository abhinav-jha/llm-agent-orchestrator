import json
import os
from uuid import uuid4

import pytest
from redis.asyncio import from_url

from app.agent.memory_manager import (
    append_completed_turn,
    history_key,
    load_short_term_history,
    run_channel,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_REDIS_URL"),
    reason="Set TEST_REDIS_URL to run Redis integration tests",
)


@pytest.mark.asyncio
async def test_redis_history_and_run_event_round_trip():
    client = from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    session_id = f"integration-{uuid4()}"
    run_id = uuid4()
    pubsub = client.pubsub()
    key = history_key(session_id)
    channel = run_channel(run_id)
    try:
        await append_completed_turn(client, session_id, "hello", "hello back")
        history = await load_short_term_history(client, session_id)
        assert history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello back"},
        ]

        await pubsub.subscribe(channel)
        await client.publish(channel, json.dumps({"step_type": "done", "status": "completed"}))
        message = None
        for _ in range(10):
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0
            )
            if message:
                break
        assert message is not None
        assert json.loads(message["data"]) == {
            "step_type": "done",
            "status": "completed",
        }
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        await client.delete(key)
        await client.aclose()
