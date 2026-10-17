import json

import pytest

from app.services.orchestrator import _execute_tool


class MemorySession:
    async def flush(self) -> None:
        return None


@pytest.mark.asyncio
async def test_unknown_tool_returns_explicit_error() -> None:
    result = await _execute_tool("unexpected_tool", "{}", MemorySession(), user_id=None)

    assert json.loads(result) == {"error": "Unknown tool: unexpected_tool"}


@pytest.mark.asyncio
async def test_invalid_arguments_return_explicit_error() -> None:
    result = await _execute_tool("search_memory", "[]", MemorySession(), user_id=None)

    assert json.loads(result) == {"error": "Tool arguments must be a JSON object"}

