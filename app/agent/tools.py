from __future__ import annotations

import ast
import asyncio
import inspect
import json
import math
from datetime import datetime, timezone
from typing import Any, get_type_hints

import httpx
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import LongTermMemory
from app.services.memory import embed_text


def web_search(query: str) -> str:
    """Search the web using Tavily and return source-linked results."""
    normalized_query = query.strip()
    if not normalized_query or len(normalized_query) > 400:
        raise ValueError("query must contain 1 to 400 non-whitespace characters")
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY must be configured to use web search")

    try:
        with httpx.Client(timeout=settings.web_search_timeout_seconds) as client:
            response = client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
                json={
                    "query": normalized_query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": False,
                },
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError("Web search provider timed out") from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Web search provider request failed") from exc

    if response.status_code == 429:
        raise RuntimeError("Web search provider rate limit exceeded")
    if response.is_error:
        raise RuntimeError(f"Web search provider returned HTTP {response.status_code}")
    try:
        results = response.json()["results"]
        if not isinstance(results, list):
            raise TypeError("results must be an array")
        normalized_results = []
        for result in results:
            if not isinstance(result, dict):
                raise TypeError("result must be an object")
            title, url, content = result.get("title"), result.get("url"), result.get("content")
            if not all(isinstance(value, str) for value in (title, url, content)):
                raise TypeError("result is missing a title, URL, or content")
            normalized_results.append({"title": title, "url": url, "snippet": content})
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Web search provider returned an invalid response") from exc
    return json.dumps(normalized_results, ensure_ascii=False)


def _calculate(node: ast.AST) -> int | float:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        if not math.isfinite(node.value):
            raise ValueError("expression contains a non-finite number")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 100 or abs(left) > 1e100:
                raise ValueError("exponent or base is too large")
            result = left**right
        else:
            raise ValueError("unsupported arithmetic operator")
        if not isinstance(result, (int, float)) or abs(result) > 1e100:
            raise ValueError("result is too large")
        return result
    raise ValueError("expression contains an unsupported operation")


def calculator(expression: str) -> str:
    """Evaluate a numeric expression using basic arithmetic operators."""
    if not expression.strip() or len(expression) > 500:
        raise ValueError("expression must contain 1 to 500 characters")
    tree = ast.parse(expression, mode="eval")
    return str(_calculate(tree.body))


def get_current_datetime() -> str:
    """Return the current UTC time as an ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def summarise_text(text: str, max_words: int = 100) -> str:
    """Summarise text with the configured Gemini chat model."""
    if not text.strip() or len(text) > 100_000:
        raise ValueError("text must contain 1 to 100000 characters")
    if not 1 <= max_words <= 1000:
        raise ValueError("max_words must be between 1 and 1000")
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY must be configured to summarise text")
    with OpenAI(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        timeout=settings.gemini_request_timeout_seconds,
    ) as client:
        response = client.chat.completions.create(
            model=settings.gemini_model,
            messages=[
                {
                    "role": "system",
                    "content": f"Summarise the user's text in no more than {max_words} words.",
                },
                {"role": "user", "content": text},
            ],
        )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Summary model returned no text")
    return answer


def remember_fact(fact: str) -> str:
    """Save a fact the user asked the agent to remember."""
    if not fact.strip() or len(fact) > 4000:
        raise ValueError("fact must contain 1 to 4000 characters")
    return fact.strip()


TOOL_REGISTRY = {
    function.__name__: function
    for function in (web_search, calculator, get_current_datetime, summarise_text, remember_fact)
}


def _json_type(annotation: Any) -> str:
    return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(
        annotation, "string"
    )


def tool_definitions(enabled_tools: list[str]) -> list[dict[str, Any]]:
    definitions = []
    for name in enabled_tools:
        function = TOOL_REGISTRY.get(name)
        if function is None:
            raise ValueError(f"Unknown tool: {name}")
        signature = inspect.signature(function)
        annotations = get_type_hints(function)
        properties = {
            parameter_name: {
                "type": _json_type(annotations.get(parameter_name, str)),
                "description": parameter_name.replace("_", " "),
            }
            for parameter_name in signature.parameters
        }
        required = [
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        ]
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": inspect.getdoc(function) or name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return definitions


async def dispatch_tool(
    name: str,
    raw_arguments: str,
    session: AsyncSession,
    user_id,
    run_id,
    enabled_tools: list[str],
) -> str:
    try:
        function = TOOL_REGISTRY.get(name)
        if function is None or name not in enabled_tools:
            raise ValueError(f"Unknown or disabled tool: {name}")
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        signature = inspect.signature(function)
        bound = signature.bind(**arguments)
        bound.apply_defaults()
        annotations = get_type_hints(function)
        for argument_name, value in bound.arguments.items():
            expected_type = annotations.get(argument_name, str)
            if expected_type is int and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{argument_name} must be an integer")
            if expected_type is str and not isinstance(value, str):
                raise ValueError(f"{argument_name} must be a string")

        result = await asyncio.to_thread(function, **bound.arguments)
        if name == "remember_fact":
            vector = await embed_text(result, "RETRIEVAL_DOCUMENT")
            session.add(
                LongTermMemory(
                    user_id=user_id,
                    content=result,
                    embedding=vector,
                    embedding_model=settings.gemini_embedding_model,
                    source_run_id=run_id,
                )
            )
            await session.flush()
            return f"Remembered: {result}"
        return str(result)
    except Exception as exc:
        return f"ERROR: {exc}"
