# LLM Agent Orchestrator

FastAPI agent backend using PostgreSQL + pgvector, Redis, Celery, Gemini, and an optional React
chat frontend. Gemini is the deliberate model provider for chat, tool calling, summaries, and
embeddings. This intentionally differs from the supplied case study's OpenAI provider choice.

## Features

- RS256 bearer authentication, registration, and token issuance.
- Gemini Flash tool-calling agent sessions with configurable prompts and enabled tools.
- Gemini `gemini-embedding-001` vectors (1536 dimensions) for user-scoped semantic memory.
- Persisted run steps, token counts, background Celery runs, status/trace APIs, cancellation,
  and Redis-published SSE events.
- Redis short-term conversation history and long-term pgvector memory.
- Supplementary React/Vite chat frontend served by Nginx with an API/SSE proxy. The backend
  remains the primary case-study deliverable.

## Run with Docker Compose

Requirements: Docker Compose, a Gemini API key, and a Tavily API key for live web search.

```sh
cp -n .env.example .env
mkdir -p .secrets
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out .secrets/jwt-private.pem
openssl pkey -in .secrets/jwt-private.pem -pubout -out .secrets/jwt-public.pem
# Set GEMINI_API_KEY and TAVILY_API_KEY in .env.
docker compose up --build
```

Get the Gemini key from Google AI Studio and the search key from Tavily. Without
`TAVILY_API_KEY`, the `web_search` tool returns an explicit configuration error; it does not
invent search results.

The backend API docs are at <http://localhost:8000/docs> and health check at
<http://localhost:8000/health>. The supplementary frontend is at <http://localhost:3000>.

## Run the frontend separately

With the backend running on port 8000:

```sh
cd frontend
npm install
npm run dev
```

The Vite development server proxies `/api` to `http://localhost:8000`. Set
`VITE_BACKEND_ORIGIN` to use a different backend origin.

## Agent APIs

Routes are under `/api/v1`. Protected routes require a bearer token.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/auth/register` | Register and receive an access token |
| `POST` | `/auth/token` | Obtain an access token |
| `POST` | `/chat/stream` | Run an interactive tool-using conversation as SSE |
| `GET`, `POST` | `/sessions` | List or create owned agent sessions |
| `GET`, `DELETE` | `/sessions/{id}` | Read or delete an owned session |
| `POST` | `/sessions/{id}/run` | Enqueue a run for an agent session |
| `GET` | `/runs/{id}/status` | Read run status, token usage, and step count |
| `GET` | `/runs/{id}/steps` | Read persisted execution steps |
| `GET` | `/runs/{id}/stream` | Stream run events over SSE |
| `DELETE` | `/runs/{id}` | Cancel an owned queued or running run |
| `GET` | `/memory`, `/memory/search` | List or semantically search owned memories |
| `DELETE` | `/memory/{id}` | Delete an owned long-term memory |

Legacy conversation browsing and background-run endpoints remain available for compatibility.
The older `/auth/login` route remains as an alias for `/auth/token`.

## Gemini and search configuration

The chat model defaults to `gemini-2.5-flash`; embeddings use `gemini-embedding-001` with an
explicit 1536-dimensional output, supported by Google's embedding API. The existing vector
columns are 1536 dimensions. Chat uses Google's documented OpenAI-compatible Gemini endpoint
through the `openai` Python protocol client; this is a client adapter only. `GEMINI_API_KEY` is
the only model credential; there is no OpenAI API key or OpenAI service configuration.

Document and query embeddings use `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types,
respectively, and memory search filters by both user and embedding model.
`TAVILY_API_KEY` configures the real `web_search` tool. The request timeout is controlled by
`WEB_SEARCH_TIMEOUT_SECONDS` (default 12 seconds); Gemini requests use
`GEMINI_REQUEST_TIMEOUT_SECONDS` (default 30 seconds).

### Preserving and re-embedding older vectors

Migration `0004_gemini_memory_model` tags all pre-existing `long_term_memories` rows as legacy
OpenAI embeddings. They remain intact and are excluded from Gemini retrieval. To re-embed them
for one user, run the service helper in bounded batches from an authenticated maintenance
context; each row is committed only after its Gemini vector and model tag are updated:

```python
from app.services.memory import reembed_legacy_memories

while await reembed_legacy_memories(session, user_id, batch_size=50):
    pass
```

The helper processes both prior `long_term_memories` records and the original `memories` table.
For old-table rows, it writes a newly embedded copy into `long_term_memories` and marks the
source row as migrated; the original text and vector remain in place. Rows tagged with a legacy
model are never compared against Gemini vectors.

## Migrations and tests

The API container applies Alembic migrations on startup. For a manual deployment, apply them
with `alembic upgrade head`. Migrations are additive; no existing memory rows are deleted.

```sh
pip install -e '.[dev]'
pytest
```

Optional PostgreSQL/pgvector and Redis integration tests require dedicated test services:

```sh
TEST_DATABASE_URL=postgresql+asyncpg://... TEST_REDIS_URL=redis://... pytest tests/integration
```

The PostgreSQL integration test creates and drops only a uniquely named test schema and
requires permission to create the pgvector extension if it is not installed already.

## Case-study differences

This implementation preserves the requested backend architecture and agent features, but
deliberately uses Gemini rather than the original case study's required OpenAI chat and
embedding providers. Tavily is an additional search service needed for genuine web search.
The frontend is supplementary rather than a case-study requirement.
