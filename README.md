# LLM Agent Orchestrator

A small AI-agent backend built with FastAPI, async SQLAlchemy, PostgreSQL + pgvector, Redis, Celery, and OpenAI tool calling.

## Features

- JWT registration and login.
- Conversation history and an OpenAI tool-calling loop with a configurable round limit.
- User-scoped semantic memory using `text-embedding-3-small` and pgvector.
- Server-Sent Events (SSE) for interactive chat.
- Celery-backed agent runs with status polling.

## Run locally

Requirements: Docker Compose and an OpenAI API key.

```sh
# .env is provided for local Compose use. If recreating it, start from:
cp .env.example .env
# Generate a local RS256 key pair (keep .secrets private and untracked):
mkdir -p .secrets
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out .secrets/jwt-private.pem
openssl pkey -in .secrets/jwt-private.pem -pubout -out .secrets/jwt-public.pem
# Set OPENAI_API_KEY in .env.
docker compose up --build
```

The local `.env` and `.secrets/` are excluded from Git and the Docker build context. Compose loads `.env` for the API, worker, and database settings, and mounts the RS256 key pair read-only into the API and worker containers.

The API and worker wait for PostgreSQL and Redis. The API container applies Alembic migrations before startup. OpenAPI docs are at <http://localhost:8000/docs>; the health endpoint is <http://localhost:8000/health>.

## API overview

All application routes use the `/api/v1` prefix.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/auth/register` | Create a user |
| `POST` | `/auth/login` | Obtain a bearer token |
| `POST` | `/chat/stream` | Run an agent turn as SSE events |
| `GET` | `/conversations` | List the current user's conversations |
| `GET` | `/conversations/{id}` | Read a conversation and its user/assistant messages |
| `POST` | `/runs` | Enqueue a background agent run |
| `GET` | `/runs/{id}` | Read a background run status and result |

For `/chat/stream`, send `{"prompt":"...", "conversation_id": null}` with a bearer token. The stream emits JSON `data:` events: `token`, `tool_start`, `tool_complete`, `done`, or `error`. A new conversation is created when `conversation_id` is omitted.

## Tests

```sh
pip install -e '.[dev]'
pytest
```

See [NOTES.md](NOTES.md) for the orchestration design, trade-offs, and next steps.
