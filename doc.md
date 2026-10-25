# LLM Agent Orchestrator: Codebase Walkthrough and Review

## 1. Purpose and scope

This project is a backend for authenticated, tool-using AI conversations. It offers:

- FastAPI endpoints for account registration, login, chat streaming, conversation reads, and background runs.
- PostgreSQL persistence through async SQLAlchemy, including user-scoped vector memories in pgvector.
- Gemini `gemini-2.5-flash` chat completion and `gemini-embedding-001` embeddings.
- Server-Sent Events (SSE) for interactive agent turns.
- Celery workers, backed by Redis, for asynchronous agent runs.

This walkthrough follows the implementation in `app/`, its migration, tests, and local Docker Compose setup. Local environment and key files are deliberately not inspected or reproduced here. The Compose files appear aimed at local development; this document does not certify the service for production.

## 2. Architecture at a glance

```text
Client
  ├── /api/v1/auth/* ────────────────> FastAPI auth routes
  ├── /api/v1/chat/stream ───────────> FastAPI agent route ──> orchestrator ──> Gemini Flash
  │                                                           ├──> PostgreSQL messages
  │                                                           └──> memory service ──> Gemini embeddings
  └── /api/v1/runs ──────────────────> FastAPI agent route ──> Redis/Celery
                                                                └──> worker ──> orchestrator
                                                                       └──> PostgreSQL run state
```

The API and worker share configuration, database models, and orchestration logic. PostgreSQL stores users, conversations, messages, memories, and run status. Redis serves as both the Celery broker and result backend.

## 3. Main request flows

### 3.1 Registration and login

1. `POST /api/v1/auth/register` validates an email and password through Pydantic.
2. The route lowercases the email, checks for an existing user, hashes the password, and commits a new `User`.
3. `POST /api/v1/auth/login` looks up the normalized email and verifies the password hash.
4. Successful login returns a signed RS256 bearer token. Protected routes decode the token and load the user from the database.

Relevant files: [`app/api/auth.py`](app/api/auth.py), [`app/api/deps.py`](app/api/deps.py), [`app/core/security.py`](app/core/security.py), [`app/schemas.py`](app/schemas.py).

### 3.2 Interactive streaming chat

1. `POST /api/v1/chat/stream` requires an authenticated user and validates a non-empty prompt.
2. The route creates a new conversation or confirms that the requested conversation belongs to the user.
3. It commits the request session, then opens a separate database session inside the SSE generator.
4. The orchestrator persists the user message, loads recent conversation messages, constructs the model request, and streams model output.
5. Tool calls are assembled from streamed deltas, checked against an allow-list, executed, persisted, and sent as `tool_start` / `tool_complete` events.
6. The final response is persisted and emitted as a `done` event. Errors are logged and returned as a generic `error` event.

The SSE payload is JSON in `data:` frames. Event types currently include `token`, `tool_start`, `tool_complete`, `done`, and `error`.

Relevant files: [`app/api/agent.py`](app/api/agent.py), [`app/services/orchestrator.py`](app/services/orchestrator.py).

### 3.3 Background agent run

1. `POST /api/v1/runs` creates or validates a conversation, inserts an `AgentRun` with `queued` status, and commits it.
2. The route dispatches `process_agent_run` to Celery through Redis and then stores the Celery task ID.
3. The worker loads the run, marks it `running`, loads its user and conversation, and invokes the same orchestration generator as interactive chat.
4. It accumulates token events into a result and records `completed`, or records `failed` on an exception.
5. `GET /api/v1/runs/{run_id}` returns the run only when it belongs to the authenticated user.

Relevant files: [`app/api/agent.py`](app/api/agent.py), [`app/tasks/agent_tasks.py`](app/tasks/agent_tasks.py), [`app/tasks/celery_app.py`](app/tasks/celery_app.py).

### 3.4 Memory

The current agent-session memory tools write 1536-dimension Gemini vectors to `long_term_memories`; retrieval uses cosine distance and filters by authenticated user and embedding model. Migration `0004_gemini_memory_model` tags pre-existing rows from the earlier OpenAI implementation as legacy and preserves them without mixing vector spaces. The re-embedding helper can migrate those rows in per-record commits.

The agent-session tool layer checks arguments against function signatures and only dispatches tools enabled for the session. The model receives up to three semantic memory results. Authenticated memory list, search, and delete routes are available; similarity cutoff, deduplication, and expiry are not implemented. `web_search` calls Tavily and requires `TAVILY_API_KEY`.

Relevant files: [`app/services/memory.py`](app/services/memory.py), [`app/services/orchestrator.py`](app/services/orchestrator.py), [`app/models.py`](app/models.py).

## 4. File-by-file guide

| File or area | Responsibility |
| --- | --- |
| [`app/main.py`](app/main.py) | Creates the FastAPI app, mounts auth and agent routers under `/api/v1`, and exposes `/health`. |
| [`app/core/config.py`](app/core/config.py) | Loads environment-backed settings for PostgreSQL, Redis, Gemini, Tavily, JWT keys, and agent limits. |
| [`app/core/security.py`](app/core/security.py) | Password hashing/verification and RS256 access-token creation/decoding. |
| [`app/api/deps.py`](app/api/deps.py) | Session dependency and bearer-token-to-current-user dependency. |
| [`app/api/auth.py`](app/api/auth.py) | Registration and login routes. |
| [`app/api/agent.py`](app/api/agent.py) | Chat streaming, conversation reads, background run creation, and run polling. |
| [`app/schemas.py`](app/schemas.py) | Request and response models, field constraints, and ORM serialization settings. |
| [`app/models.py`](app/models.py) | SQLAlchemy models and relationships for users, conversations, messages, memories, and runs. |
| [`app/db/base.py`](app/db/base.py) | Declarative SQLAlchemy metadata base. |
| [`app/db/session.py`](app/db/session.py) | Async database engine, session factory, and request session dependency. |
| [`app/services/orchestrator.py`](app/services/orchestrator.py) | System prompt, Gemini-compatible function schemas/streaming, tool dispatch, and message persistence. |
| [`app/services/memory.py`](app/services/memory.py) | Gemini embedding requests, legacy-vector migration, long-term memory inserts, and cosine-distance search. |
| [`app/tasks/celery_app.py`](app/tasks/celery_app.py) | Celery broker/backend configuration. |
| [`app/tasks/agent_tasks.py`](app/tasks/agent_tasks.py) | Worker-side run lifecycle and invocation of the orchestration loop. |
| [`migrations/env.py`](migrations/env.py) | Alembic configuration using application settings and SQLAlchemy model metadata. |
| [`migrations/versions/0001_initial.py`](migrations/versions/0001_initial.py) | Initial schema and pgvector extension migration. |
| [`tests/test_agent_api.py`](tests/test_agent_api.py) | Unit-level coverage of chat stream session setup and event forwarding. |
| [`tests/test_orchestrator.py`](tests/test_orchestrator.py) | Tool-dispatch behavior for unknown tools and malformed arguments. |
| [`tests/test_security.py`](tests/test_security.py) | Password hashing, JWT, and OpenAPI bearer scheme checks. |
| [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml) | Local container build and API/worker/PostgreSQL/Redis topology. |
| [`pyproject.toml`](pyproject.toml) | Runtime and test dependencies, Python floor, and Ruff/Pytest configuration. |
| [`README.md`](README.md), [`NOTES.md`](NOTES.md) | Local startup and API overview; implementation notes and known limitations. |

## 5. Data model and API surface

### Data model

- **User** owns conversations, memories, and runs.
- **Conversation** belongs to a user and has ordered messages.
- **Message** stores a role, optional content, optional tool-call payload, optional tool-call ID, and a per-conversation sequence number. `(conversation_id, sequence)` is unique.
- **Memory** is the historical user-owned text/vector table; current agent memories are stored separately in `LongTermMemory`. Legacy rows can be copied and re-embedded with the documented helper.
- **AgentSession** stores the owner, system prompt, and enabled tools.
- **AgentRun** stores owner, conversation/session, prompt, status, answer, token count, error, and Celery task ID.
- **RunStep** stores durable model/tool/final-answer execution trace events.

The initial migration enables the PostgreSQL `vector` extension and creates these tables and indexes. Conversation and memory ownership is enforced in route queries; foreign keys enforce parent existence and configured delete behavior. PostgreSQL/pgvector and Redis integration tests are opt-in under `tests/integration`.

### Routes

| Method | Route | Authentication | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/auth/register` | No | Create a user. |
| `POST` | `/api/v1/auth/token` | No | Return a bearer token (`/auth/login` remains a compatibility alias). |
| `POST` | `/api/v1/chat/stream` | Yes | Execute an interactive agent turn over SSE. |
| `GET` | `/api/v1/conversations` | Yes | List the current user's conversations. |
| `GET` | `/api/v1/conversations/{conversation_id}` | Yes | Read an owned conversation and non-tool messages. |
| `POST` | `/api/v1/runs` | Yes | Queue a background agent run. |
| `GET` | `/api/v1/runs/{run_id}` | Yes | Poll an owned background run. |
| `GET`, `POST` | `/api/v1/sessions` | Yes | List or create owned agent sessions. |
| `GET`, `DELETE` | `/api/v1/sessions/{session_id}` | Yes | Read or delete an owned session. |
| `POST` | `/api/v1/sessions/{session_id}/run` | Yes | Queue a session run. |
| `GET` | `/api/v1/runs/{run_id}/status` | Yes | Read run status and step count. |
| `GET` | `/api/v1/runs/{run_id}/steps` | Yes | Read persisted execution steps. |
| `GET` | `/api/v1/runs/{run_id}/stream` | Yes | Stream run events over SSE. |
| `DELETE` | `/api/v1/runs/{run_id}` | Yes | Cancel an owned queued/running run. |
| `GET` | `/api/v1/memory`, `/api/v1/memory/search` | Yes | List or search owned long-term memories. |
| `DELETE` | `/api/v1/memory/{memory_id}` | Yes | Delete an owned memory. |
| `GET` | `/health` | No | Return a simple process health response. |

There are currently no routes for account deletion, conversation deletion/renaming, or message pagination.

## 6. Agent behavior and constraints

- Gemini `gemini-2.5-flash` is the configured chat model; `gemini-embedding-001` produces 1536-dimensional vectors. Gemini chat uses Google's OpenAI-compatible endpoint and `GEMINI_API_KEY`; this deliberately differs from the original case study's OpenAI provider requirement.
- Interactive Gemini chat uses `AGENT_MAX_TOOL_ROUNDS` and limits eight tool calls per round. Agent-session runs stop after ten model iterations.
- The legacy conversation route has an eight-tool allow-list. Agent sessions expose five tools (`web_search`, `calculator`, `get_current_datetime`, `summarise_text`, and `remember_fact`); web search uses Tavily and requires `TAVILY_API_KEY`. Tool dispatch rechecks the session's enabled list and validates arguments.
- Legacy conversation history loads up to 60 persisted messages; agent-session runs use Redis short-term history capped at ten messages, with up to three semantically relevant long-term memories.
- New agent runs persist each trace step before publishing it and store token usage/final answers in PostgreSQL. The agent runner fails rather than returning a fabricated answer when it exhausts its iteration limit or receives no final text.
- Agent errors are logged server-side. SSE callers receive a generic error message; failed background runs store a generic failure string.
- Memory search is scoped to the user and embedding model and returns up to three nearest results regardless of absolute similarity.

## 7. Review questions for a code walkthrough

These are questions to use in a product/engineering review, rather than assumptions about intended behavior.

### Product and user expectations

1. **What does “remember this” mean as a user-consent contract?** Is an LLM-selected `save_memory` tool call enough, or should the UI/API require an explicit confirmation or deterministic signal?
2. **Can users inspect, correct, and delete saved memories?** If not, what retention and deletion policy should the product promise?
3. **What should the user see when an agent run fails?** Is a generic error sufficient, and should a failed/partial answer be recoverable?
4. **Should new conversations get a generated title, or is “New conversation” intended to remain the permanent title?**
5. **What are the expected limits for account registration and prompting?** Current prompts are capped at 20,000 characters, while request rate and account-level quotas are not defined here.

### Agent and conversation semantics

6. **What context window policy is intended?** Should history be limited by message count, model token budget, or a summarization/compaction strategy?
7. **What should happen when a turn reaches the maximum tool-round limit?** The current request reports failure rather than a partial or synthesized final answer.
8. **Can the same conversation receive simultaneous interactive and background turns?** The streaming route commits before it starts orchestration, so its initial row lock is not held through the model turn; independent turns may contend for the unique message sequence or observe interleaved history.
9. **What is the supported SSE contract for clients?** Are heartbeats, cancellation behavior, retry/resume, and stable event schemas required?
10. **Should tool results and tool-call messages appear in conversation history or only in internal context?** The conversation detail route currently filters out `tool` role messages but returns assistant tool-call messages as messages without exposing their `tool_calls` field.

### Reliability, operations, and data

11. **How should a queued run be reconciled if the database commit succeeds but broker dispatch or the later task-ID commit fails?** The current route does not atomically commit the run and publish the task.
12. **What are the expected retry and timeout policies for Gemini, Tavily, Redis, and PostgreSQL failures?**
13. **Is `/health` intended as liveness only, or should deployment also have a readiness check for database, Redis, and provider configuration?**
14. **Which deployment target is supported?** Compose publishes database and Redis ports and uses development defaults, which should be reviewed before exposing the stack outside a local environment.
15. **What are data retention, backup, and account-deletion requirements for conversations, prompts, run results, and memories?**
16. **What traffic and memory volume should the initial implementation support?** This determines pagination, database/vector indexing, and worker concurrency needs.

## 8. Suggested project modifications

Priorities below are relative to preparing a dependable user-facing service; confirm product requirements before changing behavior.

### P0 — Resolve before production exposure

1. **Make conversation turns safe under concurrency.** Serialize interactive and background turns consistently and allocate message sequence numbers without relying on a `MAX(sequence) + 1` read that can race. Add tests for concurrent requests on the same conversation and verify ordering and failure recovery.
2. **Make background dispatch durable.** Implement the transactional-outbox pattern already noted in `NOTES.md`, or an equivalent reconciliation mechanism, so a persisted queued run cannot be stranded when broker publication fails or the API loses the task-ID update.
3. **Define deployment secrets and network boundaries.** Require non-default database credentials, restrict externally published database/Redis ports, define TLS termination and secret delivery, and document which Compose settings are local-only.
4. **Add operational run-state recovery.** Define handling for worker crashes and runs left in `running`, plus retry/idempotency rules and bounded timeouts for provider calls.

### P1 — Reliability and product completeness

1. **Add backend-enforced memory controls.** Establish explicit save-consent behavior, input limits, similarity threshold/deduplication policy, and user-visible list/delete/expiry behavior.
2. **Use a token-aware context budget.** Preserve complete assistant-tool-call/tool-result groups when truncating history, and test histories that exceed the retained window.
3. **Handle registration uniqueness races.** The pre-insert existence check can race another registration; convert the database unique constraint violation into the intended conflict response.
4. **Add pagination and lifecycle APIs.** Paginate conversations/messages/runs and decide how users delete conversations or accounts, including cascade/retention expectations.
5. **Extend database-backed integration coverage.** PostgreSQL/pgvector and Redis opt-in integration tests are available; expand them to cover migrations, transaction rollback, Celery worker transitions, and SSE error paths. Gemini and Tavily calls remain mocked in unit tests.
6. **Improve readiness and observability.** Add readiness checks and structured metrics/log context for provider latency/errors, token usage/cost, run queue wait/execution duration, and stuck/failed runs. Avoid logging prompts or saved memories unless explicitly approved by privacy policy.

### P2 — Maintainability and UX polish

1. **Keep docs aligned with behavior.** Resolve the 21-versus-60 message-history mismatch and document model/tool limits and error semantics in one source of truth.
2. **Strengthen data constraints.** Consider database-level checks or typed enums for message roles and run statuses, plus explicit validation of tool-call history shape.
3. **Define API compatibility and event schemas.** Add tests for OpenAPI and SSE payloads, then version or otherwise manage changes clients depend on.
4. **Add a reproducible CI/developer workflow.** Pin/lock resolved dependencies for deployable builds and run formatting/lint, type checks where adopted, unit tests, and migration checks in CI.
5. **Measure before adding vector indexes.** Keep the current user filter and exact search while volumes are small; benchmark realistic data before selecting an ANN index and tuning its recall/latency trade-off.

## 9. Tests and validation

Unit tests mock Gemini, Tavily, Redis, Celery, and database session behavior. They cover chat
stream and tool-call assembly, argument validation, enabled-tool enforcement, agent iteration
limits, memory task types/dimensions/model filters, memory re-embedding, run state transitions,
SSE cleanup, ownership filtering, cancellation, and task dispatch failure.

PostgreSQL/pgvector persistence and Redis history/pubsub round-trip tests are opt-in and require
dedicated `TEST_DATABASE_URL` and `TEST_REDIS_URL`. Live Gemini and Tavily provider calls are
not asserted by mocked unit tests; service credentials are required to validate those endpoints
against the network.

## 10. Local startup summary

The README's local workflow is to create `.env` from `.env.example`, generate an RS256 key pair under `.secrets/`, set `GEMINI_API_KEY` and `TAVILY_API_KEY`, and start the Compose stack. The API container applies Alembic migrations before starting Uvicorn; the worker starts Celery. The React frontend is supplementary; `/docs` and `/health` expose the backend contract.

The `.gitignore` and `.dockerignore` exclude local environment/key material. Keep that separation when adapting the container build or deploying; do not copy real local credentials into an image or commit them.
