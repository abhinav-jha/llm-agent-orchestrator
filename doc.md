# LLM Agent Orchestrator: Codebase Walkthrough and Review

## 1. Purpose and scope

This project is a backend for authenticated, tool-using AI conversations. It offers:

- FastAPI endpoints for account registration, login, chat streaming, conversation reads, and background runs.
- PostgreSQL persistence through async SQLAlchemy, including user-scoped vector memories in pgvector.
- Gemini 2.5 Flash chat completion and Gemini Embedding 001 calls.
- Server-Sent Events (SSE) for interactive agent turns.
- Celery workers, backed by Redis, for asynchronous agent runs.

This walkthrough follows the implementation in `app/`, its migration, tests, and local Docker Compose setup. Local environment and key files are deliberately not inspected or reproduced here. The Compose files appear aimed at local development; this document does not certify the service for production.

## 2. Architecture at a glance

```text
Client
  ├── /api/v1/auth/* ────────────────> FastAPI auth routes
  ├── /api/v1/chat/stream ───────────> FastAPI agent route ──> orchestrator ──> Gemini 2.5 Flash
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

Memory is written only through memory tools, which embed content with Gemini Embedding 001 and store the content and 1536-dimension vector. `search_memory` embeds a query and orders that user's memories by cosine distance. Both queries filter by authenticated user ID and embedding model. Migration `0002_gemini_embeddings` marks existing OpenAI vectors as legacy and leaves them preserved, but they are not searched with Gemini vectors.

The tool layer validates argument names and basic content shape before calling the memory service. The model receives up to five results; the current implementation has no similarity cutoff, deduplication, expiry, or direct user-facing memory management route.

Relevant files: [`app/services/memory.py`](app/services/memory.py), [`app/services/orchestrator.py`](app/services/orchestrator.py), [`app/models.py`](app/models.py).

## 4. File-by-file guide

| File or area | Responsibility |
| --- | --- |
| [`app/main.py`](app/main.py) | Creates the FastAPI app, mounts auth and agent routers under `/api/v1`, and exposes `/health`. |
| [`app/core/config.py`](app/core/config.py) | Loads environment-backed settings for PostgreSQL, Redis, Gemini, JWT keys, and agent limits. |
| [`app/core/security.py`](app/core/security.py) | Password hashing/verification and RS256 access-token creation/decoding. |
| [`app/api/deps.py`](app/api/deps.py) | Session dependency and bearer-token-to-current-user dependency. |
| [`app/api/auth.py`](app/api/auth.py) | Registration and login routes. |
| [`app/api/agent.py`](app/api/agent.py) | Chat streaming, conversation reads, background run creation, and run polling. |
| [`app/schemas.py`](app/schemas.py) | Request and response models, field constraints, and ORM serialization settings. |
| [`app/models.py`](app/models.py) | SQLAlchemy models and relationships for users, conversations, messages, memories, and runs. |
| [`app/db/base.py`](app/db/base.py) | Declarative SQLAlchemy metadata base. |
| [`app/db/session.py`](app/db/session.py) | Async database engine, session factory, and request session dependency. |
| [`app/services/orchestrator.py`](app/services/orchestrator.py) | System prompt, tool schemas and dispatch, message history construction, Gemini streaming loop, and message persistence. |
| [`app/services/memory.py`](app/services/memory.py) | Gemini embedding requests, memory inserts, and cosine-distance search. |
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
- **Memory** stores user-owned text and a pgvector embedding.
- **AgentRun** stores the owner, conversation, prompt, status, result/error, Celery task ID, and creation time.

The initial migration enables the PostgreSQL `vector` extension and creates these tables and indexes. Conversation and memory ownership is enforced in route queries; foreign keys enforce parent existence and configured delete behavior.

### Routes

| Method | Route | Authentication | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/v1/auth/register` | No | Create a user. |
| `POST` | `/api/v1/auth/login` | No | Return a bearer token. |
| `POST` | `/api/v1/chat/stream` | Yes | Execute an interactive agent turn over SSE. |
| `GET` | `/api/v1/conversations` | Yes | List the current user's conversations. |
| `GET` | `/api/v1/conversations/{conversation_id}` | Yes | Read an owned conversation and non-tool messages. |
| `POST` | `/api/v1/runs` | Yes | Queue a background agent run. |
| `GET` | `/api/v1/runs/{run_id}` | Yes | Poll an owned background run. |
| `GET` | `/health` | No | Return a simple process health response. |

There are currently no routes for account deletion, conversation deletion/renaming, message pagination, or memory list/update/delete.

## 6. Agent behavior and constraints

- The configured chat model defaults to `gemini-2.5-flash`; embeddings use `gemini-embedding-001`. Chat calls use Google's OpenAI-compatible endpoint with the existing OpenAI Python client as a protocol adapter; API keys are sent to Google, not OpenAI.
- The default tool-round limit is five, and the orchestrator allows at most eight tool calls in one round.
- Available tools are `search_memory`, `save_memory`, and `current_time`; dispatch is allow-listed.
- Agent history is loaded as the latest 60 persisted messages. A system message is prepended, and any leading non-user messages in the selected history are discarded.
- The final response is committed to PostgreSQL before the `done` event is emitted.
- Agent errors are logged server-side. SSE callers receive a generic error message; failed background runs store a generic failure string.
- Memory search is scoped to the user but currently returns the nearest five results regardless of their absolute similarity.

**Documentation mismatch:** `NOTES.md` says each request loads the latest 21 persisted messages. The implementation in `app/services/orchestrator.py` loads up to 60. Update one side after deciding the intended context policy.

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
12. **What are the expected retry and timeout policies for Gemini, Redis, and PostgreSQL failures?**
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
5. **Add database-backed integration coverage.** Exercise migrations, ownership isolation, transaction rollback, pgvector operations, queue dispatch failures, worker status transitions, and SSE disconnect/error paths using PostgreSQL/pgvector, Redis, and mocked Gemini endpoints.
6. **Improve readiness and observability.** Add readiness checks and structured metrics/log context for provider latency/errors, token usage/cost, run queue wait/execution duration, and stuck/failed runs. Avoid logging prompts or saved memories unless explicitly approved by privacy policy.

### P2 — Maintainability and UX polish

1. **Keep docs aligned with behavior.** Resolve the 21-versus-60 message-history mismatch and document model/tool limits and error semantics in one source of truth.
2. **Strengthen data constraints.** Consider database-level checks or typed enums for message roles and run statuses, plus explicit validation of tool-call history shape.
3. **Define API compatibility and event schemas.** Add tests for OpenAPI and SSE payloads, then version or otherwise manage changes clients depend on.
4. **Add a reproducible CI/developer workflow.** Pin/lock resolved dependencies for deployable builds and run formatting/lint, type checks where adopted, unit tests, and migration checks in CI.
5. **Measure before adding vector indexes.** Keep the current user filter and exact search while volumes are small; benchmark realistic data before selecting an ANN index and tuning its recall/latency trade-off.

## 9. Existing tests and validation gaps

The repository has three test modules covering password hashing (including long passwords), token round-tripping/rejection, OpenAPI bearer auth, chat stream session setup, and two tool-dispatch error cases. These are useful unit-level checks but do not exercise the full database, provider, broker, worker, or multi-request lifecycle.

`NOTES.md` already identifies several appropriate follow-ups: PostgreSQL/pgvector + Redis integration tests, rate limits and account recovery, outbox dispatch, streaming cancellation cleanup, pagination, memory lifecycle, token/cost observability, row-level security, vector indexing, backups, and deployment configuration.

## 10. Local startup summary

The README's local workflow is to create `.env` from `.env.example`, generate an RS256 key pair under `.secrets/`, set a Gemini API key, and start the Compose stack. The API container applies Alembic migrations before starting Uvicorn; the worker starts Celery. The README documents `/docs` and `/health` as local endpoints.

The `.gitignore` and `.dockerignore` exclude local environment/key material. Keep that separation when adapting the container build or deploying; do not copy real local credentials into an image or commit them.
