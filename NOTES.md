# Implementation notes

## Provider and execution

Gemini is the deliberate provider choice, deviating from the original case study's OpenAI
provider requirement. Chat, function calls, streaming, summaries, and agent reasoning use
`gemini-2.5-flash` through Google's documented OpenAI-compatible endpoint and a protocol client.
The adapter package does not use OpenAI's API or require an OpenAI key. Request timeouts are
configured through `GEMINI_REQUEST_TIMEOUT_SECONDS`.

Agent sessions persist ownership, system prompts, and enabled tool names. The runner derives
tool schemas from the five-tool registry, only sends enabled functions to Gemini, and validates
the returned tool name against the session's enabled list again before execution. Tool
arguments are JSON-decoded and signature-bound; errors are returned to the model as explicit
tool results. The loop is bounded to ten model iterations.

`web_search` uses Tavily's real search API, validates the query and result shape, has a bounded
timeout, reports rate limits/provider failures explicitly, and returns source URLs. Configure
`TAVILY_API_KEY`; without it the tool returns a configuration error, never synthetic results.
Calculator evaluation uses a restricted AST. Summaries use the configured Gemini model.

The agent-session run API persists model, tool-call, tool-result, final-answer, and error steps;
tracks model token usage; updates run status through queued/running/completed/failed/cancelled;
and publishes trace events over Redis. Celery task startup is idempotent for non-queued runs.
The API exposes session/run ownership-checked status, steps, SSE, cancellation, and memory
management routes. The legacy conversation streaming API remains available.

## Memory and data compatibility

Gemini `gemini-embedding-001` supports `outputDimensionality`; this project requests 1536
dimensions to match existing pgvector columns. Stored text uses `RETRIEVAL_DOCUMENT`, queries
use `RETRIEVAL_QUERY`, and retrieval is limited to the top three results by cosine distance,
filtered by owner and embedding model. Redis short-term context reads the newest five complete
turns (up to ten messages).

Migration `0004_gemini_memory_model` adds a provider tag to the agent `long_term_memories`
table. Existing rows are labelled `legacy-openai-text-embedding-3-small` and are preserved but
excluded from Gemini search. `reembed_legacy_memories` safely updates prior rows from
`long_term_memories` and copies rows from the original `memories` table into the current table,
tagging each successful source row in a per-row commit. No source row is deleted, and old
vectors are not compared with Gemini vectors. Historical migration files were not rewritten.

## Verification and known limitations

- Unit tests use mocks for Gemini, Tavily, Celery, Redis, and database session behavior.
- Optional real PostgreSQL/pgvector and Redis tests are in `tests/integration`; they require
  dedicated URLs via `TEST_DATABASE_URL` and `TEST_REDIS_URL`.
- Live Gemini and Tavily calls require credentials. They were not verified in the test suite.
- A transactional outbox, retry policy, rate limiting, and production deployment hardening are
  optional reliability/production enhancements, not blockers for the core case-study API.
- The React/Vite frontend remains supplementary. Full npm build and Compose service startup
  need npm and a running Docker daemon.

The case-study authentication text explicitly says refresh tokens are not required, while its
API table also mentions a refresh token. This implementation follows the explicit access-token
only requirement.
