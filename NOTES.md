# Implementation notes

## Agent loop

Each request loads the latest persisted messages, prepends a system prompt, and calls Gemini 2.5 Flash through Google's OpenAI-compatible endpoint with streamed output and tool schemas. Tool-call deltas are assembled by index before dispatch. The loop persists tool requests/results, then makes another model call; it stops after the configured tool-round limit and reports failure rather than returning a fabricated answer. Tool implementations are allow-listed and validate arguments; memory operations always include the authenticated user's ID.

Interactive requests emit tokens and tool lifecycle events as SSE. Background requests run the same orchestration in a Celery worker and expose queued/running/completed/failed states through polling. Conversation and run lookups are restricted to the authenticated owner.

## Memory strategy

Only explicitly requested facts are saved. Each memory is embedded with Gemini Embedding 001 (1536 dimensions), stored in pgvector with its embedding model recorded, and retrieved by cosine distance with user and model filters. Existing OpenAI vectors are retained and marked as legacy by migration `0002`; they are excluded from Gemini search until facts are saved again. Retrieval count is currently limited to five, without a similarity cutoff or deduplication.

## Known limitations / next steps

- Add integration tests with PostgreSQL/pgvector, Redis, and a mocked Gemini endpoint; current tests cover token/password behavior, Gemini provider configuration, embedding request shape, and tool argument handling.
- Add rate limits, email verification, password reset, refresh-token rotation, and stronger deployment secret management.
- Add a transactional outbox for reliable Celery dispatch; a broker outage after the database write requires an API retry/reconciliation strategy.
- Add streaming cancellation cleanup, pagination, conversation deletion, memory expiry/deduplication, and token/cost observability.
- Consider using row-level security and an ANN vector index after measuring realistic memory volumes.
- Configure production CORS, TLS termination, database backups, and worker concurrency explicitly.
