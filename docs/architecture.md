# Architecture

Knowledge-Agent is organized as a FastAPI application with explicit domain,
service, gateway and infrastructure boundaries. Construction is side-effect
free; dependency initialization and readiness checks happen during the
application lifespan.

## Main components

| Component | Responsibility |
| --- | --- |
| FastAPI API | Authentication context, validation, rate limits and HTTP/SSE contracts |
| Import coordinator | Idempotent admissions, durable tasks, leases, retries and cleanup |
| MinerU gateway | Isolates the upstream parsing token and validates remote downloads |
| BGE-M3 backend | Dense and sparse embedding generation |
| Milvus | Tenant-filtered vector storage and hybrid retrieval |
| MongoDB | Import state, authoritative visibility, sessions and messages |
| Redis Streams | Bounded SSE replay and resumable events |
| Keycloak | OIDC Authorization Code + PKCE and optional verified registration |
| MinIO | Private source objects and extracted assets |

## Import flow

1. The API derives tenant and user identity from the authentication context.
2. An idempotency key reserves an admission and private object name.
3. The source is streamed to MinIO with size and digest validation.
4. A durable worker leases the task and materializes an isolated workspace.
5. MinerU parses the source, safe nodes normalize the result, and the splitter
   creates bounded chunks.
6. BGE-M3 creates dense and sparse vectors.
7. Milvus receives a fenced import version; MongoDB records the authoritative
   completed-import visibility state.
8. Failed attempts are retried with bounded backoff and stale work is cleaned.

## Search and answer flow

1. The query is validated and embedded.
2. Milvus performs dense and sparse retrieval with a tenant expression.
3. Candidate document versions are checked against authoritative completed
   imports in MongoDB.
4. Ranked evidence is passed to the LLM answer backend.
5. The response streams through SSE and includes stable citation identifiers.

## Trust boundaries

- Authentication claims are verified before constructing a request identity.
- Client-provided tenant or owner identifiers are ignored.
- Every persistence adapter applies tenant scope; user-owned resources also
  apply the immutable JWT subject.
- MinIO objects are private and named with normalized, non-traversable keys.
- Redis identity keys are HMAC-derived.
- Metrics use bounded labels and never include tenant IDs, user IDs or prompts.
- Parser downloads are host-allowlisted and archive extraction is bounded.

## Availability model

`/health/live` reports process liveness. `/health/ready` reports whether the
enabled dependencies can serve traffic. A reverse proxy or orchestrator should
route traffic only to ready instances and should not restart a process solely
because a long CPU inference temporarily delays a readiness probe.
