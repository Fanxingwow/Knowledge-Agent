# Configuration

Knowledge-Agent reads configuration from environment variables through
`pydantic-settings`. Real `.env` files are intentionally ignored.

## Templates

| Template | Purpose |
| --- | --- |
| `.env.example` | Direct Python development reference |
| `deploy/compose.env.example` | Complete local Compose stack |
| `deploy/server-reuse.env.example` | API stack reusing host MinIO and Milvus |

Copy a template to its non-example name and replace all `change-me` values.
Keep each credential independent; do not reuse database, storage, metrics,
gateway or SMTP secrets.

## Capability switches

| Variable | Effect |
| --- | --- |
| `IMPORT_ENABLED` | Enables upload admissions and the import processor |
| `SEARCH_ENABLED` | Enables BGE-M3/Milvus hybrid search |
| `CHAT_ENABLED` | Enables sessions, streamed answers and event history |
| `AUTH_MODE` | `disabled` for isolated development or `jwt` for OIDC |
| `METRICS_ENABLED` | Enables the protected Prometheus endpoint |

Keep import and search disabled until BGE-M3, Milvus, MinIO and MinerU are all
configured. Production must use `AUTH_MODE=jwt`.

## Model configuration

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_DEFAULT_MODEL`: OpenAI-compatible
  answer model.
- `BGE_M3_HOST_PATH`: host model directory mounted by Compose.
- `BGE_DEVICE`: usually `cpu` or `cuda`.
- `BGE_MODEL_BATCH_SIZE`: embedding batch size; tune against memory.
- `BGE_RERANKER_LARGE`: optional local reranker path.

Model weights must remain outside Git. Pin model revisions and verify file
digests for controlled deployments.

## MinerU gateway

Two different secrets are used intentionally:

- `MINERU_GATEWAY_AUTH_TOKEN` authenticates the private API-to-gateway hop.
- `MINERU_UPSTREAM_API_TOKEN` authenticates the gateway to the cloud service.

Separating them prevents the main API container from receiving the cloud token
and allows internal credentials to be rotated independently.

## OIDC and email registration

Keycloak settings live in `deploy/server-reuse.env.example`. For public use:

- publish console and Keycloak through HTTPS origins;
- keep Keycloak's internal JWKS URL on the Docker network;
- configure an SMTP sender account and provider authorization code;
- enable verified registration only after a successful test message;
- do not confuse the sender SMTP account with end-user registration emails.

## Secret generation

On Linux, generate independent URL-safe values, for example:

```bash
openssl rand -hex 32
```

Never paste a complete environment file into an issue or support request.
