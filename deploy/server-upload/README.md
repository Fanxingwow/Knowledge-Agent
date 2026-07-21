# CPU/GPU server upload bundle

This bundle deploys either the default Linux CPU image or the optional NVIDIA
GPU image while reusing MinIO and Milvus that already run on the Docker host.
It never contains `.env` or `deploy/server-reuse.env`.

Ordinary Python packages are locked to the Tencent Cloud PyPI mirror. CPU
builds use the official PyTorch CPU index. `compose.gpu.yaml` replaces those
three runtime wheels with an official CUDA build and assigns one NVIDIA GPU to
the API container. The image build allows slow links up to 300 seconds per
HTTP read and retains the uv download cache between builds.

MongoDB is pinned to `mongo:7.0.37-jammy`. MongoDB 8.0 and newer can crash on
Linux 6.19 and later because of an upstream TCMalloc incompatibility. Never
attach a data volume written by MongoDB 8.x directly to MongoDB 7.x; use a
logical dump and restore, or create a new volume for an empty deployment.

Upload the entire bundle contents to `/opt/rag_agent`. Preserve the existing
`/opt/rag_agent/deploy/server-reuse.env`; it contains the server-only secrets
that must not be copied back into source control.

Before enabling imports, rotate any previously exposed MinerU cloud token and
add these values to `deploy/server-reuse.env`:

```dotenv
MINERU_GATEWAY_AUTH_TOKEN=<independent value from openssl rand -hex 32>
MINERU_UPSTREAM_API_TOKEN=<new mineru.net cloud token>
MINERU_UPSTREAM_BASE_URL=https://mineru.net/api/v4
MINERU_UPSTREAM_MODEL_VERSION=vlm
MINERU_BASE_URL=http://mineru-gateway:8080

# Required by the retrieval-grounded Agent response endpoint and Web console.
OPENAI_API_KEY=<OpenAI-compatible API key>
OPENAI_BASE_URL=<OpenAI-compatible base URL>
LLM_DEFAULT_MODEL=<chat model name>
```

Do not reuse the cloud token as `MINERU_GATEWAY_AUTH_TOKEN`, and remove the
legacy `MINERU_API_TOKEN` key from the Compose environment file.

The Web console now uses Keycloak OIDC Authorization Code + PKCE. For a first
SSH-only validation, add these values before starting Compose:

```dotenv
APP_ENV=development
CONSOLE_PUBLIC_ORIGIN=http://127.0.0.1:8010
KEYCLOAK_PUBLIC_URL=http://127.0.0.1:8081
KEYCLOAK_REALM=enterprise-knowledge
KEYCLOAK_CLIENT_ID=enterprise-knowledge-console
AUTH_TENANT_ID=enterprise-default
KEYCLOAK_ADMIN_USERNAME=kc-admin
KEYCLOAK_ADMIN_PASSWORD=<new random password>
KEYCLOAK_DB_NAME=keycloak
KEYCLOAK_DB_USERNAME=keycloak
KEYCLOAK_DB_PASSWORD=<new random password>
KEYCLOAK_REGISTRATION_ENABLED=true
KEYCLOAK_EMAIL_VERIFICATION_LIFESPAN_SECONDS=900
KEYCLOAK_SMTP_HOST=<SMTP host>
KEYCLOAK_SMTP_PORT=465
KEYCLOAK_SMTP_FROM=<sender email>
KEYCLOAK_SMTP_FROM_DISPLAY_NAME=企业智能知识库问答
KEYCLOAK_SMTP_AUTH=true
KEYCLOAK_SMTP_USER=<SMTP username>
KEYCLOAK_SMTP_PASSWORD=<SMTP authorization code, not the mailbox password>
KEYCLOAK_SMTP_SSL=true
KEYCLOAK_SMTP_STARTTLS=false
KEYCLOAK_SMTP_ALLOW_UTF8=false
AUTH_MODE=jwt
AUTH_ISSUER=http://127.0.0.1:8081/realms/enterprise-knowledge
AUTH_AUDIENCE=enterprise-knowledge-console
AUTH_OIDC_CLIENT_ID=enterprise-knowledge-console
AUTH_JWKS_URL=http://keycloak:8080/realms/enterprise-knowledge/protocol/openid-connect/certs
API_CORS_ORIGINS=http://127.0.0.1:8010
```

Use two HTTPS domains and the supplied Nginx example for production. Do not
publish either loopback port directly. New Realm users receive `reader` and
`chat-user`; grant `editor` only to users who may upload shared knowledge.
Every user's sessions and questions are isolated by the JWT `sub` claim, while
the knowledge corpus is shared through the Realm's fixed `tenant_id` claim.

Validate the configuration:

```bash
cd /opt/rag_agent

docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  config --quiet
```

Build and start the default CPU image:

```bash
docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  build --progress=plain api

docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  up -d
```

For an NVIDIA GPU server, first install a compatible NVIDIA driver and NVIDIA
Container Toolkit. The default GPU build uses `cu130`; add
`PYTORCH_GPU_VARIANT=cu126` to `deploy/server-reuse.env` when an older driver
or GPU requires CUDA 12.6. Build and start with the overlay:

```bash
nvidia-smi

docker compose \
  -f compose.server-reuse.yaml \
  -f compose.gpu.yaml \
  --env-file deploy/server-reuse.env \
  config --quiet

docker compose \
  -f compose.server-reuse.yaml \
  -f compose.gpu.yaml \
  --env-file deploy/server-reuse.env \
  up -d --build
```

Check status, PyTorch profile, and health:

```bash
docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  ps

docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  logs --tail=100 mineru-gateway

docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  exec api python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'

curl -fsS http://127.0.0.1:8010/health/live
curl -fsS http://127.0.0.1:8010/health/ready
```

Keep both SSH tunnels open from the workstation:

```bash
ssh -N \
  -L 8010:127.0.0.1:8010 \
  -L 8081:127.0.0.1:8081 \
  ubuntu@<server-ip>
```

Open `http://127.0.0.1:8010/console` for the unified chat, upload, import
progress, citation, and retrieval interface. Swagger remains available at
`http://127.0.0.1:8010/docs`. With verified registration enabled, the Keycloak
login page displays a registration link and sends a verification email before
letting the user set a password. Keycloak's native flow uses a verification
link, not a numeric email OTP. Administrators can still manage users at
`http://127.0.0.1:8081/admin/` in the `enterprise-knowledge` Realm.

The PyTorch check must report a `+cpu` version and `False` in CPU mode. In GPU
mode it must report a CUDA variant and `True`; additionally print
`torch.cuda.get_device_name(0)` to confirm the assigned device. Do not run
`docker compose down -v` unless deletion of this application's MongoDB and
Redis data is intentional.

`MANIFEST.sha256` records the expected SHA-256 digest of every bundle file.
Validate it on Ubuntu with:

```bash
sha256sum --check MANIFEST.sha256
```
