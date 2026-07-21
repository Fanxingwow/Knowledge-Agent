# Deployment

## CPU and GPU image policy

Linux builds are CPU-only unless `compose.gpu.yaml` is explicitly applied.
The overlay selects an official CUDA PyTorch wheel, configures BGE-M3 for
`cuda:0`/FP16, and reserves one NVIDIA GPU for the API container. It can be
combined with either base Compose file:

```bash
# CPU, self-contained dependencies
docker compose --env-file deploy/compose.env \
  -f compose.yaml up -d --build

# GPU, self-contained dependencies
docker compose --env-file deploy/compose.env \
  -f compose.yaml -f compose.gpu.yaml up -d --build

# CPU, existing MinIO/Milvus
docker compose --env-file deploy/server-reuse.env \
  -f compose.server-reuse.yaml up -d --build

# GPU, existing MinIO/Milvus
docker compose --env-file deploy/server-reuse.env \
  -f compose.server-reuse.yaml -f compose.gpu.yaml up -d --build
```

GPU hosts require the NVIDIA driver and NVIDIA Container Toolkit. The default
GPU wheel is `cu130`; set `PYTORCH_GPU_VARIANT=cu126` or `cu128` when the host
compatibility matrix requires it. Always validate `torch.cuda.is_available()`
inside the running API container. The GPU overlay accelerates local embeddings
and the optional reranker, not an external LLM endpoint.

## Local Compose stack

```bash
cp deploy/compose.env.example deploy/compose.env
docker compose --env-file deploy/compose.env config --quiet
docker compose --env-file deploy/compose.env up -d --build
```

The local stack includes project-owned MongoDB, Redis, MinIO, Milvus and
optional Prometheus services. The example is for development and single-host
evaluation.

## Reuse existing MinIO and Milvus

Use this mode when storage and vector services already run on the Docker host:

```bash
cp deploy/server-reuse.env.example deploy/server-reuse.env
chmod 600 deploy/server-reuse.env

docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  config --quiet

docker compose \
  -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  up -d --build
```

This profile creates project-owned MongoDB, Redis, Keycloak and PostgreSQL
services while reaching MinIO/Milvus through `host.docker.internal`.

## Schema administration

Always run a dry-run first:

```bash
docker compose -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  exec -T api python scripts/schema_admin.py --target all
```

Review the plan before adding `--apply`. Back up MongoDB and Milvus before
production migrations.

## Health and logs

```bash
curl -fsS http://127.0.0.1:8010/health/live
curl -fsS http://127.0.0.1:8010/health/ready

docker compose -f compose.server-reuse.yaml \
  --env-file deploy/server-reuse.env \
  logs --tail=200 api
```

One-shot initialization services may exit with code `0`; that is expected.

## Public deployment checklist

- Terminate TLS at a maintained reverse proxy or ingress.
- Keep API, Keycloak admin, MongoDB, Redis, MinIO and Milvus off the public
  network unless explicitly required.
- Use `APP_ENV=production`, JWT validation and exact CORS origins.
- Store secrets in a secret manager instead of source-controlled files.
- Use least-privilege MinIO and database accounts.
- Back up MongoDB, MinIO and Milvus and test restoration.
- Configure log retention, metrics authentication and alerting.
- Pin images by digest and apply dependency/security updates.
- Scale CPU/GPU workers based on measured parsing and embedding latency.

The Compose files are not a high-availability topology. Multi-node production
deployments should use managed dependencies or an orchestrator with disruption
budgets, persistent volumes, rolling updates and workload isolation.
