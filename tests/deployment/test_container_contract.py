from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _compose(filename: str = "compose.yaml") -> dict[str, object]:
    payload = yaml.safe_load((PROJECT_ROOT / filename).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _services(filename: str = "compose.yaml") -> dict[str, dict[str, object]]:
    services = _compose(filename).get("services")
    assert isinstance(services, Mapping)
    return {
        str(name): cast(dict[str, object], service)
        for name, service in services.items()
        if isinstance(service, Mapping)
    }


def test_compose_contains_the_complete_enterprise_runtime() -> None:
    services = _services()

    assert {
        "api",
        "mineru-gateway",
        "mongo",
        "mongo-rs-init",
        "redis",
        "minio",
        "minio-init",
        "etcd",
        "milvus",
        "prometheus",
        "prometheus-init",
    } <= set(services)
    assert services["api"]["read_only"] is True
    assert services["api"]["cap_drop"] == ["ALL"]
    assert services["api"]["security_opt"] == ["no-new-privileges:true"]
    assert services["api"]["stop_grace_period"] == "60s"
    assert services["api"]["healthcheck"]
    assert services["mineru-gateway"]["read_only"] is True
    assert services["mineru-gateway"]["cap_drop"] == ["ALL"]
    assert "ports" not in services["mineru-gateway"]


def test_dependencies_are_loopback_bound_by_default() -> None:
    services = _services()

    for service_name in ("mongo", "redis", "minio", "milvus", "prometheus"):
        ports = services[service_name].get("ports")
        assert isinstance(ports, Sequence)
        assert not isinstance(ports, (str, bytes))
        assert ports
        for port in ports:
            assert isinstance(port, str)
            assert ":-127.0.0.1}" in port


def test_images_are_pinned_and_observability_is_opt_in() -> None:
    services = _services()

    for service in services.values():
        image = service.get("image")
        if image is None:
            continue
        assert isinstance(image, str)
        assert ":latest" not in image
    assert services["prometheus"]["profiles"] == ["observability"]
    assert services["prometheus-init"]["profiles"] == ["observability"]
    assert services["prometheus"]["read_only"] is True
    assert services["prometheus"]["user"] == "65532:65532"


def test_mongodb_is_pinned_to_the_kernel_compatible_release() -> None:
    expected = "${MONGO_IMAGE:-mongo:7.0.37-jammy}"

    for filename in ("compose.yaml", "compose.server-reuse.yaml"):
        services = _services(filename)
        assert services["mongo"]["image"] == expected
        assert services["mongo-rs-init"]["image"] == expected

    for filename in (
        "deploy/compose.env.example",
        "deploy/server-reuse.env.example",
    ):
        content = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "MONGO_IMAGE=mongo:7.0.37-jammy" in content
        assert "mongo:8." not in content


def test_api_requires_distributed_and_metrics_secrets() -> None:
    services = _services()
    environment = services["api"].get("environment")
    milvus_environment = services["milvus"].get("environment")
    assert isinstance(environment, Mapping)
    assert isinstance(milvus_environment, Mapping)

    assert ":?set REDIS_CHAT_KEY_HMAC_SECRET}" in str(environment["REDIS_CHAT_KEY_HMAC_SECRET"])
    assert ":?set METRICS_AUTH_TOKEN}" in str(environment["METRICS_AUTH_TOKEN"])
    assert environment["METRICS_PATH"] == "/metrics"
    assert environment["MONGO_REQUIRE_REPLICA_SET"] == "${MONGO_REQUIRE_REPLICA_SET:-true}"
    assert environment["MINIO_ACCESS_KEY"] == "${MINIO_APP_ACCESS_KEY:?set MINIO_APP_ACCESS_KEY}"
    assert environment["MINIO_SECRET_KEY"] == "${MINIO_APP_SECRET_KEY:?set MINIO_APP_SECRET_KEY}"  # noqa: S105
    assert milvus_environment["MINIO_ACCESS_KEY_ID"] == "${MINIO_MILVUS_ACCESS_KEY:?set MINIO_MILVUS_ACCESS_KEY}"
    assert (
        milvus_environment["MINIO_SECRET_ACCESS_KEY"] == "${MINIO_MILVUS_SECRET_KEY:?set MINIO_MILVUS_SECRET_KEY}"  # noqa: S105
    )


def test_server_reuse_compose_owns_only_application_stateful_dependencies() -> None:
    services = _services("compose.server-reuse.yaml")

    assert {
        "api",
        "keycloak",
        "keycloak-db",
        "keycloak-realm-config",
        "mineru-gateway",
        "mongo-keyfile-init",
        "mongo",
        "mongo-rs-init",
        "redis",
        "prometheus",
        "prometheus-init",
    } <= set(services)
    assert {"minio", "minio-init", "etcd", "milvus"}.isdisjoint(services)
    assert "ports" not in services["mongo"]
    assert "ports" not in services["redis"]


def test_server_reuse_api_uses_host_gateway_for_external_object_and_vector_stores() -> None:
    services = _services("compose.server-reuse.yaml")
    api = services["api"]
    environment = api.get("environment")
    dependencies = api.get("depends_on")

    assert isinstance(environment, Mapping)
    assert isinstance(dependencies, Mapping)
    assert set(dependencies) == {
        "keycloak",
        "keycloak-realm-config",
        "mineru-gateway",
        "mongo-rs-init",
        "redis",
    }
    assert api["extra_hosts"] == ["host.docker.internal:host-gateway"]
    assert environment["MINIO_ENDPOINT"] == "${MINIO_ENDPOINT:?set MINIO_ENDPOINT}"
    assert environment["MILVUS_URL"] == "${MILVUS_URL:?set MILVUS_URL}"
    assert "minio:9000" not in str(environment)
    assert "milvus:19530" not in str(environment)
    assert "203.0.113.10" not in str(environment)


def test_mineru_cloud_token_is_isolated_to_the_private_gateway() -> None:
    for filename in ("compose.yaml", "compose.server-reuse.yaml"):
        services = _services(filename)
        api_environment = services["api"].get("environment")
        gateway = services["mineru-gateway"]
        gateway_environment = gateway.get("environment")

        assert isinstance(api_environment, Mapping)
        assert isinstance(gateway_environment, Mapping)
        assert api_environment["MINERU_API_TOKEN"] == "${MINERU_GATEWAY_AUTH_TOKEN:-}"  # noqa: S105
        assert api_environment["MINERU_BASE_URL"] == "${MINERU_BASE_URL:-http://mineru-gateway:8080}"
        assert "MINERU_UPSTREAM_API_TOKEN" not in api_environment
        assert gateway_environment["MINERU_UPSTREAM_API_TOKEN"] == "${MINERU_UPSTREAM_API_TOKEN:-}"  # noqa: S105
        assert gateway_environment["MINERU_GATEWAY_AUTH_TOKEN"] == "${MINERU_GATEWAY_AUTH_TOKEN:-}"  # noqa: S105
        assert gateway["expose"] == ["8080"]
        assert "ports" not in gateway
        assert gateway["read_only"] is True
        assert gateway["cap_drop"] == ["ALL"]


def test_server_reuse_defaults_avoid_host_port_conflicts_and_preserve_hardening() -> None:
    services = _services("compose.server-reuse.yaml")
    api = services["api"]

    assert api["ports"] == ["${API_BIND_ADDRESS:-127.0.0.1}:${API_PORT:-8010}:8000"]
    assert api["read_only"] is True
    assert api["cap_drop"] == ["ALL"]
    assert api["security_opt"] == ["no-new-privileges:true"]
    assert api["stop_grace_period"] == "60s"
    assert api["healthcheck"]
    assert all("container_name" not in service for service in services.values())


def test_compose_builds_use_the_configured_debian_mirror() -> None:
    expected = "${DEBIAN_MIRROR:-https://mirrors.cloud.tencent.com}"

    for filename in ("compose.yaml", "compose.server-reuse.yaml"):
        build = _services(filename)["api"].get("build")
        assert isinstance(build, Mapping)
        args = build.get("args")
        assert isinstance(args, Mapping)
        assert args["DEBIAN_MIRROR"] == expected


def test_server_reuse_requires_application_credentials_not_external_root_secrets() -> None:
    services = _services("compose.server-reuse.yaml")
    environment = services["api"].get("environment")
    serialized = json.dumps(_compose("compose.server-reuse.yaml"), sort_keys=True)

    assert isinstance(environment, Mapping)
    assert ":?set MINIO_ACCESS_KEY}" in str(environment["MINIO_ACCESS_KEY"])
    assert ":?set MINIO_SECRET_KEY}" in str(environment["MINIO_SECRET_KEY"])
    assert ":?set REDIS_CHAT_KEY_HMAC_SECRET}" in str(environment["REDIS_CHAT_KEY_HMAC_SECRET"])
    assert ":?set METRICS_AUTH_TOKEN}" in str(environment["METRICS_AUTH_TOKEN"])
    assert "MINIO_ROOT_" not in serialized
    assert "MINIO_MILVUS_" not in serialized


def test_server_reuse_environment_example_is_safe_and_complete() -> None:
    content = (PROJECT_ROOT / "deploy/server-reuse.env.example").read_text(encoding="utf-8")
    values = {
        key.strip(): value.strip()
        for line in content.splitlines()
        if line and not line.lstrip().startswith("#") and "=" in line
        for key, value in (line.split("=", maxsplit=1),)
    }

    assert values["API_BIND_ADDRESS"] == "127.0.0.1"
    assert values["API_PORT"] == "8010"
    assert values["KEYCLOAK_BIND_ADDRESS"] == "127.0.0.1"
    assert values["KEYCLOAK_PORT"] == "8081"
    assert values["DEBIAN_MIRROR"] == "https://mirrors.cloud.tencent.com"
    assert values["MINIO_ENDPOINT"] == "host.docker.internal:9000"
    assert values["MILVUS_URL"] == "http://host.docker.internal:19530"
    assert values["MINIO_BUCKET_NAME"] == "knowledge-base-files"
    assert values["CHUNKS_COLLECTION"] == "knowledge_agent_kb_chunks_v2"
    assert values["ITEM_NAME_COLLECTION"] == "knowledge_agent_kb_item_names_v2"
    assert "change-me" in values["MINIO_ACCESS_KEY"]
    assert "change-me" in values["MINIO_SECRET_KEY"]
    assert "change" in values["MONGO_REPLICA_KEY"].lower()
    assert values["KEYCLOAK_REGISTRATION_ENABLED"] == "false"
    assert values["KEYCLOAK_EMAIL_VERIFICATION_LIFESPAN_SECONDS"] == "900"
    assert values["KEYCLOAK_SMTP_HOST"] == "smtp.example.com"
    assert "change-me" in values["KEYCLOAK_SMTP_PASSWORD"]
    assert "203.0.113.10" not in content


def test_server_reuse_keycloak_is_private_hardened_and_persistent() -> None:
    services = _services("compose.server-reuse.yaml")
    keycloak = services["keycloak"]
    database = services["keycloak-db"]
    api_environment = services["api"].get("environment")
    keycloak_environment = keycloak.get("environment")
    database_volumes = database.get("volumes")

    assert keycloak["image"] == "${KEYCLOAK_IMAGE:-quay.io/keycloak/keycloak:26.7.0}"
    assert database["image"] == "${KEYCLOAK_POSTGRES_IMAGE:-postgres:17.6-alpine}"
    assert keycloak["ports"] == ["${KEYCLOAK_BIND_ADDRESS:-127.0.0.1}:${KEYCLOAK_PORT:-8081}:8080"]
    assert keycloak["cap_drop"] == ["ALL"]
    assert keycloak["security_opt"] == ["no-new-privileges:true"]
    assert keycloak["healthcheck"]
    assert database["healthcheck"]
    assert isinstance(keycloak_environment, Mapping)
    assert "JAVA_OPTS_KC_HEAP" not in keycloak_environment
    assert isinstance(database_volumes, Sequence)
    assert not isinstance(database_volumes, (str, bytes))
    assert "keycloak_postgres_data:/var/lib/postgresql/data" in database_volumes
    assert isinstance(api_environment, Mapping)
    assert api_environment["AUTH_OIDC_CLIENT_ID"] == "${AUTH_OIDC_CLIENT_ID:-}"


def test_keycloak_verified_registration_is_an_idempotent_hardened_init_job() -> None:
    services = _services("compose.server-reuse.yaml")
    configurator = services["keycloak-realm-config"]
    environment = configurator.get("environment")
    dependencies = configurator.get("depends_on")

    assert configurator["image"] == "${API_IMAGE:-knowledge-agent:server}"
    assert configurator["entrypoint"] == ["/usr/bin/tini", "--"]
    assert configurator["restart"] == "no"
    assert configurator["read_only"] is True
    assert configurator["cap_drop"] == ["ALL"]
    assert configurator["security_opt"] == ["no-new-privileges:true"]
    assert "ports" not in configurator
    assert isinstance(dependencies, Mapping)
    assert dependencies == {"keycloak": {"condition": "service_healthy"}}
    assert isinstance(environment, Mapping)
    assert environment["KEYCLOAK_INTERNAL_URL"] == "http://keycloak:8080"
    assert environment["KEYCLOAK_REGISTRATION_ENABLED"] == "${KEYCLOAK_REGISTRATION_ENABLED:-false}"
    assert environment["KEYCLOAK_SMTP_PASSWORD"] == "${KEYCLOAK_SMTP_PASSWORD:-}"  # noqa: S105
    assert configurator["command"] == ["python", "scripts/keycloak_realm_admin.py"]


def test_keycloak_realm_enforces_pkce_and_emits_isolation_claims() -> None:
    realm = json.loads((PROJECT_ROOT / "deploy/keycloak/enterprise-knowledge-realm.json").read_text(encoding="utf-8"))
    assert realm["registrationAllowed"] is False
    assert realm["bruteForceProtected"] is True
    assert realm["defaultRoles"] == ["reader", "chat-user"]
    assert "users" not in realm

    client = realm["clients"][0]
    assert client["publicClient"] is True
    assert client["standardFlowEnabled"] is True
    assert client["implicitFlowEnabled"] is False
    assert client["directAccessGrantsEnabled"] is False
    assert client["attributes"]["pkce.code.challenge.method"] == "S256"
    assert client["redirectUris"] == ["${CONSOLE_PUBLIC_ORIGIN}/console*"]

    mappers = {mapper["name"]: mapper for mapper in client["protocolMappers"]}
    assert mappers["enterprise-tenant"]["config"] == {
        "claim.name": "tenant_id",
        "claim.value": "${AUTH_TENANT_ID}",
        "jsonType.label": "String",
        "access.token.claim": "true",
        "id.token.claim": "true",
        "userinfo.token.claim": "true",
        "introspection.token.claim": "true",
    }
    assert mappers["enterprise-realm-roles"]["config"]["claim.name"] == "roles"
    assert mappers["enterprise-api-audience"]["config"]["included.client.audience"] == "${KEYCLOAK_CLIENT_ID}"


def test_every_compose_bind_mount_source_exists() -> None:
    for filename in ("compose.yaml", "compose.server-reuse.yaml"):
        for service in _services(filename).values():
            volumes = service.get("volumes", ())
            assert isinstance(volumes, Sequence)
            assert not isinstance(volumes, (str, bytes))
            for volume in volumes:
                if not isinstance(volume, str) or not volume.startswith("./"):
                    continue
                source = volume.split(":", maxsplit=1)[0]
                assert (PROJECT_ROOT / source).exists(), source


def test_dockerfile_has_reproducible_non_root_runtime_contract() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "USER ${APP_UID}:${APP_GID}" in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/tini"' in dockerfile
    assert "STOPSIGNAL SIGTERM" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "scripts ./scripts" in dockerfile
    assert "COPY . " not in dockerfile
    assert dockerfile.count("ARG DEBIAN_MIRROR") == 3
    assert dockerfile.count("/etc/apt/sources.list.d/debian.sources") == 2
    assert "https://mirrors.cloud.tencent.com" in dockerfile
    assert "UV_PYTHON_DOWNLOADS=never \\" in dockerfile
    assert "UV_HTTP_TIMEOUT=300 \\" in dockerfile
    assert "UV_HTTP_RETRIES=10 \\" in dockerfile
    assert "UV_CONCURRENT_DOWNLOADS=4" in dockerfile


def test_linux_uses_cpu_pytorch_while_windows_keeps_cuda_development() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    uv_config = project["tool"]["uv"]
    expected_sources = [
        {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"},
        {"index": "pytorch-cu130", "marker": "sys_platform == 'win32'"},
    ]

    for package in ("torch", "torchvision", "torchaudio"):
        assert uv_config["sources"][package] == expected_sources

    indexes = {index["name"]: index for index in uv_config["index"]}
    assert indexes["pytorch-cpu"] == {
        "name": "pytorch-cpu",
        "url": "https://download.pytorch.org/whl/cpu",
        "explicit": True,
    }
    assert indexes["pytorch-cu130"] == {
        "name": "pytorch-cu130",
        "url": "https://download.pytorch.org/whl/cu130",
        "explicit": True,
    }
    assert indexes["tencent-pypi"] == {
        "name": "tencent-pypi",
        "url": "https://mirrors.cloud.tencent.com/pypi/simple",
        "default": True,
    }

    lock_content = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    lock = tomllib.loads(lock_content)
    packages = lock["package"]
    registries = {package["source"]["registry"] for package in packages if "registry" in package["source"]}
    assert "https://mirrors.cloud.tencent.com/pypi/simple" in registries
    assert "https://pypi.org/simple" not in registries
    assert "files.pythonhosted.org" not in lock_content
    torch_variants = {
        (package["version"], package["source"]["registry"]) for package in packages if package["name"] == "torch"
    }
    assert (
        "2.10.0+cpu",
        "https://download.pytorch.org/whl/cpu",
    ) in torch_variants
    assert (
        "2.10.0+cu130",
        "https://download.pytorch.org/whl/cu130",
    ) in torch_variants
    assert not {
        package["name"] for package in packages if package["name"].startswith("nvidia-") or package["name"] == "triton"
    }


def test_prometheus_uses_a_file_backed_bearer_token() -> None:
    configuration = yaml.safe_load((PROJECT_ROOT / "deploy/prometheus/prometheus.yml").read_text(encoding="utf-8"))
    scrape_configs = configuration["scrape_configs"]
    assert scrape_configs[0]["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/run/metrics/token",
    }
    assert scrape_configs[0]["static_configs"][0]["targets"] == ["api:8000"]


def test_container_entrypoint_preserves_signal_delivery() -> None:
    entrypoint = (PROJECT_ROOT / "deploy/docker/entrypoint.sh").read_text(encoding="utf-8")

    assert "set -eu" in entrypoint
    assert 'exec "$@"' in entrypoint


def test_mongo_application_user_can_reconcile_runtime_schema() -> None:
    script = (PROJECT_ROOT / "deploy/mongo/init-replica-set.js").read_text(encoding="utf-8")

    assert '{ role: "readWrite", db: appDatabaseName }' in script
    assert '{ role: "dbAdmin", db: appDatabaseName }' in script
    assert 'role: "dbOwner"' not in script


def test_minio_init_provisions_distinct_least_privilege_identities() -> None:
    script = (PROJECT_ROOT / "deploy/minio/init.sh").read_text(encoding="utf-8")

    assert "knowledge-agent-app-v1" in script
    assert "knowledge-agent-milvus-v1" in script
    assert 'mc admin user info local "$MINIO_APP_ACCESS_KEY"' in script
    assert 'mc admin user info local "$MINIO_MILVUS_ACCESS_KEY"' in script
    assert 'mc admin policy attach local knowledge-agent-app-v1 --user "$MINIO_APP_ACCESS_KEY"' in script
    assert ('mc admin policy attach local knowledge-agent-milvus-v1 --user "$MINIO_MILVUS_ACCESS_KEY"') in script

    policy_match = re.search(
        r'''printf '%s\\n' "(\{.*\})" > "\$output"''',
        script,
    )
    assert policy_match is not None
    policy = json.loads(policy_match.group(1).replace(r"\"", '"').replace("${bucket}", "bucket-under-test"))
    resources = {resource for statement in policy["Statement"] for resource in statement["Resource"]}
    assert resources == {
        "arn:aws:s3:::bucket-under-test",
        "arn:aws:s3:::bucket-under-test/*",
    }
    assert all("*" not in statement["Action"] for statement in policy["Statement"])
