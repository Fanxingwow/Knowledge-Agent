import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gitignore_blocks_secrets_and_runtime_artifacts() -> None:
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    required = {".env", ".venv/", "logs/", "output/", "__pycache__/", "*.pyc"}
    assert required.issubset(set(rules))


def test_env_example_contains_no_values() -> None:
    for raw in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        assert key.strip()
        assert value.strip() == ""


def test_direct_dependencies_are_declared() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(data["project"]["dependencies"]).lower()
    assert "pydantic-settings" in dependencies
    assert "pyjwt" in dependencies
    assert "neo4j" in dependencies


def test_project_metadata_is_not_placeholder_text() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "knowledge-agent"
    assert "add your description" not in data["project"]["description"].lower()
