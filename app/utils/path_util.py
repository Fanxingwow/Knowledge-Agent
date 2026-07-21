import os
from pathlib import Path


def get_project_root() -> Path:
    configured = os.getenv("PROJECT_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "pyproject.toml").is_file():
            return candidate
        raise FileNotFoundError(f"PROJECT_ROOT is not a project directory: {candidate}")

    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate project root containing pyproject.toml")


PROJECT_ROOT = get_project_root()
