from app.utils.path_util import PROJECT_ROOT, get_project_root


def test_project_root_uses_pyproject_marker() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert get_project_root() == PROJECT_ROOT
