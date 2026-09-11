"""project_store 测试：.projects.toml 读写、删除、权限。"""

import stat
from pathlib import Path

from agent.core.project_store import load_projects, remove_project, save_project


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = tmp_path / ".projects.toml"
    save_project(store, "p1", {"name": "项目一", "path": "/tmp/a"})
    save_project(store, "p2", {"name": "项目二", "path": "/tmp/b"})

    loaded = load_projects(store)
    assert loaded["p1"] == {"name": "项目一", "path": "/tmp/a"}
    assert loaded["p2"]["path"] == "/tmp/b"


def test_update_keeps_others(tmp_path: Path) -> None:
    store = tmp_path / ".projects.toml"
    save_project(store, "p1", {"name": "a", "path": "/x"})
    save_project(store, "p2", {"name": "b", "path": "/y"})
    save_project(store, "p1", {"name": "a2", "path": "/x2"})

    loaded = load_projects(store)
    assert loaded["p1"]["name"] == "a2"
    assert loaded["p2"]["name"] == "b"


def test_remove_project(tmp_path: Path) -> None:
    store = tmp_path / ".projects.toml"
    save_project(store, "p1", {"name": "a", "path": "/x"})
    save_project(store, "p2", {"name": "b", "path": "/y"})
    remove_project(store, "p1")
    assert list(load_projects(store)) == ["p2"]


def test_file_permission_is_600(tmp_path: Path) -> None:
    store = tmp_path / ".projects.toml"
    save_project(store, "p1", {"name": "a", "path": "/x"})
    assert stat.S_IMODE(store.stat().st_mode) == 0o600


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert load_projects(tmp_path / "nope.toml") == {}
