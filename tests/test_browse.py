"""目录浏览端点测试。"""

from pathlib import Path

from fastapi.testclient import TestClient

from agent.core.orchestrator import Orchestrator
from agent.providers import ChatResult, FakeProvider
from agent.server.app import create_app


def make_client(tmp_path: Path) -> TestClient:
    orch = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=FakeProvider([ChatResult(text="y")]),
        root=tmp_path,
    )
    return TestClient(create_app(orch))


def test_browse_lists_only_dirs(tmp_path: Path) -> None:
    (tmp_path / "dirA").mkdir()
    (tmp_path / "dirB").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.txt").write_text("x")
    with make_client(tmp_path) as client:
        resp = client.get("/api/browse", params={"path": str(tmp_path)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == str(tmp_path.resolve())
    assert data["dirs"] == ["dirA", "dirB"]  # 仅目录、无隐藏、无文件、按字母序
    assert data["parent"] == str(tmp_path.resolve().parent)


def test_browse_default_is_home() -> None:
    with TestClient(
        create_app(
            Orchestrator(
                main_provider=FakeProvider([ChatResult(text="x")]),
                executor_provider=FakeProvider([ChatResult(text="y")]),
                root=".",
            )
        )
    ) as client:
        resp = client.get("/api/browse")
    assert resp.status_code == 200
    assert resp.json()["path"] == str(Path.home())


def test_browse_nonexistent_returns_400(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        resp = client.get("/api/browse", params={"path": "/nonexistent/xyz"})
    assert resp.status_code == 400
    assert "不存在" in resp.json()["error"]
