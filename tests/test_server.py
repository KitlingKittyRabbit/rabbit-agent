"""server WebSocket 管道测试：stub orchestrator 验证收发接线（逻辑由 orchestrator 测试覆盖）。"""

import asyncio

from fastapi.testclient import TestClient

from agent.server.app import create_app


class StubOrchestrator:
    def __init__(self) -> None:
        self.received: list[dict] = []
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def subscribe(self) -> asyncio.Queue:
        return self.outbox

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        pass

    def handle_client_message(self, data: dict) -> None:
        self.received.append(data)
        self.outbox.put_nowait({"type": "text_delta", "text": "回显:" + data.get("text", "")})


def test_ws_roundtrip() -> None:
    stub = StubOrchestrator()
    app = create_app(stub)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "user", "text": "hi"})
            assert ws.receive_json() == {"type": "text_delta", "text": "回显:hi"}
            ws.send_json({"type": "set_plan_mode", "on": True})
            assert ws.receive_json() == {"type": "text_delta", "text": "回显:"}
    assert stub.received == [
        {"type": "user", "text": "hi"},
        {"type": "set_plan_mode", "on": True},
    ]


def test_static_assets_no_store() -> None:
    """静态资源必须 no-store：避免旧缓存模块与新 app.js 不匹配导致整页不执行。"""
    app = create_app(StubOrchestrator(), token="t")
    with TestClient(app) as client:
        for path in ["/", "/app.js", "/timeline_logic.mjs", "/styles.css"]:
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert resp.headers.get("cache-control") == "no-store", path


def test_katex_assets_served() -> None:
    """数学渲染资源必须本地可用（零外联）。"""
    app = create_app(StubOrchestrator(), token="t")
    with TestClient(app) as client:
        for path in ["/katex/katex.min.js", "/katex/katex.min.css",
                     "/katex/contrib/auto-render.min.js"]:
            resp = client.get(path)
            assert resp.status_code == 200, path


def test_list_project_files_filters_and_ignores(tmp_path) -> None:
    """@ 候选：忽略重目录、空查询只列根子项、查询按前缀/子串过滤。"""
    from agent.server.app import list_project_files

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("x", encoding="utf-8")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "main.py").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("x", encoding="utf-8")

    assert list_project_files(tmp_path) == ["README.md", "src/", "web/"]  # 空查询=字母序
    assert list_project_files(tmp_path, "app") == ["src/app.js"]
    assert list_project_files(tmp_path, "main", 1) == ["web/main.py"]
    assert all(".git" not in p and "node_modules" not in p
               for p in list_project_files(tmp_path, "j"))


def test_api_files_route(tmp_path) -> None:
    class Orch(StubOrchestrator):
        def project_root(self, project: str):
            return tmp_path if project == "p1" else None

    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    app = create_app(Orch(), token="t")
    with TestClient(app) as client:
        ok = client.get("/api/files?project=p1&q=note&token=t")
        assert ok.status_code == 200 and ok.json()["paths"] == ["note.txt"]
        assert client.get("/api/files?project=p1&token=bad").status_code == 401
        assert client.get("/api/files?project=nope&token=t").status_code == 404


def test_api_skills_route(tmp_path) -> None:
    class Orch(StubOrchestrator):
        def project_root(self, project: str):
            return tmp_path if project == "p1" else None

    app = create_app(Orch(), token="t")
    with TestClient(app) as client:
        assert client.get("/api/skills?project=p1&token=t").json() == {"skills": []}
        assert client.get("/api/skills?project=p1&token=bad").status_code == 401
        assert client.get("/api/skills?project=nope&token=t").status_code == 404


def test_api_upload_saves_and_dedupes(tmp_path) -> None:
    """拖放上传：落到 <项目>/.agent/attachments/，重名自动加序号，非法名被清洗。"""
    class Orch(StubOrchestrator):
        def project_root(self, project: str):
            return tmp_path if project == "p1" else None

    app = create_app(Orch(), token="t")
    with TestClient(app) as client:
        url = "/api/upload?project=p1&filename=note.txt&token=t"
        r1 = client.post(url, content=b"hello")
        assert r1.status_code == 200
        assert r1.json()["path"] == ".agent/attachments/note.txt"
        r2 = client.post(url, content=b"again")
        assert r2.json()["path"] == ".agent/attachments/note-1.txt"
        assert (tmp_path / ".agent" / "attachments" / "note.txt").read_bytes() == b"hello"

        bad = client.post("/api/upload?project=p1&filename=../evil.sh&token=t", content=b"x")
        assert bad.json()["path"] == ".agent/attachments/evil.sh"
        long_name = client.post(
            f"/api/upload?project=p1&filename={'a' * 200}.md&token=t", content=b"x")
        assert long_name.json()["path"].endswith(".md")       # 截断保留扩展名
        assert not (tmp_path.parent / "evil.sh").exists()

        assert client.post(url.replace("token=t", "token=bad"), content=b"x").status_code == 401
        assert client.post("/api/upload?project=nope&filename=a&token=t",
                           content=b"x").status_code == 404
        assert client.post(url, content=b"").status_code == 400
