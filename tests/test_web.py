"""web 轮后端测试：多项目隔离、subtask_step 事件、项目管理、文件端点、页面服务、预设解析。"""

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from agent.core.orchestrator import Orchestrator
from agent.core.session import SessionStore
from agent.providers import ChatResult, FakeProvider, ToolCall
from agent.server.app import create_app


def make_orch(
    tmp_path: Path, main=None, executor=None, projects=None, store=None, **kwargs
) -> Orchestrator:
    return Orchestrator(
        main_provider=main or FakeProvider([ChatResult(text="x")]),
        executor_provider=executor or FakeProvider([ChatResult(text="y")]),
        root=tmp_path,
        store=store,
        projects=projects,
        **kwargs,
    )


def two_projects(tmp_path: Path) -> dict:
    pa = tmp_path / "projA"
    pb = tmp_path / "projB"
    pa.mkdir()
    pb.mkdir()
    (pa / "a.txt").write_text("A 项目文件", encoding="utf-8")
    (pb / "b.txt").write_text("B 项目文件", encoding="utf-8")
    return {
        "pa": {"name": "A", "path": str(pa)},
        "pb": {"name": "B", "path": str(pb)},
    }


async def _until(queue: asyncio.Queue, pred, timeout: float = 5.0):
    async def _wait():
        while True:
            event = await queue.get()
            if pred(event):
                return event

    return await asyncio.wait_for(_wait(), timeout)


# ---------- 多项目隔离 ----------


async def test_conversation_tools_rooted_at_own_project(tmp_path: Path) -> None:
    projects = two_projects(tmp_path)
    orch = make_orch(tmp_path, projects=projects)
    conv_a = orch.create_session(project_id="pa", title="A会话")
    conv_b = orch.create_session(project_id="pb", title="B会话")

    assert conv_a._root() == Path(projects["pa"]["path"])
    assert conv_b._root() == Path(projects["pb"]["path"])
    # A 会话的读工具只能看到 A 项目文件
    output = await conv_a._main_tools.call("read_file", {"path": "a.txt"})
    assert "A 项目文件" in output
    output = await conv_a._main_tools.call_safe("read_file", {"path": "b.txt"})
    assert "不存在" in output


async def test_project_session_isolation_in_store(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.create_session("s1", "pa", "A会话")
    store.create_session("s2", "pb", "B会话")
    sessions = store.list_sessions()
    assert sessions[0]["project_id"] == "pa"
    assert sessions[1]["project_id"] == "pb"
    store.close()


# ---------- 项目管理消息 ----------


async def test_create_project_and_list(tmp_path: Path) -> None:
    orch = make_orch(tmp_path)
    queue = orch.subscribe()
    orch.handle_client_message(
        {"type": "create_project", "name": "测试项目", "path": str(tmp_path)}
    )
    created = await _until(queue, lambda e: e.get("type") == "project_created")
    assert created["name"] == "测试项目"
    pid = created["project"]

    orch.handle_client_message({"type": "list_projects"})
    plist = await _until(queue, lambda e: e.get("type") == "project_list")
    assert any(p["id"] == pid for p in plist["projects"])


async def test_create_project_bad_path(tmp_path: Path) -> None:
    orch = make_orch(tmp_path)
    result = orch.create_project(name="x", path="/nonexistent/path/xyz")
    assert result["ok"] is False
    assert "不存在" in result["message"]


# ---------- subtask_step 事件 ----------


async def test_subtask_step_events_flow(tmp_path: Path) -> None:
    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="call_subagent", arguments={"prompt": "干活"})],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发"),
            ChatResult(text="确认"),
        ]
    )
    executor = FakeProvider(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(
                        id="c2", name="write_file", arguments={"path": "f.txt", "content": "hi"}
                    )
                ],
                stop_reason="tool_use",
            ),
            ChatResult(text="写完了"),
        ]
    )
    orch = make_orch(tmp_path, main, executor)
    queue = orch.subscribe()
    await orch.start()
    try:
        session = next(iter(orch.conversations))
        orch.handle_client_message({"type": "user", "session": session, "text": "开始"})
        steps = []

        async def collect() -> None:
            turn_ends = 0
            while turn_ends < 2:
                event = await queue.get()
                if event.get("type") == "subtask_step":
                    steps.append(event)
                elif event.get("type") == "turn_end":
                    turn_ends += 1

        await asyncio.wait_for(collect(), timeout=5)
    finally:
        await orch.stop()

    kinds = [s["kind"] for s in steps]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "text" in kinds
    assert all(s["task"] == 1 for s in steps)
    tool_call_step = next(s for s in steps if s["kind"] == "tool_call")
    assert "write_file" in tool_call_step["content"]


async def test_list_tasks(tmp_path: Path) -> None:
    gate = asyncio.Event()

    class BlockingExecutor:
        async def chat(self, messages, tools=None, on_text=None):
            await gate.wait()
            return ChatResult(text="x")

    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="call_subagent", arguments={"prompt": "干活"})],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发"),
        ]
    )
    orch = make_orch(tmp_path, main, BlockingExecutor())
    queue = orch.subscribe()
    await orch.start()
    try:
        session = next(iter(orch.conversations))
        orch.handle_client_message({"type": "user", "session": session, "text": "开始"})
        await _until(queue, lambda e: e.get("type") == "turn_end")
        orch.handle_client_message({"type": "list_tasks", "session": session})
        task_list = await _until(queue, lambda e: e.get("type") == "task_list")
        assert task_list["tasks"] == [{"id": 1, "status": "running"}]
    finally:
        gate.set()
        await orch.stop()


# ---------- HTTP 端点与页面 ----------


def test_http_api_and_page(tmp_path: Path) -> None:
    projects = two_projects(tmp_path)
    orch = make_orch(tmp_path, projects=projects)
    app = create_app(orch)
    with TestClient(app) as client:
        # 页面
        resp = client.get("/")
        assert resp.status_code == 200
        assert "project-tree" in resp.text
        assert "marked.min.js" in resp.text
        # 文件列表
        resp = client.get("/api/ls", params={"project": "pa", "path": "."})
        assert resp.status_code == 200
        assert "a.txt" in resp.json()["result"]
        # 读文件
        resp = client.get("/api/read", params={"project": "pb", "path": "b.txt"})
        assert "B 项目文件" in resp.json()["result"]
        # 越界拒绝
        resp = client.get("/api/read", params={"project": "pa", "path": "../projB/b.txt"})
        assert resp.status_code == 400
        # 未知项目 404
        resp = client.get("/api/ls", params={"project": "nope", "path": "."})
        assert resp.status_code == 404
        # marked 静态资源
        resp = client.get("/marked.min.js")
        assert resp.status_code == 200


# ---------- 预设解析 ----------


async def test_connect_with_preset_resolution(tmp_path: Path) -> None:
    received = {}

    def factory(**kw):
        received.update(kw)
        return FakeProvider([ChatResult(text="pong")])

    orch = make_orch(tmp_path, provider_factory=factory)
    queue = orch.subscribe()
    orch.handle_client_message(
        {
            "type": "connect_provider",
            "role": "main",
            "preset": "deepseek",
            "model": "",
            "api_key": "sk-x",
        }
    )
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    assert received["protocol"] == "openai"
    assert received["base_url"] == "https://api.deepseek.com/v1"
    assert received["model"] == "deepseek-chat"  # 空模型回落预设默认


async def test_connect_with_preset_no_key_needed(tmp_path: Path) -> None:
    received = {}

    def factory(**kw):
        received.update(kw)
        return FakeProvider([ChatResult(text="pong")])

    orch = make_orch(tmp_path, provider_factory=factory)
    queue = orch.subscribe()
    orch.handle_client_message(
        {
            "type": "connect_provider",
            "role": "executor",
            "preset": "ollama",
            "model": "",
            "api_key": "",
        }
    )
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    assert received["api_key"] == "unused"


async def test_restore_with_deleted_project_falls_back(tmp_path: Path) -> None:
    """审核缺陷回归：会话引用已删项目时回退默认项目，server 照常可用。"""
    store = SessionStore(tmp_path / "s.db")
    store.create_session("s1", "deleted-proj", "旧会话")
    store.close()
    orch = make_orch(tmp_path, store=SessionStore(tmp_path / "s.db"))
    conv = orch.conversations["s1"]
    assert conv.project_id == "default"
    # 工具可用（沙箱根已回退到默认项目）
    output = await conv._main_tools.call("ls", {})
    assert isinstance(output, str)


async def test_new_session_unknown_project_errors(tmp_path: Path) -> None:
    """审核缺陷回归：伪造 project 不炸穿连接，只回错误事件。"""
    orch = make_orch(tmp_path)
    queue = orch.subscribe()
    orch.handle_client_message({"type": "new_session", "project": "bogus", "title": ""})
    error = await _until(queue, lambda e: e.get("type") == "error")
    assert "项目不存在" in error["message"]
    orch.handle_client_message({"type": "list_projects"})
    plist = await _until(queue, lambda e: e.get("type") == "project_list")
    assert plist["projects"]


# ---------- 迁移 ----------


def test_migrate_drops_sessions_without_project_id(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT DEFAULT '', created_at REAL)"
    )
    conn.execute("INSERT INTO sessions (id, title, created_at) VALUES ('old', '旧会话', 1)")
    conn.execute(
        "CREATE TABLE messages (idx INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,"
        " role TEXT, content TEXT DEFAULT '', tool_calls TEXT, tool_call_id TEXT)"
    )
    conn.commit()
    conn.close()

    store = SessionStore(db)
    assert store.list_sessions() == []
    store.create_session("new", "p1", "新")
    assert store.list_sessions()[0]["id"] == "new"
    store.close()
