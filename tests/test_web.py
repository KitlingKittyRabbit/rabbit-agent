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
    # 显式隔离 store/keys：绝不触碰真实 ~/.rabbit-agent/keys.json
    kwargs.setdefault("store_path", tmp_path / ".providers.toml")
    kwargs.setdefault("keys_path", tmp_path / "keys.json")
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
    (pa / "sub").mkdir()
    (pa / "sub" / "inner.txt").write_text("内层文件", encoding="utf-8")
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


async def test_rename_session(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    orch = make_orch(tmp_path, store=SessionStore(db))
    queue = orch.subscribe()
    sid = next(iter(orch.conversations))
    orch.handle_client_message({"type": "rename_session", "session": sid, "title": "改名了"})
    event = await _until(queue, lambda e: e.get("type") == "session_updated")
    assert event["title"] == "改名了"
    assert orch.conversations[sid].title == "改名了"
    assert SessionStore(db).list_sessions()[0]["title"] == "改名了"


async def test_delete_session(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    orch = make_orch(tmp_path, store=SessionStore(db))
    queue = orch.subscribe()
    await orch.start()
    try:
        sid = next(iter(orch.conversations))
        orch.handle_client_message({"type": "delete_session", "session": sid})
        event = await _until(queue, lambda e: e.get("type") == "session_deleted")
        assert event["session"] == sid
        assert sid not in orch.conversations
        assert SessionStore(db).list_sessions() == []
    finally:
        await orch.stop()


async def test_delete_nonexistent_session(tmp_path: Path) -> None:
    orch = make_orch(tmp_path)
    queue = orch.subscribe()
    orch.handle_client_message({"type": "delete_session", "session": "nope"})
    event = await _until(queue, lambda e: e.get("type") == "error")
    assert "不存在" in event["message"]


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
    """执行事件流：subagent 的 queued/started/tool_started/tool_finished/completed。"""
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
        event_types = []

        async def collect() -> None:
            turn_ends = 0
            while turn_ends < 2:
                event = await queue.get()
                if event["type"] != "turn_end":
                    event_types.append(event["type"])
                else:
                    turn_ends += 1

        await asyncio.wait_for(collect(), timeout=5)
    finally:
        await orch.stop()

    assert "subagent_queued" in event_types
    assert "subagent_started" in event_types
    assert "subagent_tool_started" in event_types
    assert "subagent_tool_finished" in event_types
    assert "subagent_completed" in event_types


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
        # 时间线纯逻辑模块（ES module，浏览器端 app.js import）
        resp = client.get("/timeline_logic.mjs")
        assert resp.status_code == 200
        assert "export function hasWork" in resp.text
        assert "text/javascript" in resp.headers["content-type"]


def test_ls_single_level_and_errors(tmp_path: Path) -> None:
    """文件栏后端：子目录只返回本层；坏路径/缺文件给出错误。"""
    projects = two_projects(tmp_path)
    orch = make_orch(tmp_path, projects=projects)
    with TestClient(create_app(orch)) as client:
        sub = client.get("/api/ls", params={"project": "pa", "path": "sub"})
        assert sub.status_code == 200
        assert "inner.txt" in sub.json()["result"]
        assert "a.txt" not in sub.json()["result"]  # 不再混入父层内容

        bad_dir = client.get("/api/ls", params={"project": "pa", "path": "sub/nope"})
        assert bad_dir.status_code == 400
        assert "error" in bad_dir.json()

        bad_file = client.get("/api/read", params={"project": "pa", "path": "sub/nope.txt"})
        assert bad_file.status_code == 400
        assert "error" in bad_file.json()


# ---------- 历史 API（timeline/task） ----------


def test_timeline_and_task_api(tmp_path: Path) -> None:
    from agent.core.events import SUBAGENT_COMPLETED, TOOL_STARTED

    store = SessionStore(tmp_path / "s.db")
    store.create_session("s1", "default", "会话")
    turn = store.create_turn("s1", "写文件", "running")
    store.add_event(
        "s1", TOOL_STARTED, 1.0, turn_id=turn, actor="main",
        name="call_subagent", arguments="{}",
    )
    store.finish_turn(turn, "completed", "完成")
    store.create_task(1, "s1", turn, "任务一", "prompt", "mock", "Fake", "done")
    store.add_event(
        "s1", SUBAGENT_COMPLETED, 2.0, turn_id=turn, task_id=1,
        actor="subagent", text="ok",
    )

    with TestClient(create_app(make_orch(tmp_path, store=store))) as client:
        resp = client.get("/api/timeline", params={"session": "s1"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["turns"]) == 1
        assert data["turns"][0]["final_text"] == "完成"
        assert [e["type"] for e in data["turns"][0]["events"]] == [TOOL_STARTED, SUBAGENT_COMPLETED]
        assert data["tasks"][0]["title"] == "任务一"

        resp = client.get("/api/task", params={"session": "s1", "task": 1})
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["task"]["status"] == "done"
        assert detail["events"][0]["type"] == SUBAGENT_COMPLETED

        assert client.get("/api/task", params={"session": "s1", "task": 99}).status_code == 404


def test_timeline_and_task_without_store(tmp_path: Path) -> None:
    with TestClient(create_app(make_orch(tmp_path))) as client:
        resp = client.get("/api/timeline", params={"session": "s1"})
        assert resp.json() == {"turns": [], "tasks": [], "context": None}
        assert client.get("/api/task", params={"session": "s1", "task": 1}).status_code == 404


def test_timeline_context_differs_old_vs_new(tmp_path: Path) -> None:
    """切换/加载即返回上下文环数据（含 system）：旧会话明显大于新会话，未知模型不编造百分比。"""
    from agent.providers import ChatResult, FakeProvider, Message

    store = SessionStore(tmp_path / "s.db")
    store.create_session("old", "default", "旧会话")
    store.create_session("fresh", "default", "新会话")
    store.replace("old", [Message(role="user", content="长" * 4000)])
    store.replace("fresh", [])
    store.close()

    main = FakeProvider([ChatResult(text="x")])
    orch = make_orch(tmp_path, main=main, store=SessionStore(tmp_path / "s.db"))
    orch.set_context_window("main", 100_000)  # 用户显式覆盖（无 provider 元数据）
    with TestClient(create_app(orch)) as client:
        old = client.get("/api/timeline", params={"session": "old"}).json()["context"]
        fresh = client.get("/api/timeline", params={"session": "fresh"}).json()["context"]
        assert old["window"] == 100_000 and old["window_source"] == "user"
        assert old["messages"] == 1
        assert old["used_tokens"] > fresh["used_tokens"]  # system 也计入新会话
        assert fresh["used_tokens"] > 0
        assert old["percent"] == round(old["used_tokens"] / 100_000 * 100)
        assert old["exact"] is False  # 历史未经过本次请求 → 估算
        assert old["compact_at"] and old["compact_at"] < 100_000


def test_timeline_context_unknown_model(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.create_session("s1", "default", "会话")
    store.close()
    orch = make_orch(tmp_path, store=SessionStore(tmp_path / "s.db"))
    with TestClient(create_app(orch)) as client:
        context = client.get("/api/timeline", params={"session": "s1"}).json()["context"]
        assert context["window"] is None
        assert context["window_source"] == "unknown"
        assert context["percent"] is None  # 不编造百分比


async def test_session_list_includes_created_at(tmp_path: Path) -> None:
    orch = make_orch(tmp_path, store=SessionStore(tmp_path / "s.db"))
    queue = orch.subscribe()
    orch.handle_client_message({"type": "list_sessions"})
    event = await _until(queue, lambda e: e.get("type") == "session_list")
    assert event["sessions"]
    assert all("created_at" in s and s["created_at"] for s in event["sessions"])


async def test_completed_turn_persisted(tmp_path: Path) -> None:
    """跑完一轮后 turns/events 必须落库（UI 历史的数据源）。"""
    from agent.core.events import FINAL_TEXT_DELTA, TURN_STARTED

    db = tmp_path / "s.db"
    orch = make_orch(tmp_path, store=SessionStore(db))
    await orch.start()
    try:
        session = next(iter(orch.conversations))
        orch.handle_client_message({"type": "user", "session": session, "text": "你好"})
        await asyncio.sleep(0.3)
    finally:
        await orch.stop()

    store = SessionStore(db)
    turns = store.list_turns(session)
    assert len(turns) == 1
    assert turns[0]["user_message"] == "你好"
    assert turns[0]["status"] == "completed"
    assert turns[0]["final_text"] == "x"
    event_types = [e["type"] for e in store.list_events(session)]
    assert TURN_STARTED in event_types
    assert FINAL_TEXT_DELTA in event_types
    store.close()


async def test_cancelled_turn_persisted(tmp_path: Path) -> None:
    class BlockingProvider:
        async def chat(self, messages, tools=None, on_text=None):
            await asyncio.Event().wait()
            return ChatResult(text="不应到达")

    db = tmp_path / "s.db"
    orch = make_orch(tmp_path, main=BlockingProvider(), store=SessionStore(db))
    await orch.start()
    try:
        session = next(iter(orch.conversations))
        orch.handle_client_message({"type": "user", "session": session, "text": "跑"})
        await asyncio.sleep(0.1)
        orch.handle_client_message({"type": "stop", "session": session})
        await asyncio.sleep(0.3)
    finally:
        await orch.stop()

    store = SessionStore(db)
    turns = store.list_turns(session)
    assert len(turns) == 1
    assert turns[0]["status"] == "cancelled"
    store.close()


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
