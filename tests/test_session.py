"""会话存储测试：多会话 SQLite 读写、隔离、整段重写、旧库迁移。"""

import sqlite3
from pathlib import Path

from agent.core.session import SessionStore
from agent.providers import Message, ToolCall

MESSAGES = [
    Message(role="user", content="你好"),
    Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})],
    ),
    Message(role="tool", content="内容", tool_call_id="c1"),
    Message(role="assistant", content="看完了"),
]


def test_create_and_list_sessions(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.create_session("aaa", "p1", "第一个")
    store.create_session("bbb", "p2", "第二个")
    sessions = store.list_sessions()
    assert [s["id"] for s in sessions] == ["aaa", "bbb"]
    assert sessions[0]["title"] == "第一个"
    store.close()


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.append("s1", MESSAGES)
    store.close()
    assert SessionStore(tmp_path / "s.db").load("s1") == MESSAGES


def test_sessions_isolated(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.append("s1", [Message(role="user", content="A")])
    store.append("s2", [Message(role="user", content="B")])
    assert [m.content for m in store.load("s1")] == ["A"]
    assert [m.content for m in store.load("s2")] == ["B"]
    store.close()


def test_replace_rewrites_history(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.append("s1", [Message(role="user", content="旧1"), Message(role="user", content="旧2")])
    store.replace("s1", [Message(role="user", content="新摘要")])
    store.append("s2", [Message(role="user", content="别会话")])
    store.close()
    assert [m.content for m in SessionStore(tmp_path / "s.db").load("s1")] == ["新摘要"]
    store.close()


def test_load_empty(tmp_path: Path) -> None:
    assert SessionStore(tmp_path / "s.db").load("any") == []


def test_set_title_and_delete(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.create_session("s1", "p1", "旧名")
    store.append("s1", [Message(role="user", content="hi")])
    store.set_title("s1", "新名")
    assert store.list_sessions()[0]["title"] == "新名"
    store.delete_session("s1")
    assert store.list_sessions() == []
    assert store.load("s1") == []
    store.close()


def test_migrate_drops_legacy_schema(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "s.db")
    conn.execute(
        "CREATE TABLE messages (idx INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT,"
        " content TEXT DEFAULT '', tool_calls TEXT, tool_call_id TEXT)"
    )
    conn.execute("INSERT INTO messages (role, content) VALUES ('user', '旧')")
    conn.commit()
    conn.close()
    store = SessionStore(tmp_path / "s.db")
    assert store.load("any") == []
    store.append("s1", [Message(role="user", content="新")])
    assert [m.content for m in store.load("s1")] == ["新"]
    store.close()


def test_messages_persist_reasoning_and_content_blocks(tmp_path: Path) -> None:
    """协议块（thinking+signature/tool_use 顺序）与 reasoning 完整入库并可恢复。"""
    from agent.providers import Message, ToolCall

    store = SessionStore(tmp_path / "s.db")
    blocks = [
        {"type": "thinking", "thinking": "先读", "signature": "sig-1"},
        {"type": "text", "text": "我来看看"},
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
    ]
    messages = [
        Message(role="user", content="修复 a.py"),
        Message(
            role="assistant", content="我来看看",
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a.py"})],
            reasoning="先读", content_blocks=blocks,
        ),
        Message(role="tool", content="内容", tool_call_id="t1"),
    ]
    store.append("s1", messages)
    loaded = store.load("s1")
    assert loaded[1].reasoning == "先读"
    assert loaded[1].content_blocks == blocks  # 顺序与字段完全一致
    store.close()


def test_messages_invalid_blocks_json_degrades(tmp_path: Path) -> None:
    """坏 JSON 的旧记录安全降级为 None，不阻塞读取。"""
    store = SessionStore(tmp_path / "s.db")
    store._conn.execute(
        "INSERT INTO messages (session_id, role, content, content_blocks)"
        " VALUES ('s1', 'assistant', 'x', '{not-json')"
    )
    store._conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls)"
        " VALUES ('s1', 'assistant', 'y', '{bad')"
    )
    store._conn.commit()
    loaded = store.load("s1")
    assert len(loaded) == 2
    assert loaded[0].content_blocks is None and loaded[1].tool_calls is None
    store.close()


def test_old_messages_table_migrates_columns_without_data_loss(tmp_path: Path) -> None:
    """旧库 messages 无 reasoning/content_blocks：自动补列且不丢行。"""
    import sqlite3

    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, project_id TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
        CREATE TABLE messages (idx INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '', tool_calls TEXT, tool_call_id TEXT);
        INSERT INTO messages (session_id, role, content) VALUES ('s1', 'user', '旧消息');
        """
    )
    conn.commit()
    conn.close()
    store = SessionStore(db)
    loaded = store.load("s1")
    assert [m.content for m in loaded] == ["旧消息"]
    assert loaded[0].reasoning is None and loaded[0].content_blocks is None
    store.close()


def test_messages_stream_isolated(tmp_path):
    """双流：main/executor 各自独立读写，互不干扰。"""
    store = SessionStore(tmp_path / "s.db")
    store.append("s1", [Message(role="user", content="指挥者消息")], "main")
    store.append("s1", [Message(role="user", content="任务A"),
                        Message(role="assistant", content="执行结果")], "executor")
    store.append("s1", [Message(role="assistant", content="指挥者回复")], "main")
    assert [m.content for m in store.load("s1", "main")] == ["指挥者消息", "指挥者回复"]
    assert [m.content for m in store.load("s1", "executor")] == ["任务A", "执行结果"]
    # 整段重写只影响本流
    store.replace("s1", [Message(role="user", content="重写后")], "executor")
    assert [m.content for m in store.load("s1", "executor")] == ["重写后"]
    assert [m.content for m in store.load("s1", "main")] == ["指挥者消息", "指挥者回复"]


def test_messages_stream_migration_from_old_db(tmp_path):
    """旧库（无 stream 列）打开后补列，旧消息全部归入 main 流。"""
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE messages (idx INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
        " role TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', tool_calls TEXT,"
        " tool_call_id TEXT);")
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('s1','user','旧消息')")
    conn.commit()
    conn.close()
    store = SessionStore(path)
    rows = store.load("s1")
    assert [m.content for m in rows] == ["旧消息"]
    assert rows[0].content_blocks is None


def test_reconcile_stale_tasks(tmp_path: Path) -> None:
    """启动对账：running/queued 遗留任务标记 cancelled，终态任务不受影响。"""
    store = SessionStore(tmp_path / "s.db")
    try:
        store.create_session("s1", "", "t")
        store.create_task(1, "s1", None, "t", "p", "m", "prov", "running")
        store.create_task(2, "s1", None, "t", "p", "m", "prov", "queued")
        store.create_task(3, "s1", None, "t", "p", "m", "prov", "done")
        assert store.reconcile_stale_tasks("s1") == 2
        assert store.get_task("s1", 1)["status"] == "cancelled"
        assert store.get_task("s1", 2)["status"] == "cancelled"
        assert store.get_task("s1", 3)["status"] == "done"
    finally:
        store.close()
