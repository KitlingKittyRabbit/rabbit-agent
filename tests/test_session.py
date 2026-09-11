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
