"""会话存储测试：SQLite 读写往返。"""

from pathlib import Path

from agent.core.session import SessionStore
from agent.providers import Message, ToolCall


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    messages = [
        Message(role="user", content="你好"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})],
        ),
        Message(role="tool", content="内容", tool_call_id="c1"),
        Message(role="assistant", content="看完了"),
    ]
    store.append(messages)
    store.close()

    loaded = SessionStore(tmp_path / "s.db").load()
    assert loaded == messages


def test_load_empty(tmp_path: Path) -> None:
    assert SessionStore(tmp_path / "s.db").load() == []


def test_replace_rewrites_history(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.append([Message(role="user", content="旧1"), Message(role="user", content="旧2")])
    store.replace([Message(role="user", content="新摘要"), Message(role="user", content="新消息")])
    store.close()

    loaded = SessionStore(tmp_path / "s.db").load()
    assert [m.content for m in loaded] == ["新摘要", "新消息"]
