"""Codex 会话导入：定位、解析（只导对话）、落库。"""

import json
from pathlib import Path

import pytest

import agent.core.codex_import as codex
from agent.core.session import SessionStore

SESSION_ID = "01a034c4-db7a-7e70-92c6-8fb5993e03dc"


def _msg(role: str, text: str) -> dict:
    kind = "input_text" if role in ("user", "developer") else "output_text"
    return {"type": "response_item",
            "payload": {"type": "message", "role": role,
                        "content": [{"type": kind, "text": text}]}}


def _rollout(tmp_path: Path, session_id: str = SESSION_ID, records: list | None = None) -> Path:
    day = tmp_path / "sessions" / "2026" / "09" / "21"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-2026-09-21T10-00-00-{session_id}.jsonl"
    rows = records if records is not None else [
        {"type": "session_meta", "payload": {"id": session_id, "cwd": "/proj"}},
        _msg("user", "第一个问题"),
        {"type": "response_item",
         "payload": {"type": "function_call", "name": "shell", "arguments": "{}"}},
        _msg("assistant", "第一个回答"),
        _msg("developer", "系统提醒"),
        _msg("assistant", "补充回答"),
        _msg("user", "第二个问题"),
        "坏行不是 JSON",
        _msg("assistant", "第二个回答"),
    ]
    path.write_text("\n".join(
        row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) for row in rows
    ), encoding="utf-8")
    return path


def test_find_session_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(codex, "sessions_dir", lambda: tmp_path / "sessions")
    path = _rollout(tmp_path)
    assert codex.find_session_file(SESSION_ID) == path
    assert codex.find_session_file("no-such-id") is None
    assert codex.find_session_file("../escape") is None
    assert codex.find_session_file("") is None


def test_parse_session_pairs_turns_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(codex, "sessions_dir", lambda: tmp_path / "sessions")
    turns = codex.parse_session(_rollout(tmp_path))
    assert [t["user"] for t in turns] == ["第一个问题", "第二个问题"]
    assert turns[0]["assistant"] == "第一个回答\n\n补充回答"   # developer/工具被忽略
    assert turns[1]["assistant"] == "第二个回答"


def test_import_conversation_creates_session(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(codex, "sessions_dir", lambda: tmp_path / "sessions")
    path = _rollout(tmp_path)
    store = SessionStore(tmp_path / "s.db")
    try:
        info = codex.import_conversation(store, session_id=SESSION_ID, project_id="proj")
        assert info["turns"] == 2 and info["messages"] == 4 and path == Path(info["path"])
        session = next(s for s in store.list_sessions() if s["id"] == info["session"])
        assert session["title"] == f"codex · {SESSION_ID[:8]}"
        turns = store.list_turns(info["session"])
        assert [t["user_message"] for t in turns] == ["第一个问题", "第二个问题"]
        assert turns[0]["final_text"] == "第一个回答\n\n补充回答"
        assert turns[0]["status"] == "completed"
        msgs = store.load(info["session"])
        assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    finally:
        store.close()


def test_import_missing_or_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(codex, "sessions_dir", lambda: tmp_path / "sessions")
    store = SessionStore(tmp_path / "s.db")
    try:
        with pytest.raises(FileNotFoundError):
            codex.import_conversation(store, session_id="missing", project_id="p")
        _rollout(tmp_path, records=[
            {"type": "session_meta", "payload": {"id": SESSION_ID}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "x"}},
        ])
        with pytest.raises(ValueError):
            codex.import_conversation(store, session_id=SESSION_ID, project_id="p")
    finally:
        store.close()


def test_imported_turns_have_boundaries_and_undo(tmp_path, monkeypatch) -> None:
    """导入会话也带消息边界：撤销最后一条只回滚该轮，不清空全部。"""
    monkeypatch.setattr(codex, "sessions_dir", lambda: tmp_path / "sessions")
    _rollout(tmp_path)
    store = SessionStore(tmp_path / "s.db")
    try:
        info = codex.import_conversation(store, session_id=SESSION_ID, project_id="proj")
        turns = store.list_turns(info["session"])
        assert [t["msg_count"] for t in turns] == [0, 2]
        assert store.undo_from(info["session"], turns[1]["id"])["unsupported"] is False
        store.truncate_messages(info["session"], turns[1]["msg_count"])
        kept = [m.content for m in store.load(info["session"])]
        assert kept == ["第一个问题", "第一个回答\n\n补充回答"]
        assert [t["id"] for t in store.list_turns(info["session"])] == [turns[0]["id"]]
    finally:
        store.close()
