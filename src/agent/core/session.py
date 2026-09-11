"""SQLite 会话存储：多会话。system 提示词不入库（每次启动重建）。"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ..providers import Message, ToolCall

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    idx INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    tool_call_id TEXT
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """旧库（messages 无 session_id 列，或 sessions 无 project_id 列）丢弃重建（开发阶段）。"""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if cols and "session_id" not in cols:
        conn.execute("DROP TABLE messages")
    s_cols = [row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if s_cols and "project_id" not in s_cols:
        conn.execute("DROP TABLE sessions")
        conn.execute("DROP TABLE IF EXISTS messages")
    conn.commit()


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        _migrate(self._conn)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- 会话表 ----
    def create_session(self, session_id: str, project_id: str, title: str = "") -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (id, project_id, title, created_at)"
            " VALUES (?, ?, ?, ?)",
            (session_id, project_id, title, time.time()),
        )
        self._conn.commit()

    def list_sessions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, project_id, title, created_at FROM sessions ORDER BY created_at"
        ).fetchall()
        return [{"id": r[0], "project_id": r[1], "title": r[2], "created_at": r[3]} for r in rows]

    def set_title(self, session_id: str, title: str) -> None:
        self._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    # ---- 消息表 ----
    def append(self, session_id: str, messages: list[Message]) -> None:
        self._conn.executemany(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (
                    session_id,
                    m.role,
                    m.content,
                    (
                        json.dumps([tc.__dict__ for tc in m.tool_calls], ensure_ascii=False)
                        if m.tool_calls
                        else None
                    ),
                    m.tool_call_id,
                )
                for m in messages
            ],
        )
        self._conn.commit()

    def replace(self, session_id: str, messages: list[Message]) -> None:
        """整段重写某会话（上下文压缩/截断后历史被改写，append 不再适用）。"""
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.append(session_id, messages)

    def load(self, session_id: str) -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages"
            " WHERE session_id = ? ORDER BY idx",
            (session_id,),
        ).fetchall()
        out: list[Message] = []
        for role, content, tool_calls_json, tool_call_id in rows:
            tool_calls = (
                [ToolCall(**tc) for tc in json.loads(tool_calls_json)] if tool_calls_json else None
            )
            out.append(
                Message(
                    role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id
                )
            )
        return out

    def close(self) -> None:
        self._conn.close()
