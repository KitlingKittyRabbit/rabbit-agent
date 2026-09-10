"""SQLite 会话存储：重启不丢对话。system 提示词不入库（每次启动重建）。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..providers import Message, ToolCall

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    idx INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    tool_call_id TEXT
)
"""


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def append(self, messages: list[Message]) -> None:
        self._conn.executemany(
            "INSERT INTO messages (role, content, tool_calls, tool_call_id) VALUES (?, ?, ?, ?)",
            [
                (
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

    def replace(self, messages: list[Message]) -> None:
        """整段重写（上下文压缩/截断后历史被改写，append 不再适用）。"""
        self._conn.execute("DELETE FROM messages")
        self.append(messages)

    def load(self) -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages ORDER BY idx"
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
