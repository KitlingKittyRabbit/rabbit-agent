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
    tool_call_id TEXT,
    reasoning TEXT,
    content_blocks TEXT,
    stream TEXT NOT NULL DEFAULT 'main'
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_message TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    final_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS task_runs (
    id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    parent_turn_id INTEGER,
    title TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    final_output TEXT NOT NULL DEFAULT '',
    steps_used INTEGER NOT NULL DEFAULT 0,
    max_steps INTEGER NOT NULL DEFAULT 0,
    last_action TEXT NOT NULL DEFAULT '',
    stop_reason TEXT NOT NULL DEFAULT '',
    actions_used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, id)
);
CREATE TABLE IF NOT EXISTS session_flags (
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);
CREATE TABLE IF NOT EXISTS execution_events (
    idx INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id INTEGER,
    task_id INTEGER,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'system',
    name TEXT,
    arguments TEXT,
    result TEXT,
    status TEXT,
    text TEXT
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """旧库（messages 无 session_id 列，或 sessions 无 project_id 列）丢弃重建（开发阶段）。"""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if cols and "session_id" not in cols:
        conn.execute("DROP TABLE messages")
        cols = []
    if cols:
        if "stream" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN stream TEXT NOT NULL DEFAULT 'main'")
        for column in ("reasoning", "content_blocks"):
            if column not in cols:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {column} TEXT")
    s_cols = [row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if s_cols and "project_id" not in s_cols:
        conn.execute("DROP TABLE sessions")
        conn.execute("DROP TABLE IF EXISTS messages")
    # task_runs 进度列：向后兼容补列，不动已有数据
    t_cols = [row[1] for row in conn.execute("PRAGMA table_info(task_runs)").fetchall()]
    if t_cols:
        for column, ddl in (
            ("steps_used", "INTEGER NOT NULL DEFAULT 0"),
            ("max_steps", "INTEGER NOT NULL DEFAULT 0"),
            ("last_action", "TEXT NOT NULL DEFAULT ''"),
            ("stop_reason", "TEXT NOT NULL DEFAULT ''"),
            ("actions_used", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if column not in t_cols:
                conn.execute(f"ALTER TABLE task_runs ADD COLUMN {column} {ddl}")
    conn.commit()


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
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
        """删除会话及其全部关联记录（messages/turns/task_runs/execution_events）。"""
        for table in ("messages", "turns", "task_runs", "execution_events"):
            self._conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    # ---- Turns ----
    def create_turn(self, session_id: str, user_message: str, status: str) -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO turns (session_id, user_message, status, created_at, started_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, user_message, status, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def finish_turn(self, turn_id: int, status: str, final_text: str = "") -> None:
        self._conn.execute(
            "UPDATE turns SET status = ?, completed_at = ?, final_text = ? WHERE id = ?",
            (status, time.time(), final_text, turn_id),
        )
        self._conn.commit()

    def list_turns(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, user_message, status, created_at, started_at, completed_at, final_text"
            " FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "user_message": r[1],
                "status": r[2],
                "created_at": r[3],
                "started_at": r[4],
                "completed_at": r[5],
                "final_text": r[6],
            }
            for r in rows
        ]

    # ---- TaskRuns ----
    def max_task_id(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM task_runs WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row[0])

    def create_task(
        self,
        task_id: int,
        session_id: str,
        parent_turn_id: int | None,
        title: str,
        prompt: str,
        model: str,
        provider: str,
        status: str,
        max_steps: int = 0,
    ) -> None:
        self._conn.execute(
            "INSERT INTO task_runs (id, session_id, parent_turn_id, title, prompt, model,"
            " provider, status, created_at, max_steps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, session_id, parent_turn_id, title, prompt, model, provider, status,
             time.time(), max_steps),
        )
        self._conn.commit()

    def update_task(
        self,
        task_id: int,
        session_id: str,
        status: str,
        final_output: str | None = None,
        started: bool = False,
        steps_used: int | None = None,
        last_action: str | None = None,
        stop_reason: str | None = None,
        actions_used_delta: int = 0,
    ) -> None:
        from .events import TASK_CANCELLED, TASK_DONE, TASK_ERROR

        sets = ["status = ?"]
        params: list = [status]
        if status in (TASK_DONE, TASK_ERROR, TASK_CANCELLED):
            # 只有终态写 completed_at；COALESCE 保证一经写入不再被后续更新改变
            sets.append("completed_at = COALESCE(completed_at, ?)")
            params.append(time.time())
        if started:
            sets.append("started_at = COALESCE(started_at, ?)")
            params.append(time.time())
        if final_output is not None:
            sets.append("final_output = ?")
            params.append(final_output)
        if steps_used is not None:
            sets.append("steps_used = ?")
            params.append(steps_used)
        if last_action is not None:
            sets.append("last_action = ?")
            params.append(last_action)
        if stop_reason is not None:
            sets.append("stop_reason = ?")
            params.append(stop_reason)
        if actions_used_delta:
            sets.append("actions_used = actions_used + ?")
            params.append(actions_used_delta)
        params.extend([task_id, session_id])
        self._conn.execute(
            f"UPDATE task_runs SET {', '.join(sets)} WHERE id = ? AND session_id = ?", params
        )
        self._conn.commit()

    def get_flag(self, session_id: str, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM session_flags WHERE session_id = ? AND key = ?",
            (session_id, key),
        ).fetchone()
        return row[0] if row else default

    def set_flag(self, session_id: str, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO session_flags (session_id, key, value) VALUES (?, ?, ?)"
            " ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value",
            (session_id, key, value),
        )
        self._conn.commit()

    def reconcile_stale_tasks(self, session_id: str) -> int:
        """启动对账：把上次进程遗留的 running/queued 任务标记为 cancelled，返回条数。"""
        from .events import TASK_CANCELLED

        cur = self._conn.execute(
            "UPDATE task_runs SET status = ?, completed_at = COALESCE(completed_at, ?)"
            " WHERE session_id = ? AND status IN ('running', 'queued')",
            (TASK_CANCELLED, time.time(), session_id),
        )
        self._conn.commit()
        return cur.rowcount

    def get_task(self, session_id: str, task_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, parent_turn_id, title, prompt, model, provider, status, created_at,"
            " started_at, completed_at, final_output, steps_used, max_steps, last_action,"
            " stop_reason, actions_used FROM task_runs WHERE session_id = ? AND id = ?",
            (session_id, task_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "parent_turn_id": row[1],
            "title": row[2],
            "prompt": row[3],
            "model": row[4],
            "provider": row[5],
            "status": row[6],
            "created_at": row[7],
            "started_at": row[8],
            "completed_at": row[9],
            "final_output": row[10],
            "steps_used": row[11],
            "max_steps": row[12],
            "last_action": row[13],
            "stop_reason": row[14],
            "actions_used": row[15],
        }

    def list_tasks(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, parent_turn_id, title, model, provider, status, created_at,"
            " started_at, completed_at, final_output, steps_used, max_steps, last_action,"
            " stop_reason, actions_used FROM task_runs WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "parent_turn_id": r[1],
                "title": r[2],
                "model": r[3],
                "provider": r[4],
                "status": r[5],
                "created_at": r[6],
                "started_at": r[7],
                "completed_at": r[8],
                "final_output": r[9],
                "steps_used": r[10],
                "max_steps": r[11],
                "last_action": r[12],
                "stop_reason": r[13],
                "actions_used": r[14],
            }
            for r in rows
        ]

    # ---- ExecutionEvents ----
    def add_event(
        self,
        session_id: str,
        type: str,
        ts: float,
        turn_id: int | None = None,
        task_id: int | None = None,
        actor: str = "system",
        name: str | None = None,
        arguments: str | None = None,
        result: str | None = None,
        status: str | None = None,
        text: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO execution_events (session_id, turn_id, task_id, ts, type, actor,"
            " name, arguments, result, status, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_id, task_id, ts, type, actor, name, arguments, result, status,
             text),
        )
        self._conn.commit()

    def list_events(
        self,
        session_id: str,
        turn_id: int | None = None,
        task_id: int | None = None,
    ) -> list[dict]:
        sql = (
            "SELECT idx, turn_id, task_id, ts, type, actor, name, arguments, result, status,"
            " text FROM execution_events WHERE session_id = ?"
        )
        params: list = [session_id]
        if turn_id is not None:
            sql += " AND turn_id = ?"
            params.append(turn_id)
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY idx"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "idx": r[0],
                "turn_id": r[1],
                "task_id": r[2],
                "ts": r[3],
                "type": r[4],
                "actor": r[5],
                "name": r[6],
                "arguments": r[7],
                "result": r[8],
                "status": r[9],
                "text": r[10],
            }
            for r in rows
        ]

    # ---- 消息表（stream 双流：main=指挥者 / executor=执行者，各自独立历史） ----
    def append(self, session_id: str, messages: list[Message], stream: str = "main") -> None:
        self._conn.executemany(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id,"
            " reasoning, content_blocks, stream) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
                    m.reasoning,
                    (
                        json.dumps(m.content_blocks, ensure_ascii=False)
                        if m.content_blocks
                        else None
                    ),
                    stream,
                )
                for m in messages
            ],
        )
        self._conn.commit()

    def replace(self, session_id: str, messages: list[Message], stream: str = "main") -> None:
        """整段重写某条流（上下文压缩/截断后历史被改写，append 不再适用）。"""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND stream = ?", (session_id, stream)
        )
        self.append(session_id, messages, stream)

    @staticmethod
    def _load_blocks(raw: str | None) -> list[dict] | None:
        """坏 JSON / 旧记录安全降级为 None，绝不阻塞启动。"""
        if not raw:
            return None
        try:
            blocks = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(blocks, list):
            return None
        return [b for b in blocks if isinstance(b, dict)]

    def load(self, session_id: str, stream: str = "main") -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, reasoning, content_blocks"
            " FROM messages WHERE session_id = ? AND stream = ? ORDER BY idx",
            (session_id, stream),
        ).fetchall()
        out: list[Message] = []
        for role, content, tool_calls_json, tool_call_id, reasoning, blocks_json in rows:
            try:
                parsed_calls = json.loads(tool_calls_json) if tool_calls_json else None
            except (ValueError, TypeError):
                parsed_calls = None
            tool_calls = (
                [ToolCall(**tc) for tc in parsed_calls] if parsed_calls else None
            )
            out.append(
                Message(
                    role=role, content=content, tool_calls=tool_calls,
                    tool_call_id=tool_call_id, reasoning=reasoning,
                    content_blocks=self._load_blocks(blocks_json),
                )
            )
        return out

    def close(self) -> None:
        self._conn.close()
