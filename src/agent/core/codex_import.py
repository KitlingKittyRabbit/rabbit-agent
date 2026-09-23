"""导入 Codex CLI 会话（只导对话）。

会话文件：~/.codex/sessions/YYYY/MM/DD/rollout-<时间戳>-<会话id>.jsonl
只取 response_item 里 role=user/assistant 的 message 文本，忽略工具调用与 developer 消息。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from uuid import uuid4

from ..providers import Message

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")
_TEXT_KINDS = ("input_text", "output_text")


def sessions_dir() -> Path:
    return Path(os.path.expanduser("~/.codex/sessions"))


def find_session_file(session_id: str) -> Path | None:
    sid = (session_id or "").strip()
    if not sid or not _SESSION_ID_RE.match(sid):
        return None
    base = sessions_dir()
    if not base.is_dir():
        return None
    matches = sorted(base.glob(f"**/rollout-*{sid}*.jsonl"))
    return matches[-1] if matches else None


def parse_session(path: str | Path) -> list[dict]:
    """解析成 [{user, assistant}]；连续 user 合并到同一轮。"""
    turns: list[dict] = []
    user: str | None = None
    parts: list[str] = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload") or {}
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in ("user", "assistant"):
            continue
        chunks = [
            str(item.get("text") or "")
            for item in payload.get("content") or []
            if isinstance(item, dict) and item.get("type") in _TEXT_KINDS
        ]
        text = "".join(chunks).strip()
        if not text:
            continue
        if role == "user":
            if user is not None:
                turns.append({"user": user, "assistant": "\n\n".join(parts)})
            user, parts = text, []
        else:
            if user is None:
                user = ""          # 孤儿 assistant（如 developer 消息之后）
            parts.append(text)
    if user is not None:
        turns.append({"user": user, "assistant": "\n\n".join(parts)})
    return turns


def import_conversation(store, *, session_id: str, project_id: str,
                        title: str = "") -> dict:
    """把 Codex 会话落成一个新会话（turns + messages），返回导入统计。"""
    path = find_session_file(session_id)
    if path is None:
        raise FileNotFoundError(
            f"未找到 Codex 会话 {session_id or '（空）'}（查找目录 {sessions_dir()}）")
    turns = parse_session(path)
    if not turns:
        raise ValueError(f"会话 {session_id} 里没有可导入的对话")
    new_id = uuid4().hex[:8]
    store.create_session(new_id, project_id, title or f"codex · {session_id[:8]}")
    messages: list[Message] = []
    for turn in turns:
        if turn["user"]:
            user_text = turn["user"]
        else:
            user_text = "（无用户消息）"
        turn_id = store.create_turn(new_id, user_text, "completed", msg_count=len(messages))
        store.finish_turn(turn_id, "completed", turn["assistant"])
        if turn["user"]:
            messages.append(Message(role="user", content=turn["user"]))
        if turn["assistant"]:
            messages.append(Message(role="assistant", content=turn["assistant"]))
    if messages:
        store.append(new_id, messages)
    return {"session": new_id, "turns": len(turns), "messages": len(messages),
            "path": str(path)}
