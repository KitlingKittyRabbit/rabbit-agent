"""操作审计日志：写操作与 shell 命令的阶段化记录（JSONL），事后可回看。读操作不记。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..tools.base import scrub_secrets

# 写操作与外部变更操作：gh_command / git_remote 会产生远端或本地变更，一并入审计
_AUDIT_TOOLS = {"write_file", "edit_file", "run_shell", "gh_command", "git_remote"}


def _truncate(text: str, limit: int = 300) -> str:
    # 兜底清洗：即使某个工具漏洗，凭据也不落进审计文件
    text = scrub_secrets(text)
    return text if len(text) <= limit else text[:limit] + "…"


class AuditLogger:
    """path 为 None 时完全静默（测试/未配置）。"""

    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None

    def log(
        self,
        *,
        session: str,
        tool: str,
        phase: str,
        status: str,
        args: dict,
        result: str,
        turn: int | None = None,
        task: int | None = None,
    ) -> None:
        if self._path is None or tool not in _AUDIT_TOOLS:
            return
        record = {
            "ts": round(time.time(), 3),
            "session": session,
            "tool": tool,
            "phase": phase,
            "status": status,
            "args": _truncate(str(args)),
            "result": _truncate(result),
        }
        if turn is not None:
            record["turn"] = turn
        if task is not None:
            record["task"] = task
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
