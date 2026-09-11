"""审计日志测试：只记写操作、静默模式、截断。"""

import json
from pathlib import Path

from agent.core.audit import AuditLogger


def test_audit_writes_only_write_tools(tmp_path: Path) -> None:
    log = tmp_path / "audit.log"
    audit = AuditLogger(log)
    audit.log(session="s1", tool="write_file", args={"path": "a"}, result="ok")
    audit.log(session="s1", tool="run_shell", args={"command": "ls"}, result="exit code: 0")
    audit.log(session="s1", tool="read_file", args={"path": "a"}, result="内容")
    audit.log(session="s1", tool="grep", args={"pattern": "x"}, result="(无匹配)")

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["tool"] == "write_file"
    assert record["session"] == "s1"
    assert "ts" in record


def test_audit_noop_without_path() -> None:
    AuditLogger(None).log(session="s", tool="write_file", args={}, result="x")


def test_audit_truncates_long_fields(tmp_path: Path) -> None:
    log = tmp_path / "audit.log"
    AuditLogger(log).log(session="s", tool="write_file", args={"content": "x" * 1000}, result="ok")
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert len(record["args"]) <= 301
