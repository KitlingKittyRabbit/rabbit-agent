"""审计日志测试：只记写操作、阶段化状态、静默模式、截断。"""

import json
from pathlib import Path

from agent.core.audit import AuditLogger


def test_audit_writes_only_write_tools(tmp_path: Path) -> None:
    log = tmp_path / "audit.log"
    audit = AuditLogger(log)
    audit.log(
        session="s1", tool="write_file", phase="started", status="started",
        args={"path": "a"}, result="",
    )
    audit.log(
        session="s1", tool="write_file", phase="finished", status="success",
        args={"path": "a"}, result="ok",
    )
    audit.log(
        session="s1", tool="run_shell", phase="finished", status="denied",
        args={"command": "rm -rf x"}, result="已被用户拒绝执行",
    )
    audit.log(
        session="s1", tool="read_file", phase="finished", status="success",
        args={"path": "a"}, result="内容",
    )

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # read_file 不记
    started, finished, denied = (json.loads(line) for line in lines)
    assert started["phase"] == "started" and started["status"] == "started"
    assert finished["phase"] == "finished" and finished["status"] == "success"
    assert denied["status"] == "denied"
    assert denied["tool"] == "run_shell"
    assert "ts" in started and "session" in started


def test_audit_records_turn_and_task(tmp_path: Path) -> None:
    log = tmp_path / "audit.log"
    AuditLogger(log).log(
        session="s1", tool="write_file", phase="finished", status="success",
        args={"path": "a"}, result="ok", turn=7, task=3,
    )
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["turn"] == 7
    assert record["task"] == 3


def test_audit_noop_without_path() -> None:
    AuditLogger(None).log(
        session="s", tool="write_file", phase="finished", status="success", args={}, result="x"
    )


def test_audit_truncates_long_fields(tmp_path: Path) -> None:
    log = tmp_path / "audit.log"
    AuditLogger(log).log(
        session="s", tool="write_file", phase="finished", status="success",
        args={"content": "x" * 1000}, result="ok",
    )
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert len(record["args"]) <= 301


async def test_registry_on_call_phases() -> None:
    """ToolRegistry 按 started/finished/error 三阶段回调（审计挂点）。"""
    import pytest

    from agent.providers import ToolSpec
    from agent.tools.base import Tool, ToolRegistry

    async def ok(args: dict) -> str:
        return "结果"

    async def bad(args: dict) -> str:
        raise RuntimeError("炸了")

    records: list = []
    registry = ToolRegistry(
        [Tool(ToolSpec("ok", "", {}), ok), Tool(ToolSpec("bad", "", {}), bad)],
        on_call=lambda *args: records.append(args),
    )
    assert await registry.call("ok", {}) == "结果"
    assert [(r[2], r[3]) for r in records] == [
        ("started", None),
        ("finished", "结果"),
    ]
    with pytest.raises(RuntimeError):
        await registry.call("bad", {})
    assert records[-2][2] == "started"
    assert records[-1][2] == "error" and "炸了" in records[-1][3]


def test_classify_status_branches() -> None:
    from agent.core.conversation import _classify_status

    assert _classify_status("started", None) == "started"
    assert _classify_status("error", "boom") == "error"
    assert _classify_status("finished", "已被用户拒绝执行") == "denied"
    assert _classify_status("finished", "超时：无人确认") == "timeout"
    assert _classify_status("finished", "错误: x") == "error"
    assert _classify_status("finished", "ok") == "success"
