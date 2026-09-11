"""shell 工具：在工作目录执行命令，带超时、输出截断、取消杀进程、危险命令确认。"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..providers import ToolSpec
from .base import Tool

_MAX_OUTPUT = 8192

_DANGEROUS_PATTERNS = [
    r"\brm\s+[^|;&]*-[a-zA-Z]*[rf]",
    r"\bgit\s+push[^|;&]*--force",
    r"\bgit\s+reset\s+--hard",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bchmod\s+-R\s+777",
]


def is_dangerous(command: str) -> bool:
    return any(re.search(p, command) for p in _DANGEROUS_PATTERNS)


def _kill_tree(proc) -> None:
    """杀整个进程组（shell 的子进程也占管道，单杀 sh 无法释放 EOF）。"""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


def make_shell_tool(root: Path, confirm: Callable[[str], Awaitable[bool]] | None = None) -> Tool:
    async def run_shell(args: dict) -> str:
        command = args["command"]
        if confirm is not None and is_dangerous(command):
            if not await confirm(command):
                return f"已被用户拒绝执行（危险命令）: {command}"
        timeout = float(args.get("timeout", 60))
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # 独立进程组，killpg 的前提
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            _kill_tree(proc)
            await proc.wait()
            return f"超时（{timeout:g} 秒），进程已终止"
        except asyncio.CancelledError:
            _kill_tree(proc)
            with contextlib.suppress(Exception):
                await proc.wait()
            raise
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > _MAX_OUTPUT:
            half = _MAX_OUTPUT // 2
            output = output[:half] + "\n...（输出过长，中间截断）...\n" + output[-half:]
        return f"exit code: {proc.returncode}\n{output}".rstrip()

    return Tool(
        ToolSpec(
            name="run_shell",
            description="在工作目录执行 shell 命令，返回退出码与输出",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "timeout": {"type": "number", "description": "超时秒数，默认 60"},
                },
                "required": ["command"],
            },
        ),
        run_shell,
    )
