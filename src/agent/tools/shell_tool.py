"""shell 工具：在工作目录执行命令，带超时与输出截断。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..providers import ToolSpec
from .base import Tool

_MAX_OUTPUT = 8192


def make_shell_tool(root: Path) -> Tool:
    async def run_shell(args: dict) -> str:
        command = args["command"]
        timeout = float(args.get("timeout", 60))
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return f"超时（{timeout:g} 秒），进程已终止"
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
