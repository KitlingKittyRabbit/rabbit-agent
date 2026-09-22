"""shell 工具：项目目录内执行命令。

执行边界（不是 regex 假装安全）：
- 环境变量：最小 allowlist，不继承 server 的 provider 密钥等敏感变量
- OS 沙箱：bwrap 可用时，项目 root 可写、系统其余只读、临时 /tmp
  不可用时明确标记 unsandboxed（仅 env 隔离，不谎称沙箱）
- 超时杀整个进程组；危险命令确认作为 UX 防护（≠ 沙箱）
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import shutil
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..providers import ToolSpec
from .base import Tool, cap_output

_MAX_OUTPUT = 8192

# 敏感环境变量隔离：executor shell 只允许这些（provider key 等一律不继承）
_ENV_ALLOWLIST = (
    "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
    "TZ", "USER", "LOGNAME", "SYSTEMROOT", "WINDIR",
)

_BWRAP = shutil.which("bwrap")

# 敏感 home/config：沙箱内遮蔽（文件用 /dev/null 覆盖，目录用 tmpfs 覆盖）
_MASK_PATHS = (
    ".ssh", ".gnupg", ".aws", ".kube", ".docker", ".config",
    ".netrc", ".agent_token", ".providers.toml", ".projects.toml",
)

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


def sandbox_mode() -> str:
    """当前 shell 沙箱状态：bwrap | unsandboxed。"""
    return "bwrap" if _BWRAP else "unsandboxed"


def base_mount_args() -> list[str]:
    """bwrap 通用挂载（argv 形式）：系统只读 + 临时 /tmp。

    供 run_shell（拼成 shell 字符串）与宿主凭据代理（直接以 argv 启动）共用，
    顺序不可调换：--ro-bind / / 必须在 --dev-bind /dev 之前，否则 /dev 被盖住。
    """
    return [
        "--die-with-parent",
        "--ro-bind", "/", "/",
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
    ]


def root_bind_args(root: Path) -> list[str]:
    """把项目根挂回可写。必须排在 --tmpfs /tmp 与其它遮蔽之后：更晚的挂载才生效。"""
    return ["--bind", str(root), str(root)]


def _sandbox_env() -> dict:
    return {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}


def _mask_args() -> str:
    """敏感路径的 bwrap 遮蔽参数（仅对存在的路径生效）。"""
    home = Path.home()
    parts: list[str] = []
    for name in _MASK_PATHS:
        path = home / name
        if path.is_dir():
            parts.append(f"--tmpfs {shlex.quote(str(path))}")
        elif path.exists():
            parts.append(f"--ro-bind /dev/null {shlex.quote(str(path))}")
    return " ".join(parts)


def _wrap_command(command: str, root: Path) -> str:
    """bwrap 包裹：root 可写，其余只读，临时 /tmp，敏感 home 路径遮蔽。"""
    if _BWRAP is None:
        return command
    quoted_root = shlex.quote(str(root))
    mounts = " ".join(
        shlex.quote(part) for part in (*base_mount_args(), *root_bind_args(root))
    )
    # 不用 --new-session：沙箱内进程须留在我们的进程组，killpg 才能整组杀
    # mask 放在 root bind 之后：遮蔽必须最后生效（root bind 可能意外覆盖敏感路径）
    return (
        f"{_BWRAP} {mounts} {_mask_args()}"
        f" --chdir {quoted_root}"
        f" sh -c {shlex.quote(command)}"
    )


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
            _wrap_command(command, root),
            cwd=root,
            env=_sandbox_env(),
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
        return f"exit code: {proc.returncode}\n{cap_output(output, _MAX_OUTPUT)}".rstrip()

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
