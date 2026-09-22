"""工具层：读/写/shell 工具与按角色组装注册表。"""

from collections.abc import Awaitable, Callable
from pathlib import Path

from .base import Tool, ToolError, ToolRegistry, cap_output, scrub_secrets
from .file_tools import make_read_tools, make_write_tools
from .host_github import (
    Account,
    HostCredentialRunner,
    make_gh_accounts_tool,
    make_gh_command_tool,
    make_git_remote_tool,
    make_host_tools,
    parse_auth_status,
)
from .shell_tool import make_shell_tool


def build_registry(
    root: Path,
    *,
    write: bool,
    shell: bool,
    on_call: Callable[[str, dict, str, str | None], None] | None = None,
    confirm: Callable[[str], Awaitable[bool]] | None = None,
    host_github: str = "off",
    host_git: bool = False,
    host_runner: HostCredentialRunner | None = None,
) -> ToolRegistry:
    """按角色组装：write/shell 关闭时对应工具物理缺席（plan 模式与主 agent 的保障）。

    on_call：审计挂点（每次成功调用回调）；confirm：危险命令确认挂点。
    host_github：off（无）/ read（只看账号）/ write（可执行 gh 子命令）；
    host_git：是否提供 git 远程工具（fetch/pull/push）。
    宿主工具在 server 进程内按指定账号执行，凭据不进入模型上下文；
    普通 run_shell 仍在沙箱内、仍拿不到宿主凭据。
    """
    tools = make_read_tools(root)
    if write:
        tools += make_write_tools(root)
    if shell:
        tools.append(make_shell_tool(root, confirm=confirm))
    if host_github != "off" or host_git:
        tools += make_host_tools(
            root,
            runner=host_runner or HostCredentialRunner(),
            github=host_github,
            git=host_git,
        )
    return ToolRegistry(tools, on_call=on_call)


__all__ = [
    "Account",
    "HostCredentialRunner",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "build_registry",
    "cap_output",
    "make_gh_accounts_tool",
    "make_gh_command_tool",
    "make_git_remote_tool",
    "make_host_tools",
    "make_read_tools",
    "make_shell_tool",
    "make_write_tools",
    "parse_auth_status",
    "scrub_secrets",
]
