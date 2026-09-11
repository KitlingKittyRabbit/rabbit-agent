"""工具层：读/写/shell 工具与按角色组装注册表。"""

from collections.abc import Awaitable, Callable
from pathlib import Path

from .base import Tool, ToolError, ToolRegistry
from .file_tools import make_read_tools, make_write_tools
from .shell_tool import make_shell_tool


def build_registry(
    root: Path,
    *,
    write: bool,
    shell: bool,
    on_call: Callable[[str, dict, str], None] | None = None,
    confirm: Callable[[str], Awaitable[bool]] | None = None,
) -> ToolRegistry:
    """按角色组装：write/shell 关闭时对应工具物理缺席（plan 模式与主 agent 的保障）。

    on_call：审计挂点（每次成功调用回调）；confirm：危险命令确认挂点。
    """
    tools = make_read_tools(root)
    if write:
        tools += make_write_tools(root)
    if shell:
        tools.append(make_shell_tool(root, confirm=confirm))
    return ToolRegistry(tools, on_call=on_call)


__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "build_registry",
    "make_read_tools",
    "make_shell_tool",
    "make_write_tools",
]
