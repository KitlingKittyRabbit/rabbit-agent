"""工具与注册表：一个工具 = 定义（ToolSpec）+ 异步处理函数。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from ..providers import ToolSpec

ToolHandler = Callable[[dict], Awaitable[str]]


class ToolError(Exception):
    """工具执行失败；信息会以文本形式反馈给模型，供其自我纠正。"""


@dataclass
class Tool:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """按名索引的工具表。主/sub agent 的差异就是挂载不同的注册表。"""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {t.spec.name: t for t in tools}

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in self.names()]

    def add(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    async def call(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            raise ToolError(f"未知工具: {name}")
        return await self._tools[name].handler(arguments)

    async def call_safe(self, name: str, arguments: dict) -> str:
        """供 loop 使用：任何失败都转为错误文本，让模型看到并自我纠正。"""
        try:
            return await self.call(name, arguments)
        except Exception as e:
            return f"错误: {e}"
