"""工具与注册表：一个工具 = 定义（ToolSpec）+ 异步处理函数。"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from ..providers import ToolSpec

ToolHandler = Callable[[dict], Awaitable[str]]

MAX_OUTPUT_CHARS = 30_000  # 工具输出统一上限（字符）：防止单次结果撑爆上下文

# 凭据形态兜底：拿不到确切值（如子进程回显）时也能挡住常见 token/URL 内嵌凭据
_SECRET_VALUE_SHAPES = re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{10,}")
_SECRET_URL_SHAPES = re.compile(r"(\bhttps?://)[^/\s@]+:[^/\s@]+@")


def scrub_secrets(text: str, *secrets: str) -> str:
    """清洗意外出现的凭据：已知值精确替换 + token 形态/URL 内嵌凭据兜底。

    工具结果、异常文本、审计记录都必须经此出口，保证凭据不进模型上下文与日志。
    """
    out = text or ""
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***")
    out = _SECRET_VALUE_SHAPES.sub("***", out)
    return _SECRET_URL_SHAPES.sub(r"\1***@", out)


def cap_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """统一输出预算：超限时保留头尾，中间以标记明示省略了多少字符。"""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n"
        f"……（输出过长，中间省略 {omitted} 字符；完整共 {len(text)} 字符）\n"
        f"{text[-tail:]}"
    )


class ToolError(Exception):
    """工具执行失败；信息会以文本形式反馈给模型，供其自我纠正。"""


@dataclass
class Tool:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """按名索引的工具表。主/sub agent 的差异就是挂载不同的注册表。

    on_call 非空时按阶段回调 (name, args, phase, payload)：
    started / finished(成功) / error(异常)——审计日志的挂点。
    """

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        on_call: Callable[[str, dict, str, str | None], None] | None = None,
    ) -> None:
        self._tools = {t.spec.name: t for t in tools}
        self._on_call = on_call

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in self.names()]

    def add(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    async def call(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            raise ToolError(f"未知工具: {name}")
        if self._on_call is not None:
            self._on_call(name, arguments, "started", None)
        try:
            result = await self._tools[name].handler(arguments)
        except Exception as e:
            if self._on_call is not None:
                self._on_call(name, arguments, "error", str(e))
            raise
        result = cap_output(result)
        if self._on_call is not None:
            self._on_call(name, arguments, "finished", result)
        return result

    async def call_safe(self, name: str, arguments: dict) -> str:
        """供 loop 使用：任何失败都转为错误文本，让模型看到并自我纠正。"""
        try:
            return await self.call(name, arguments)
        except Exception as e:
            return f"错误: {e}"
