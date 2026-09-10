"""provider 协议接口与公共类型。

上层（agent loop）只依赖本模块定义的接口与类型，不感知具体协议差异。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    """模型发出的一次工具调用。"""

    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    """对话消息。tool_calls 仅 assistant 使用；tool_call_id 仅 tool 使用。"""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolSpec:
    """工具定义，parameters 为 JSON Schema。"""

    name: str
    description: str
    parameters: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatResult:
    """一次对话补全的完整结果。stop_reason: stop | tool_use | length | other"""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)


class ProviderError(Exception):
    """provider 层统一异常基类。status_code 为 HTTP 状态码（如有）。"""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(ProviderError):
    """鉴权失败（401）。"""


class RateLimitError(ProviderError):
    """触发限流（429）。"""


class ContextOverflowError(ProviderError):
    """上下文超长。"""


OnText = Callable[[str], None]


class Provider(Protocol):
    """provider 协议接口：输入消息与工具定义，输出完整结果；文本增量经 on_text 流出。"""

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
    ) -> ChatResult: ...
