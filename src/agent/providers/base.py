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
    """对话消息。tool_calls 仅 assistant 使用；tool_call_id 仅 tool 使用。

    reasoning 仅本地展示；content_blocks 保存 provider 原始有序内容块
    （Anthropic thinking[含 signature]/text/tool_use），下一轮按协议原样回传。
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    content_blocks: list[dict] | None = None


@dataclass(frozen=True)
class ModelCapability:
    """单个模型的能力（只来自 provider 元数据或用户覆盖，绝不按名称/协议猜）。"""

    window: int | None = None
    reasoning_returned: bool | None = None
    reasoning_mode: str = "unknown"  # adjustable | fixed | none | unknown
    levels: tuple[str, ...] | None = None
    max_output: int | None = None
    tools: bool | None = None
    interleaved: str | None = None  # 思考回传字段名（如 reasoning_content）；目录声明才回传
    source: str = "unknown"  # provider | user | unknown

    def as_dict(self) -> dict:
        return {
            "window": self.window,
            "reasoning_returned": self.reasoning_returned,
            "reasoning_mode": self.reasoning_mode,
            "levels": list(self.levels) if self.levels is not None else None,
            "max_output": self.max_output,
            "tools": self.tools,
            "interleaved": self.interleaved,
            "source": self.source,
        }


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
    reasoning_tokens: int = 0


@dataclass
class ChatResult:
    """一次对话补全的完整结果。stop_reason: stop | tool_use | length | other

    reasoning 为 provider 明确返回的思考文本（无则为空）。
    reasoning_blocks 保存 provider 原始思考块（如 Anthropic thinking+signature），
    供下一轮按协议原样回传。
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    reasoning: str = ""
    reasoning_blocks: list[dict] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)  # 原始有序内容块（协议回传用）


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
OnReasoning = Callable[[str], None]


class Provider(Protocol):
    """provider 协议接口：输入消息与工具定义，输出完整结果；文本增量经 on_text 流出。"""

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
        on_reasoning: OnReasoning | None = None,
    ) -> ChatResult: ...


def is_opencode_host(base_url: str | None) -> bool:
    """OpenCode Zen/Go 网关主机判定（go/zen 的会话头只发给该站及其子域）。"""
    from urllib.parse import urlparse

    if not base_url:
        return False
    try:
        host = (urlparse(str(base_url)).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    return host == "opencode.ai" or host.endswith(".opencode.ai")
