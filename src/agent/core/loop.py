"""通用 agent loop：主 agent 与 subagent 共用的唯一发动机。

循环：发给模型 → 无工具调用则完工 → 有则执行并回填 → 再发。步数上限防死循环。
event_source 非空时，每轮开始前 drain 事件消息注入对话（派发不阻塞的回注通道）。
compactor 非空时，每次调模型前调用（B：摘要压缩）；
撞 ContextOverflowError 时截断重试 ≤2 次（A：兜底自愈）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..providers import ContextOverflowError, Message, OnText, Provider, ToolCall, Usage
from ..tools.base import ToolRegistry
from .context import truncate_messages

_MAX_OVERFLOW_RETRIES = 2


@dataclass
class LoopResult:
    text: str
    stop_reason: str  # "completed" | "max_steps"
    steps: int
    usage: Usage = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = Usage()


class AgentLoop:
    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        max_steps: int = 50,
        compactor: Callable[[list[Message]], Awaitable[None]] | None = None,
        on_tool_event: Callable[[ToolCall, str], None] | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_steps = max_steps
        self._compactor = compactor
        self._on_tool_event = on_tool_event

    async def run(
        self,
        messages: list[Message],
        on_text: OnText | None = None,
        event_source: asyncio.Queue[Message] | None = None,
    ) -> LoopResult:
        steps = 0
        usage = Usage()
        while True:
            if event_source is not None:
                messages.extend(_drain(event_source))
            result = await self._chat_with_fallback(messages, on_text)
            usage.input_tokens += result.usage.input_tokens
            usage.output_tokens += result.usage.output_tokens
            messages.append(
                Message(role="assistant", content=result.text, tool_calls=result.tool_calls or None)
            )
            if not result.tool_calls:
                return LoopResult(
                    text=result.text, stop_reason="completed", steps=steps, usage=usage
                )
            for tool_call in result.tool_calls:
                output = await self._tools.call_safe(tool_call.name, tool_call.arguments)
                if self._on_tool_event is not None:
                    self._on_tool_event(tool_call, output)
                messages.append(Message(role="tool", content=output, tool_call_id=tool_call.id))
            steps += 1
            if steps >= self._max_steps:
                return LoopResult(
                    text=result.text, stop_reason="max_steps", steps=steps, usage=usage
                )

    async def _chat_with_fallback(self, messages: list[Message], on_text: OnText | None):
        specs = self._tools.specs()
        retries = 0
        while True:
            if self._compactor is not None:
                await self._compactor(messages)
            try:
                return await self._provider.chat(messages, specs or None, on_text)
            except ContextOverflowError:
                retries += 1
                if retries > _MAX_OVERFLOW_RETRIES:
                    raise
                truncate_messages(messages)


def _drain(queue: asyncio.Queue[Message]) -> list[Message]:
    items: list[Message] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return items
