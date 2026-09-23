"""通用 agent loop：主 agent 与 subagent 共用的唯一发动机。

循环：发给模型 → 无工具调用则完工 → 有则执行并回填 → 再发。步数上限防死循环。
event_source 非空时，每轮开始前 drain 事件消息注入对话（派发不阻塞的回注通道）。
compactor 非空时，每次调模型前调用（B：摘要压缩）；
撞 ContextOverflowError 时截断重试 ≤2 次（A：兜底自愈）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..providers import (
    ContextOverflowError,
    Message,
    OnReasoning,
    OnText,
    Provider,
    ToolCall,
    Usage,
)
from ..tools.base import ToolRegistry
from .context import estimate_tokens, truncate_messages

_MAX_OVERFLOW_RETRIES = 2


@dataclass
class LoopResult:
    text: str
    stop_reason: str  # completed | max_steps | no_progress | context_budget
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
        on_tool_event: Callable[[ToolCall, str | None, str], None] | None = None,
        on_call_text: Callable[[str, bool], None] | None = None,
        on_step: Callable[[int], None] | None = None,
        on_reasoning: OnReasoning | None = None,
        on_llm_call: Callable[[Usage], None] | None = None,
        max_context_tokens: int | None = None,
        no_progress_limit: int = 3,
        session_id: str | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_steps = max_steps
        self._compactor = compactor
        self._on_tool_event = on_tool_event
        self._on_call_text = on_call_text
        self._on_step = on_step
        self._on_reasoning = on_reasoning
        self._on_llm_call = on_llm_call
        self._max_context_tokens = max_context_tokens
        self._no_progress_limit = max(2, no_progress_limit)
        self._session_id = session_id
        try:
            params = inspect.signature(provider.chat).parameters
            self._accepts_reasoning = "on_reasoning" in params
            self._accepts_session = "session_id" in params
        except (TypeError, ValueError):
            self._accepts_reasoning = False
            self._accepts_session = False

    async def run(
        self,
        messages: list[Message],
        on_text: OnText | None = None,
        event_source: asyncio.Queue[Message] | None = None,
    ) -> LoopResult:
        steps = 0
        usage = Usage()
        signature: tuple | None = None
        repeats = 0
        while True:
            if event_source is not None:
                messages.extend(_drain(event_source))
            if self._max_context_tokens is not None:
                estimated = estimate_tokens(messages, tool_specs=self._tools.specs())
                if estimated > self._max_context_tokens:
                    return LoopResult(
                        text="[上下文预算不足，已停止以避免溢出]",
                        stop_reason="context_budget",
                        steps=steps,
                        usage=usage,
                    )
            result = await self._chat_with_fallback(messages, on_text)
            if self._on_llm_call is not None:
                self._on_llm_call(result.usage)
            usage.input_tokens += result.usage.input_tokens
            usage.output_tokens += result.usage.output_tokens
            usage.reasoning_tokens += result.usage.reasoning_tokens
            # working/final 判定：带 tool_calls 的中间调用是 working，收尾调用是 final
            if self._on_call_text is not None and result.text:
                self._on_call_text(result.text, not result.tool_calls)
            messages.append(
                Message(
                    role="assistant",
                    content=result.text,
                    tool_calls=result.tool_calls or None,
                    reasoning=result.reasoning or None,
                    content_blocks=result.blocks or None,
                )
            )
            if not result.tool_calls:
                return LoopResult(
                    text=result.text, stop_reason="completed", steps=steps, usage=usage
                )
            round_signature: list[tuple[str, str, str]] = []
            for tool_call in result.tool_calls:
                if self._on_tool_event is not None:
                    self._on_tool_event(tool_call, None, "started")
                output = await self._tools.call_safe(tool_call.name, tool_call.arguments)
                if self._on_tool_event is not None:
                    self._on_tool_event(tool_call, output, "finished")
                messages.append(Message(role="tool", content=output, tool_call_id=tool_call.id))
                round_signature.append(
                    (
                        tool_call.name,
                        json.dumps(tool_call.arguments, sort_keys=True, default=str),
                        output,
                    )
                )
            steps += 1
            # 步数 = 已完成的模型回合数（一轮多工具算 1 步）；回调值与 LoopResult.steps 一致
            if self._on_step is not None:
                self._on_step(steps)
            if steps >= self._max_steps:
                return LoopResult(
                    text=result.text, stop_reason="max_steps", steps=steps, usage=usage
                )
            current = tuple(round_signature)
            repeats = repeats + 1 if current == signature else 1
            signature = current
            if repeats >= self._no_progress_limit:
                return LoopResult(
                    text=result.text, stop_reason="no_progress", steps=steps, usage=usage
                )

    async def _chat_with_fallback(self, messages: list[Message], on_text: OnText | None):
        specs = self._tools.specs()
        retries = 0
        while True:
            if self._compactor is not None:
                await self._compactor(messages)
            try:
                if self._accepts_reasoning:
                    if self._accepts_session:
                        return await self._provider.chat(
                            messages, specs or None, on_text, self._on_reasoning,
                            session_id=self._session_id,
                        )
                    return await self._provider.chat(
                        messages, specs or None, on_text, self._on_reasoning
                    )
                if self._accepts_session:
                    return await self._provider.chat(
                        messages, specs or None, on_text, session_id=self._session_id,
                    )
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
