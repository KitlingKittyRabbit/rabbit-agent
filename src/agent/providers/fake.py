"""脚本化 fake provider：测试 agent loop 用，零网络、零密钥。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

from .base import ChatResult, Message, OnReasoning, OnText, ToolSpec

ScriptItem = (
    ChatResult | Exception | Callable[[Sequence[Message], "Sequence[ToolSpec] | None"], ChatResult]
)


class FakeProvider:
    """按脚本依次处理每次 chat 调用。

    - ChatResult：直接返回（有 text/reasoning 时向对应回调推送全量文本）
    - Exception：原样抛出
    - callable：以 (messages, tools) 调用，返回其结果

    calls 记录每次收到的 (messages, tools)，供测试断言。
    """

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        self._script = deque(script)
        self.calls: list[tuple[list[Message], list[ToolSpec] | None]] = []
        self.context_window: int | None = None
        self.models: list[dict] | Exception = []  # list_models 脚本
        self.models_calls = 0

    async def list_models(self) -> list[dict]:
        self.models_calls += 1
        if isinstance(self.models, Exception):
            raise self.models
        return [dict(m) for m in self.models]

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
        on_reasoning: OnReasoning | None = None,
    ) -> ChatResult:
        self.calls.append((list(messages), list(tools) if tools is not None else None))
        if not self._script:
            raise AssertionError("FakeProvider 脚本已耗尽")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        result = item(messages, tools) if callable(item) else item
        if on_text is not None and result.text:
            on_text(result.text)
        if on_reasoning is not None and result.reasoning:
            on_reasoning(result.reasoning)
        return result
