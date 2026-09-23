"""Anthropic 兼容协议适配（覆盖 coding plan Anthropic 端点）。"""

from __future__ import annotations

from collections.abc import Sequence

import anthropic
import httpx2

from .base import (
    AuthError,
    ChatResult,
    ContextOverflowError,
    Message,
    OnReasoning,
    OnText,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolSpec,
    Usage,
)
from .listing import fetch_models

_STOP_MAP = {"end_turn": "stop", "tool_use": "tool_use", "max_tokens": "length"}


_EFFORT_BUDGET = {"off": 0, "low": 2_048, "medium": 8_192, "high": 16_384}


class AnthropicCompatProvider:
    """max_retries 默认 0：重试策略属于上层 loop 的职责，不在 SDK 层隐藏重试。"""

    def reasoning_reserve(self, effort: str) -> int:
        """Anthropic thinking 预算（协议级映射，与窗口无关的部分）。"""
        return _EFFORT_BUDGET.get(effort, 0)

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 8192,
        timeout: float = 600.0,
        max_retries: int = 0,
        http_client: object | None = None,
        context_window: int | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int = 0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = base_url
        self._api_key = api_key
        self.context_window = context_window
        self._reasoning_effort = reasoning_effort
        self._thinking_budget = thinking_budget
        client_kwargs: dict = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        else:
            # trust_env=False：不吃环境代理变量（桌面代理 socks:// 等会让 SDK 初始化崩溃）
            client_kwargs["http_client"] = httpx2.AsyncClient(trust_env=False)
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    async def list_models(self) -> list[dict]:
        """拉取 Anthropic 模型列表（内存凭据，SDK/HTTP 均不带 key 入 URL）。"""
        return await fetch_models("anthropic", self._base_url, self._api_key or "")

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
        on_reasoning: OnReasoning | None = None,
    ) -> ChatResult:
        system, converted = self._convert_messages(messages)
        kwargs: dict = {"model": self._model, "messages": converted, "max_tokens": self._max_tokens}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        budget = 0
        if self._reasoning_effort and self._reasoning_effort != "off":
            budget = self.reasoning_reserve(self._reasoning_effort)
            if budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                kwargs["max_tokens"] = max(self._max_tokens, budget + 1_024)
        try:
            reasoning_parts: list[str] = []
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "text":
                        if on_text is not None:
                            on_text(event.text)
                    elif etype == "thinking":
                        reasoning_parts.append(event.thinking)
                        if on_reasoning is not None:
                            on_reasoning(event.thinking)
                final = await stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise AuthError(str(e), status_code=401) from e
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), status_code=429) from e
        except anthropic.BadRequestError as e:
            if self._is_context_overflow(e):
                raise ContextOverflowError(str(e), status_code=400) from e
            raise ProviderError(str(e), status_code=400) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(str(e), status_code=e.status_code) from e
        except anthropic.APIError as e:
            raise ProviderError(str(e)) from e
        result = self._convert_result(final)
        if not result.reasoning and reasoning_parts:
            result.reasoning = "".join(reasoning_parts)
        return result

    @staticmethod
    def _is_context_overflow(error: anthropic.BadRequestError) -> bool:
        message = str(error).lower()
        return "context" in message or "too long" in message

    @classmethod
    def _convert_messages(cls, messages: Sequence[Message]) -> tuple[str, list[dict]]:
        """拆分 system；连续 tool 结果合并进同一个 user 消息（Anthropic 协议要求）。"""
        system_parts: list[str] = []
        out: list[dict] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
            elif message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                prev = out[-1] if out else None
                if (
                    prev is not None
                    and prev["role"] == "user"
                    and isinstance(prev["content"], list)
                ):
                    prev["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif message.role == "assistant":
                if message.content_blocks:
                    # 原样回传（thinking+signature / text / tool_use 顺序不变）
                    out.append({
                        "role": "assistant",
                        "content": [dict(block) for block in message.content_blocks],
                    })
                else:
                    out.append({"role": "assistant", "content": cls._assistant_content(message)})
            else:
                out.append({"role": "user", "content": message.content})
        return "\n\n".join(system_parts), out

    @staticmethod
    def _assistant_content(message: Message) -> list[dict]:
        content: list[dict] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        for tc in message.tool_calls or []:
            content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        return content or [{"type": "text", "text": ""}]

    @staticmethod
    def _block_to_dict(block) -> dict:
        if isinstance(block, dict):
            return dict(block)
        btype = getattr(block, "type", "")
        if btype == "thinking":
            return {
                "type": "thinking",
                "thinking": getattr(block, "thinking", "") or "",
                "signature": getattr(block, "signature", "") or "",
            }
        if btype == "redacted_thinking":
            return {"type": "redacted_thinking", "data": getattr(block, "data", "") or ""}
        if btype == "text":
            return {"type": "text", "text": getattr(block, "text", "") or ""}
        if btype == "tool_use":
            return {
                "type": "tool_use",
                "id": getattr(block, "id", "") or "",
                "name": getattr(block, "name", "") or "",
                "input": getattr(block, "input", {}) or {},
            }
        return {"type": btype} if btype else {}

    @classmethod
    def _convert_result(cls, final) -> ChatResult:
        blocks = [cls._block_to_dict(block) for block in final.content]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tool_calls = [
            ToolCall(id=b.get("id", ""), name=b.get("name", ""), arguments=b.get("input") or {})
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        reasoning = "".join(
            b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
        )
        reasoning_blocks = [
            b for b in blocks if b.get("type") in ("thinking", "redacted_thinking")
        ]
        usage = final.usage
        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=_STOP_MAP.get(final.stop_reason or "end_turn", "other"),
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                reasoning_tokens=getattr(usage, "reasoning_tokens", 0) or 0,
            ),
            reasoning=reasoning,
            reasoning_blocks=reasoning_blocks,
            blocks=blocks,
        )
