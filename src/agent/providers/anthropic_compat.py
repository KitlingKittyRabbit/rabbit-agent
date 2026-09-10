"""Anthropic 兼容协议适配（覆盖 coding plan Anthropic 端点）。"""

from __future__ import annotations

from collections.abc import Sequence

import anthropic

from .base import (
    AuthError,
    ChatResult,
    ContextOverflowError,
    Message,
    OnText,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolSpec,
    Usage,
)

_STOP_MAP = {"end_turn": "stop", "tool_use": "tool_use", "max_tokens": "length"}


class AnthropicCompatProvider:
    """max_retries 默认 0：重试策略属于上层 loop 的职责，不在 SDK 层隐藏重试。"""

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
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        client_kwargs: dict = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
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
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if on_text is not None:
                        on_text(text)
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
        return self._convert_result(final)

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
    def _convert_result(final) -> ChatResult:
        text = "".join(block.text for block in final.content if block.type == "text")
        tool_calls = [
            ToolCall(id=block.id, name=block.name, arguments=block.input)
            for block in final.content
            if block.type == "tool_use"
        ]
        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=_STOP_MAP.get(final.stop_reason or "end_turn", "other"),
            usage=Usage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            ),
        )
