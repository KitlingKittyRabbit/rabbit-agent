"""OpenAI 兼容协议适配（覆盖按量 API / coding plan OpenAI 端点 / Ollama）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx2
import openai

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

_STOP_MAP = {"stop": "stop", "tool_calls": "tool_use", "length": "length"}


class OpenAICompatProvider:
    """max_retries 默认 0：重试策略属于上层 loop 的职责，不在 SDK 层隐藏重试。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str,
        model: str,
        timeout: float = 600.0,
        max_retries: int = 0,
        http_client: object | None = None,
    ) -> None:
        self._model = model
        client_kwargs: dict = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        else:
            # trust_env=False：不吃环境代理变量（桌面代理 socks:// 等会让 SDK 初始化崩溃）
            client_kwargs["http_client"] = httpx2.AsyncClient(trust_env=False)
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
    ) -> ChatResult:
        kwargs: dict = {
            "model": self._model,
            "messages": [self._convert_message(m) for m in messages],
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            return await self._consume(stream, on_text)
        except openai.AuthenticationError as e:
            raise AuthError(str(e), status_code=401) from e
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), status_code=429) from e
        except openai.BadRequestError as e:
            if self._is_context_overflow(e):
                raise ContextOverflowError(str(e), status_code=400) from e
            raise ProviderError(str(e), status_code=400) from e
        except openai.APIStatusError as e:
            raise ProviderError(str(e), status_code=e.status_code) from e
        except openai.APIError as e:
            raise ProviderError(str(e)) from e

    @staticmethod
    def _is_context_overflow(error: openai.BadRequestError) -> bool:
        message = str(error).lower()
        return "context" in message or "too long" in message

    @staticmethod
    def _convert_message(message: Message) -> dict:
        if message.role == "assistant" and message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in message.tool_calls
                ],
            }
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _convert_tool(tool: ToolSpec) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    async def _consume(self, stream: AsyncIterator, on_text: OnText | None) -> ChatResult:
        text_parts: list[str] = []
        tool_slots: dict[int, dict[str, str]] = {}
        finish: str | None = None
        usage = Usage()
        async for chunk in stream:
            if chunk.usage is not None:
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    text_parts.append(delta.content)
                    if on_text is not None:
                        on_text(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = tool_slots.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                slot["name"] = tc.function.name
                            if tc.function.arguments:
                                slot["args"] += tc.function.arguments
            if choice.finish_reason:
                finish = choice.finish_reason
        tool_calls = [self._build_tool_call(tool_slots[i]) for i in sorted(tool_slots)]
        return ChatResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=_STOP_MAP.get(finish or "stop", "other"),
            usage=usage,
        )

    @staticmethod
    def _build_tool_call(slot: dict[str, str]) -> ToolCall:
        if not slot["args"]:
            arguments = {}
        else:
            try:
                arguments = json.loads(slot["args"])
            except json.JSONDecodeError as e:
                raise ProviderError(f"工具参数不是合法 JSON: {slot['args']!r}") from e
        return ToolCall(id=slot["id"], name=slot["name"], arguments=arguments)
