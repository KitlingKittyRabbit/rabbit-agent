"""OpenAI 兼容协议适配（覆盖按量 API / coding plan OpenAI 端点 / Ollama）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import httpx2
import openai

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
    is_opencode_host,
)
from .listing import fetch_models

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
        context_window: int | None = None,
        reasoning_effort: str | None = None,
        echo_reasoning_field: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self.context_window = context_window
        self._reasoning_effort = reasoning_effort
        # 目录声明 interleaved 的模型：assistant 消息按该字段回传思考，否则网关 400
        self._echo_reasoning_field = echo_reasoning_field
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
        self._opencode_host = is_opencode_host(base_url)
        self._session_fallback = uuid4().hex

    def _session_headers(self, session_id: str | None) -> dict | None:
        """OpenCode Go/Zen 网关要求自定义 UA + 每会话稳定的 x-opencode-session。"""
        if not self._opencode_host:
            return None
        return {
            "User-Agent": "rabbit-agent/0.1",
            "x-opencode-session": session_id or self._session_fallback,
        }

    async def list_models(self) -> list[dict]:
        """拉取 provider 真实模型列表（内存凭据，不进 URL/日志）。"""
        return await fetch_models("openai", self._base_url, self._api_key,
                                  session_id=self._session_fallback)

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
        on_reasoning: OnReasoning | None = None,
        session_id: str | None = None,
    ) -> ChatResult:
        kwargs: dict = {
            "model": self._model,
            "messages": [self._convert_message(m) for m in messages],
            "stream": True,
        }
        headers = self._session_headers(session_id)
        if headers:
            kwargs["extra_headers"] = headers
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if self._reasoning_effort and self._reasoning_effort != "off":
            kwargs["reasoning_effort"] = self._reasoning_effort
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            return await self._consume(stream, on_text, on_reasoning)
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

    def _convert_message(self, message: Message) -> dict:
        if message.role == "assistant" and message.tool_calls:
            payload = {
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
        elif message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        else:
            payload = {"role": message.role, "content": message.content}
        if (
            self._echo_reasoning_field
            and message.role == "assistant"
            and message.reasoning
        ):
            payload[self._echo_reasoning_field] = message.reasoning
        return payload

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

    async def _consume(
        self, stream: AsyncIterator, on_text: OnText | None, on_reasoning: OnReasoning | None = None
    ) -> ChatResult:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_slots: dict[int, dict[str, str]] = {}
        finish: str | None = None
        usage = Usage()
        async for chunk in stream:
            if chunk.usage is not None:
                details = getattr(chunk.usage, "completion_tokens_details", None)
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                    reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
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
                # provider 明确返回的思考内容（DeepSeek/Kimi 等兼容字段）
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    if on_reasoning is not None:
                        on_reasoning(reasoning_delta)
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
            reasoning="".join(reasoning_parts),
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


