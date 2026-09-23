"""ChatGPT(Codex) Responses 协议 provider。

对接 https://chatgpt.com/backend-api/codex/responses（与 opencode/Codex 相同的端点与头）：
- 认证：默认走本项目的 ChatGPT 登录（codex_auth），也可用显式 api_key（测试/代理用）
- 请求：Responses API（instructions / input / tools / store=false / stream / reasoning）
- 输出：把 SSE 事件翻译成统一的 ChatResult（文本/工具调用/思考/用量），
  原始 output items 存进 blocks，下一轮按协议原样回传（reasoning 的加密内容必须回传）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import httpx2

from . import codex_auth
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

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_MODELS_CACHE = "~/.codex/models_cache.json"


def load_codex_models(path: str | None = None) -> list[dict]:
    """读本机 Codex 的模型缓存（离线，实时反映账号可用模型）。

    过滤 visibility=hide / supported_in_api=false 的条目；按 priority 排序。
    缓存缺失或不可解析时回退内置清单。
    """
    raw_path = Path(path).expanduser() if path else Path(DEFAULT_MODELS_CACHE).expanduser()
    try:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [dict(m) for m in STATIC_MODELS]
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return [dict(m) for m in STATIC_MODELS]
    out: list[dict] = []
    for item in models:
        if not isinstance(item, dict) or not item.get("slug"):
            continue
        if item.get("visibility") == "hide" or item.get("supported_in_api") is False:
            continue
        levels = [lvl.get("effort") for lvl in (item.get("supported_reasoning_levels") or [])
                  if isinstance(lvl, dict) and lvl.get("effort")]
        capability: dict = {}
        if item.get("context_window"):
            capability["window"] = int(item["context_window"])
        if levels:
            capability["reasoning_mode"] = "adjustable"
            capability["levels"] = levels
        priority = item.get("priority")
        out.append({
            "id": str(item["slug"]),
            "display_name": str(item.get("display_name") or item["slug"]),
            "capability": capability,
            "_order": priority if isinstance(priority, int) else 999,
        })
    if not out:
        return [dict(m) for m in STATIC_MODELS]
    out.sort(key=lambda m: (m["_order"], m["id"]))
    for item in out:
        item.pop("_order", None)
    return out
STATIC_MODELS = [
    {"id": "gpt-5.2-codex", "display_name": "GPT-5.2 Codex (ChatGPT)",
     "capability": {"reasoning_mode": "adjustable", "levels": ["low", "medium", "high"]}},
    {"id": "gpt-5.1-codex-max", "display_name": "GPT-5.1 Codex Max (ChatGPT)",
     "capability": {"reasoning_mode": "adjustable", "levels": ["low", "medium", "high"]}},
    {"id": "gpt-5.1-codex", "display_name": "GPT-5.1 Codex (ChatGPT)",
     "capability": {"reasoning_mode": "adjustable", "levels": ["low", "medium", "high"]}},
    {"id": "gpt-5-codex", "display_name": "GPT-5 Codex (ChatGPT)",
     "capability": {"reasoning_mode": "adjustable", "levels": ["low", "medium", "high"]}},
    {"id": "gpt-5.1-codex-mini", "display_name": "GPT-5.1 Codex Mini (ChatGPT)",
     "capability": {"reasoning_mode": "adjustable", "levels": ["low", "medium", "high"]}},
    {"id": "gpt-5", "display_name": "GPT-5 (ChatGPT)",
     "capability": {"reasoning_mode": "adjustable", "levels": ["low", "medium", "high"]}},
]


class CodexResponsesProvider:
    """max_retries 默认 0：重试策略属于上层 loop。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str = "",
        model: str,
        timeout: float = 600.0,
        max_retries: int = 0,
        http_client: object | None = None,
        context_window: int | None = None,
        reasoning_effort: str | None = None,
        token_path: str | None = None,
        account_id: str = "",
        use_oauth: bool = True,
        models: Sequence[dict] | None = None,
        models_path: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url or DEFAULT_BASE_URL
        self._endpoint = self._base_url.rstrip("/") + "/responses"
        self._api_key = api_key
        self._account_id = account_id
        self._token_path = token_path
        self._use_oauth = use_oauth
        self._explicit_models = list(models) if models else None
        self._models_path = models_path
        self.context_window = context_window
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout
        if http_client is not None:
            self._client = http_client
        else:
            self._client = httpx2.AsyncClient(trust_env=False, timeout=timeout)

    async def list_models(self) -> list[dict]:
        """Codex 后端无公开 /models：读本机 Codex 缓存；缺失时回退内置清单。"""
        if self._explicit_models is not None:
            return [dict(m) for m in self._explicit_models]
        return load_codex_models(self._models_path)

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        on_text: OnText | None = None,
        on_reasoning: OnReasoning | None = None,
        session_id: str | None = None,
    ) -> ChatResult:
        access, account = await self._credentials()
        payload = self._build_payload(messages, tools, session_id)
        headers = {
            "Authorization": f"Bearer {access}",
            "ChatGPT-Account-Id": account or self._account_id,
            "originator": codex_auth.ORIGINATOR,
            "User-Agent": codex_auth.USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if session_id:
            headers["session-id"] = session_id
        try:
            async with self._client.stream("POST", self._endpoint, json=payload,
                                           headers=headers) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise self._http_error(resp.status_code, detail)
                return await self._consume(resp, on_text, on_reasoning)
        except httpx2.HTTPError as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e

    async def _credentials(self) -> tuple[str, str]:
        if not self._use_oauth:
            return self._api_key, self._account_id
        return await codex_auth.valid_access_token(self._token_path or None)

    @staticmethod
    def _http_error(status: int, detail: str) -> ProviderError:
        if status in (401, 403):
            return AuthError(f"ChatGPT 登录已失效或无权访问（HTTP {status}）：{detail}")
        if status == 429:
            return RateLimitError(f"请求过于频繁（HTTP 429）：{detail}")
        if status == 400 and ("context" in detail.lower() or "too long" in detail.lower()):
            return ContextOverflowError(f"上下文超限：{detail}", status_code=400)
        return ProviderError(f"HTTP {status}: {detail}", status_code=status)

    def _build_payload(self, messages: Sequence[Message],
                       tools: Sequence[ToolSpec] | None,
                       session_id: str | None) -> dict:
        instructions = "\n\n".join(m.content for m in messages if m.role == "system").strip()
        items: list[dict] = []
        for message in messages:
            if message.role == "system":
                continue
            items.extend(self._convert_message(message))
        reasoning: dict = {"summary": "auto"}
        if self._reasoning_effort and self._reasoning_effort != "off":
            reasoning["effort"] = self._reasoning_effort
        body: dict = {
            "model": self._model,
            "input": items,
            "store": False,
            "stream": True,
            "parallel_tool_calls": True,
            "reasoning": reasoning,
            # store=false 时必须显式请求加密思考内容，否则无状态回传会丢推理上下文
            "include": ["reasoning.encrypted_content"],
        }
        if instructions:
            body["instructions"] = instructions
        if tools:
            body["tools"] = [
                {"type": "function", "name": t.name, "description": t.description,
                 "parameters": t.parameters}
                for t in tools
            ]
        if session_id:
            body["prompt_cache_key"] = session_id
        return body

    @staticmethod
    def _convert_message(message: Message) -> list[dict]:
        if message.role == "user":
            return [{"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": message.content}]}]
        if message.role == "tool":
            return [{"type": "function_call_output", "call_id": message.tool_call_id,
                     "output": message.content}]
        if message.role != "assistant":
            return []
        if message.content_blocks:
            return [dict(block) for block in message.content_blocks]   # 原始块原样回传
        items: list[dict] = []
        if message.content:
            items.append({"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": message.content}]})
        for call in message.tool_calls or []:
            items.append({"type": "function_call", "call_id": call.id, "name": call.name,
                          "arguments": json.dumps(call.arguments, ensure_ascii=False)})
        return items

    async def _consume(self, resp, on_text: OnText | None,
                       on_reasoning: OnReasoning | None) -> ChatResult:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        blocks: list[dict] = []
        usage = Usage()
        failed: str | None = None
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            etype = event.get("type") or ""
            if etype == "response.output_text.delta":
                delta = event.get("delta") or ""
                if delta:
                    text_parts.append(delta)
                    if on_text is not None:
                        on_text(delta)
            elif etype in ("response.reasoning_summary_text.delta",
                           "response.reasoning_text.delta"):
                delta = event.get("delta") or ""
                if delta:
                    reasoning_parts.append(delta)
                    if on_reasoning is not None:
                        on_reasoning(delta)
            elif etype == "response.output_item.done":
                item = event.get("item") or {}
                if not isinstance(item, dict):
                    continue
                blocks.append(item)
                if item.get("type") == "function_call":
                    tool_calls.append(ToolCall(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=_load_arguments(item.get("arguments")),
                    ))
            elif etype == "response.completed":
                payload = event.get("response") or {}
                usage = _usage_from(payload.get("usage") or {})
            elif etype in ("response.failed", "error", "response.error"):
                failed = str(event.get("message")
                             or (event.get("response") or {}).get("error")
                             or event)
        if failed:
            raise ProviderError(f"ChatGPT 返回失败：{failed}")
        return ChatResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "stop",
            usage=usage,
            reasoning="".join(reasoning_parts),
            reasoning_blocks=[b for b in blocks if b.get("type") == "reasoning"],
            blocks=blocks,
        )


def _load_arguments(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw or "{}"))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _usage_from(raw: dict) -> Usage:
    details = raw.get("output_tokens_details") or {}
    return Usage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
    )
