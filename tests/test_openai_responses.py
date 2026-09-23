"""Codex Responses provider：请求构造、SSE 解析、错误映射、reasoning 回传（离线）。"""

from __future__ import annotations

import json

import httpx2
import pytest

from agent.providers import (
    AuthError,
    CodexResponsesProvider,
    ContextOverflowError,
    Message,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolSpec,
)

SSE_HEADERS = {"content-type": "text/event-stream"}


def make_provider(handler, **kwargs) -> CodexResponsesProvider:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return CodexResponsesProvider(
        model="gpt-5.2-codex", use_oauth=False, api_key="tok-test",
        account_id="acct-test", http_client=client, **kwargs)


def sse(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events).encode()


async def test_payload_and_headers() -> None:
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, content=sse(
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"usage": {}}},
        ), headers=SSE_HEADERS)

    provider = make_provider(handler, reasoning_effort="low")
    tools = [ToolSpec("read_file", "读文件", {"type": "object",
                                              "properties": {"path": {"type": "string"}}})]
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="你好"),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="read_file",
                                                       arguments={"path": "a.txt"})]),
        Message(role="tool", content="文件内容", tool_call_id="c1"),
    ]
    result = await provider.chat(messages, tools, session_id="sess-1")

    assert seen["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert seen["headers"]["authorization"] == "Bearer tok-test"
    assert seen["headers"]["chatgpt-account-id"] == "acct-test"
    assert seen["headers"]["originator"] == "rabbit-agent"
    assert seen["headers"]["session-id"] == "sess-1"
    assert seen["headers"]["user-agent"].startswith("rabbit-agent/")

    body = seen["body"]
    assert body["model"] == "gpt-5.2-codex" and body["store"] is False
    assert body["stream"] is True and body["parallel_tool_calls"] is True
    assert body["instructions"] == "sys"
    assert body["prompt_cache_key"] == "sess-1"
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert body["include"] == ["reasoning.encrypted_content"]   # store=false 必须显式请求
    assert body["tools"][0]["name"] == "read_file" and "parameters" in body["tools"][0]
    kinds = [item["type"] for item in body["input"]]
    assert kinds == ["message", "function_call", "function_call_output"]
    assert body["input"][0]["content"][0] == {"type": "input_text", "text": "你好"}
    assert body["input"][1]["arguments"] == json.dumps({"path": "a.txt"})
    assert body["input"][2]["call_id"] == "c1"
    assert result.text == "ok"


async def test_stream_maps_text_reasoning_tools_usage() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=sse(
            {"type": "response.output_item.added",
             "item": {"type": "reasoning", "id": "r1", "summary": []}},
            {"type": "response.reasoning_summary_text.delta", "delta": "想"},
            {"type": "response.output_text.delta", "delta": "你"},
            {"type": "response.output_text.delta", "delta": "好"},
            {"type": "response.output_item.done",
             "item": {"type": "reasoning", "id": "r1", "encrypted_content": "ENC",
                      "summary": [{"type": "summary_text", "text": "想"}]}},
            {"type": "response.output_item.done",
             "item": {"type": "function_call", "call_id": "call_1", "name": "read_file",
                      "arguments": "{\"path\": \"a.txt\"}"}},
            {"type": "response.completed", "response": {"usage": {
                "input_tokens": 10, "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 3}}}},
        ), headers=SSE_HEADERS)

    provider = make_provider(handler)
    texts: list[str] = []
    reasons: list[str] = []
    result = await provider.chat([Message(role="user", content="hi")],
                                 on_text=texts.append, on_reasoning=reasons.append)
    assert result.text == "你好"
    assert result.reasoning == "想"
    assert texts == ["你", "好"] and reasons == ["想"]
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == \
        [("call_1", "read_file", {"path": "a.txt"})]
    assert result.stop_reason == "tool_use"
    assert (result.usage.input_tokens, result.usage.output_tokens,
            result.usage.reasoning_tokens) == (10, 5, 3)
    assert [b["type"] for b in result.blocks] == ["reasoning", "function_call"]
    assert [b["type"] for b in result.reasoning_blocks] == ["reasoning"]


async def test_blocks_roundtrip_in_next_request() -> None:
    """原始块（含 reasoning 加密内容）必须原样回传，否则工具轮会失败。"""
    captured: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, content=sse(
            {"type": "response.completed", "response": {"usage": {}}}), headers=SSE_HEADERS)

    reasoning_block = {"type": "reasoning", "id": "r1", "encrypted_content": "ENC",
                       "summary": [{"type": "summary_text", "text": "想"}]}
    call_block = {"type": "function_call", "call_id": "call_1", "name": "read_file",
                  "arguments": "{\"path\": \"a.txt\"}"}
    provider = make_provider(handler)
    await provider.chat([
        Message(role="user", content="hi"),
        Message(role="assistant", content="", content_blocks=[reasoning_block, call_block]),
        Message(role="tool", content="内容", tool_call_id="call_1"),
    ])
    assert captured["body"]["input"][1] == reasoning_block
    assert captured["body"]["input"][2] == call_block


async def test_login_hint_when_not_logged_in(tmp_path) -> None:
    provider = CodexResponsesProvider(model="gpt-5.2-codex", token_path=str(tmp_path / "keys.json"))
    with pytest.raises(Exception, match="尚未登录"):
        await provider.chat([Message(role="user", content="hi")])
    await provider._client.aclose()


async def test_error_mapping() -> None:
    def handler(status: int, detail: str):
        def inner(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(status, json={"error": detail})
        return inner

    cases = [(401, "unauthorized", AuthError), (429, "slow down", RateLimitError),
             (400, "context length exceeded", ContextOverflowError),
             (500, "boom", ProviderError)]
    for status, detail, expected in cases:
        provider = make_provider(handler(status, detail))
        with pytest.raises(expected):
            await provider.chat([Message(role="user", content="hi")])
        await provider._client.aclose()


async def test_oauth_mode_uses_codex_auth(monkeypatch) -> None:
    import agent.providers.codex_auth as codex_auth

    async def fake_token(path=None, *, client=None):
        return "at-oauth", "acct-oauth"

    monkeypatch.setattr(codex_auth, "valid_access_token", fake_token)
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["headers"] = dict(request.headers)
        return httpx2.Response(200, content=sse(
            {"type": "response.completed", "response": {"usage": {}}}), headers=SSE_HEADERS)

    provider = make_provider(handler)
    provider._use_oauth = True
    await provider.chat([Message(role="user", content="hi")])
    assert seen["headers"]["authorization"] == "Bearer at-oauth"
    assert seen["headers"]["chatgpt-account-id"] == "acct-oauth"
    await provider._client.aclose()


def test_load_codex_models_from_cache(tmp_path) -> None:
    """读本机 Codex 缓存：映射 id/显示名/窗口/推理档位，过滤 hide 与非 API 模型。"""
    from agent.providers.openai_responses import load_codex_models

    cache = tmp_path / "models_cache.json"
    cache.write_text(json.dumps({"models": [
        {"slug": "hidden", "display_name": "H", "visibility": "hide"},
        {"slug": "no-api", "display_name": "N", "supported_in_api": False},
        {"slug": "b-model", "display_name": "B", "priority": 2, "context_window": 1000,
         "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}]},
        {"slug": "a-model", "display_name": "A", "priority": 1, "context_window": 272000,
         "supported_reasoning_levels": [{"effort": "medium"}]},
    ]}, ensure_ascii=False), encoding="utf-8")

    models = load_codex_models(str(cache))
    assert [m["id"] for m in models] == ["a-model", "b-model"]     # 按 priority 排序
    assert models[0]["capability"] == {"window": 272000,
                                       "reasoning_mode": "adjustable", "levels": ["medium"]}
    assert models[1]["display_name"] == "B"


def test_load_codex_models_fallback(tmp_path) -> None:
    from agent.providers.openai_responses import load_codex_models

    assert [m["id"] for m in load_codex_models(str(tmp_path / "missing.json"))]  # 回退内置
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert load_codex_models(str(bad))                       # 坏文件同样回退


async def test_list_models_reads_codex_cache(tmp_path) -> None:
    cache = tmp_path / "models_cache.json"
    cache.write_text(json.dumps({"models": [
        {"slug": "gpt-5.6-sol", "display_name": "GPT-5.6-Sol", "priority": 1,
         "context_window": 272000,
         "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}]},
    ]}), encoding="utf-8")
    provider = CodexResponsesProvider(model="gpt-5.6-sol", use_oauth=False, api_key="x",
                                      models_path=str(cache))
    models = await provider.list_models()
    assert models[0]["id"] == "gpt-5.6-sol" and models[0]["capability"]["window"] == 272000
    await provider._client.aclose()
