"""Anthropic 兼容适配的契约测试：MockTransport 注入，零网络、零密钥。"""

import json
from collections.abc import Callable

import httpx2
import pytest

from agent.providers import (
    AnthropicCompatProvider,
    AuthError,
    ContextOverflowError,
    Message,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolSpec,
    Usage,
)

BASE = "https://api.test"
SSE_HEADERS = {"content-type": "text/event-stream"}

Handler = Callable[[httpx2.Request], httpx2.Response]


def make_provider(handler: Handler) -> AnthropicCompatProvider:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return AnthropicCompatProvider(
        base_url=BASE, api_key="sk-ant-test", model="claude-test", http_client=client
    )


def sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def message_start(input_tokens: int) -> tuple[str, dict]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        },
    )


def text_events(stop_reason: str = "end_turn") -> bytes:
    return sse(
        message_start(12),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "你好"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "，世界"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def tool_use_events() -> bytes:
    return sse(
        message_start(9),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.txt"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 8},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def sse_handler(body: bytes) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/messages"
        return httpx2.Response(200, headers=SSE_HEADERS, content=body)

    return handler


def capture(handler: Handler, captured: list[httpx2.Request]) -> Handler:
    def wrapped(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return handler(request)

    return wrapped


def error_handler(status: int, error_type: str, message: str) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, json={"type": "error", "error": {"type": error_type, "message": message}}
        )

    return handler


async def test_text_stream_usage_and_request_shape() -> None:
    tools = [
        ToolSpec(
            name="read_file",
            description="读文件",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    captured: list[httpx2.Request] = []
    provider = make_provider(capture(sse_handler(text_events()), captured))
    deltas: list[str] = []
    result = await provider.chat(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        tools=tools,
        on_text=deltas.append,
    )

    assert result.text == "你好，世界"
    assert deltas == ["你好", "，世界"]
    assert result.stop_reason == "stop"
    assert result.usage == Usage(input_tokens=12, output_tokens=4)
    payload = json.loads(captured[0].content)
    assert payload["model"] == "claude-test"
    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 8192
    assert payload["tools"] == [
        {
            "name": "read_file",
            "description": "读文件",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


async def test_tool_use_parsed() -> None:
    provider = make_provider(sse_handler(tool_use_events()))
    result = await provider.chat([Message(role="user", content="x")])

    assert result.stop_reason == "tool_use"
    assert result.tool_calls == [
        ToolCall(id="toolu_1", name="read_file", arguments={"path": "a.txt"})
    ]


async def test_consecutive_tool_results_merged_into_one_user_message() -> None:
    captured: list[httpx2.Request] = []
    provider = make_provider(capture(sse_handler(text_events()), captured))
    await provider.chat(
        [
            Message(role="user", content="u"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="t1", name="f", arguments={"a": 1}),
                    ToolCall(id="t2", name="g", arguments={}),
                ],
            ),
            Message(role="tool", content="r1", tool_call_id="t1"),
            Message(role="tool", content="r2", tool_call_id="t2"),
        ]
    )

    messages = json.loads(captured[0].content)["messages"]
    assert messages[0] == {"role": "user", "content": "u"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
        {"type": "tool_use", "id": "t2", "name": "g", "input": {}},
    ]
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
        ],
    }


@pytest.mark.parametrize(("status", "error"), [(401, AuthError), (429, RateLimitError)])
async def test_auth_and_rate_limit_mapping(status: int, error: type[ProviderError]) -> None:
    with pytest.raises(error):
        await make_provider(error_handler(status, "authentication_error", "bad key")).chat(
            [Message(role="user", content="x")]
        )


async def test_server_error_maps_to_provider_error_with_status() -> None:
    with pytest.raises(ProviderError) as exc_info:
        await make_provider(error_handler(500, "api_error", "boom")).chat(
            [Message(role="user", content="x")]
        )
    assert exc_info.value.status_code == 500


async def test_context_overflow_detection() -> None:
    handler = error_handler(
        400, "invalid_request_error", "prompt is too long: 200000 tokens > 100000 maximum"
    )
    with pytest.raises(ContextOverflowError):
        await make_provider(handler).chat([Message(role="user", content="x")])


async def test_connection_error_maps_to_provider_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    with pytest.raises(ProviderError):
        await make_provider(handler).chat([Message(role="user", content="x")])


async def test_opencode_anthropic_host_sends_session_header() -> None:
    """Anthropic 协议走 opencode.ai 时同样带 x-opencode-session + 自定义 UA。"""
    from agent.providers.base import is_opencode_host

    assert is_opencode_host("https://opencode.ai/zen/v1") is True
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["ua"] = request.headers.get("user-agent")
        seen["sid"] = request.headers.get("x-opencode-session")
        return httpx2.Response(200, content=text_events(), headers=SSE_HEADERS)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    provider = AnthropicCompatProvider(
        base_url="https://opencode.ai/zen/v1", api_key="sk-zen",
        model="qwen3.8-max", http_client=client,
    )
    await provider.chat([Message(role="user", content="ping")], session_id="sess-9")

    assert seen["sid"] == "sess-9"
    assert seen["ua"] and "rabbit-agent" in seen["ua"]


async def test_non_opencode_anthropic_host_sends_no_session_header() -> None:
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["sid"] = request.headers.get("x-opencode-session")
        return httpx2.Response(200, content=text_events(), headers=SSE_HEADERS)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    provider = AnthropicCompatProvider(
        base_url="https://api.anthropic.com", api_key="sk-a",
        model="claude-test", http_client=client,
    )
    await provider.chat([Message(role="user", content="ping")], session_id="sess-1")
    assert seen["sid"] is None
