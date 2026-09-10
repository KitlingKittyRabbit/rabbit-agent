"""OpenAI 兼容适配的契约测试：MockTransport 注入，零网络、零密钥。"""

import json
from collections.abc import Callable

import httpx2
import pytest

from agent.providers import (
    AuthError,
    ContextOverflowError,
    Message,
    OpenAICompatProvider,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolSpec,
    Usage,
)

BASE = "https://api.test/v1"
SSE_HEADERS = {"content-type": "text/event-stream"}

Handler = Callable[[httpx2.Request], httpx2.Response]


def make_provider(handler: Handler) -> OpenAICompatProvider:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAICompatProvider(
        base_url=BASE, api_key="sk-test", model="test-model", http_client=client
    )


def sse(*data_lines: str) -> bytes:
    return "".join(f"data: {line}\n\n" for line in data_lines).encode()


def chunk(delta: dict, finish: str | None = None) -> str:
    return json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def usage_chunk() -> str:
    return json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    )


def sse_handler(body: bytes) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx2.Response(200, headers=SSE_HEADERS, content=body)

    return handler


def capture(handler: Handler, captured: list[httpx2.Request]) -> Handler:
    def wrapped(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return handler(request)

    return wrapped


async def test_text_stream_usage_and_request_shape() -> None:
    body = sse(
        chunk({"role": "assistant", "content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="stop"),
        usage_chunk(),
        "[DONE]",
    )
    captured: list[httpx2.Request] = []
    provider = make_provider(capture(sse_handler(body), captured))
    deltas: list[str] = []
    result = await provider.chat([Message(role="user", content="hi")], on_text=deltas.append)

    assert result.text == "Hello"
    assert deltas == ["Hel", "lo"]
    assert result.stop_reason == "stop"
    assert result.usage == Usage(input_tokens=10, output_tokens=2)
    payload = json.loads(captured[0].content)
    assert payload["model"] == "test-model"
    assert payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in payload


async def test_tool_calls_accumulated_and_tools_serialized() -> None:
    body = sse(
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": ""},
                    }
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path":'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a.txt"}'}}]}),
        chunk({}, finish="tool_calls"),
        "[DONE]",
    )
    tools = [
        ToolSpec(
            name="read_file",
            description="读文件",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    captured: list[httpx2.Request] = []
    provider = make_provider(capture(sse_handler(body), captured))
    result = await provider.chat([Message(role="user", content="x")], tools=tools)

    assert result.text == ""
    assert result.stop_reason == "tool_use"
    assert result.tool_calls == [
        ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})
    ]
    payload = json.loads(captured[0].content)
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读文件",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


async def test_message_conversion() -> None:
    captured: list[httpx2.Request] = []
    provider = make_provider(
        capture(sse_handler(sse(chunk({"content": "ok"}, finish="stop"), "[DONE]")), captured)
    )
    await provider.chat(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="u"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="call_1", name="f", arguments={"a": 1})],
            ),
            Message(role="tool", content="结果", tool_call_id="call_1"),
        ]
    )

    messages = json.loads(captured[0].content)["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "u"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] is None
    tool_call = messages[2]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "f"
    assert json.loads(tool_call["function"]["arguments"]) == {"a": 1}
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "结果"}


@pytest.mark.parametrize(("status", "error"), [(401, AuthError), (429, RateLimitError)])
async def test_auth_and_rate_limit_mapping(status: int, error: type[ProviderError]) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "boom"}})

    with pytest.raises(error):
        await make_provider(handler).chat([Message(role="user", content="x")])


async def test_server_error_maps_to_provider_error_with_status() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(ProviderError) as exc_info:
        await make_provider(handler).chat([Message(role="user", content="x")])
    assert exc_info.value.status_code == 500


async def test_context_overflow_detection() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400, json={"error": {"message": "This model's maximum context length is 8192"}}
        )

    with pytest.raises(ContextOverflowError):
        await make_provider(handler).chat([Message(role="user", content="x")])


async def test_bad_request_without_context_word_is_plain_provider_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": {"message": "invalid model"}})

    with pytest.raises(ProviderError) as exc_info:
        await make_provider(handler).chat([Message(role="user", content="x")])
    assert not isinstance(exc_info.value, ContextOverflowError)


async def test_connection_error_maps_to_provider_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    with pytest.raises(ProviderError):
        await make_provider(handler).chat([Message(role="user", content="x")])


async def test_malformed_tool_arguments_raise_provider_error() -> None:
    body = sse(
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{bad"},
                    }
                ]
            }
        ),
        chunk({}, finish="tool_calls"),
        "[DONE]",
    )
    with pytest.raises(ProviderError, match="JSON"):
        await make_provider(sse_handler(body)).chat([Message(role="user", content="x")])
