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


async def test_interleaved_reasoning_echoed_for_assistant_messages() -> None:
    """目录声明 interleaved 的模型：assistant 消息按字段回传思考，否则网关 400。"""
    captured: list[httpx2.Request] = []
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            capture(sse_handler(sse(chunk({"content": "ok"}, finish="stop"), "[DONE]")), captured)
        )
    )
    provider = OpenAICompatProvider(
        base_url=BASE, api_key="sk-test", model="m", http_client=client,
        echo_reasoning_field="reasoning_content",
    )
    await provider.chat(
        [
            Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="call_1", name="f", arguments={})],
                reasoning="思考一",
            ),
            Message(role="tool", content="结果", tool_call_id="call_1"),
            Message(role="assistant", content="答复", reasoning="思考二"),
        ]
    )

    messages = json.loads(captured[0].content)["messages"]
    assert messages[0]["reasoning_content"] == "思考一"
    assert messages[2]["reasoning_content"] == "思考二"


async def test_interleaved_reasoning_omitted_without_declaration() -> None:
    captured: list[httpx2.Request] = []
    provider = make_provider(
        capture(sse_handler(sse(chunk({"content": "ok"}, finish="stop"), "[DONE]")), captured)
    )
    await provider.chat([Message(role="assistant", content="答复", reasoning="思考")])

    messages = json.loads(captured[0].content)["messages"]
    assert "reasoning_content" not in messages[0]


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


def test_is_opencode_host_matches_only_opencode() -> None:
    from agent.providers.base import is_opencode_host

    assert is_opencode_host("https://opencode.ai/zen/go/v1") is True
    assert is_opencode_host("https://api.opencode.ai/v1") is True
    assert is_opencode_host("https://api.deepseek.com/v1") is False
    assert is_opencode_host("https://fakeopencode.ai/v1") is False
    assert is_opencode_host(None) is False
    assert is_opencode_host("not a url") is False


async def test_opencode_host_sends_session_and_ua() -> None:
    """OpenCode Go 网关要求：自定义 UA + 每会话稳定的 x-opencode-session。"""
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["ua"] = request.headers.get("user-agent")
        seen["sid"] = request.headers.get("x-opencode-session")
        return httpx2.Response(
            200,
            content=sse(chunk({"role": "assistant", "content": "你好"}), chunk({}, "stop")),
            headers=SSE_HEADERS,
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    provider = OpenAICompatProvider(
        base_url="https://opencode.ai/zen/go/v1", api_key="sk-go",
        model="deepseek-v4.1-flash", http_client=client,
    )
    result = await provider.chat([Message(role="user", content="ping")], session_id="sess-42")

    assert result.text == "你好"
    assert seen["sid"] == "sess-42"
    assert seen["ua"] and "rabbit-agent" in seen["ua"]


async def test_non_opencode_host_sends_no_session_header() -> None:
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["sid"] = request.headers.get("x-opencode-session")
        return httpx2.Response(
            200,
            content=sse(chunk({"role": "assistant", "content": "ok"}), chunk({}, "stop")),
            headers=SSE_HEADERS,
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1", api_key="sk-d",
        model="deepseek-flash", http_client=client,
    )
    await provider.chat([Message(role="user", content="ping")], session_id="sess-1")

    assert seen["sid"] is None


async def test_opencode_session_fallback_stable_per_instance() -> None:
    """不传 session_id（如连接 ping/压缩兜底）时用实例级 fallback：实例内稳定、实例间不同。"""
    def make(handler_calls: list):
        def handler(request: httpx2.Request) -> httpx2.Response:
            handler_calls.append(request.headers.get("x-opencode-session"))
            return httpx2.Response(
                200,
                content=sse(chunk({"role": "assistant", "content": "ok"}), chunk({}, "stop")),
                headers=SSE_HEADERS,
            )
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        return OpenAICompatProvider(
            base_url="https://opencode.ai/zen/go/v1", api_key="sk-go",
            model="m", http_client=client,
        )

    calls_a: list = []
    provider_a = make(calls_a)
    await provider_a.chat([Message(role="user", content="1")])
    await provider_a.chat([Message(role="user", content="2")])
    assert calls_a[0] and calls_a[0] == calls_a[1]        # 实例内稳定

    calls_b: list = []
    await make(calls_b).chat([Message(role="user", content="1")])
    assert calls_b[0] != calls_a[0]                         # 实例间不同
