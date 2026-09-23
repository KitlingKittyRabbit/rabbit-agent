"""provider reasoning 解析与 Anthropic thinking 协议保真测试（零网络，假流）。

覆盖：
- OpenAI 兼容 reasoning_content 与 text 分离、reasoning_tokens、effort 参数门控
- Anthropic thinking 事件、thinking block + signature 保存与二轮回传
- signature 丢失必须被协议校验拒绝（回归保护）
"""

import asyncio
from types import SimpleNamespace

import httpx2
import pytest

from agent.providers.base import Message, ProviderError, ToolCall
from agent.providers.openai_compat import OpenAICompatProvider


def _openai_provider(**kw) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url="http://127.0.0.1:1/v1", api_key="unused", model="m",
        http_client=httpx2.AsyncClient(trust_env=False), **kw,
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()


def _chunk(content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    choices = [SimpleNamespace(delta=delta, finish_reason=finish)]
    return SimpleNamespace(choices=choices, usage=usage)


def test_openai_consume_separates_reasoning_and_text() -> None:
    provider = _openai_provider()
    chunks = [
        _chunk(reasoning="思考1"),
        _chunk(reasoning="思考2"),
        _chunk(content="答案"),
        _chunk(content="。", finish="stop"),
    ]
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    result = asyncio.run(
        provider._consume(_FakeStream(chunks), text_parts.append, reasoning_parts.append)
    )
    assert result.text == "答案。"
    assert result.reasoning == "思考1思考2"
    assert text_parts == ["答案", "。"]
    assert reasoning_parts == ["思考1", "思考2"]


def test_openai_consume_usage_reasoning_tokens() -> None:
    provider = _openai_provider()
    details = SimpleNamespace(reasoning_tokens=17)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=40,
                            completion_tokens_details=details)
    result = asyncio.run(
        provider._consume(_FakeStream([_chunk(content="x", usage=usage)]), None, None)
    )
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 40
    assert result.usage.reasoning_tokens == 17


def test_openai_no_reasoning_field_is_empty() -> None:
    provider = _openai_provider()
    result = asyncio.run(
        provider._consume(_FakeStream([_chunk(content="普通")]), None, None)
    )
    assert result.reasoning == ""
    assert result.blocks == []


def test_openai_reasoning_effort_kwarg_gated_by_value() -> None:
    captured: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeStream([_chunk(content="x")])

    provider = _openai_provider(reasoning_effort="high")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    asyncio.run(provider.chat([Message(role="user", content="hi")]))
    assert captured["reasoning_effort"] == "high"

    off = _openai_provider(reasoning_effort="off")
    captured.clear()
    off._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    asyncio.run(off.chat([Message(role="user", content="hi")]))
    assert "reasoning_effort" not in captured


# ---------- Anthropic thinking 协议 ----------

def _anthropic_provider(effort="medium"):
    from agent.providers.anthropic_compat import AnthropicCompatProvider

    return AnthropicCompatProvider(
        api_key="unused", model="claude-sonnet-4-5", base_url="http://127.0.0.1:1",
        http_client=httpx2.AsyncClient(trust_env=False), reasoning_effort=effort,
    )


class _AnthropicStream:
    """假 Anthropic 流：产出 thinking/text 事件，并严格校验回传的 thinking signature。"""

    def __init__(self, blocks, calls, kwargs):
        self._blocks = blocks
        self._calls = calls
        self._kwargs = kwargs
        self._events = [
            SimpleNamespace(type="thinking", thinking=b["thinking"])
            for b in blocks if b.get("type") == "thinking"
        ] + [
            SimpleNamespace(type="text", text=b["text"])
            for b in blocks if b.get("type") == "text"
        ]

    def _validate_assistant_thinking(self):
        for message in self._kwargs.get("messages", []):
            if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if block.get("type") == "thinking" and not block.get("signature"):
                    raise ProviderError("thinking block 缺少 signature，协议拒绝")

    async def __aenter__(self):
        self._validate_assistant_thinking()
        self._calls.append(self._kwargs)
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()

    async def get_final_message(self):
        return SimpleNamespace(
            content=[
                SimpleNamespace(type=b["type"], **{
                    k: v for k, v in b.items() if k != "type"
                })
                for b in self._blocks
            ],
            stop_reason=(
                "tool_use" if any(b["type"] == "tool_use" for b in self._blocks)
                else "end_turn"
            ),
            usage=SimpleNamespace(input_tokens=5, output_tokens=6),
        )


def _make_stream_factory(provider, rounds):
    calls: list[dict] = []
    queue = list(rounds)

    def factory(**kwargs):
        blocks = queue.pop(0)
        return _AnthropicStream(blocks, calls, kwargs)

    provider._client = SimpleNamespace(messages=SimpleNamespace(stream=factory))
    return calls


def test_anthropic_thinking_blocks_and_signature_roundtrip() -> None:
    provider = _anthropic_provider()
    calls = _make_stream_factory(provider, [
        [  # 第一轮：thinking + text + tool_use（顺序：thinking→text→tool_use）
            {"type": "thinking", "thinking": "先读", "signature": "sig-1"},
            {"type": "text", "text": "我来看看"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
        ],
        [  # 第二轮：纯文本收尾
            {"type": "thinking", "thinking": "再改", "signature": "sig-2"},
            {"type": "text", "text": "完成"},
        ],
    ])
    reasoning: list[str] = []
    first = asyncio.run(provider.chat(
        [Message(role="user", content="修复 a.py")], on_reasoning=reasoning.append
    ))
    assert first.reasoning == "先读"
    assert first.reasoning_blocks[0]["signature"] == "sig-1"
    assert [b["type"] for b in first.blocks] == ["thinking", "text", "tool_use"]
    assert first.tool_calls == [ToolCall(id="t1", name="read_file", arguments={"path": "a.py"})]

    # 模拟 loop：把原始 blocks 放回 assistant 消息，再带 tool 结果请求第二轮
    messages = [
        Message(role="user", content="修复 a.py"),
        Message(role="assistant", content=first.text, tool_calls=first.tool_calls,
                reasoning=first.reasoning, content_blocks=first.blocks),
        Message(role="tool", content="已读取", tool_call_id="t1"),
    ]
    second = asyncio.run(provider.chat(messages))
    assert second.text == "完成"
    # 第二轮请求里 assistant 的 thinking block 仍带 signature（协议要求）
    second_call = calls[1]
    assistant = next(m for m in second_call["messages"] if m["role"] == "assistant")
    assert assistant["content"][0] == {
        "type": "thinking", "thinking": "先读", "signature": "sig-1",
    }
    assert [b["type"] for b in assistant["content"]] == ["thinking", "text", "tool_use"]


def test_anthropic_missing_signature_is_rejected() -> None:
    """回归保护：实现若丢失 signature，假 provider 会拒绝（测试失败）。"""
    provider = _anthropic_provider()
    _make_stream_factory(provider, [[{"type": "text", "text": "ok"}]])
    stripped = [{"type": "thinking", "thinking": "想", "signature": ""}]
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="", content_blocks=stripped),
    ]
    with pytest.raises(ProviderError, match="signature"):
        asyncio.run(provider.chat(messages))


def test_anthropic_non_thinking_model_has_no_thinking_param() -> None:
    provider = _anthropic_provider(effort="off")
    calls = _make_stream_factory(provider, [[{"type": "text", "text": "ok"}]])
    asyncio.run(provider.chat([Message(role="user", content="hi")]))
    assert "thinking" not in calls[0]


def test_anthropic_thinking_budget_reaches_request() -> None:
    """思考强度必须真实进入请求参数（medium→8192，low→2048）。"""
    medium = _anthropic_provider(effort="medium")
    calls_m = _make_stream_factory(medium, [[{"type": "text", "text": "ok"}]])
    asyncio.run(medium.chat([Message(role="user", content="hi")]))
    assert calls_m[0]["thinking"] == {"type": "enabled", "budget_tokens": 8_192}

    low = _anthropic_provider(effort="low")
    calls_l = _make_stream_factory(low, [[{"type": "text", "text": "ok"}]])
    asyncio.run(low.chat([Message(role="user", content="hi")]))
    assert calls_l[0]["thinking"]["budget_tokens"] == 2_048
