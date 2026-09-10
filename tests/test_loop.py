"""agent loop 行为测试：FakeProvider 脚本化驱动。"""

import asyncio

import pytest

from agent.core.loop import AgentLoop
from agent.providers import (
    ChatResult,
    ContextOverflowError,
    FakeProvider,
    Message,
    ToolCall,
    ToolSpec,
)
from agent.tools.base import Tool, ToolError, ToolRegistry


def make_registry() -> ToolRegistry:
    async def add(args: dict) -> str:
        return str(args["a"] + args["b"])

    async def boom(args: dict) -> str:
        raise ToolError("工具炸了")

    return ToolRegistry(
        [
            Tool(ToolSpec("add", "加法", {"type": "object", "properties": {}}), add),
            Tool(ToolSpec("boom", "总是失败", {"type": "object", "properties": {}}), boom),
        ]
    )


async def test_text_reply_completes_immediately() -> None:
    provider = FakeProvider([ChatResult(text="你好")])
    messages = [Message(role="user", content="hi")]
    result = await AgentLoop(provider, make_registry()).run(messages)

    assert result.text == "你好"
    assert result.stop_reason == "completed"
    assert result.steps == 0
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "你好"


async def test_tool_cycle_feeds_output_back() -> None:
    provider = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
                stop_reason="tool_use",
            ),
            ChatResult(text="答案是 3"),
        ]
    )
    messages = [Message(role="user", content="1+2?")]
    result = await AgentLoop(provider, make_registry()).run(messages)

    assert result.text == "答案是 3"
    assert result.steps == 1
    tool_message = messages[2]
    assert tool_message.role == "tool"
    assert tool_message.content == "3"
    assert tool_message.tool_call_id == "c1"
    # 第二次调用时 provider 应看到工具结果
    second_call = provider.calls[1][0]
    assert second_call[-1].content == "3"
    # 工具定义应随调用下发
    assert [t.name for t in provider.calls[0][1]] == ["add", "boom"]


async def test_tool_error_becomes_feedback_text() -> None:
    provider = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="boom", arguments={})], stop_reason="tool_use"
            ),
            ChatResult(text="工具失败，换个办法"),
        ]
    )
    messages = [Message(role="user", content="x")]
    result = await AgentLoop(provider, make_registry()).run(messages)

    assert result.text == "工具失败，换个办法"
    assert "工具炸了" in messages[2].content


async def test_max_steps_force_stops_infinite_tool_calls() -> None:
    script = [
        ChatResult(
            tool_calls=[ToolCall(id=f"c{i}", name="add", arguments={"a": i, "b": 0})],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    provider = FakeProvider(script)
    result = await AgentLoop(provider, make_registry(), max_steps=2).run(
        [Message(role="user", content="x")]
    )

    assert result.stop_reason == "max_steps"
    assert result.steps == 2


async def test_event_source_injected_between_steps() -> None:
    provider = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 0, "b": 0})],
                stop_reason="tool_use",
            ),
            ChatResult(text="好"),
        ]
    )
    events: asyncio.Queue[Message] = asyncio.Queue()
    events.put_nowait(Message(role="user", content="[任务 #9 完成]\n输出"))
    messages = [Message(role="user", content="x")]
    await AgentLoop(provider, make_registry()).run(messages, event_source=events)

    # 事件在第二轮调用前注入，provider 第一次调用就看到（首轮开始前 drain）
    first_call = provider.calls[0][0]
    assert first_call[-1].content == "[任务 #9 完成]\n输出"
    assert events.empty()


async def test_no_tools_passes_none_to_provider() -> None:
    provider = FakeProvider([ChatResult(text="好")])
    await AgentLoop(provider, ToolRegistry()).run([Message(role="user", content="x")])
    assert provider.calls[0][1] is None


async def test_context_overflow_truncates_and_retries() -> None:
    provider = FakeProvider([ContextOverflowError("超长"), ChatResult(text="恢复")])
    messages = [Message(role="system", content="sys")] + [
        Message(role="user", content=f"消息{i}") for i in range(10)
    ]
    result = await AgentLoop(provider, ToolRegistry()).run(messages)

    assert result.text == "恢复"
    # 第二次调用时历史已被砍半，system 保留
    second_call = provider.calls[1][0]
    assert len(second_call) < 11
    assert second_call[0].content == "sys"


async def test_context_overflow_retries_bounded() -> None:
    provider = FakeProvider([ContextOverflowError("x") for _ in range(3)])
    messages = [Message(role="user", content=f"消息{i}") for i in range(10)]
    with pytest.raises(ContextOverflowError):
        await AgentLoop(provider, ToolRegistry()).run(messages)
    assert len(provider.calls) == 3  # 首次 + 重试 2 次后放弃


async def test_overflow_truncation_never_leaves_orphan_tool() -> None:
    provider = FakeProvider([ContextOverflowError("x"), ChatResult(text="ok")])
    messages = [
        Message(role="user", content="u1"),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="f", arguments={})]),
        Message(role="tool", content="r", tool_call_id="c1"),
        Message(role="user", content="u2"),
    ]
    await AgentLoop(provider, ToolRegistry()).run(messages)
    second_call = provider.calls[1][0]
    assert second_call[0].role != "tool"


async def test_compactor_hook_called_before_chat() -> None:
    calls = []

    async def compactor(messages: list[Message]) -> None:
        calls.append(len(messages))
        messages.append(Message(role="user", content="注入"))

    provider = FakeProvider([ChatResult(text="好")])
    await AgentLoop(provider, ToolRegistry(), compactor=compactor).run(
        [Message(role="user", content="x")]
    )
    assert calls == [1]
    assert provider.calls[0][0][-1].content == "注入"
