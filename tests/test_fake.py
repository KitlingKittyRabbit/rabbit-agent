"""FakeProvider 的行为测试。"""

import pytest

from agent.providers import AuthError, ChatResult, FakeProvider, Message, ToolSpec


async def test_returns_scripted_in_order_and_records_calls() -> None:
    provider = FakeProvider([ChatResult(text="一"), ChatResult(text="二", stop_reason="tool_use")])
    first = await provider.chat([Message(role="user", content="a")])
    second = await provider.chat(
        [Message(role="user", content="b")], tools=[ToolSpec("t", "d", {})]
    )

    assert first.text == "一"
    assert second.stop_reason == "tool_use"
    assert provider.calls[0][0][0].content == "a"
    assert provider.calls[0][1] is None
    assert provider.calls[1][1][0].name == "t"


async def test_scripted_exception_is_raised() -> None:
    provider = FakeProvider([AuthError("no")])
    with pytest.raises(AuthError):
        await provider.chat([])


async def test_callable_item_receives_messages_and_tools() -> None:
    seen = {}

    def handler(messages, tools):
        seen["messages"] = len(messages)
        seen["tools"] = tools
        return ChatResult(text="ok")

    provider = FakeProvider([handler])
    result = await provider.chat(
        [Message(role="user", content="x"), Message(role="user", content="y")]
    )

    assert result.text == "ok"
    assert seen["messages"] == 2
    assert seen["tools"] is None


async def test_on_text_receives_full_text() -> None:
    deltas: list[str] = []
    provider = FakeProvider([ChatResult(text="hello")])
    await provider.chat([], on_text=deltas.append)
    assert deltas == ["hello"]


async def test_exhausted_script_raises() -> None:
    provider = FakeProvider([])
    with pytest.raises(AssertionError, match="耗尽"):
        await provider.chat([])
