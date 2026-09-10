"""上下文管理纯函数测试：估算、截断（含工具链对齐）、摘要序列化。"""

from agent.core.context import estimate_chars, serialize_for_summary, truncate_messages
from agent.providers import Message, ToolCall


def make_messages(count: int) -> list[Message]:
    return [Message(role="user", content=f"消息{i}") for i in range(count)]


def test_estimate_counts_content_and_tool_args() -> None:
    messages = [
        Message(role="user", content="12345"),
        Message(
            role="assistant",
            content="ab",
            tool_calls=[ToolCall(id="c", name="tool", arguments={"x": 1})],
        ),
    ]
    assert estimate_chars(messages) == 5 + 2 + 4 + len(str({"x": 1}))


def test_truncate_drops_oldest_half() -> None:
    messages = make_messages(10)
    truncate_messages(messages)
    assert len(messages) == 5
    assert messages[0].content == "消息5"


def test_truncate_preserves_system() -> None:
    messages = [Message(role="system", content="sys"), *make_messages(10)]
    truncate_messages(messages)
    assert messages[0].content == "sys"
    assert len(messages) == 6


def test_truncate_drops_orphan_tool_results() -> None:
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u1"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="f", arguments={})],
        ),
        Message(role="tool", content="r1", tool_call_id="c1"),
        Message(role="user", content="u2"),
        Message(role="user", content="u3"),
    ]
    truncate_messages(messages)
    # 砍半后剩下的第一条不得是 tool（其调用已被砍掉）
    assert messages[0].role == "system"
    assert messages[1].role != "tool"


def test_truncate_tiny_list_is_noop() -> None:
    messages = make_messages(2)
    truncate_messages(messages)
    assert len(messages) == 2


def test_serialize_formats_and_truncates() -> None:
    messages = [
        Message(role="user", content="问题"),
        Message(
            role="assistant",
            content="好",
            tool_calls=[ToolCall(id="c", name="read_file", arguments={"path": "a"})],
        ),
        Message(role="tool", content="x" * 800, tool_call_id="c"),
    ]
    text = serialize_for_summary(messages, max_chars=10_000)
    assert "user: 问题" in text
    assert "调用工具: read_file" in text
    assert "x" * 500 in text and "x" * 501 not in text  # tool 结果截断 500

    long_text = serialize_for_summary(messages * 1000, max_chars=100)
    assert long_text.startswith("……（更早的已省略）")
