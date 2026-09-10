"""上下文管理：字符估算、截断、摘要序列化（纯函数，零依赖）。

B（默认）：摘要压缩——接近阈值时把旧历史压成 [前情摘要] 替换；
A（兜底）：撞 ContextOverflowError 后 truncate_messages 砍半重试。
"""

from __future__ import annotations

from ..providers import Message

COMPACT_PROMPT = """把以下对话历史压缩为一份要点摘要，供后续对话延续使用。
保留：用户目标、已完成的动作及其结果、关键文件路径、未决事项。
逐条列出，不超过 500 字。直接输出摘要，不要解释。"""

SUMMARY_PREFIX = "[前情摘要]"


def estimate_chars(messages: list[Message]) -> int:
    """粗估上下文占用（字符数）：内容 + 工具调用参数。"""
    total = 0
    for m in messages:
        total += len(m.content)
        for tc in m.tool_calls or []:
            total += len(tc.name) + len(str(tc.arguments))
    return total


def truncate_messages(messages: list[Message]) -> None:
    """兜底截断：砍最旧一半（原位修改）。保留 system；清理切割后孤立的 tool 结果。"""
    if len(messages) <= 2:
        return
    keep_head = 1 if messages[0].role == "system" else 0
    body = messages[keep_head:]
    remaining = body[len(body) // 2 :]
    # 切割点处的 tool 结果，其调用已被砍掉，必须一并丢弃
    while remaining and remaining[0].role == "tool":
        remaining.pop(0)
    messages[keep_head:] = remaining


def serialize_for_summary(messages: list[Message], max_chars: int) -> str:
    """把旧历史序列化为压缩提示词的输入；超长时丢弃最早的部分。"""
    lines: list[str] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = "、".join(f"{tc.name}({str(tc.arguments)[:100]})" for tc in m.tool_calls)
            lines.append(f"assistant: {m.content} [调用工具: {calls}]")
        elif m.role == "tool":
            lines.append(f"tool: {m.content[:500]}")
        else:
            lines.append(f"{m.role}: {m.content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "……（更早的已省略）\n" + text[-max_chars:]
    return text
