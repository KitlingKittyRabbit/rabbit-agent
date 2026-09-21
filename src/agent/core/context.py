"""上下文估算、预算与截断（纯函数，零依赖）。

预算数学在 model_catalog.context_budget（窗口未知 = 不压缩，仅保留溢出兜底）。
精确 usage 优先；估算一律标「约」。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..providers import Message, ToolSpec, Usage
from .model_catalog import context_budget, next_output_reserve  # noqa: F401  (re-export)

COMPACT_PROMPT = """把以下对话历史压缩为一份要点摘要，供后续对话延续使用。
保留：用户目标、已完成的动作及其结果、关键文件路径、未决事项。
逐条列出，不超过 500 字。直接输出摘要，不要解释。"""

SUMMARY_PREFIX = "[前情摘要]"


def estimate_chars(messages: list[Message]) -> int:
    """粗估消息字符数：内容 + 工具调用参数 + 工具结果（tool 消息即结果）。"""
    total = 0
    for m in messages:
        total += len(m.content)
        for tc in m.tool_calls or []:
            total += len(tc.name) + len(str(tc.arguments))
    return total


def tool_schema_chars(tool_specs: Sequence[ToolSpec] | None) -> int:
    """工具 schema 的字符量（名称+描述+参数 JSON）。"""
    total = 0
    for spec in tool_specs or []:
        total += len(spec.name) + len(spec.description) + len(str(spec.parameters))
    return total


def estimate_request_chars(
    messages: list[Message],
    system: str = "",
    tool_specs: Sequence[ToolSpec] | None = None,
) -> int:
    """完整请求的字符估算：system + 历史/工具结果 + 工具 schema。"""
    return len(system) + estimate_chars(messages) + tool_schema_chars(tool_specs)


def approximate_tokens(chars: int) -> int:
    """字符数粗估 token（约 4 字符/token）；UI 展示必须标「约」。"""
    return chars // 4


def estimate_tokens(messages: list[Message], system: str = "", tool_specs=None) -> int:
    """完整请求的 token 估算（混合中英文按 ~3.5 字符/token），仅供预算与「约」显示。"""
    return int(estimate_request_chars(messages, system, tool_specs) / 3.5)


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


async def compact_messages(
    messages: list[Message],
    *,
    provider,
    budget_tokens: int,
    tool_specs: Sequence[ToolSpec] | None,
    on_compacted: Callable[[dict], None] | None = None,
    usage: Usage | None = None,
    preserve: Message | None = None,
) -> bool:
    """接近动态阈值时把旧历史压成 [前情摘要] 替换（只在 user 边界切割）。

    指挥者与执行者会话共用。`preserve` 是必须留在 recent 的锚点（当前任务/本回合
    起点，按对象身份匹配）；未给则保护最后一条 user 消息。返回是否发生了压缩。
    """
    if provider is None:
        return False
    if estimate_tokens(messages, tool_specs=tool_specs) < budget_tokens:
        return False
    keep_head = 1 if messages and messages[0].role == "system" else 0
    body = messages[keep_head:]
    budget_chars = max(2_000, int((budget_tokens // 4) * 3.5))
    cut = len(body)
    size = 0
    while cut > 1 and size < budget_chars:
        m = body[cut - 1]
        size += len(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls or [])
        if size <= budget_chars:
            cut -= 1
    # 切割下界：锚点（当前任务/本回合起点）及其之后的工具链必须留在 recent
    anchor = None
    if preserve is not None:
        anchor = next((i for i, m in enumerate(body) if m is preserve), None)
    if anchor is None:
        anchor = next(
            (i for i in range(len(body) - 1, -1, -1) if body[i].role == "user"), None
        )
    if anchor is None:
        return False  # 没有干净的 user 边界：本轮不压缩
    cut = min(cut, anchor)
    # recent 不得以 tool 结果开头：其 assistant 调用会被压进 old，形成孤链
    while cut < len(body) and body[cut].role == "tool":
        cut += 1
    old, recent = body[:cut], body[cut:]
    if not old or not recent:
        return False
    before = estimate_chars(messages)
    serialized = serialize_for_summary(old, before)  # old ⊆ messages，不再二次截断
    result = await provider.chat(
        [Message(role="user", content=f"{COMPACT_PROMPT}\n\n{serialized}")]
    )
    if usage is not None:
        usage.input_tokens += result.usage.input_tokens
        usage.output_tokens += result.usage.output_tokens
    messages[keep_head:] = [
        Message(role="user", content=f"{SUMMARY_PREFIX}\n{result.text}"),
        *recent,
    ]
    after = estimate_chars(messages)
    if on_compacted is not None:
        on_compacted({
            "type": "compacted",
            "before": before,
            "after": after,
            "before_tokens": approximate_tokens(before),
            "after_tokens": approximate_tokens(after),
        })
    return True
