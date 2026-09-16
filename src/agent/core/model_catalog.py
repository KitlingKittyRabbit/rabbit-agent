"""上下文预算数学与内部思考强度语义（纯函数，不做任何模型名称猜测）。

窗口来源由 orchestrator 决定：用户显式覆盖 > provider 元数据 > 未知（协议不提供能力）。
未知窗口不计算预算（不压缩，仅保留 provider overflow 兜底）。
"""

from __future__ import annotations

EFFORT_LEVELS = ("off", "low", "medium", "high")


def next_output_reserve(window: int) -> int:
    """下一次模型输出的预留（保守 4096，窗口很小则按比例下调）。"""
    return min(4_096, max(1_024, window // 16))


def context_budget(window: int | None, reasoning_reserve: int = 0) -> dict | None:
    """动态压缩预算；窗口未知返回 None（UI 显示 ?，不编造阈值）。"""
    if not window or window <= 0:
        return None
    safety = int(window * 0.05)
    reserve = next_output_reserve(window) + max(0, reasoning_reserve) + safety
    compact_at = max(1_024, min(int(window * 0.85), window - reserve))
    return {
        "window": window,
        "compact_at": compact_at,
        "reserve": reserve,
        "safety": safety,
        "reasoning": max(0, reasoning_reserve),
        "next_output": next_output_reserve(window),
    }
