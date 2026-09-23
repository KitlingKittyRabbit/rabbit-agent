"""执行事件模型（常量定义）：事件类型、Turn/Task 状态、actor。

LLM messages（模型上下文，会压缩截断）与 UI-visible execution history（稳定持久）
是两套数据，本模块定义后者的事件字典 schema；持久化在 SessionStore 的
turns/task_runs/execution_events 表。
"""

from __future__ import annotations

# ---- Turn 状态 ----
TURN_RUNNING = "running"
TURN_COMPLETED = "completed"
TURN_CANCELLED = "cancelled"
TURN_ERROR = "error"

# ---- Task 状态 ----
TASK_QUEUED = "queued"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_ERROR = "error"
TASK_CANCELLED = "cancelled"
TASK_UNCONFIGURED = "unconfigured"  # executor 未配置，任务未执行
TASK_INCOMPLETE = "incomplete"  # 撞上限/无进展等，未完成（不是成功）

# ---- 事件类型 ----
TURN_STARTED = "turn_started"
ACTIVITY_TEXT_DELTA = "activity_text_delta"  # working 文本（中间模型调用）
TOOL_STARTED = "tool_started"
TOOL_FINISHED = "tool_finished"
SUBAGENT_QUEUED = "subagent_queued"
SUBAGENT_STARTED = "subagent_started"
SUBAGENT_STEP = "subagent_step"
SUBAGENT_TEXT_DELTA = "subagent_text_delta"
SUBAGENT_TOOL_STARTED = "subagent_tool_started"
SUBAGENT_TOOL_FINISHED = "subagent_tool_finished"
SUBAGENT_COMPLETED = "subagent_completed"
TASK_DIFF = "task_diff"  # 任务文件改动汇总（写/编辑工具的行数统计）
SUBAGENT_FAILED = "subagent_failed"
FINAL_STARTED = "final_started"
FINAL_TEXT_DELTA = "final_text_delta"
FINAL_COMPLETED = "final_completed"
TURN_COMPLETED_EVENT = "turn_completed"
TURN_CANCELLED_EVENT = "turn_cancelled"
TURN_FAILED_EVENT = "turn_failed"
REASONING_DELTA = "reasoning_delta"  # provider 明确返回的思考增量

ACTOR_MAIN = "main"
ACTOR_SUBAGENT = "subagent"
ACTOR_SYSTEM = "system"

USER_TO_EXECUTOR = "user_to_executor"  # 用户直接对执行者说话（主时间线灰色提示，不入指挥者上下文）

STOP_REASON_TEXT = {
    "max_steps": "已达最大步数上限",
    "no_progress": "连续重复相同操作，无进展",
    "context_budget": "上下文预算不足",
    "cancelled": "已被用户中断",
}


def _preview(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _classify_status(phase: str, payload: str | None) -> str:
    """从阶段与结果文本推断状态（审计与 execution events 共用）。"""
    if phase == "started":
        return "started"
    if phase == "error":
        return "error"
    text = payload or ""
    if text.startswith("已被用户拒绝"):
        return "denied"
    if text.startswith("超时"):
        return "timeout"
    if text.startswith("错误"):
        return "error"
    return "success"
