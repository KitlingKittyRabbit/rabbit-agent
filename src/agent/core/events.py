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
