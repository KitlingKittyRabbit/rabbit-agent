"""执行者会话：会话级持续上下文 + 任务执行 + 插话 + 压缩。

与指挥者对称的一等会话：自己的消息历史（store 流 "executor"），跨任务累积、
跨重启保留。任务由派发器按序调用 spawn 执行；执行中用户插话经 event_source
每轮回注，下一个模型回合即生效。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..providers import Message
from ..tools import build_registry
from .context import compact_messages
from .diff_stats import accrue, diff_payload, file_change
from .dispatch import ExecutorUnconfigured, TaskIncomplete
from .events import (
    ACTOR_SUBAGENT,
    REASONING_DELTA,
    STOP_REASON_TEXT,
    SUBAGENT_STEP,
    SUBAGENT_TEXT_DELTA,
    SUBAGENT_TOOL_FINISHED,
    SUBAGENT_TOOL_STARTED,
    TASK_DIFF,
    TASK_RUNNING,
    _classify_status,
    _preview,
)
from .loop import AgentLoop
from .prompts import SUBAGENT_SYSTEM

logger = logging.getLogger(__name__)


class ExecutorSession:
    """执行者的一等会话：持续上下文 + 压缩 + 插话（指挥者任务的镜像）。"""

    def __init__(
        self,
        *,
        session_id: str,
        project_id: str,
        registry,
        record_event: Callable,
        audit_call: Callable,
        confirm: Callable,
    ) -> None:
        self.id = session_id
        self.project_id = project_id
        self._registry = registry
        self._record_event = record_event
        self._audit_call = audit_call
        self._confirm = confirm
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()  # 插话（循环内 drain）
        store = registry.store
        self._messages: list[Message] = store.load(session_id, "executor") if store else []
        self._tools = None
        self._working: list[Message] | None = None   # 进行中任务的工作集（含未持久化尾部）
        self._inflight_anchor: Message | None = None  # 当前任务消息对象（inflight 锚点）

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def _persist_interrupted(
        self, working: list[Message], store, text: str, reasoning: str
    ) -> None:
        """中断保留：把已产生的半截输出写进执行者历史，用户可据此纠偏。

        两种中断点：
        - 工具执行中断：末条 assistant 的 tool_calls 可能只跑了一部分（工具串行，
          缺的必是尾部），为所有缺结果的 call 补占位，保证下一轮请求序列合法；
        - 模型流式中断：本轮半截文本/思考尚未入 messages，补一条半截 assistant。
        同一轮内不会既补占位又追加半截（文本已在原 assistant 消息里）。
        """
        pending: Message | None = None
        pending_idx = -1
        for i in range(len(working) - 1, -1, -1):
            m = working[i]
            if m.role == "assistant" and m.tool_calls:
                pending, pending_idx = m, i
                break
        if pending is not None:
            answered = {
                m.tool_call_id for m in working[pending_idx + 1:]
                if m.role == "tool"
            }
            missing = [c for c in pending.tool_calls if c.id not in answered]
            for call in missing:
                working.append(Message(
                    role="tool",
                    content="（任务被用户中断，该工具未完成）",
                    tool_call_id=call.id,
                ))
            if missing:
                text = reasoning = ""   # 本轮流式文本已在 pending 里，勿重复
        if text or reasoning:
            working.append(Message(
                role="assistant", content=text, reasoning=reasoning or None,
            ))
        self._messages = working[1:]
        if store is not None:
            try:
                store.replace(self.id, self._messages, "executor")
            except Exception as e:
                # 关机时 store 可能已关：中断保留尽力而为，绝不掩盖 CancelledError
                logger.warning("中断保留落盘失败: %s", e)

    def inflight_messages(self) -> list[Message]:
        """进行中任务的未持久化消息（从任务消息对象起；截断/压缩不影响）。

        空闲为空。_working_base 已弃用（保留字段兼容历史读取）。
        """
        if self._working is None:
            return []
        for i, m in enumerate(self._working):
            if m is self._inflight_anchor:
                return self._working[i:]
        return []

    def steer(self, text: str) -> None:
        """执行中插话：下一个模型回合生效（AgentLoop 每轮 drain event_source）。"""
        self._inbox.put_nowait(Message(role="user", content=f"[用户对执行者插话]\n{text}"))

    def take_undelivered(self) -> list[str]:
        """取走未送达的插话（任务结束时仍留在收件箱的），交给派发器转成新任务。"""
        out: list[str] = []
        while True:
            try:
                out.append(self._inbox.get_nowait().content)
            except asyncio.QueueEmpty:
                return out

    def _root(self) -> Path:
        root = self._registry.project_root(self.project_id)
        if root is None:
            raise RuntimeError(f"会话所属项目不存在: {self.project_id}")
        return root

    async def _compact(self, messages: list[Message]) -> None:
        registry = self._registry
        budget = registry.context_budget("executor")
        if budget is None or self._tools is None:
            return
        await compact_messages(
            messages, provider=registry.executor_provider,
            budget_tokens=budget["compact_at"], tool_specs=self._tools.specs(),
            on_compacted=None, usage=None, preserve=self._inflight_anchor,
            session_id=self.id,
        )

    def spawn(
        self, task_id: int, prompt: str, extra_tools: list, *, turn: int | None
    ) -> Awaitable[str]:
        """执行一个任务：历史 + 新任务追加进执行者上下文，完成后持久化。"""
        registry = self._registry
        if registry.executor_provider is None:
            raise ExecutorUnconfigured("executor 未配置，任务未执行（请先连接 executor）")
        plan = registry.plan_mode
        tools = build_registry(
            self._root(), write=not plan, shell=not plan,
            on_call=self._audit_call, confirm=self._confirm,
            # 宿主 GitHub/Git：普通执行者可用（每次显式指定账号）；
            # plan 模式保持纯本地只读探索，不挂任何 GitHub/Git 工具
            host_github="write" if not plan else "off",
            host_git=not plan,
        )
        for tool in extra_tools:
            tools.add(tool)
        self._tools = tools
        store = registry.store
        sid = self.id
        record = self._record_event
        partial_text: list[str] = []       # 当前模型回合已流出的文本（每轮清空）
        partial_reasoning: list[str] = []  # 当前模型回合已流出的思考
        task_diff: dict[str, list[int]] = {}   # path -> [added, removed]（写/编辑工具）

        def record_diff() -> None:
            payload = diff_payload(task_diff)
            if payload is None:
                return
            task_diff.clear()
            record(TASK_DIFF, actor=ACTOR_SUBAGENT, task_id=task_id, turn_id=turn,
                   text=json.dumps(payload, ensure_ascii=False))

        def on_text(text: str) -> None:
            partial_text.append(text)
            record(SUBAGENT_TEXT_DELTA, actor=ACTOR_SUBAGENT, task_id=task_id,
                   turn_id=turn, text=text)

        def on_tool(call, payload: str | None, phase: str) -> None:
            if phase == "finished" and not (payload or "").startswith("错误"):
                # call_safe 把失败也回成 finished（文本以「错误」开头）：失败不计改动
                change = file_change(call.name, call.arguments)
                if change is not None:
                    accrue(task_diff, change)
            if phase == "started":
                if store is not None:
                    # 最近动作属 actions 维度；步数（模型回合）由 on_step 单独维护
                    store.update_task(
                        task_id, sid, TASK_RUNNING,
                        last_action=f"{call.name} {_preview(str(call.arguments), 60)}",
                        actions_used_delta=1,
                    )
                record(SUBAGENT_TOOL_STARTED, actor=ACTOR_SUBAGENT, task_id=task_id,
                       turn_id=turn, name=call.name, arguments=_preview(str(call.arguments)))
            else:
                record(SUBAGENT_TOOL_FINISHED, actor=ACTOR_SUBAGENT, task_id=task_id,
                       turn_id=turn, name=call.name,
                       status=_classify_status("finished", payload),
                       result=_preview(payload or ""))

        def on_reasoning(chunk: str) -> None:
            partial_reasoning.append(chunk)
            record(REASONING_DELTA, actor=ACTOR_SUBAGENT, task_id=task_id,
                   turn_id=turn, text=chunk)

        def on_step(step: int) -> None:
            """步数 = 已完成的模型回合数；落库 + 广播，运行中与结束值同一口径。"""
            partial_text.clear()          # 上一轮的文本已进入 messages，避免重复补写
            partial_reasoning.clear()
            if store is not None:
                store.update_task(task_id, sid, TASK_RUNNING, steps_used=step)
            registry.emit({
                "type": SUBAGENT_STEP, "session": sid, "turn_id": turn,
                "actor": ACTOR_SUBAGENT, "task_id": task_id,
                "steps_used": step, "max_steps": registry.max_steps_executor,
            })

        # 执行者上下文是持续历史：system 不入库，每次启动重建
        # 与主 agent 同策略：压缩优先（compactor），不做硬停，溢出截断是最后兜底
        task_msg = Message(role="user", content=prompt)
        working = [
            Message(role="system", content=SUBAGENT_SYSTEM),
            *self._messages,
            task_msg,
        ]

        async def go() -> str:
            self._working = working                    # 在协程内登记，防未启动即泄漏
            self._inflight_anchor = task_msg           # 对象身份锚定：截断/压缩不影响
            loop = AgentLoop(
                registry.executor_provider,
                tools,
                session_id=self.id,
                max_steps=registry.max_steps_executor,
                compactor=self._compact,
                on_tool_event=on_tool,
                on_step=on_step,
                on_reasoning=on_reasoning,
            )
            try:
                result = await loop.run(working, on_text=on_text, event_source=self._inbox)
            except asyncio.CancelledError:
                self._persist_interrupted(
                    working, store, "".join(partial_text), "".join(partial_reasoning),
                )
                raise
            finally:
                record_diff()          # 成功/失败/中断都汇总已产生的文件改动
                self._working = None
                self._inflight_anchor = None
            # 压缩/截断可能已改写 working：整段替换（去掉 system），持久化同步整段重写
            self._messages = working[1:]
            if store is not None:
                store.replace(sid, self._messages, "executor")
                store.update_task(task_id, sid, TASK_RUNNING,
                                  steps_used=result.steps, stop_reason=result.stop_reason)
            self._working = None
            self._inflight_anchor = None
            if result.stop_reason == "completed":
                return result.text
            raise TaskIncomplete(
                result.stop_reason, STOP_REASON_TEXT.get(result.stop_reason, result.stop_reason)
            )

        return go()
