"""会话运行时：一份消息历史、一个收件队列、一个驱动循环、一张任务表。

多会话的基本单元；共享资源（provider/配置/审计/事件总线）经 registry 读取。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from uuid import uuid4

from ..providers import Message, Usage
from ..tools import build_registry
from .context import (
    compact_messages,
    estimate_chars,
    estimate_tokens,
)
from .dispatch import Dispatcher, SubtaskEvent
from .events import (
    ACTIVITY_TEXT_DELTA,
    ACTOR_MAIN,
    ACTOR_SUBAGENT,
    FINAL_COMPLETED,
    FINAL_STARTED,
    FINAL_TEXT_DELTA,
    REASONING_DELTA,
    SUBAGENT_COMPLETED,
    SUBAGENT_FAILED,
    SUBAGENT_QUEUED,
    SUBAGENT_STARTED,
    TASK_CANCELLED,
    TASK_DONE,
    TASK_ERROR,
    TASK_INCOMPLETE,
    TASK_QUEUED,
    TASK_RUNNING,
    TASK_UNCONFIGURED,
    TOOL_FINISHED,
    TOOL_STARTED,
    TURN_CANCELLED,
    TURN_CANCELLED_EVENT,
    TURN_COMPLETED,
    TURN_COMPLETED_EVENT,
    TURN_ERROR,
    TURN_FAILED_EVENT,
    TURN_RUNNING,
    TURN_STARTED,
    USER_TO_EXECUTOR,
    _classify_status,
    _preview,
)
from .executor_session import ExecutorSession
from .loop import AgentLoop
from .prompts import build_main_system

_CONFIRM_TIMEOUT = 120
_UNSET: object = object()  # _record_event 的 turn_id 未指定哨兵


class Conversation:
    def __init__(self, *, session_id: str, title: str, project_id: str, registry) -> None:
        self.id = session_id
        self.title = title
        self.project_id = project_id
        self._registry = registry
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        start_id = registry.store.max_task_id(session_id) + 1 if registry.store else 1
        self._dispatcher = Dispatcher(
            spawn=self._spawn_subagent,
            on_event=self._on_subtask_event,
            on_question=self._on_task_question,
            on_created=self._on_task_created,
            writer_locks=registry.writer_locks,
            project_key=project_id,
            start_id=start_id,
        )
        self._main_tools = build_registry(
            self._root(), write=False, shell=False, on_call=self._audit_call,
            # 只读账号查询：主 agent 看得到已登录账号，但拿不到任何可写 GitHub/Git 能力
            host_github="read",
        )
        self._main_tools.add(self._dispatcher.make_tool())
        self._main_tools.add(self._dispatcher.make_answer_tool())
        self._messages: list[Message] = registry.store.load(session_id) if registry.store else []
        self._system = build_main_system(self._load_discipline())
        self._driver_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._executor = ExecutorSession(
            session_id=session_id, project_id=project_id, registry=registry,
            record_event=self._record_event, audit_call=self._audit_call,
            confirm=self._confirm,
        )
        self._compact_usage = Usage()
        self._current_turn_id: int | None = None
        self._turn_anchor: Message | None = None  # 本回合首条 user 消息（压缩保护锚点）
        self._direct_tasks: set[int] = set()      # 用户直连执行者的任务（不回报指挥者）
        self._direct_dispatch = False             # 正在派发直连任务（供 _on_task_created 判定来源）
        self._interrupted_task: int | None = None  # 当前故事里被用户中断的指挥者任务号
        # 中断介入的派生直连任务 → 它对应的被中断任务号（报告按此取号，避免连续中断串号）
        self._intervention_origin: dict[int, int] = {}
        # 每个中断故事里用户对执行者说的话（按被中断任务号归档，交付后弹出）
        self._intervention_prompts: dict[int, list[str]] = {}
        stored_flag = (
            registry.store.get_flag(session_id, "report_executor")
            if registry.store is not None
            else None
        )
        self._report_executor = stored_flag != "0"   # 默认回报；可在执行者栏开关
        self._task_turns: dict[int, int | None] = {}  # 任务派发时固定的 parent turn
        self._last_prompt_tokens: int | None = None  # 上一次请求的精确 input tokens
        self._last_prompt_epoch: int = -1
        self._epoch: int = 0  # 消息集变更计数（用于判断精确值是否仍适用）

    def _root(self) -> Path:
        """工具沙箱根 = 所属项目路径。"""
        root = self._registry.project_root(self.project_id)
        if root is None:
            raise RuntimeError(f"会话所属项目不存在: {self.project_id}")
        return root

    def context_chars(self) -> int:
        """完整请求上下文的字符估算：system prompt + 历史消息（兼容旧接口）。"""
        return len(self._system) + estimate_chars(self._messages)

    def request_tokens(self) -> int:
        """下一次请求的 token 估算：messages（含 system） + 主工具 schema。"""
        return estimate_tokens(
            self._messages, system=self._system, tool_specs=self._main_tools.specs()
        )

    def context_payload(self) -> dict:
        """上下文环数据（单一来源，WS 事件与 /api/timeline 共用）。

        used_tokens/exact：消息集自上次成功请求后未变化时为精确 prompt_tokens，
        否则为估算（UI 必须加「约」）。
        window 未知时 percent=None（UI 显示 ?，不编造百分比）。
        """
        registry = self._registry
        window, source = registry.context_window("main")
        budget = registry.context_budget("main")
        estimated = self.request_tokens()
        exact = self._last_prompt_tokens is not None and self._last_prompt_epoch == self._epoch
        used = self._last_prompt_tokens if exact else estimated
        return {
            "used_tokens": used,
            "exact": bool(exact),
            "last_prompt_tokens": self._last_prompt_tokens,
            "estimated_tokens": estimated,
            "window": window,
            "window_source": source,
            "percent": round(used / window * 100) if window else None,
            "compact_at": budget["compact_at"] if budget else None,
            "messages": len(self._messages),
        }

    def _emit_context(self) -> None:
        self._emit({"type": "context", **self.context_payload()})

    def _sub_turn(self, task_id: int) -> int | None:
        """subagent 事件归属：任务派发时固定的 parent turn（不再受新 turn 影响）。"""
        return self._task_turns.get(task_id, self._current_turn_id)

    def _load_discipline(self) -> str | None:
        try:
            path = self._root() / "AGENTS.md"
        except RuntimeError:
            return None
        return path.read_text(encoding="utf-8") if path.is_file() else None

    # ---------- 生命周期 ----------

    def start(self) -> None:
        self._driver_task = asyncio.create_task(self._driver())

    async def shutdown(self) -> None:
        self._dispatcher.cancel_all("agent 服务关闭，任务被中断")
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        if self._driver_task is not None:
            self._driver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._driver_task

    # ---------- 外部接口 ----------

    def enqueue_user(self, text: str) -> None:
        self._inbox.put_nowait(Message(role="user", content=text))

    def stop(self) -> None:
        """/stop：中断当前 turn 与全部运行中的 subagent。"""
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        self._dispatcher.cancel_all("用户手动中断（点击了指挥者栏的停止按钮）")

    def executor_stop(self) -> None:
        """/executor_stop：只中断执行者的运行/排队任务，不动指挥者回合。"""
        self._dispatcher.cancel_all("用户手动中断（点击了执行者栏的停止按钮）")

    def executor_report(self) -> bool:
        """执行者结果（含中断介入）是否回报指挥者。"""
        return self._report_executor

    def set_executor_report(self, on: bool) -> None:
        self._report_executor = bool(on)
        if not self._report_executor:
            # 关闭回报=不让指挥者卷进来：清掉未交付的中断故事，避免以后补发旧话
            self._interrupted_task = None
            self._intervention_origin.clear()
            self._intervention_prompts.clear()
        if self._registry.store is not None:
            self._registry.store.set_flag(self.id, "report_executor", "1" if on else "0")

    def executor_inflight(self) -> list[Message]:
        """执行者进行中任务的未持久化消息（供 /api/executor_messages 拼接展示）。"""
        return self._executor.inflight_messages()

    def _dispatch_direct(self, text: str) -> int:
        """派发"用户直连执行者"的任务：默认只留在窗口；中断介入的派生任务会回报指挥者。"""
        self._direct_dispatch = True
        try:
            return self._dispatcher.dispatch(text)
        finally:
            self._direct_dispatch = False

    def _anchor_turn(self) -> int | None:
        """灰色提示的归属 turn：当前 turn 优先，否则最近一个（刷新后仍在主时间线）。"""
        if self._current_turn_id is not None:
            return self._current_turn_id
        store = self._registry.store
        if store is not None:
            turns = store.list_turns(self.id)
            return turns[-1]["id"] if turns else None
        return None

    def executor_message(self, text: str) -> None:
        """用户直接对执行者说话：执行中=插话（下一生效）；空闲=新任务。

        主时间线只留一条灰色提示（不入指挥者上下文）。
        """
        self._record_event(USER_TO_EXECUTOR, actor="user", text=text,
                           turn_id=self._anchor_turn())
        if self._interrupted_task is not None and self._report_executor:
            # 中断介入：按故事归档用户对执行者说的话（含执行中插话）
            self._intervention_prompts.setdefault(self._interrupted_task, []).append(text)
        busy = any(s in ("running", "queued") for s in self._dispatcher.tasks.values())
        if busy:
            self._executor.steer(text)
            return
        for leftover in self._executor.take_undelivered():
            self._dispatch_direct(leftover)  # 迟到的插话：转成新任务
        self._dispatch_direct(f"[用户直接对执行者说]\n{text}")

    def resolve_confirm(self, confirm_id: str, allow: bool) -> None:
        fut = self._pending_confirms.pop(confirm_id, None)
        if fut is not None and not fut.done():
            fut.set_result(allow)

    # ---------- 内部 ----------

    def _emit(self, event: dict) -> None:
        self._registry.emit({**event, "session": self.id})

    def _record_event(
        self,
        type: str,
        *,
        actor: str,
        turn_id: int | None | object = _UNSET,
        name: str | None = None,
        arguments: str | None = None,
        result: str | None = None,
        status: str | None = None,
        text: str | None = None,
        task_id: int | None = None,
    ) -> None:
        """执行事件：持久化到 SessionStore 并广播到事件总线（UI 数据源）。

        turn_id 未指定时归属当前 turn；subagent 事件必须显式传固定 parent turn。
        """
        effective_turn = self._current_turn_id if turn_id is _UNSET else turn_id
        store = self._registry.store
        if store is not None:
            store.add_event(
                self.id, type, time.time(),
                turn_id=effective_turn, task_id=task_id, actor=actor,
                name=name, arguments=arguments, result=result, status=status, text=text,
            )
        event: dict = {"type": type, "session": self.id, "turn_id": effective_turn, "actor": actor}
        if task_id is not None:
            event["task_id"] = task_id
        for key, value in (
            ("name", name), ("arguments", arguments),
            ("result", result), ("status", status), ("text", text),
        ):
            if value is not None:
                event[key] = value
        self._registry.emit(event)

    def _audit_call(self, tool: str, args: dict, phase: str, payload: str | None) -> None:
        self._registry.audit.log(
            session=self.id, tool=tool, phase=phase,
            status=_classify_status(phase, payload), args=args, result=payload or "",
            turn=self._current_turn_id,
        )

    async def _confirm(self, command: str) -> bool:
        confirm_id = uuid4().hex[:8]
        fut = asyncio.get_running_loop().create_future()
        self._pending_confirms[confirm_id] = fut
        self._emit({"type": "confirm_request", "id": confirm_id, "command": command})
        try:
            return await asyncio.wait_for(fut, timeout=_CONFIRM_TIMEOUT)
        except TimeoutError:
            return False
        finally:
            self._pending_confirms.pop(confirm_id, None)

    async def _driver(self) -> None:
        while True:
            first = await self._inbox.get()
            batch = [first]
            while True:
                try:
                    batch.append(self._inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._turn_task = asyncio.create_task(self._run_turn(batch))
            try:
                await self._turn_task
            except asyncio.CancelledError:
                self._emit({"type": "stopped"})
                # 取消来自 /stop（取消的是 turn）则继续循环；
                # 来自 shutdown（取消的是 driver 自己）则必须退出，否则挂死
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    raise
            except Exception as e:
                self._emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                self._turn_task = None
                self._emit({"type": "turn_end"})

    async def _run_turn(self, batch: list[Message]) -> None:
        registry = self._registry
        if registry.main_provider is None:
            self._emit(
                {"type": "error", "message": "主 agent 未连接，请发送 /connect_provider 配置"}
            )
            return
        # Turn 生命周期：开始
        user_text = "\n".join(
            m.content
            for m in batch
            if m.role == "user" and not m.content.startswith("[任务 #")
        )
        store = registry.store
        turn_id = store.create_turn(self.id, user_text, TURN_RUNNING) if store else None
        self._current_turn_id = turn_id
        self._dispatcher.current_turn_id = turn_id
        self._record_event(TURN_STARTED, actor=ACTOR_MAIN, text=user_text)

        working = [Message(role="system", content=self._system), *self._messages, *batch]
        self._turn_anchor = next((m for m in batch if m.role == "user"), None)
        self._compact_usage = Usage()
        loop = AgentLoop(
            registry.main_provider,
            self._main_tools,
            max_steps=registry.max_steps_main,
            compactor=self._compact,
            on_tool_event=self._on_main_tool,
            on_call_text=self._on_call_text,
            on_reasoning=self._on_main_reasoning,
            on_llm_call=self._on_llm_call,
            # 主 agent：压缩优先，不做硬停（溢出截断是最后兜底）
        )

        def on_text(text: str) -> None:
            self._emit({"type": "text_delta", "text": text})

        try:
            result = await loop.run(working, on_text=on_text, event_source=self._inbox)
        except asyncio.CancelledError:
            if store and turn_id is not None:
                store.finish_turn(turn_id, TURN_CANCELLED)
            self._record_event(TURN_CANCELLED_EVENT, actor=ACTOR_MAIN)
            raise
        except Exception as e:
            if store and turn_id is not None:
                store.finish_turn(turn_id, TURN_ERROR)
            self._record_event(TURN_FAILED_EVENT, actor=ACTOR_MAIN, text=f"{type(e).__name__}: {e}")
            raise
        self._turn_anchor = None
        # 压缩/截断可能已改写 working：整段替换（去掉 system），持久化同步整段重写
        self._messages = working[1:]
        self._epoch += 1
        if store is not None:
            store.replace(self.id, self._messages)
            if turn_id is not None:
                store.finish_turn(turn_id, TURN_COMPLETED, final_text=result.text)
        self._record_event(TURN_COMPLETED_EVENT, actor=ACTOR_MAIN)
        usage = result.usage
        self._emit(
            {
                "type": "usage",
                "input": usage.input_tokens + self._compact_usage.input_tokens,
                "output": usage.output_tokens + self._compact_usage.output_tokens,
                "reasoning": usage.reasoning_tokens + self._compact_usage.reasoning_tokens,
                "scope": "本轮累计消耗",
            }
        )
        self._emit_context()

    def _on_main_tool(self, call, payload: str | None, phase: str) -> None:
        """主 agent 工具行为 → execution events（需求：主 agent 工具必须可见）。"""
        if phase == "started":
            self._record_event(
                TOOL_STARTED, actor=ACTOR_MAIN, name=call.name,
                arguments=_preview(str(call.arguments)),
            )
        else:
            self._record_event(
                TOOL_FINISHED, actor=ACTOR_MAIN, name=call.name,
                status=_classify_status("finished", payload), result=_preview(payload or ""),
            )

    def _on_main_reasoning(self, chunk: str) -> None:
        """provider 明确返回的思考增量（主 agent）→ working「思考」区。"""
        self._record_event(REASONING_DELTA, actor=ACTOR_MAIN, text=chunk)

    def _on_llm_call(self, usage: Usage) -> None:
        """每次模型调用的精确 usage：作为「消息集未变化时」的当前占用。"""
        self._last_prompt_tokens = usage.input_tokens
        self._last_prompt_epoch = self._epoch
        self._emit_context()
        self._epoch += 1  # 下一次请求的消息集已变化，精确值不再适用

    def _on_call_text(self, text: str, final: bool) -> None:
        """working/final 分离：中间调用文本进 working，收尾调用文本进 final。"""
        if final:
            self._record_event(FINAL_STARTED, actor=ACTOR_MAIN)
            self._record_event(FINAL_TEXT_DELTA, actor=ACTOR_MAIN, text=text)
            self._record_event(FINAL_COMPLETED, actor=ACTOR_MAIN)
        else:
            self._record_event(ACTIVITY_TEXT_DELTA, actor=ACTOR_MAIN, text=text)

    async def _compact(self, messages: list[Message]) -> None:
        """接近动态阈值时把旧历史压成 [前情摘要] 替换。只在 user 边界切割。"""
        registry = self._registry
        budget = registry.context_budget("main")
        if registry.main_provider is None or budget is None:
            return
        compacted = await compact_messages(
            messages, provider=registry.main_provider,
            budget_tokens=budget["compact_at"], tool_specs=self._main_tools.specs(),
            on_compacted=self._emit, usage=self._compact_usage,
            preserve=self._turn_anchor,
        )
        if compacted:
            self._epoch += 1

    def _on_subtask_event(self, event: SubtaskEvent) -> None:
        # TaskRun 持久化 + 新版执行事件（turn 归属固定为派发时的 parent turn）
        store = self._registry.store
        turn = self._sub_turn(event.id)
        terminal = True
        if event.status == "running":
            terminal = False
            if store is not None:
                store.update_task(event.id, self.id, TASK_RUNNING, started=True)
            self._record_event(
                SUBAGENT_STARTED, actor=ACTOR_SUBAGENT, task_id=event.id, turn_id=turn
            )
        elif event.status == "done":
            if store is not None:
                store.update_task(event.id, self.id, TASK_DONE, final_output=event.output)
            self._record_event(
                SUBAGENT_COMPLETED, actor=ACTOR_SUBAGENT, task_id=event.id,
                turn_id=turn, text=event.output,
            )
        elif event.status == "unconfigured":
            if store is not None:
                store.update_task(
                    event.id, self.id, TASK_UNCONFIGURED, final_output=event.output
                )
            self._record_event(
                SUBAGENT_FAILED, actor=ACTOR_SUBAGENT, task_id=event.id,
                turn_id=turn, status=event.status, text=event.output,
            )
        elif event.status == "incomplete":
            if store is not None:
                store.update_task(
                    event.id, self.id, TASK_INCOMPLETE, final_output=event.output
                )
            self._record_event(
                SUBAGENT_FAILED, actor=ACTOR_SUBAGENT, task_id=event.id,
                turn_id=turn, status=event.status, text=event.output,
            )
        elif event.status in ("error", "cancelled"):
            if store is not None:
                store.update_task(
                    event.id, self.id,
                    TASK_ERROR if event.status == "error" else TASK_CANCELLED,
                    final_output=event.output,
                )
            self._record_event(
                SUBAGENT_FAILED, actor=ACTOR_SUBAGENT, task_id=event.id,
                turn_id=turn, status=event.status, text=event.output,
            )
        direct = event.id in self._direct_tasks
        if (terminal and not direct and event.status == "cancelled"
                and self._report_executor):
            self._interrupted_task = event.id            # 记下被用户中断的任务，等介入结果
            self._intervention_prompts.setdefault(event.id, [])
        if terminal:
            self._task_turns.pop(event.id, None)
            self._direct_tasks.discard(event.id)
            if event.status in ("done", "incomplete", "error"):
                # 任务正常/异常结束：未送达的插话转成新任务（用户主动停止时不自动重开）
                for leftover in self._executor.take_undelivered():
                    self._dispatch_direct(leftover)
        # 旧总线事件（CLI 兼容）
        status_text = {
            "done": "完成", "error": "出错", "cancelled": "中断", "unconfigured": "未配置",
            "incomplete": "未完成",
        }.get(event.status, event.status)
        self._emit(
            {"type": "task_update", "id": event.id, "status": event.status, "output": event.output}
        )
        if event.status != "running" and not direct:
            # running 仅作 UI/持久化通知；唤醒 main 会白跑一轮 LLM
            # 直连执行者的任务不回报指挥者（它没派过，避免"任务完成"误导）
            self._inbox.put_nowait(
                Message(role="user", content=f"[任务 #{event.id} {status_text}]\n{event.output}")
            )
        elif event.status != "running" and event.id in self._intervention_origin:
            # 中断介入：把"被中断的任务 + 用户的话 + 执行者输出"一并回报指挥者
            interrupted = self._intervention_origin.pop(event.id)
            prompts = "；".join(self._intervention_prompts.pop(interrupted, [])) or "（无文字指令）"
            if self._report_executor:
                self._inbox.put_nowait(Message(role="user", content=(
                    f"[任务 #{interrupted} 被用户中断介入]\n"
                    f"用户对执行者说：{prompts}\n"
                    f"执行者输出：{event.output}"
                )))

    def _on_task_created(self, task_id: int, title: str, prompt: str) -> None:
        """派发即登记 TaskRun（queued）并广播；parent turn 此刻固定。"""
        if self._direct_dispatch:
            self._direct_tasks.add(task_id)
            if self._interrupted_task is not None and self._report_executor:
                # 记住来源任务号：之后即使又发生新的中断，本任务仍回报它纠偏的那个
                self._intervention_origin[task_id] = self._interrupted_task
        else:
            self._interrupted_task = None               # 指挥者重新派活：上一段中断故事翻篇
        registry = self._registry
        model = getattr(registry.executor_provider, "_model", "") or ""
        provider = type(registry.executor_provider).__name__ if registry.executor_provider else ""
        self._task_turns[task_id] = self._current_turn_id
        if registry.store is not None:
            registry.store.create_task(
                task_id, self.id, self._current_turn_id, title, prompt,
                model, provider, TASK_QUEUED,
                max_steps=registry.max_steps_executor,
            )
        self._record_event(
            SUBAGENT_QUEUED, actor=ACTOR_SUBAGENT, task_id=task_id,
            turn_id=self._current_turn_id, text=title,
        )

    def _on_task_question(self, task_id: int, question: str) -> None:
        self._emit({"type": "task_question", "id": task_id, "question": question})
        content = (
            f"[任务 #{task_id} 提问]\n{question}\n（用 answer_task 工具回答，任务号 {task_id}）"
        )
        self._inbox.put_nowait(Message(role="user", content=content))

    def _spawn_subagent(self, task_id: int, prompt: str, extra_tools: list):
        """构造 subagent 的一次执行：委托给持续上下文的执行者会话。"""
        return self._executor.spawn(task_id, prompt, extra_tools, turn=self._sub_turn(task_id))
