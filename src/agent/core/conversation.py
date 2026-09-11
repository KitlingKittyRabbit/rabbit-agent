"""会话运行时：一份消息历史、一个收件队列、一个驱动循环、一张任务表。

多会话的基本单元；共享资源（provider/配置/审计/事件总线）经 registry 读取。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from uuid import uuid4

from ..providers import Message, Usage
from ..tools import build_registry
from .context import COMPACT_PROMPT, SUMMARY_PREFIX, estimate_chars, serialize_for_summary
from .dispatch import Dispatcher, SubtaskEvent
from .loop import AgentLoop
from .prompts import SUBAGENT_SYSTEM, build_main_system

_CONFIRM_TIMEOUT = 120


class Conversation:
    def __init__(self, *, session_id: str, title: str, project_id: str, registry) -> None:
        self.id = session_id
        self.title = title
        self.project_id = project_id
        self._registry = registry
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._dispatcher = Dispatcher(
            spawn=self._spawn_subagent,
            on_event=self._on_subtask_event,
            on_question=self._on_task_question,
        )
        self._main_tools = build_registry(
            self._root(), write=False, shell=False, on_call=self._audit_call
        )
        self._main_tools.add(self._dispatcher.make_tool())
        self._main_tools.add(self._dispatcher.make_answer_tool())
        self._messages: list[Message] = registry.store.load(session_id) if registry.store else []
        self._system = build_main_system(self._load_discipline())
        self._driver_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._compact_usage = Usage()

    def _root(self) -> Path:
        """工具沙箱根 = 所属项目路径。"""
        root = self._registry.project_root(self.project_id)
        if root is None:
            raise RuntimeError(f"会话所属项目不存在: {self.project_id}")
        return root

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
        self._dispatcher.cancel_all()
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
        self._dispatcher.cancel_all()

    def resolve_confirm(self, confirm_id: str, allow: bool) -> None:
        fut = self._pending_confirms.pop(confirm_id, None)
        if fut is not None and not fut.done():
            fut.set_result(allow)

    # ---------- 内部 ----------

    def _emit(self, event: dict) -> None:
        self._registry.emit({**event, "session": self.id})

    def _audit_call(self, tool: str, args: dict, result: str) -> None:
        self._registry.audit.log(session=self.id, tool=tool, args=args, result=result)

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
        working = [Message(role="system", content=self._system), *self._messages, *batch]
        self._compact_usage = Usage()
        loop = AgentLoop(
            registry.main_provider,
            self._main_tools,
            max_steps=registry.max_steps_main,
            compactor=self._compact,
        )

        def on_text(text: str) -> None:
            self._emit({"type": "text_delta", "text": text})

        result = await loop.run(working, on_text=on_text, event_source=self._inbox)
        # 压缩/截断可能已改写 working：整段替换（去掉 system），持久化同步整段重写
        self._messages = working[1:]
        if registry.store is not None:
            registry.store.replace(self.id, self._messages)
        usage = result.usage
        self._emit(
            {
                "type": "usage",
                "input": usage.input_tokens + self._compact_usage.input_tokens,
                "output": usage.output_tokens + self._compact_usage.output_tokens,
            }
        )
        self._emit(
            {
                "type": "context",
                "chars": estimate_chars(self._messages),
                "threshold": registry.compact_threshold,
            }
        )

    async def _compact(self, messages: list[Message]) -> None:
        """B（默认）：接近阈值时把旧历史压成 [前情摘要] 替换。只在 user 边界切割。"""
        provider = self._registry.main_provider
        threshold = self._registry.compact_threshold
        if provider is None or estimate_chars(messages) < threshold:
            return
        keep_head = 1 if messages and messages[0].role == "system" else 0
        body = messages[keep_head:]
        budget = threshold // 4
        cut = len(body)
        size = 0
        while cut > 1 and size < budget:
            m = body[cut - 1]
            size += len(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls or [])
            if size <= budget:
                cut -= 1
        # recent 不得以 tool 结果开头（其调用在 old 里，压缩后会成孤链）
        while cut < len(body) and body[cut].role == "tool":
            cut += 1
        old, recent = body[:cut], body[cut:]
        if not old:
            return
        before = estimate_chars(messages)
        serialized = serialize_for_summary(old, threshold)
        result = await provider.chat(
            [Message(role="user", content=f"{COMPACT_PROMPT}\n\n{serialized}")]
        )
        self._compact_usage.input_tokens += result.usage.input_tokens
        self._compact_usage.output_tokens += result.usage.output_tokens
        messages[keep_head:] = [
            Message(role="user", content=f"{SUMMARY_PREFIX}\n{result.text}"),
            *recent,
        ]
        self._emit({"type": "compacted", "before": before, "after": estimate_chars(messages)})

    def _on_subtask_event(self, event: SubtaskEvent) -> None:
        status_text = {"done": "完成", "error": "出错", "cancelled": "中断"}.get(
            event.status, event.status
        )
        self._emit(
            {"type": "task_update", "id": event.id, "status": event.status, "output": event.output}
        )
        self._inbox.put_nowait(
            Message(role="user", content=f"[任务 #{event.id} {status_text}]\n{event.output}")
        )

    def _on_task_question(self, task_id: int, question: str) -> None:
        self._emit({"type": "task_question", "id": task_id, "question": question})
        content = (
            f"[任务 #{task_id} 提问]\n{question}\n（用 answer_task 工具回答，任务号 {task_id}）"
        )
        self._inbox.put_nowait(Message(role="user", content=content))

    def _spawn_subagent(self, task_id: int, prompt: str, extra_tools: list):
        """构造 subagent 的一次执行。plan 模式下写与 shell 工具物理缺席。"""
        registry = self._registry
        if registry.executor_provider is None:

            async def noop() -> str:
                return "[执行 subagent 未连接，请 /connect_provider 配置]"

            return noop()
        tools = build_registry(
            self._root(),
            write=not registry.plan_mode,
            shell=not registry.plan_mode,
            on_call=self._audit_call,
            confirm=self._confirm,
        )
        for tool in extra_tools:
            tools.add(tool)

        def on_text(text: str) -> None:
            self._emit({"type": "subtask_step", "task": task_id, "kind": "text", "content": text})

        def on_tool(call, result: str) -> None:
            args = str(call.arguments)[:200]
            self._emit(
                {
                    "type": "subtask_step",
                    "task": task_id,
                    "kind": "tool_call",
                    "content": f"{call.name}({args})",
                }
            )
            self._emit(
                {
                    "type": "subtask_step",
                    "task": task_id,
                    "kind": "tool_result",
                    "content": result[:500],
                }
            )

        loop = AgentLoop(
            registry.executor_provider,
            tools,
            max_steps=registry.max_steps_executor,
            on_tool_event=on_tool,
        )
        messages = [
            Message(role="system", content=SUBAGENT_SYSTEM),
            Message(role="user", content=prompt),
        ]

        async def go() -> str:
            result = await loop.run(messages, on_text=on_text)
            if result.stop_reason == "max_steps":
                return f"[已达最大步数上限，任务可能未完成]\n{result.text}"
            return result.text

        return go()
