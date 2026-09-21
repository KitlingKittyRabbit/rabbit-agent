"""派发器：call_subagent / ask / answer_task 工具，后台任务表，中断取消，writer 串行。

任务生命周期：dispatch → queued →(拿到项目 writer 锁)→ running → done/error/cancelled。
同一项目同时最多一个 write-capable subagent（asyncio.Lock 按项目串行）。
ask 回答经 answers 队列回注 subagent；answer_task 供主 agent 回复提问。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..providers import ToolSpec
from ..tools.base import Tool

logger = logging.getLogger(__name__)

SpawnFn = Callable[[int, str, list], Awaitable[str]]  # (task_id, prompt, extra_tools)

_ASK_TIMEOUT = 300


@dataclass
class SubtaskEvent:
    id: int
    status: str  # "queued" | "running" | "done" | "error" | "cancelled" | "unconfigured"
    output: str


class ExecutorUnconfigured(Exception):
    """executor 未配置：任务未执行，不得当作成功。"""


class TaskIncomplete(Exception):
    """任务达到终止条件但未完成（步数上限/无进展/上下文预算），不得当作成功。"""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class _Task:
    def __init__(self, handle: asyncio.Task) -> None:
        self.handle = handle
        self.answers: asyncio.Queue[str] = asyncio.Queue()


class Dispatcher:
    def __init__(
        self,
        spawn: SpawnFn,
        on_event: Callable[[SubtaskEvent], None],
        on_question: Callable[[int, str], None] | None = None,
        on_created: Callable[[int, str, str], None] | None = None,
        writer_locks: dict[str, asyncio.Lock] | None = None,
        project_key: str = "",
        start_id: int = 1,
    ) -> None:
        self._spawn = spawn
        self._on_event = on_event
        self._on_question = on_question
        self._on_created = on_created
        self._writer_locks = writer_locks
        self._project_key = project_key
        # 会话内任务号必须持久单调（task_runs 主键含 id），从已用最大值续号
        self._next_id = start_id
        self.tasks: dict[int, str] = {}
        self.current_turn_id: int | None = None
        self._running: dict[int, _Task] = {}
        self._cancel_reasons: dict[int, str] = {}

    def make_tool(self) -> Tool:
        """call_subagent：主 agent 的派发工具。"""

        async def call_subagent(args: dict) -> str:
            task_id = self.dispatch(args["prompt"], title=str(args.get("title") or ""))
            return f"已派发，任务号 #{task_id}。其输出将在完成时以事件消息送达，无需等待。"

        return Tool(
            ToolSpec(
                name="call_subagent",
                description=(
                    "派发一个 subagent 在后台执行写操作任务。"
                    "立即返回任务号不阻塞；结果稍后以事件消息送达。"
                    "提示词必须完整、可独立执行，并写明验证方式。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "任务的简短可读标题"},
                        "prompt": {"type": "string", "description": "给 subagent 的完整任务提示词"},
                    },
                    "required": ["prompt"],
                },
            ),
            call_subagent,
        )

    def make_answer_tool(self) -> Tool:
        """answer_task：主 agent 回答 subagent 提问的工具。"""

        async def answer_task(args: dict) -> str:
            task_id = int(args["task_id"])
            task = self._running.get(task_id)
            if task is None or self.tasks.get(task_id) != "running":
                return f"任务 #{task_id} 不存在或已结束"
            task.answers.put_nowait(str(args["answer"]))
            return f"已把回答送达任务 #{task_id}"

        return Tool(
            ToolSpec(
                name="answer_task",
                description="回答正在等待的 subagent 的提问（任务号在提问消息中给出）",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer", "description": "任务号"},
                        "answer": {"type": "string", "description": "回答内容"},
                    },
                    "required": ["task_id", "answer"],
                },
            ),
            answer_task,
        )

    def dispatch(self, prompt: str, title: str = "") -> int:
        task_id = self._next_id
        self._next_id += 1
        self.tasks[task_id] = "queued"
        if not title:
            title = prompt.splitlines()[0][:50] if prompt else f"任务 {task_id}"
        if self._on_created is not None:
            self._on_created(task_id, title, prompt)
        handle = asyncio.get_running_loop().create_task(self._run(task_id, prompt))
        self._running[task_id] = _Task(handle)
        return task_id

    def cancel_all(self, reason: str = "用户手动中断") -> None:
        """中断所有运行中/排队中的任务（各自走 cancelled 分支发事件）。"""
        for task_id, task in self._running.items():
            if self.tasks.get(task_id) in ("running", "queued"):
                self._cancel_reasons[task_id] = reason
                task.handle.cancel()

    async def _run(self, task_id: int, prompt: str) -> None:
        ask_tool = Tool(
            ToolSpec(
                name="ask",
                description="规格有歧义时向主 agent 提问澄清，拿到回答前会等待",
                parameters={
                    "type": "object",
                    "properties": {"question": {"type": "string", "description": "要问的问题"}},
                    "required": ["question"],
                },
            ),
            lambda args: self._ask(task_id, args["question"]),
        )
        lock = (
            self._writer_locks.setdefault(self._project_key, asyncio.Lock())
            if self._writer_locks is not None
            else None
        )
        acquired = False
        coro: Awaitable[str] | None = None
        try:
            try:
                if lock is not None:
                    await lock.acquire()  # 同一项目 writer 串行：拿不到锁就保持 queued
                    acquired = True
                # 先构造 spawn：executor 未配置等构造期异常不留下 running/started_at 痕迹
                coro = self._spawn(task_id, prompt, [ask_tool])
                self.tasks[task_id] = "running"
                # running 回调故意不包裹：它失败必须走 error 分支并关闭未 await 的协程
                self._on_event(SubtaskEvent(id=task_id, status="running", output=""))
                output = await coro
                self.tasks[task_id] = "done"
                self._notify(SubtaskEvent(id=task_id, status="done", output=output))
            except asyncio.CancelledError:
                self.tasks[task_id] = "cancelled"
                self._notify(SubtaskEvent(
                    id=task_id, status="cancelled",
                    output=self._cancel_reasons.pop(task_id, "用户手动中断"),
                ))
            except ExecutorUnconfigured as e:
                self.tasks[task_id] = "unconfigured"
                self._notify(SubtaskEvent(id=task_id, status="unconfigured", output=str(e)))
            except TaskIncomplete as e:
                self.tasks[task_id] = "incomplete"
                self._notify(
                    SubtaskEvent(id=task_id, status="incomplete", output=f"[{e.reason}] {e}")
                )
            except Exception as e:
                self.tasks[task_id] = "error"
                self._notify(
                    SubtaskEvent(id=task_id, status="error", output=f"{type(e).__name__}: {e}")
                )
        finally:
            if coro is not None:
                coro.close()  # 未 await 的协程必须关闭，防 "never awaited" 告警
            if acquired and lock is not None:
                lock.release()
            self._cancel_reasons.pop(task_id, None)
            self._running.pop(task_id, None)

    def _notify(self, event: SubtaskEvent) -> None:
        """终态事件回调失败不击穿派发流程（任务状态已先行落定，仅记日志）。"""
        try:
            self._on_event(event)
        except Exception as e:
            logger.warning("任务 %s 的事件回调失败: %s", event.id, e)

    async def _ask(self, task_id: int, question: str) -> str:
        if self._on_question is not None:
            self._on_question(task_id, question)
        task = self._running[task_id]
        try:
            return await asyncio.wait_for(task.answers.get(), timeout=_ASK_TIMEOUT)
        except TimeoutError:
            return "（主 agent 未在限时内回答，请自行判断并继续）"
