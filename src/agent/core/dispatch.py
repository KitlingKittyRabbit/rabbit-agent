"""派发器：call_subagent 工具的实现——立即返回任务号，后台运行，完成事件回调。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..providers import ToolSpec
from ..tools.base import Tool

SpawnFn = Callable[[str], Awaitable[str]]


@dataclass
class SubtaskEvent:
    id: int
    status: str  # "done" | "error"
    output: str


class Dispatcher:
    def __init__(self, spawn: SpawnFn, on_event: Callable[[SubtaskEvent], None]) -> None:
        self._spawn = spawn
        self._on_event = on_event
        self._next_id = 1
        self.tasks: dict[int, str] = {}

    def make_tool(self) -> Tool:
        async def call_subagent(args: dict) -> str:
            task_id = self.dispatch(args["prompt"])
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
                        "prompt": {"type": "string", "description": "给 subagent 的完整任务提示词"}
                    },
                    "required": ["prompt"],
                },
            ),
            call_subagent,
        )

    def dispatch(self, prompt: str) -> int:
        task_id = self._next_id
        self._next_id += 1
        self.tasks[task_id] = "running"
        asyncio.get_running_loop().create_task(self._run(task_id, prompt))
        return task_id

    async def _run(self, task_id: int, prompt: str) -> None:
        try:
            output = await self._spawn(prompt)
            self.tasks[task_id] = "done"
            self._on_event(SubtaskEvent(id=task_id, status="done", output=output))
        except Exception as e:
            self.tasks[task_id] = "error"
            self._on_event(
                SubtaskEvent(id=task_id, status="error", output=f"{type(e).__name__}: {e}")
            )
