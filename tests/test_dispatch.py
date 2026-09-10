"""派发器测试：不阻塞证明、完成/错误事件回调、call_subagent 工具形态。"""

import asyncio

from agent.core.dispatch import Dispatcher, SubtaskEvent


def make_dispatcher(spawn, events: list[SubtaskEvent]) -> Dispatcher:
    return Dispatcher(spawn=spawn, on_event=events.append)


async def test_dispatch_returns_immediately_without_blocking() -> None:
    gate = asyncio.Event()
    events: list[SubtaskEvent] = []

    async def spawn(prompt: str) -> str:
        await gate.wait()
        return f"产出:{prompt}"

    dispatcher = make_dispatcher(spawn, events)
    task_id = dispatcher.dispatch("干活")

    assert task_id == 1
    assert dispatcher.tasks[1] == "running"
    assert events == []  # 未完成前无事件，证明未等待

    gate.set()
    await asyncio.sleep(0.05)
    assert dispatcher.tasks[1] == "done"
    assert events == [SubtaskEvent(id=1, status="done", output="产出:干活")]


async def test_spawn_error_becomes_error_event() -> None:
    events: list[SubtaskEvent] = []

    async def spawn(prompt: str) -> str:
        raise RuntimeError("爆了")

    dispatcher = make_dispatcher(spawn, events)
    dispatcher.dispatch("x")
    await asyncio.sleep(0.05)

    assert dispatcher.tasks[1] == "error"
    assert events[0].status == "error"
    assert "爆了" in events[0].output


async def test_sequential_ids() -> None:
    events: list[SubtaskEvent] = []

    async def spawn(prompt: str) -> str:
        return prompt

    dispatcher = make_dispatcher(spawn, events)
    assert dispatcher.dispatch("a") == 1
    assert dispatcher.dispatch("b") == 2
    await asyncio.sleep(0.05)
    assert [e.id for e in events] == [1, 2]


async def test_make_tool_returns_task_id_text() -> None:
    events: list[SubtaskEvent] = []

    async def spawn(prompt: str) -> str:
        return "ok"

    dispatcher = make_dispatcher(spawn, events)
    tool = dispatcher.make_tool()

    assert tool.spec.name == "call_subagent"
    assert "prompt" in tool.spec.parameters["required"]
    output = await tool.handler({"prompt": "任务"})
    assert "#1" in output
    await asyncio.sleep(0.05)
    assert events[0].output == "ok"
