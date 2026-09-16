"""writer 串行测试（Phase E）：同一项目同时最多一个 write-capable subagent。"""

import asyncio

from agent.core.dispatch import Dispatcher, SubtaskEvent


async def test_writer_serialization_same_project() -> None:
    locks: dict[str, asyncio.Lock] = {}
    events: list[SubtaskEvent] = []
    first_started = asyncio.Event()
    finish_first = asyncio.Event()
    order: list[str] = []

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        order.append(f"start#{task_id}")
        if task_id == 1:
            first_started.set()
            await finish_first.wait()
        order.append(f"end#{task_id}")
        return f"产出{task_id}"

    dispatcher = Dispatcher(
        spawn=spawn,
        on_event=events.append,
        writer_locks=locks,
        project_key="projA",
    )
    dispatcher.dispatch("任务一")
    await first_started.wait()
    dispatcher.dispatch("任务二")
    await asyncio.sleep(0.05)

    # 第二个 writer 在第一个完成前必须保持 queued
    assert dispatcher.tasks[2] == "queued"
    assert order == ["start#1"]

    finish_first.set()
    await asyncio.sleep(0.1)

    # 第一个完成后第二个才开始，严格串行
    assert order == ["start#1", "end#1", "start#2", "end#2"]
    assert dispatcher.tasks[1] == "done"
    assert dispatcher.tasks[2] == "done"


async def test_writers_parallel_across_different_projects() -> None:
    locks: dict[str, asyncio.Lock] = {}
    both_started = asyncio.Event()
    started_count = {"n": 0}

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        started_count["n"] += 1
        if started_count["n"] == 2:
            both_started.set()
        await asyncio.sleep(0.05)
        return "ok"

    d1 = Dispatcher(spawn=spawn, on_event=lambda e: None, writer_locks=locks, project_key="projA")
    d2 = Dispatcher(spawn=spawn, on_event=lambda e: None, writer_locks=locks, project_key="projB")
    d1.dispatch("A 任务")
    d2.dispatch("B 任务")

    # 不同项目的 writer 并行，不互相阻塞
    await asyncio.wait_for(both_started.wait(), timeout=2)
