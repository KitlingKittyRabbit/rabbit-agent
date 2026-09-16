"""派发器测试：不阻塞、完成/错误/中断事件、ask/answer 往返、call_subagent 形态。"""

import asyncio
import logging

from agent.core.dispatch import Dispatcher, SubtaskEvent


def make_dispatcher(spawn, events: list, questions: list | None = None) -> Dispatcher:
    on_question = (lambda tid, q: questions.append((tid, q))) if questions is not None else None
    return Dispatcher(spawn=spawn, on_event=events.append, on_question=on_question)


async def test_dispatch_returns_immediately_without_blocking() -> None:
    gate = asyncio.Event()
    events: list[SubtaskEvent] = []

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        await gate.wait()
        return f"产出:{prompt}"

    dispatcher = make_dispatcher(spawn, events)
    task_id = dispatcher.dispatch("干活")

    assert task_id == 1
    assert dispatcher.tasks[1] == "queued"
    assert events == []

    gate.set()
    await asyncio.sleep(0.05)
    assert dispatcher.tasks[1] == "done"
    assert events == [
        SubtaskEvent(id=1, status="running", output=""),
        SubtaskEvent(id=1, status="done", output="产出:干活"),
    ]


async def test_spawn_error_becomes_error_event() -> None:
    events: list[SubtaskEvent] = []

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        raise RuntimeError("爆了")

    dispatcher = make_dispatcher(spawn, events)
    dispatcher.dispatch("x")
    await asyncio.sleep(0.05)

    assert dispatcher.tasks[1] == "error"
    assert events[-1].status == "error"
    assert "爆了" in events[-1].output


async def test_running_callback_error_closes_coro_and_continues() -> None:
    """running 回调抛异常：未 await 协程被关闭（无告警）、任务 error、后续任务正常。"""
    import gc
    import warnings

    events: list[SubtaskEvent] = []

    def bad_running(event: SubtaskEvent) -> None:
        if event.id == 1 and event.status == "running":
            raise RuntimeError("回调坏了")
        events.append(event)

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        return f"产出:{prompt}"

    dispatcher = Dispatcher(spawn=spawn, on_event=bad_running)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dispatcher.dispatch("第一次")
        await asyncio.sleep(0.05)
        assert dispatcher.tasks[1] == "error"
        assert events and events[-1].status == "error"
        assert "回调坏了" in events[-1].output

        dispatcher.dispatch("第二次")  # 不受前一次回调故障影响
        await asyncio.sleep(0.05)
        assert dispatcher.tasks[2] == "done"
        assert 1 not in dispatcher._running and 2 not in dispatcher._running
        gc.collect()

    never_awaited = [str(w.message) for w in caught if "never awaited" in str(w.message)]
    assert never_awaited == [], never_awaited


async def test_terminal_callback_error_logged_and_next_task_ok(caplog) -> None:
    """终态回调抛异常：不击穿流程、任务状态保持、记警告、后续任务可执行。"""
    events: list[SubtaskEvent] = []

    def bad_done(event: SubtaskEvent) -> None:
        if event.status == "done":
            raise RuntimeError("终态回调坏")
        events.append(event)

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        return f"产出:{prompt}"

    dispatcher = Dispatcher(spawn=spawn, on_event=bad_done)
    with caplog.at_level(logging.WARNING, logger="agent.core.dispatch"):
        dispatcher.dispatch("第一次")
        await asyncio.sleep(0.05)
        assert dispatcher.tasks[1] == "done"  # 状态先落定，不伪装也不回退
        assert "任务 1 的事件回调失败" in caplog.text
        dispatcher.dispatch("第二次")
        await asyncio.sleep(0.05)
        assert dispatcher.tasks[2] == "done"
        assert 1 not in dispatcher._running and 2 not in dispatcher._running


async def test_cancel_all_marks_cancelled() -> None:
    gate = asyncio.Event()
    events: list[SubtaskEvent] = []

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        await gate.wait()
        return "x"

    dispatcher = make_dispatcher(spawn, events)
    dispatcher.dispatch("a")
    dispatcher.dispatch("b")
    await asyncio.sleep(0.02)
    dispatcher.cancel_all()
    await asyncio.sleep(0.05)

    assert [e.status for e in events if e.status == "cancelled"] == ["cancelled", "cancelled"]
    assert dispatcher.tasks[1] == "cancelled"


async def test_ask_and_answer_roundtrip() -> None:
    events: list[SubtaskEvent] = []
    questions: list[tuple] = []

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        return await extra_tools[0].handler({"question": "用哪个文件？"})

    dispatcher = make_dispatcher(spawn, events, questions)
    answer_tool = dispatcher.make_answer_tool()
    dispatcher.dispatch("干活")
    await asyncio.sleep(0.05)

    assert questions == [(1, "用哪个文件？")]
    result = await answer_tool.handler({"task_id": 1, "answer": "a.txt"})
    assert "已把回答送达" in result
    await asyncio.sleep(0.05)
    assert events[-1].status == "done"
    assert events[-1].output == "a.txt"


async def test_answer_to_unknown_task() -> None:
    dispatcher = make_dispatcher(lambda p, t: asyncio.sleep(0), [])
    result = await dispatcher.make_answer_tool().handler({"task_id": 99, "answer": "x"})
    assert "不存在" in result


async def test_make_tool_returns_task_id_text() -> None:
    events: list[SubtaskEvent] = []

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        return "ok"

    dispatcher = make_dispatcher(spawn, events)
    tool = dispatcher.make_tool()

    assert tool.spec.name == "call_subagent"
    assert "prompt" in tool.spec.parameters["required"]
    output = await tool.handler({"prompt": "任务"})
    assert "#1" in output
    await asyncio.sleep(0.05)
    assert events[-1].output == "ok"
