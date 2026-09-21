"""编排器/会话端到端测试：闭环、多会话隔离、中断、plan 模式、压缩、广播。"""

import asyncio
from pathlib import Path

import pytest

from agent.core.orchestrator import Orchestrator
from agent.providers import ChatResult, FakeProvider, Message, ToolCall


async def _until(queue: asyncio.Queue, pred, timeout: float = 5.0):
    async def _wait():
        while True:
            event = await queue.get()
            if pred(event):
                return event

    return await asyncio.wait_for(_wait(), timeout)


def _is_turn_end(event: dict) -> bool:
    return event.get("type") == "turn_end"


def make_orch(tmp_path: Path, main, executor, **kwargs) -> Orchestrator:
    return Orchestrator(
        main_provider=main, executor_provider=executor, root=tmp_path, **kwargs
    )


def sid(orch: Orchestrator) -> str:
    return next(iter(orch.conversations))


async def test_full_closed_loop(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("纪律全文：先批后落盘", encoding="utf-8")
    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="call_subagent", arguments={"prompt": "写 a.txt"})
                ],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发，等待结果"),
            ChatResult(text="结果已确认"),
        ]
    )
    executor = FakeProvider([ChatResult(text="写完了，pytest 全绿")])
    orch = make_orch(tmp_path, main, executor)
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "帮我写 a.txt"})

        # subagent 可能在第一轮结束前就完成了：按序消费，不假设事件先后
        done_update = None
        turn_ends = 0

        async def collect() -> None:
            nonlocal done_update, turn_ends
            while done_update is None or turn_ends < 2:
                event = await queue.get()
                if event.get("type") == "task_update" and event.get("status") == "done":
                    done_update = event
                elif event.get("type") == "turn_end":
                    turn_ends += 1

        await asyncio.wait_for(collect(), timeout=5)
        assert "写完了" in done_update["output"]
        assert done_update["session"] == session
    finally:
        await orch.stop()

    third_call = main.calls[2][0]
    assert any("[任务 #1 完成]" in m.content and "写完了" in m.content for m in third_call)
    assert executor.calls[0][0][-1].content == "写 a.txt"
    main_tool_names = [t.name for t in main.calls[0][1]]
    assert "call_subagent" in main_tool_names
    assert "answer_task" in main_tool_names
    assert "write_file" not in main_tool_names
    assert "run_shell" not in main_tool_names
    system = main.calls[0][0][0]
    assert system.role == "system"
    assert "先批后落盘" in system.content
    executor_tool_names = [t.name for t in executor.calls[0][1]]
    assert "write_file" in executor_tool_names
    assert "run_shell" in executor_tool_names
    assert "ask" in executor_tool_names  # 澄清通道工具


async def test_running_notification_does_not_wake_main(tmp_path: Path) -> None:
    """running 状态只广播不唤醒 main；终态才唤醒（少跑一轮白费 LLM）。"""
    from agent.core.dispatch import SubtaskEvent

    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), FakeProvider([]))
    conv = orch.conversations[sid(orch)]
    conv._on_subtask_event(SubtaskEvent(id=1, status="running", output=""))
    assert conv._inbox.empty()
    conv._on_subtask_event(SubtaskEvent(id=1, status="done", output="产出"))
    message = conv._inbox.get_nowait()
    assert "[任务 #1 完成]" in message.content and "产出" in message.content


async def test_multi_session_isolation(tmp_path: Path) -> None:
    main = FakeProvider([ChatResult(text="会话一回复"), ChatResult(text="会话二回复")])
    orch = make_orch(tmp_path, main, FakeProvider([ChatResult(text="x")]))
    queue = orch.subscribe()
    await orch.start()
    try:
        s1 = sid(orch)
        s2 = orch.create_session(title="第二").id
        orch.handle_client_message({"type": "user", "session": s1, "text": "一"})
        await _until(queue, lambda e: e.get("type") == "turn_end" and e.get("session") == s1)
        orch.handle_client_message({"type": "user", "session": s2, "text": "二"})
        await _until(queue, lambda e: e.get("type") == "turn_end" and e.get("session") == s2)
    finally:
        await orch.stop()

    conv1 = orch.conversations[s1]
    conv2 = orch.conversations[s2]
    assert [m.content for m in conv1._messages if m.role == "user"] == ["一"]
    assert [m.content for m in conv2._messages if m.role == "user"] == ["二"]
    # 两个会话的历史互不包含对方内容
    assert main.calls[0][0][-1].content == "一"
    assert all(m.content != "一" for m in main.calls[1][0])


async def test_stop_interrupts_turn_and_subagents(tmp_path: Path) -> None:
    gate = asyncio.Event()

    class BlockingProvider:
        async def chat(self, messages, tools=None, on_text=None):
            await gate.wait()
            return ChatResult(text="不应到达")

    orch = make_orch(tmp_path, BlockingProvider(), FakeProvider([ChatResult(text="x")]))
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "跑"})
        await asyncio.sleep(0.1)  # turn 已阻塞在 provider 调用上
        orch.handle_client_message({"type": "stop", "session": session})
        stopped = await _until(queue, lambda e: e.get("type") == "stopped")
        assert stopped["session"] == session
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()


class BlockingProvider:
    """永远阻塞的 provider（测试中断/shutdown 用）。"""

    def __init__(self) -> None:
        self.gate = asyncio.Event()

    async def chat(self, messages, tools=None, on_text=None):
        await self.gate.wait()
        return ChatResult(text="不应到达")


async def test_shutdown_during_turn_completes(tmp_path: Path) -> None:
    """审核缺陷回归：turn 进行中 shutdown 必须能返回（driver 不挂死）。"""
    orch = make_orch(tmp_path, BlockingProvider(), FakeProvider([ChatResult(text="x")]))
    await orch.start()
    orch.handle_client_message({"type": "user", "session": sid(orch), "text": "跑"})
    await asyncio.sleep(0.1)  # turn 已阻塞
    stop_task = asyncio.create_task(orch.stop())
    # asyncio.wait 超时不取消任务：旧代码挂死时 pending 非空，断言必失败（真锁定）
    done, pending = await asyncio.wait({stop_task}, timeout=2)
    for task in pending:
        task.cancel()
    assert done, "shutdown 在 turn 进行中挂死"


async def test_default_session_persisted_across_restarts(tmp_path: Path) -> None:
    """审核缺陷回归：default 会话写 sessions 表，重启后历史仍在。"""
    from agent.core.session import SessionStore

    db = tmp_path / "s.db"
    main = FakeProvider([ChatResult(text="第一句回复")])
    orch = Orchestrator(
        main_provider=main,
        executor_provider=FakeProvider([ChatResult(text="x")]),
        root=tmp_path,
        store=SessionStore(db),
    )
    await orch.start()
    first_id = sid(orch)
    orch.handle_client_message({"type": "user", "session": first_id, "text": "记住我"})
    await asyncio.sleep(0.2)
    await orch.stop()

    orch2 = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="y")]),
        executor_provider=FakeProvider([ChatResult(text="z")]),
        root=tmp_path,
        store=SessionStore(db),
    )
    assert first_id in orch2.conversations
    contents = [m.content for m in orch2.conversations[first_id]._messages]
    assert "记住我" in contents


async def test_confirm_allow_and_timeout(tmp_path: Path, monkeypatch) -> None:
    import agent.core.conversation as conversation_mod

    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), FakeProvider([]))
    queue = orch.subscribe()
    conv = orch.conversations[sid(orch)]

    # allow 路径：确认请求发出后被 resolve 为 True
    task = asyncio.create_task(conv._confirm("rm -rf x"))
    request = await _until(queue, lambda e: e.get("type") == "confirm_request")
    orch.handle_client_message({"type": "confirm_response", "id": request["id"], "allow": True})
    assert await task is True

    # 超时路径：限时内无人回答自动拒绝
    monkeypatch.setattr(conversation_mod, "_CONFIRM_TIMEOUT", 0.05)
    assert await conv._confirm("rm -rf y") is False


async def test_ask_timeout_returns_fallback(tmp_path: Path, monkeypatch) -> None:
    import agent.core.dispatch as dispatch_mod
    from agent.core.dispatch import Dispatcher

    monkeypatch.setattr(dispatch_mod, "_ASK_TIMEOUT", 0.05)
    events: list = []

    async def spawn(task_id: int, prompt: str, extra_tools: list) -> str:
        return await extra_tools[0].handler({"question": "没人理我"})

    dispatcher = Dispatcher(spawn=spawn, on_event=events.append)
    dispatcher.dispatch("干活")
    await asyncio.sleep(0.3)
    assert events[-1].status == "done"
    assert "自行判断" in events[-1].output


async def test_plan_mode_strips_write_and_shell_from_subagent(tmp_path: Path) -> None:
    executor = FakeProvider([ChatResult(text="只读探索结果")])
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor, plan_mode=True)
    conv = orch.conversations[sid(orch)]
    output = await conv._spawn_subagent(1, "探索一下", [])

    assert output == "只读探索结果"
    tool_names = [t.name for t in executor.calls[0][1]]
    assert tool_names == ["glob", "grep", "ls", "read_file"]


async def test_subagent_max_steps_marks_output(tmp_path: Path) -> None:
    from agent.core.dispatch import TaskIncomplete

    executor = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="ls", arguments={})], stop_reason="tool_use"
            ),
            ChatResult(
                tool_calls=[ToolCall(id="c2", name="ls", arguments={})], stop_reason="tool_use"
            ),
        ]
    )
    orch = make_orch(
        tmp_path,
        FakeProvider([ChatResult(text="x")]),
        executor,
        max_steps_executor=1,
    )
    conv = orch.conversations[sid(orch)]
    with pytest.raises(TaskIncomplete) as exc_info:
        await conv._spawn_subagent(1, "干活", [])
    assert exc_info.value.reason == "max_steps"
    assert "最大步数" in str(exc_info.value)


async def test_subagent_max_steps_progress_persisted(tmp_path: Path) -> None:
    """撞步数上限：步数/最近动作/原因落库，且不自动重派（任务只此一个）。"""
    from agent.core.session import SessionStore

    executor = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="c1", name="ls", arguments={"path": "."})],
                stop_reason="tool_use",
            ),
            ChatResult(text="不应到达"),
        ]
    )
    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="m1", name="call_subagent", arguments={"prompt": "干活"})],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发"),
            ChatResult(text="确认"),
        ]
    )
    db = tmp_path / "s.db"
    orch = Orchestrator(
        main_provider=main,
        executor_provider=executor,
        root=tmp_path,
        store=SessionStore(db),
        max_steps_executor=1,
    )
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "开始"})
        await _until(
            queue, lambda e: e.get("type") == "task_update" and e.get("status") == "incomplete"
        )
        await _until(queue, _is_turn_end)
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()

    store = SessionStore(db)
    tasks = store.list_tasks(session)
    assert [t["id"] for t in tasks] == [1]  # 未自动重派
    task = tasks[0]
    assert task["status"] == "incomplete"  # 撞上限不是成功
    assert task["stop_reason"] == "max_steps"
    assert task["steps_used"] == 1
    assert task["max_steps"] == 1
    assert "ls" in task["last_action"]
    assert "最大步数" in task["final_output"]
    store.close()


async def test_background_subagent_events_keep_parent_turn(tmp_path: Path) -> None:
    """旧 subagent 跨 turn 运行：其事件与任务归属固定为派发时的 turn。"""
    from agent.core.session import SessionStore

    gate = asyncio.Event()

    class GatedExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, on_text=None):
            self.calls += 1
            if self.calls == 1:
                await gate.wait()
                return ChatResult(
                    tool_calls=[ToolCall(id="e1", name="ls", arguments={"path": "."})],
                    stop_reason="tool_use",
                )
            return ChatResult(text="子代理完成")

    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="m1", name="call_subagent", arguments={"prompt": "慢活"})],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发"),
            ChatResult(text="第二轮的回复"),
            ChatResult(text="确认"),
        ]
    )
    db = tmp_path / "s.db"
    orch = Orchestrator(
        main_provider=main,
        executor_provider=GatedExecutor(),
        root=tmp_path,
        store=SessionStore(db),
        max_steps_executor=5,
    )
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "A"})
        await _until(queue, _is_turn_end)  # turn A 结束（subagent 仍被 gate 卡住）
        orch.handle_client_message({"type": "user", "session": session, "text": "B"})
        await _until(queue, _is_turn_end)  # turn B 结束
        gate.set()  # 旧 subagent 在 turn B 之后才产生工具/完成事件
        await _until(queue, lambda e: e.get("type") == "task_update" and e.get("status") == "done")
        await asyncio.sleep(0.05)
    finally:
        await orch.stop()

    store = SessionStore(db)
    turns = store.list_turns(session)
    assert len(turns) >= 2
    turn_a, turn_b = turns[0]["id"], turns[1]["id"]
    events = store.list_events(session, task_id=1)
    assert events
    assert all(e["turn_id"] == turn_a for e in events), [
        (e["type"], e["turn_id"]) for e in events
    ]
    assert all(e["turn_id"] != turn_b for e in events)
    assert store.get_task(session, 1)["parent_turn_id"] == turn_a
    store.close()


async def test_unconfigured_executor_task_not_done(tmp_path: Path) -> None:
    """executor 未配置：任务状态 unconfigured（非 done），主会话收到明确失败事件。"""
    from agent.core.events import TASK_UNCONFIGURED
    from agent.core.session import SessionStore

    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="m1", name="call_subagent", arguments={"prompt": "干活"})],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发"),
            ChatResult(text="确认"),
        ]
    )
    db = tmp_path / "s.db"
    orch = Orchestrator(
        main_provider=main, executor_provider=None, root=tmp_path, store=SessionStore(db)
    )
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "开始"})
        update = await _until(
            queue,
            lambda e: (
                e.get("type") == "task_update"
                and e.get("id") == 1
                and e.get("status") == "unconfigured"
            ),
        )
        assert update["status"] == "unconfigured"
        assert "未配置" in update["output"] and "未执行" in update["output"]
        await _until(queue, _is_turn_end)
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()

    store = SessionStore(db)
    task = store.get_task(session, 1)
    assert task["status"] == TASK_UNCONFIGURED != "done"
    assert task["started_at"] is None  # 未执行：不留开始时间痕迹
    assert "未配置" in task["final_output"]
    events = {e["type"]: e for e in store.list_events(session, task_id=1)}
    assert events["subagent_failed"]["status"] == "unconfigured"
    assert orch.provider_status()["roles"]["executor"]["configured"] is False
    store.close()


async def test_subagent_step_event_and_store_progress(tmp_path: Path) -> None:
    """on_step 接线：一轮多工具 → 运行中广播 subagent_step(1) 且落库 steps_used。"""
    from agent.core.session import SessionStore

    executor = FakeProvider(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="c1", name="ls", arguments={"path": "."}),
                    ToolCall(id="c2", name="ls", arguments={"path": "."}),
                ],
                stop_reason="tool_use",
            ),
            ChatResult(text="完成"),
        ]
    )
    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[ToolCall(id="m1", name="call_subagent", arguments={"prompt": "干活"})],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发"),
            ChatResult(text="确认"),
        ]
    )
    db = tmp_path / "s.db"
    orch = Orchestrator(
        main_provider=main,
        executor_provider=executor,
        root=tmp_path,
        store=SessionStore(db),
        max_steps_executor=5,
    )
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "开始"})
        step = await _until(queue, lambda e: e.get("type") == "subagent_step")
        assert step["task_id"] == 1
        assert step["steps_used"] == 1  # 一轮两个工具 = 1 步
        assert step["max_steps"] == 5
        await _until(queue, lambda e: e.get("type") == "task_update" and e.get("status") == "done")
        await _until(queue, _is_turn_end)
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()

    store = SessionStore(db)
    task = store.get_task(session, 1)
    assert task["status"] == "done"
    assert task["steps_used"] == 1 and task["max_steps"] == 5
    assert "ls" in task["last_action"]
    store.close()


async def test_set_plan_mode_ack(tmp_path: Path) -> None:
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), FakeProvider([]))
    queue = orch.subscribe()
    orch.handle_client_message({"type": "set_plan_mode", "on": True})
    event = queue.get_nowait()
    assert event == {"type": "plan_mode", "on": True}
    assert orch.plan_mode is True


async def test_broadcast_reaches_all_subscribers(tmp_path: Path) -> None:
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), FakeProvider([]))
    q1 = orch.subscribe()
    q2 = orch.subscribe()
    orch.emit({"type": "plan_mode", "on": True})
    assert q1.get_nowait()["type"] == "plan_mode"
    assert q2.get_nowait()["type"] == "plan_mode"
    orch.unsubscribe(q1)
    orch.emit({"type": "plan_mode", "on": False})
    assert q1.empty()
    assert q2.get_nowait()["on"] is False


async def test_compaction_replaces_old_history(tmp_path: Path) -> None:
    main = FakeProvider([ChatResult(text="这是摘要"), ChatResult(text="回复你")])
    orch = make_orch(tmp_path, main, FakeProvider([ChatResult(text="x")]))
    orch.set_context_window("main", 8_000)  # 用户覆盖窗口，触发动态阈值
    conv = orch.conversations[sid(orch)]
    conv._messages = [
        Message(role="user", content="旧消息" + "长" * 100),
        *[Message(role="user", content="填充" + "长" * 100) for _ in range(300)],
        Message(role="assistant", content="短回复"),
    ]
    queue = orch.subscribe()
    await orch.start()
    try:
        conv.enqueue_user("新消息")
        compacted = await _until(queue, lambda e: e.get("type") == "compacted")
        assert compacted["before_tokens"] >= compacted["after_tokens"]
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()

    compact_call = main.calls[0][0]
    assert "压缩为一份要点摘要" in compact_call[0].content
    assert "旧消息" in compact_call[0].content
    second_contents = [m.content for m in main.calls[1][0]]
    assert any("[前情摘要]" in c and "这是摘要" in c for c in second_contents)
    assert any("短回复" in c for c in second_contents)
    assert any("[前情摘要]" in m.content for m in conv._messages)


async def test_usage_and_context_events(tmp_path: Path) -> None:
    from agent.providers import Usage

    main = FakeProvider([ChatResult(text="回复", usage=Usage(input_tokens=11, output_tokens=7))])
    orch = make_orch(tmp_path, main, FakeProvider([ChatResult(text="x")]))
    orch.set_context_window("main", 100_000)
    queue = orch.subscribe()
    await orch.start()
    try:
        orch.handle_client_message({"type": "user", "session": sid(orch), "text": "hi"})
        usage = await _until(queue, lambda e: e.get("type") == "usage")
        assert usage["input"] == 11 and usage["output"] == 7
        assert usage["scope"] == "本轮累计消耗" and usage["reasoning"] == 0
        context = await _until(queue, lambda e: e.get("type") == "context")
        assert context["window"] == 100_000 and context["window_source"] == "user"
        assert context["used_tokens"] > 0 and context["estimated_tokens"] > 0
        assert context["compact_at"] and context["compact_at"] < context["window"]
        assert context["percent"] is not None and 0 <= context["percent"] <= 100
        assert context["messages"] >= 1
        # 消息集已变化（本 turn 写入历史）→ 值为估算
        assert context["exact"] is False
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()


async def test_context_window_unknown_shows_no_percent(tmp_path: Path) -> None:
    """未知模型：window 为 None、percent 为 None，不编造上限。"""
    orch = make_orch(
        tmp_path, FakeProvider([ChatResult(text="x")]), FakeProvider([ChatResult(text="y")])
    )
    payload = orch.conversations[sid(orch)].context_payload()
    assert payload["window"] is None
    assert payload["window_source"] == "unknown"  # 未知来源不得标成 adapter
    assert payload["percent"] is None
    assert payload["compact_at"] is None


async def test_context_exact_only_when_messages_unchanged(tmp_path: Path) -> None:
    """精确 usage 只在「发出请求的那一刻」有效；随后消息集变化回到估算。"""
    from agent.providers import Usage

    main = FakeProvider([ChatResult(text="x", usage=Usage(input_tokens=123, output_tokens=4))])
    orch = make_orch(tmp_path, main, FakeProvider([ChatResult(text="y")]))
    orch.set_context_window("main", 100_000)
    conv = orch.conversations[sid(orch)]
    queue = orch.subscribe()

    # 模拟一次模型调用：这次调用自身的 context 事件应为精确值
    conv._on_llm_call(Usage(input_tokens=123, output_tokens=4))
    event = queue.get_nowait()
    assert event["type"] == "context"
    assert event["exact"] is True and event["used_tokens"] == 123
    # 调用之后消息集已变化：下一次请求回到估算
    payload2 = conv.context_payload()
    assert payload2["exact"] is False
    assert payload2["used_tokens"] == payload2["estimated_tokens"]
    assert payload2["last_prompt_tokens"] == 123  # 精确值仍作为「上次请求实际输入」保留


async def test_round_with_multiple_subagents(tmp_path: Path) -> None:
    """一轮派发两个 subagent：各自独立任务/事件/统计。"""
    from agent.core.session import SessionStore

    main = FakeProvider(
        [
            ChatResult(
                tool_calls=[
                    ToolCall(id="m1", name="call_subagent", arguments={"prompt": "活一"}),
                    ToolCall(id="m2", name="call_subagent", arguments={"prompt": "活二"}),
                ],
                stop_reason="tool_use",
            ),
            ChatResult(text="已派发"),
            ChatResult(text="确认"),
        ]
    )
    executor = FakeProvider([ChatResult(text="一"), ChatResult(text="二")])
    db = tmp_path / "s.db"
    orch = Orchestrator(
        main_provider=main, executor_provider=executor, root=tmp_path,
        store=SessionStore(db),
    )
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "开始"})
        await _until(queue, lambda e: e.get("type") == "task_update" and e.get("id") == 2
                     and e.get("status") == "done")
        await _until(queue, _is_turn_end)
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()
    store = SessionStore(db)
    tasks = store.list_tasks(session)
    assert [t["id"] for t in tasks] == [1, 2]
    assert all(t["status"] == "done" for t in tasks)
    assert all(t["steps_used"] == 0 for t in tasks)  # 无工具调用：回合数为 0
    store.close()


async def test_subagent_no_progress_tags_incomplete(tmp_path: Path) -> None:
    """executor 连续重复相同工具+结果 → no_progress → 任务 incomplete（非成功）。"""
    from agent.core.session import SessionStore

    executor = FakeProvider(
        [
            ChatResult(tool_calls=[ToolCall(id=f"c{i}", name="ls", arguments={"path": "."})],
                       stop_reason="tool_use")
            for i in range(10)
        ]
    )
    main = FakeProvider(
        [
            ChatResult(tool_calls=[ToolCall(id="m1", name="call_subagent",
                                            arguments={"prompt": "原地打转"})],
                       stop_reason="tool_use"),
            ChatResult(text="已派发"),
            ChatResult(text="确认"),
        ]
    )
    db = tmp_path / "s.db"
    orch = Orchestrator(main_provider=main, executor_provider=executor, root=tmp_path,
                        store=SessionStore(db))
    queue = orch.subscribe()
    await orch.start()
    try:
        session = sid(orch)
        orch.handle_client_message({"type": "user", "session": session, "text": "开始"})
        await _until(queue, lambda e: e.get("type") == "task_update"
                     and e.get("status") == "incomplete")
        await _until(queue, _is_turn_end)
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()
    task = SessionStore(db).get_task(session, 1)
    assert task["status"] == "incomplete"
    assert task["stop_reason"] == "no_progress"
    assert task["steps_used"] == 3
    assert task["actions_used"] == 3
    store = SessionStore(db)
    assert store.get_task(session, 1)["status"] == "incomplete"
    store.close()


async def test_executor_compacts_instead_of_bricking(tmp_path: Path) -> None:
    """executor 上下文逼近预算 → 压缩继续完成，不再硬停砖化（V1 回归）。"""
    from agent.core.session import SessionStore

    executor = FakeProvider([
        ChatResult(text="一轮完成"),
        ChatResult(text="摘要"),        # 第二个任务前的压缩调用
        ChatResult(text="二轮完成"),
    ])
    main = FakeProvider([ChatResult(text="x")])
    db = tmp_path / "s.db"
    orch = Orchestrator(main_provider=main, executor_provider=executor, root=tmp_path,
                        store=SessionStore(db))
    orch.set_context_window("executor", 100)  # 预算下限 1024 tokens
    conv = orch.conversations[sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("任务一 " + "长" * 2000)
        await _wait_for(lambda: conv._dispatcher.tasks.get(1) == "done")
        conv._dispatcher.dispatch("任务二 " + "长" * 2000)
        await _wait_for(lambda: conv._dispatcher.tasks.get(2) == "done")
    finally:
        await orch.stop()
    store = SessionStore(db)
    session = conv.id
    task1 = store.get_task(session, 1)
    task2 = store.get_task(session, 2)
    assert task1["status"] == "done" and task2["status"] == "done"
    exec_msgs = store.load(session, "executor")
    assert any("前情摘要" in m.content for m in exec_msgs)  # 压缩过
    assert any("任务二" in m.content for m in exec_msgs)    # 当前任务未被吞
    assert any("二轮完成" in m.content for m in exec_msgs)  # 继续完成
    store.close()


async def _wait_for(pred, timeout=5.0) -> None:
    import asyncio as _a

    async def _spin():
        while not pred():
            await _a.sleep(0.01)

    await _a.wait_for(_spin(), timeout)


async def test_restart_keeps_thinking_blocks_for_next_request(tmp_path: Path) -> None:
    """重启级：thinking+signature+tool_use 经数据库恢复后，下一次请求仍按序携带。"""
    from agent.core.session import SessionStore
    from agent.providers import ToolCall

    db = tmp_path / "s.db"
    store = SessionStore(db)
    store.create_session("s1", "default", "旧会话")
    blocks = [
        {"type": "thinking", "thinking": "先读", "signature": "sig-restart-1"},
        {"type": "text", "text": "我来看看"},
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
    ]
    store.replace("s1", [
        Message(role="user", content="修复 a.py"),
        Message(
            role="assistant", content="我来看看",
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a.py"})],
            reasoning="先读", content_blocks=blocks,
        ),
        Message(role="tool", content="已读取", tool_call_id="t1"),
    ])
    store.close()

    main = FakeProvider([ChatResult(text="继续完成")])
    orch = Orchestrator(
        main_provider=main, executor_provider=FakeProvider([ChatResult(text="x")]),
        root=tmp_path, store=SessionStore(db),
    )
    conv = orch.conversations["s1"]
    loaded_assistant = [m for m in conv._messages if m.role == "assistant"][0]
    assert loaded_assistant.content_blocks == blocks  # 重启后协议块完整

    queue = orch.subscribe()
    await orch.start()
    try:
        conv.enqueue_user("继续")
        await _until(queue, _is_turn_end)
    finally:
        await orch.stop()

    sent = main.calls[0][0]
    assistant = next(m for m in sent if m.role == "assistant")
    assert assistant.content_blocks == blocks
    assert [b["type"] for b in assistant.content_blocks] == ["thinking", "text", "tool_use"]
    assert assistant.content_blocks[0]["signature"] == "sig-restart-1"


async def test_compaction_never_swallows_last_user_message(tmp_path: Path) -> None:
    """回归（审核 F4）：超大单条消息也不得把当前任务 prompt 卷进摘要。"""
    from agent.core.context import compact_messages

    fake = FakeProvider([ChatResult(text="摘要")])
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="旧" + "长" * 100),
        Message(role="assistant", content="回复"),
        Message(role="user", content="当前任务" + "长" * 4000),
    ]
    ok = await compact_messages(messages, provider=fake, budget_tokens=100, tool_specs=[])
    assert ok is True
    assert any("当前任务" in m.content for m in messages)
    assert any("前情摘要" in m.content for m in messages)

    single = [Message(role="system", content="sys"),
              Message(role="user", content="长" * 5000)]
    ok2 = await compact_messages(single, provider=FakeProvider([ChatResult(text="摘要")]),
                                 budget_tokens=100, tool_specs=[])
    assert ok2 is False and single[-1].content.startswith("长")  # 没有干净边界则不压


async def test_compaction_keeps_current_task_mid_flight(tmp_path: Path) -> None:
    """回归（审核 F3）：任务进行中压缩，必须保留当前任务 prompt 与其工具链。"""
    from agent.core.context import compact_messages

    fake = FakeProvider([ChatResult(text="摘要")])
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="旧任务" + "长" * 200),
        Message(role="assistant", content="旧回复"),
        Message(role="user", content="当前任务"),
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="t1", name="read", arguments={})]),
        Message(role="tool", content="结果" + "长" * 4000, tool_call_id="t1"),
    ]
    ok = await compact_messages(messages, provider=fake, budget_tokens=100, tool_specs=[])
    assert ok is True
    assert any("当前任务" in m.content for m in messages)      # 进行中任务未被吞
    assert any("前情摘要" in m.content for m in messages)      # 旧历史被压缩
    # 链完整：每条 tool 消息前都有对应 assistant tool_call
    seen_calls: set[str] = set()
    for m in messages:
        if m.role == "assistant":
            seen_calls.update(tc.id for tc in (m.tool_calls or []))
        elif m.role == "tool":
            assert m.tool_call_id in seen_calls


async def test_compaction_no_orphan_tool_when_cut_lands_on_tool(tmp_path: Path) -> None:
    """回归（审核 G1）：切割点落在 tool 上时必须前移，recent 不得以孤儿工具结果开头。"""
    from agent.core.context import compact_messages

    fake = FakeProvider([ChatResult(text="摘要")])
    big_args = {"data": "x" * 3000}
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="旧消息"),
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="t0", name="write", arguments=big_args)]),
        Message(role="tool", content="旧结果", tool_call_id="t0"),
        Message(role="user", content="当前任务"),
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="t1", name="read", arguments={})]),
        Message(role="tool", content="当前结果", tool_call_id="t1"),
    ]
    ok = await compact_messages(messages, provider=fake, budget_tokens=100, tool_specs=[],
                                preserve=messages[4])
    assert ok is True
    assert any("当前任务" in m.content for m in messages)
    seen: set[str] = set()
    for m in messages:
        if m.role == "assistant":
            seen.update(tc.id for tc in (m.tool_calls or []))
        elif m.role == "tool":
            assert m.tool_call_id in seen, "孤儿 tool 结果"


async def test_compaction_anchor_survives_later_steer_message(tmp_path: Path) -> None:
    """回归（审核 G2）：插话（新的 user 消息）不得让任务 prompt 失去保护。"""
    from agent.core.context import compact_messages

    fake = FakeProvider([ChatResult(text="摘要")])
    task = Message(role="user", content="当前任务")
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="旧历史" + "长" * 2000),
        task,
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="t1", name="read", arguments={})]),
        Message(role="tool", content="大结果" + "长" * 2000, tool_call_id="t1"),
        Message(role="user", content="[用户对执行者插话]\n换个方式"),
    ]
    # 尺寸切割会落在任务之后：只保护"最后一条 user"（插话）会把任务 prompt 卷入摘要
    ok = await compact_messages(messages, provider=fake, budget_tokens=100, tool_specs=[],
                                preserve=task)
    assert ok is True
    assert any("当前任务" in m.content for m in messages)   # 锚点仍在
    assert any("插话" in m.content for m in messages)
    seen: set[str] = set()
    for m in messages:
        if m.role == "assistant":
            seen.update(tc.id for tc in (m.tool_calls or []))
        elif m.role == "tool":
            assert m.tool_call_id in seen


async def test_conversation_compact_uses_turn_anchor(tmp_path: Path) -> None:
    """主 agent 压缩必须使用本回合锚点（中途注入的任务完成消息不得挤掉用户指令）。"""
    main = FakeProvider([ChatResult(text="摘要")])
    orch = make_orch(tmp_path, main, FakeProvider([ChatResult(text="x")]))
    orch.set_context_window("main", 100)
    conv = orch.conversations[sid(orch)]
    conv._messages = [Message(role="user", content="旧历史" + "长" * 2000)]
    anchor = Message(role="user", content="本回合指令")
    conv._turn_anchor = anchor
    working = [
        Message(role="system", content=conv._system),
        *conv._messages,
        anchor,
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="t1", name="read", arguments={})]),
        Message(role="tool", content="大结果" + "长" * 2000, tool_call_id="t1"),
        Message(role="user", content="[任务 #1 完成]\n结果来了"),  # 注入消息（新的最后 user）
    ]
    await conv._compact(working)
    assert any("本回合指令" in m.content for m in working)     # 锚点未被吞
    assert any("前情摘要" in m.content for m in working)       # 旧历史被压缩
    assert any("任务 #1 完成" in m.content for m in working)
