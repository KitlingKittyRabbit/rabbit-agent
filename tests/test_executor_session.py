"""执行者会话：持续上下文（跨任务/重启）+ 插话 + 直连 + 结果回流指挥者。"""

import asyncio
from pathlib import Path

from agent.core.orchestrator import Orchestrator
from agent.core.session import SessionStore
from agent.providers import ChatResult, FakeProvider, Message, ToolCall


def make_orch(tmp_path: Path, main, executor, store=None) -> Orchestrator:
    return Orchestrator(main_provider=main, executor_provider=executor, root=tmp_path,
                        store=store)


def chatty_main(n: int = 10) -> FakeProvider:
    """脚本给足的主 agent fake：多次唤醒也不会耗尽。"""
    return FakeProvider([ChatResult(text=f"收到{i}") for i in range(n)])


def _sid(orch: Orchestrator) -> str:
    return next(iter(orch.conversations))


async def _wait(pred, timeout=5.0) -> None:
    async def _spin():
        while not pred():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(_spin(), timeout)


async def test_executor_context_persists_across_tasks(tmp_path: Path) -> None:
    """同一个会话里，执行者的第二个任务必须看到第一个任务的完整历史。"""
    main = FakeProvider([ChatResult(text="x")])
    executor = FakeProvider([ChatResult(text="任务一完成"), ChatResult(text="任务二完成")])
    orch = make_orch(tmp_path, main, executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("任务一")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
        conv._dispatcher.dispatch("任务二")
        await _wait(lambda: conv._dispatcher.tasks.get(2) == "done")
    finally:
        await orch.stop()

    assert len(executor.calls) == 2
    second = executor.calls[1][0]
    assert second[0].role == "system"            # system 不入库、每次重建
    contents = [m.content for m in second]
    assert any("任务一" in c for c in contents)   # 上个任务描述在历史里
    assert any("任务一完成" in c for c in contents)  # 上个任务结果在历史里
    assert contents[-1] == "任务二"


async def test_executor_context_persists_across_restart(tmp_path: Path) -> None:
    """重启后执行者上下文仍在（store 流 executor）。"""
    store_path = tmp_path / "s.db"
    store = SessionStore(store_path)
    main = FakeProvider([ChatResult(text="x")])
    executor = FakeProvider([ChatResult(text="任务一完成")])
    orch = make_orch(tmp_path, main, executor, store=store)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("任务一")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
    finally:
        await orch.stop()
        store.close()

    # 重启：执行者下次任务必须带着旧历史
    executor2 = FakeProvider([ChatResult(text="任务二完成")])
    orch2 = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor2,
                      store=SessionStore(store_path))
    conv2 = orch2.conversations[_sid(orch2)]
    await orch2.start()
    try:
        conv2._dispatcher.dispatch("任务二")
        await _wait(lambda: conv2._dispatcher.tasks.get(2) == "done")
    finally:
        await orch2.stop()
    contents = [m.content for m in executor2.calls[0][0]]
    assert any("任务一" in c and any("任务一完成" in c2 for c2 in contents) for c in contents)
    assert contents[-1] == "任务二"


async def test_steering_message_reaches_next_model_round(tmp_path: Path) -> None:
    """执行中插话：下一个模型回合就看到（不新建任务）。"""
    (tmp_path / "a.txt").write_text("文件内容", encoding="utf-8")
    executor = FakeProvider([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ])
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("读 a.txt")
        conv.executor_message("改用别的方式读")  # 派发后立刻插话（任务仍 queued/running）
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
    finally:
        await orch.stop()
    # 插话经 event_source 回注：至少一轮模型调用看到了它，且没有另起任务
    assert any("改用别的方式读" in m.content for messages, _t in executor.calls for m in messages)
    assert len(executor.calls) == 2
    assert conv._dispatcher.tasks[1] == "done"  # 没另起任务


async def test_direct_message_to_idle_executor_dispatches_task(tmp_path: Path) -> None:
    """执行者空闲时直接对它说话 = 新任务；主时间线留灰色提示、指挥者上下文不含它。"""
    executor = FakeProvider([ChatResult(text="执行者收到")])
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor,
                     store=SessionStore(tmp_path / "s.db"))
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv.executor_message("你好执行者")
        await _wait(lambda: len(executor.calls) >= 1)
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
        events = orch.store.list_events(_sid(orch)) if orch.store else []
    finally:
        await orch.stop()
    assert any("你好执行者" in m.content for m in executor.calls[0][0])
    assert not any("你好执行者" in m.content for m in conv._messages)
    assert any(e.get("type") == "user_to_executor" and "你好执行者" in (e.get("text") or "")
               for e in events)


async def test_result_message_still_reaches_commander_context(tmp_path: Path) -> None:
    """任务完成的结果仍以消息形式回到指挥者上下文（不是工具返回值）。"""
    main = FakeProvider([ChatResult(text="回你")])
    executor = FakeProvider([ChatResult(text="执行完成报告")])
    orch = make_orch(tmp_path, main, executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("干活")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
        await _wait(lambda: any("[任务 #1 完成]" in m.content for m in conv._messages))
    finally:
        await orch.stop()
    assert any("执行完成报告" in m.content for m in conv._messages)


class _BlockingFake(FakeProvider):
    """第二次 chat 被闸门拦住的 fake：用于确定性地观察执行中的 inflight 状态。"""

    def __init__(self, script, gate):
        super().__init__(script)
        self._gate = gate

    async def chat(self, messages, tools=None, on_text=None, on_reasoning=None):
        if self.calls:  # 第二次及以后的调用被拦住
            await self._gate.wait()
        return await super().chat(messages, tools, on_text, on_reasoning)


async def test_inflight_task_visible_while_running(tmp_path: Path) -> None:
    """执行者窗口必须在任务进行中看到内容（不必等任务结束才落盘可见）。"""
    (tmp_path / "a.txt").write_text("内容", encoding="utf-8")
    gate = asyncio.Event()
    executor = _BlockingFake([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ], gate)
    orch = make_orch(tmp_path, chatty_main(), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("读 a.txt")
        await _wait(lambda: len(executor.calls) >= 1)
        inflight = conv.executor_inflight()
        assert any("读 a.txt" in m.content for m in inflight)
        assert any(m.role == "assistant" and m.tool_calls for m in inflight)
        gate.set()
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
    finally:
        await orch.stop()
    # 结束后 inflight 清空，正式历史含完整任务
    assert conv.executor_inflight() == []


async def test_user_to_executor_hint_anchored_to_last_turn(tmp_path: Path) -> None:
    """灰色提示挂在最近一次 turn 下，刷新时间线后仍可见。"""
    store = SessionStore(tmp_path / "s.db")
    main = FakeProvider([ChatResult(text="你好")])
    orch = make_orch(tmp_path, main, FakeProvider([ChatResult(text="x")]), store=store)
    conv = orch.conversations[_sid(orch)]
    queue = orch.subscribe()
    await orch.start()
    try:
        orch.handle_client_message({"type": "user", "session": conv.id, "text": "你好"})
        await _wait(lambda: any(e.get("type") == "turn_end" for e in list(queue._queue)))
        conv.executor_message("给执行者的悄悄话")
        last_turn = store.list_turns(conv.id)[-1]
        events = store.list_events(conv.id)
        hint = next(e for e in events if e["type"] == "user_to_executor")
        assert hint["turn_id"] == last_turn["id"]
        assert "悄悄话" in hint["text"]
    finally:
        await orch.stop()


async def test_cancel_running_task_clears_inflight(tmp_path: Path) -> None:
    """取消执行中的任务：inflight 必须清空（不留幽灵进行中内容）。"""
    gate = asyncio.Event()
    executor = _BlockingFake([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ], gate)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("读文件")
        await _wait(lambda: len(executor.calls) >= 1)
        assert conv.executor_inflight()          # 运行中有内容
        conv.stop()                               # 取消当前任务
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "cancelled")
        assert conv.executor_inflight() == []
    finally:
        gate.set()
        await orch.stop()


async def test_spawn_registers_inflight_only_when_started(tmp_path: Path) -> None:
    """回归（审核 F2）：go() 未启动（协程被关闭）时不得留下 inflight 幽灵。"""
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]),
                     FakeProvider([ChatResult(text="x")]))
    conv = orch.conversations[_sid(orch)]
    coro = conv._executor.spawn(1, "任务", [], turn=None)
    assert conv.executor_inflight() == []   # 未 await：不应登记
    coro.close()


async def test_undelivered_steer_becomes_next_task(tmp_path: Path) -> None:
    """回归：任务结束时仍未送达的插话，必须转成下一个任务而不是静默滞留。"""
    from agent.core.dispatch import SubtaskEvent

    executor = FakeProvider([ChatResult(text="收到")])
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.tasks[1] = "done"          # 模拟任务 #1 刚结束
        conv._executor.steer("迟到的话")             # 插话晚于最后一轮，滞留收件箱
        conv._on_subtask_event(SubtaskEvent(id=1, status="done", output="完成"))
        await _wait(lambda: len(executor.calls) >= 1)
    finally:
        await orch.stop()
    assert "迟到的话" in executor.calls[0][0][-1].content


async def test_undelivered_steer_flush_status_matrix(tmp_path: Path) -> None:
    """回到正题：done/incomplete/error 转新任务；cancelled/unconfigured 不重开。"""
    from agent.core.dispatch import SubtaskEvent

    for status in ("done", "incomplete", "error"):
        executor = FakeProvider([ChatResult(text="收到")])
        orch = make_orch(tmp_path / f"t_{status}", FakeProvider([ChatResult(text="x")]),
                         executor)
        conv = orch.conversations[_sid(orch)]
        await orch.start()
        try:
            conv._executor.steer("迟到的话")
            conv._on_subtask_event(SubtaskEvent(id=1, status=status, output="x"))
            await _wait(lambda e=executor: len(e.calls) >= 1)
        finally:
            await orch.stop()
        assert "迟到的话" in executor.calls[0][0][-1].content, status

    for status in ("cancelled", "unconfigured"):
        executor = FakeProvider([ChatResult(text="不应运行")])
        orch = make_orch(tmp_path / f"t_{status}", FakeProvider([ChatResult(text="x")]),
                         executor)
        conv = orch.conversations[_sid(orch)]
        await orch.start()
        try:
            conv._executor.steer("迟到的话")
            conv._on_subtask_event(SubtaskEvent(id=1, status=status, output="x"))
            await asyncio.sleep(0.2)
            assert executor.calls == [], status  # 用户停止/未配置：不自动重开
            assert conv._executor.take_undelivered(), status  # 遗留仍留在收件箱不丢
        finally:
            await orch.stop()


async def test_idle_direct_message_flushes_leftover_first(tmp_path: Path) -> None:
    """空闲直连：先派送遗留插话，再派新话（顺序正确）。"""
    executor = FakeProvider([ChatResult(text="一"), ChatResult(text="二")])
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._executor.steer("迟到的话")
        conv.executor_message("新的直连")
        await _wait(lambda: len(executor.calls) >= 2)
    finally:
        await orch.stop()
    assert "迟到的话" in executor.calls[0][0][-1].content
    assert "新的直连" in executor.calls[1][0][-1].content


async def test_direct_executor_chat_does_not_notify_commander(tmp_path: Path) -> None:
    """直连执行者的对话只留在执行者窗口：不回报指挥者、不唤醒它。"""
    main = FakeProvider([ChatResult(text="不应被唤醒")])
    executor = FakeProvider([ChatResult(text="聊天回复")])
    orch = make_orch(tmp_path, main, executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv.executor_message("你心情怎么样")
        await _wait(lambda: len(executor.calls) >= 1)
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
        await asyncio.sleep(0.2)  # 给"若误唤醒"留出触发窗口
    finally:
        await orch.stop()
    assert main.calls == []                                   # 指挥者没被唤醒
    assert not any("任务 #" in m.content for m in conv._messages)  # 上下文无完成回报


async def test_commander_dispatched_task_still_notifies(tmp_path: Path) -> None:
    """对照：指挥者自己派的任务，完成回报必须送达（不能被直连规则误伤）。"""
    executor = FakeProvider([ChatResult(text="执行完成")])
    main = FakeProvider([ChatResult(text="收到")])
    orch = make_orch(tmp_path, main, executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("正式任务")     # 非直连：模拟 call_subagent
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
        await _wait(lambda: any("[任务 #1 完成]" in m.content for m in conv._messages))
    finally:
        await orch.stop()


async def test_interrupted_task_intervention_reported_with_prompt(tmp_path: Path) -> None:
    """中断介入：回报指挥者的消息必须带任务号、用户原话和执行者输出。"""
    from agent.core.dispatch import SubtaskEvent

    executor = FakeProvider([ChatResult(text="纠偏完成")])
    main = chatty_main()
    orch = make_orch(tmp_path, main, executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        # 指挥者的任务 #1 被用户中断
        conv._on_task_created(1, "正式任务", "正式任务")
        conv._on_subtask_event(SubtaskEvent(id=1, status="cancelled", output="已被用户中断"))
        # 用户直连执行者纠偏
        conv.executor_message("换个思路，先读 a.txt")
        await _wait(lambda: len(executor.calls) >= 1)
        await _wait(lambda: any("[任务 #1 被用户中断介入]" in m.content
                                for m in conv._messages))
    finally:
        await orch.stop()
    report = next(m.content for m in conv._messages if "[任务 #1 被用户中断介入]" in m.content)
    assert "换个思路，先读 a.txt" in report     # 用户提示词一并返回
    assert "纠偏完成" in report                # 执行者输出一并返回


async def test_intervention_steer_prompts_accumulate(tmp_path: Path) -> None:
    """介入期间的多条用户消息（含执行中插话）都要出现在回报里。"""
    from agent.core.dispatch import SubtaskEvent

    gate = asyncio.Event()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    executor = _BlockingFake([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ], gate)
    orch = make_orch(tmp_path, chatty_main(), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._on_task_created(1, "正式任务", "正式任务")
        conv._on_subtask_event(SubtaskEvent(id=1, status="cancelled", output="中断"))
        conv.executor_message("第一条：先读 a.txt")     # 派发直连任务
        await _wait(lambda: len(executor.calls) >= 1)   # 首次模型调用后卡在闸门
        conv.executor_message("第二条：改用 b.txt")     # 执行中插话
        gate.set()
        await _wait(lambda: any("[任务 #1 被用户中断介入]" in m.content
                                for m in conv._messages))
    finally:
        await orch.stop()
    report = next(m.content for m in conv._messages if "[任务 #1 被用户中断介入]" in m.content)
    assert "第一条" in report and "第二条" in report


async def test_direct_chat_without_interrupt_not_reported(tmp_path: Path) -> None:
    """对照：没有中断发生时，直连对话不回报指挥者（既有规则不被破坏）。"""
    executor = FakeProvider([ChatResult(text="闲聊回复")])
    orch = make_orch(tmp_path, FakeProvider([ChatResult(text="x")]), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv.executor_message("你心情怎么样")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
        await asyncio.sleep(0.2)
    finally:
        await orch.stop()
    assert not any("被用户中断介入" in m.content for m in conv._messages)


async def test_commander_redispatch_clears_intervention_story(tmp_path: Path) -> None:
    """指挥者重新派活后，旧的中断故事翻篇：之后直连不再算介入。"""
    from agent.core.dispatch import SubtaskEvent

    executor = FakeProvider([ChatResult(text="回答")])
    orch = make_orch(tmp_path, chatty_main(), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._on_task_created(1, "旧任务", "旧任务")
        conv._on_subtask_event(SubtaskEvent(id=1, status="cancelled", output="中断"))
        conv._on_task_created(2, "新任务", "新任务")   # 指挥者重新派活（非直连）
        conv.executor_message("随便聊聊")
        await _wait(lambda: len(executor.calls) >= 1)
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "done")
        await asyncio.sleep(0.2)
    finally:
        await orch.stop()
    assert not any("被用户中断介入" in m.content for m in conv._messages)


async def test_old_story_prompts_do_not_leak_into_new_report(tmp_path: Path) -> None:
    """回归：旧中断故事的插话不得串进后续新故事的介入报告。"""
    from agent.core.dispatch import SubtaskEvent

    executor = FakeProvider([ChatResult(text="新故事回复")])
    orch = make_orch(tmp_path, chatty_main(), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        # 故事一：任务 #1 被中断；用户在其后说了一句（此时另有任务在跑 → 走插话，不派生任务）
        conv._on_task_created(1, "t1", "t1")
        conv._on_subtask_event(SubtaskEvent(id=1, status="cancelled", output="中断"))
        conv._dispatcher.tasks[98] = "running"
        conv.executor_message("旧故事的话")
        del conv._dispatcher.tasks[98]
        # 指挥者重派活 → 故事翻篇
        conv._on_task_created(2, "t2", "t2")
        # 故事二：任务 #3 被中断，用户纠偏
        conv._on_task_created(3, "t3", "t3")
        conv._on_subtask_event(SubtaskEvent(id=3, status="cancelled", output="中断"))
        conv.executor_message("新故事纠偏")
        await _wait(lambda: any("被用户中断介入" in m.content for m in conv._messages))
    finally:
        await orch.stop()
    report = next(m.content for m in conv._messages if "被用户中断介入" in m.content)
    assert "新故事纠偏" in report
    assert "旧故事的话" not in report          # 旧故事提示词不得串入


async def test_intervention_reports_origin_task_after_chained_interrupts(
        tmp_path: Path) -> None:
    """回归：介入任务完成时，报告的任务号必须是它纠偏的那个（不被后续中断串号）。"""
    from agent.core.dispatch import SubtaskEvent

    gate = asyncio.Event()
    executor = _BlockingFake([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ], gate)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    orch = make_orch(tmp_path, chatty_main(), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._on_task_created(1, "t1", "t1")
        conv._on_subtask_event(SubtaskEvent(id=1, status="cancelled", output="中断"))
        conv.executor_message("针对任务一的纠偏")     # 介入任务派生并开始跑（被闸门卡住）
        await _wait(lambda: len(executor.calls) >= 1)
        # 期间又发生：指挥者重派 + 新任务被中断
        conv._on_task_created(2, "t2", "t2")
        conv._on_task_created(3, "t3", "t3")
        conv._on_subtask_event(SubtaskEvent(id=3, status="cancelled", output="中断"))
        gate.set()
        await _wait(lambda: any("被用户中断介入" in m.content for m in conv._messages))
    finally:
        await orch.stop()
    report = next(m.content for m in conv._messages if "被用户中断介入" in m.content)
    assert "[任务 #1 被用户中断介入]" in report   # 仍是它纠偏的那个任务号
    assert "针对任务一的纠偏" in report


async def test_executor_stop_cancels_task_only_not_commander_turn(tmp_path: Path) -> None:
    """执行者停止：只取消执行者任务（进入中断故事），指挥者回合不受影响。"""
    gate = asyncio.Event()
    executor = _BlockingFake([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ], gate)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    orch = make_orch(tmp_path, chatty_main(), executor)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    turn = asyncio.create_task(asyncio.sleep(30))
    try:
        conv._dispatcher.dispatch("慢活")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "running")
        conv._turn_task = turn
        orch.handle_client_message({"type": "executor_stop", "session": conv.id})
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "cancelled")
        assert not turn.done()                # 指挥者回合没被停
        assert conv._interrupted_task == 1    # 非直连任务被中断 → 可以走介入回报
    finally:
        turn.cancel()
        conv._turn_task = None
        await orch.stop()


async def test_restart_reconciles_stale_running_tasks(tmp_path: Path) -> None:
    """重启对账接线：上次进程遗留的 running/queued 任务在启动时标记 cancelled。"""
    store_path = tmp_path / "s.db"
    seed = SessionStore(store_path)
    seed.create_session("s-old", "", "旧会话")
    seed.create_task(1, "s-old", None, "t", "p", "m", "prov", "running")
    seed.create_task(2, "s-old", None, "t", "p", "m", "prov", "queued")
    seed.close()
    orch = make_orch(tmp_path, chatty_main(), FakeProvider([ChatResult(text="x")]),
                     store=SessionStore(store_path))
    try:
        assert orch.store.get_task("s-old", 1)["status"] == "cancelled"
        assert orch.store.get_task("s-old", 2)["status"] == "cancelled"
    finally:
        orch.store.close()


async def test_interrupt_source_named_in_commander_notice(tmp_path: Path) -> None:
    """指挥者收到的中断通知要写明来源：指挥者栏停止 / 执行者栏停止。"""
    gate = asyncio.Event()
    executor = _BlockingFake([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ], gate)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    orch = make_orch(tmp_path, chatty_main(), executor)
    conv = orch.conversations[_sid(orch)]
    queue = orch.subscribe()
    await orch.start()
    try:
        conv._dispatcher.dispatch("慢活")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "running")
        conv.executor_stop()
        await _wait(lambda: any(
            e.get("type") == "task_update" and e.get("id") == 1
            and e.get("status") == "cancelled" and "执行者栏" in e.get("output", "")
            for e in list(queue._queue)))
        await _wait(lambda: any(
            "任务 #1 中断" in m.content and "执行者栏" in m.content for m in conv._messages))
        conv._dispatcher.dispatch("慢活二")
        await _wait(lambda: conv._dispatcher.tasks.get(2) == "running")
        conv.stop()
        await _wait(lambda: any(
            "任务 #2 中断" in m.content and "指挥者栏" in m.content for m in conv._messages))
        conv._dispatcher.dispatch("慢活三")                 # 服务关闭也是用户可感知的中断来源
        await _wait(lambda: conv._dispatcher.tasks.get(3) == "running")
        await conv.shutdown()
        await _wait(lambda: any(
            e.get("type") == "task_update" and e.get("id") == 3
            and "服务关闭" in e.get("output", "") for e in list(queue._queue)))
    finally:
        await orch.stop()


async def test_executor_report_switch_controls_intervention_notice(tmp_path: Path) -> None:
    """回报开关：关=中断介入也不回报指挥者；开=恢复回报；且持久化到重启。"""
    store_path = tmp_path / "s.db"
    store = SessionStore(store_path)
    gate = asyncio.Event()
    executor = _BlockingFake([
        ChatResult(tool_calls=[ToolCall(id="t1", name="read_file",
                                        arguments={"path": "a.txt"})], stop_reason="tool_use"),
        ChatResult(text="纠偏输出"),
    ], gate)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    orch = make_orch(tmp_path, chatty_main(), executor, store=store)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        assert conv.executor_report() is True          # 默认开
        orch.handle_client_message(
            {"type": "set_executor_report", "session": conv.id, "on": False})
        assert conv.executor_report() is False
        conv._dispatcher.dispatch("慢活")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "running")
        conv.executor_stop()
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "cancelled")
        assert conv._interrupted_task is None      # 关着时不记中断故事
        orch.handle_client_message(
            {"type": "set_executor_report", "session": conv.id, "on": True})
        gate.set()
        conv.executor_message("现在回报开着")
        await _wait(lambda: conv._dispatcher.tasks.get(2) == "done")
        assert not any("被用户中断介入" in m.content for m in conv._messages)
        orch.handle_client_message(
            {"type": "set_executor_report", "session": conv.id, "on": False})  # 还原，供重启断言
    finally:
        await orch.stop()
        store.close()

    # 重启：开关仍为关
    orch2 = make_orch(tmp_path, chatty_main(), FakeProvider([ChatResult(text="纠偏输出")]),
                      store=SessionStore(store_path))
    conv2 = orch2.conversations[_sid(orch2)]
    try:
        assert conv2.executor_report() is False
        conv2.set_executor_report(True)
        assert conv2.executor_report() is True
    finally:
        await orch2.stop()


class _StreamBlockFake(FakeProvider):
    """先流出一段文本再永久等待的 fake：模拟模型流到一半被中断。"""

    def __init__(self, chunks, gate):
        super().__init__([])
        self._chunks = chunks
        self._gate = gate

    async def chat(self, messages, tools=None, on_text=None, on_reasoning=None):
        self.calls.append((list(messages), list(tools) if tools is not None else None))
        for chunk in self._chunks:
            if on_text is not None:
                on_text(chunk)
        await self._gate.wait()
        return ChatResult(text="".join(self._chunks))


async def test_cancel_mid_stream_preserves_partial_output(tmp_path: Path) -> None:
    """中断保留：流到一半的输出落进执行者历史，且下个任务能看到它（便于纠偏）。"""
    gate = asyncio.Event()
    executor = _StreamBlockFake(["我准备先检查文件；", "第一部分写好了：开头……"], gate)
    store = SessionStore(tmp_path / "s.db")
    orch = make_orch(tmp_path, chatty_main(), executor, store=store)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("写报告")
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "running")
        await _wait(lambda: len(executor.calls) >= 1)
        conv.executor_stop()
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "cancelled")
        assert conv.executor_inflight() == []
        msgs = store.load(conv.id, "executor")
        assert any(m.role == "user" and "写报告" in m.content for m in msgs)
        partial = [m for m in msgs if m.role == "assistant"]
        assert partial and "第一部分写好了" in partial[-1].content
        gate.set()
        conv.executor_message("把第一部分改成第二人称")
        await _wait(lambda: conv._dispatcher.tasks.get(2) == "done")
    finally:
        gate.set()
        await orch.stop()
    sent = executor.calls[-1][0]
    assert any(m.role == "assistant" and "第一部分写好了" in m.content for m in sent)


async def test_cancel_during_tool_call_persists_placeholder_result(tmp_path: Path) -> None:
    """中断落在工具执行中：tool_calls 补占位结果，消息序列保持合法。"""
    executor = FakeProvider([
        ChatResult(tool_calls=[ToolCall(id="ask1", name="ask",
                                        arguments={"question": "选哪个？"})],
                   stop_reason="tool_use"),
        ChatResult(text="完成"),
    ])
    store = SessionStore(tmp_path / "s.db")
    orch = make_orch(tmp_path, chatty_main(), executor, store=store)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("先问我")
        await _wait(lambda: len(executor.calls) >= 1)
        await asyncio.sleep(0.05)   # 等 ask 工具真正开始等待
        conv.executor_stop()
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "cancelled")
        msgs = store.load(conv.id, "executor")
        assert msgs[-2].role == "assistant" and msgs[-2].tool_calls
        assert msgs[-1].role == "tool" and msgs[-1].tool_call_id == "ask1"
        assert "未完成" in msgs[-1].content
    finally:
        await orch.stop()


async def test_cancel_with_partial_tool_batch_fills_only_missing(tmp_path: Path) -> None:
    """回归（审核）：一轮多工具、停在后半个：只补缺失占位，不重复半截文本、不留孤儿。"""
    (tmp_path / "a.txt").write_text("内容", encoding="utf-8")
    executor = FakeProvider([
        ChatResult(text="先读再问。", tool_calls=[
            ToolCall(id="tc_read", name="read_file", arguments={"path": "a.txt"}),
            ToolCall(id="tc_ask", name="ask", arguments={"question": "选哪个？"}),
        ], stop_reason="tool_use"),
        ChatResult(text="完成"),
    ])
    store = SessionStore(tmp_path / "s.db")
    orch = make_orch(tmp_path, chatty_main(), executor, store=store)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("先读再问")
        await _wait(lambda: len(executor.calls) >= 1)
        await asyncio.sleep(0.05)   # 等 read 完成、ask 卡住
        conv.executor_stop()
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "cancelled")
        msgs = store.load(conv.id, "executor")
        assistants = [m for m in msgs if m.role == "assistant"]
        assert len(assistants) == 1                       # 半截文本不重复追加
        assert "先读再问" in assistants[0].content
        assert msgs[-1].role == "tool" and msgs[-1].tool_call_id == "tc_ask"
        assert "未完成" in msgs[-1].content
        answered = {m.tool_call_id for m in msgs if m.role == "tool"}
        assert {c.id for c in assistants[0].tool_calls} <= answered   # 无孤儿
    finally:
        await orch.stop()


async def test_cancel_with_duplicate_tool_ids_uses_latest_assistant(tmp_path: Path) -> None:
    """加固：两轮 assistant 完全相同时（个别端点复用固定 id），占位要补在最新一轮。"""
    (tmp_path / "a.txt").write_text("内容", encoding="utf-8")
    same = ChatResult(text="一", tool_calls=[
        ToolCall(id="dup", name="read_file", arguments={"path": "a.txt"}),
    ], stop_reason="tool_use")
    executor = FakeProvider([
        ChatResult(text="一", tool_calls=[
            ToolCall(id="dup", name="read_file", arguments={"path": "a.txt"}),
        ], stop_reason="tool_use"),
        ChatResult(text="一", tool_calls=[
            ToolCall(id="dup", name="ask", arguments={"question": "选哪个？"}),
        ], stop_reason="tool_use"),
        same,
    ])
    store = SessionStore(tmp_path / "s.db")
    orch = make_orch(tmp_path, chatty_main(), executor, store=store)
    conv = orch.conversations[_sid(orch)]
    await orch.start()
    try:
        conv._dispatcher.dispatch("重复 id 场景")
        await _wait(lambda: len(executor.calls) >= 2)
        await asyncio.sleep(0.05)   # 第二轮 ask 卡住
        conv.executor_stop()
        await _wait(lambda: conv._dispatcher.tasks.get(1) == "cancelled")
        msgs = store.load(conv.id, "executor")
        assert msgs[-1].role == "tool" and msgs[-1].tool_call_id == "dup"
        assert "未完成" in msgs[-1].content          # 最新一轮补了占位
        assistants = [m for m in msgs if m.role == "assistant" and m.tool_calls]
        assert len(assistants) == 2                  # 未重复追加
    finally:
        await orch.stop()


async def test_persist_interrupted_uses_latest_assistant_identity(tmp_path: Path) -> None:
    """加固单测：两条值完全相同的 assistant（复用固定 id 的端点）也要按最后一条补占位。"""
    store = SessionStore(tmp_path / "s.db")
    orch = make_orch(tmp_path, chatty_main(), FakeProvider([ChatResult(text="x")]),
                     store=store)
    conv = orch.conversations[_sid(orch)]
    ex = conv._executor
    call = ToolCall(id="dup", name="read_file", arguments={"path": "a.txt"})
    a1 = Message(role="assistant", content="一", tool_calls=[call])
    a2 = Message(role="assistant", content="一", tool_calls=[call])
    working = [
        Message(role="system", content="s"),
        Message(role="user", content="p"),
        a1,
        Message(role="tool", content="第一轮结果", tool_call_id="dup"),
        a2,
    ]
    ex._persist_interrupted(working, store, "", "")
    msgs = store.load(conv.id, "executor")
    assert msgs[-1].role == "tool" and msgs[-1].tool_call_id == "dup"
    assert "未完成" in msgs[-1].content
    assert sum(1 for m in msgs if m.role == "assistant") == 2   # 无重复追加
    store.close()
