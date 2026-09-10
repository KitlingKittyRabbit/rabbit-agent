"""编排器端到端测试：FakeProvider 全链路——派发、不阻塞、事件回注、纪律注入、plan 模式。"""

import asyncio
from pathlib import Path

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
    orch = Orchestrator(main_provider=main, executor_provider=executor, root=tmp_path)
    await orch.start()
    try:
        orch.handle_client_message({"type": "user", "text": "帮我写 a.txt"})

        await _until(orch.outbox, _is_turn_end)  # 第一轮结束（已派发）
        update = await _until(
            orch.outbox, lambda e: e.get("type") == "task_update" and e.get("status") == "done"
        )
        assert "写完了" in update["output"]
        await _until(orch.outbox, _is_turn_end)  # 第二轮（事件回注后主 agent 汇报）
    finally:
        await orch.stop()

    # 主 agent 第三轮调用看到了回注的任务完成事件
    third_call = main.calls[2][0]
    assert any("[任务 #1 完成]" in m.content and "写完了" in m.content for m in third_call)
    # executor 收到的正是派发的提示词
    assert executor.calls[0][0][-1].content == "写 a.txt"
    # 主 agent 工具：有 call_subagent，无写工具、无 shell（物理缺席）
    main_tool_names = [t.name for t in main.calls[0][1]]
    assert "call_subagent" in main_tool_names
    assert "write_file" not in main_tool_names
    assert "run_shell" not in main_tool_names
    # 项目纪律注入主 agent 系统提示词
    system = main.calls[0][0][0]
    assert system.role == "system"
    assert "先批后落盘" in system.content
    # subagent 拿到完整工具
    executor_tool_names = [t.name for t in executor.calls[0][1]]
    assert "write_file" in executor_tool_names
    assert "run_shell" in executor_tool_names


async def test_plan_mode_strips_write_and_shell_from_subagent(tmp_path: Path) -> None:
    executor = FakeProvider([ChatResult(text="只读探索结果")])
    orch = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=executor,
        root=tmp_path,
        plan_mode=True,
    )
    output = await orch._spawn_subagent("探索一下")

    assert output == "只读探索结果"
    tool_names = [t.name for t in executor.calls[0][1]]
    assert tool_names == ["grep", "ls", "read_file"]


async def test_subagent_max_steps_marks_output(tmp_path: Path) -> None:
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
    orch = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=executor,
        root=tmp_path,
        max_steps_executor=1,
    )
    output = await orch._spawn_subagent("干活")
    assert "已达最大步数上限" in output


async def test_set_plan_mode_ack(tmp_path: Path) -> None:
    orch = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=FakeProvider([ChatResult(text="y")]),
        root=tmp_path,
    )
    orch.handle_client_message({"type": "set_plan_mode", "on": True})
    event = orch.outbox.get_nowait()
    assert event == {"type": "plan_mode", "on": True}
    assert orch.plan_mode is True


async def test_compaction_replaces_old_history(tmp_path: Path) -> None:
    main = FakeProvider([ChatResult(text="这是摘要"), ChatResult(text="回复你")])
    orch = Orchestrator(
        main_provider=main,
        executor_provider=FakeProvider([ChatResult(text="x")]),
        root=tmp_path,
        compact_threshold=250,
    )
    orch._messages = [
        Message(role="user", content="旧消息" + "长" * 50),
        Message(role="assistant", content="短回复"),
    ]
    await orch.start()
    try:
        orch.handle_client_message({"type": "user", "text": "新消息"})
        await _until(orch.outbox, _is_turn_end)
    finally:
        await orch.stop()

    # 第一次调用是压缩：提示词 + 序列化的旧历史
    compact_call = main.calls[0][0]
    assert "压缩为一份要点摘要" in compact_call[0].content
    assert "旧消息" in compact_call[0].content
    # 第二次调用：旧历史已被 [前情摘要] 替换，recent 保留
    second_contents = [m.content for m in main.calls[1][0]]
    assert any("[前情摘要]" in c and "这是摘要" in c for c in second_contents)
    assert not any("长长长" in c for c in second_contents)
    assert any("短回复" in c for c in second_contents)
    # 持久态同步整段替换
    assert any("[前情摘要]" in m.content for m in orch._messages)
    assert not any("长长长" in m.content for m in orch._messages)
