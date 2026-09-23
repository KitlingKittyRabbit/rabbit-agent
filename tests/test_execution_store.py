"""执行事件存储测试（Phase A）：turns / task_runs / execution_events 读写。"""

import asyncio
from pathlib import Path

from agent.core import events as ev
from agent.core.session import SessionStore


def test_turn_lifecycle(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    tid = store.create_turn("s1", "检查仓库", ev.TURN_RUNNING)
    store.finish_turn(tid, ev.TURN_COMPLETED, final_text="最终答复")

    turns = store.list_turns("s1")
    assert len(turns) == 1
    assert turns[0]["user_message"] == "检查仓库"
    assert turns[0]["status"] == "completed"
    assert turns[0]["final_text"] == "最终答复"
    assert turns[0]["started_at"] is not None
    assert turns[0]["completed_at"] is not None
    store.close()


def test_task_lifecycle(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    turn_id = store.create_turn("s1", "干活", ev.TURN_RUNNING)
    store.create_task(
        task_id=1, session_id="s1", parent_turn_id=turn_id, title="检查 provider",
        prompt="完整提示词", model="m1", provider="openai", status=ev.TASK_QUEUED,
    )
    task = store.get_task("s1", 1)
    assert task["status"] == "queued"
    assert task["title"] == "检查 provider"
    assert task["parent_turn_id"] == turn_id

    store.update_task(1, "s1", ev.TASK_RUNNING, started=True)
    assert store.get_task("s1", 1)["started_at"] is not None
    store.update_task(1, "s1", ev.TASK_DONE, final_output="完成摘要")
    task = store.get_task("s1", 1)
    assert task["status"] == "done"
    assert task["final_output"] == "完成摘要"
    assert task["completed_at"] is not None

    tasks = store.list_tasks("s1")
    assert len(tasks) == 1 and tasks[0]["model"] == "m1"
    assert store.get_task("s1", 99) is None
    store.close()


def test_update_task_timestamps_only_on_terminal(tmp_path: Path) -> None:
    """running/进度更新不写 completed_at，且不覆盖 started_at；终态写入后不再变。"""
    store = SessionStore(tmp_path / "s.db")
    store.create_task(1, "s1", None, "任务", "p", "m", "F", ev.TASK_QUEUED, max_steps=5)
    task = store.get_task("s1", 1)
    assert task["started_at"] is None and task["completed_at"] is None

    store.update_task(1, "s1", ev.TASK_RUNNING, started=True)
    started = store.get_task("s1", 1)["started_at"]
    assert started is not None
    assert store.get_task("s1", 1)["completed_at"] is None

    store.update_task(1, "s1", ev.TASK_RUNNING, steps_used=1, last_action="ls .")
    store.update_task(1, "s1", ev.TASK_RUNNING, steps_used=2, last_action="read a.py")
    task = store.get_task("s1", 1)
    assert task["completed_at"] is None  # 多次进度更新不提前写完成时间
    assert task["started_at"] == started  # 也不覆盖开始时间

    store.update_task(1, "s1", ev.TASK_DONE, final_output="完成")
    completed = store.get_task("s1", 1)["completed_at"]
    assert completed is not None

    store.update_task(1, "s1", ev.TASK_DONE, final_output="再写一次")
    task = store.get_task("s1", 1)
    assert task["completed_at"] == completed  # 完成时间不被后续更新改变
    assert task["started_at"] == started
    store.close()


def test_update_task_error_and_cancelled_set_completed(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    for task_id, status in ((1, ev.TASK_ERROR), (2, ev.TASK_CANCELLED)):
        store.create_task(task_id, "s1", None, "任务", "p", "m", "F", ev.TASK_QUEUED)
        store.update_task(task_id, "s1", ev.TASK_RUNNING, started=True)
        assert store.get_task("s1", task_id)["completed_at"] is None
        store.update_task(task_id, "s1", status)
        assert store.get_task("s1", task_id)["completed_at"] is not None
    store.close()


def test_events_query_by_turn_and_task(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.add_event("s1", ev.TURN_STARTED, 1.0, turn_id=10, actor=ev.ACTOR_MAIN)
    store.add_event("s1", ev.TOOL_STARTED, 2.0, turn_id=10, actor=ev.ACTOR_MAIN,
                    name="read_file", arguments='{"path":"a.py"}')
    store.add_event("s1", ev.TOOL_FINISHED, 3.0, turn_id=10, actor=ev.ACTOR_MAIN,
                    name="read_file", status="success", result="预览")
    store.add_event("s1", ev.SUBAGENT_TOOL_STARTED, 4.0, turn_id=10, task_id=1,
                    actor=ev.ACTOR_SUBAGENT, name="write_file")
    store.add_event("s1", ev.TURN_COMPLETED_EVENT, 5.0, turn_id=11, actor=ev.ACTOR_MAIN)

    all_events = store.list_events("s1")
    assert [e["type"] for e in all_events] == [
        "turn_started", "tool_started", "tool_finished",
        "subagent_tool_started", "turn_completed",
    ]
    turn10 = store.list_events("s1", turn_id=10)
    assert len(turn10) == 4
    task1 = store.list_events("s1", task_id=1)
    assert len(task1) == 1 and task1[0]["actor"] == "subagent"
    assert all_events[1]["arguments"] == '{"path":"a.py"}'
    assert all_events[2]["result"] == "预览"
    store.close()


def test_events_isolated_between_sessions(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db")
    store.add_event("s1", ev.TURN_STARTED, 1.0)
    store.add_event("s2", ev.TURN_STARTED, 2.0)
    assert len(store.list_events("s1")) == 1
    assert len(store.list_events("s2")) == 1
    store.close()


async def test_task_id_continues_after_restart(tmp_path: Path) -> None:
    """缺陷回归：会话内任务号必须持久单调，重启后派发不得撞 (session_id, id) 主键。"""
    from agent.core.orchestrator import Orchestrator
    from agent.providers import ChatResult, FakeProvider

    db = tmp_path / "s.db"
    first = SessionStore(db)
    first.create_session("s1", "default", "会话")
    first.create_task(1, "s1", None, "旧任务", "p", "m", "F", ev.TASK_DONE)
    assert first.max_task_id("s1") == 1
    first.close()

    orch = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=FakeProvider([ChatResult(text="干完了")]),
        root=tmp_path,
        store=SessionStore(db),
    )
    task_id = orch.conversations["s1"]._dispatcher.dispatch("接着干")
    assert task_id == 2
    await asyncio.sleep(0.2)
    run = SessionStore(db).get_task("s1", 2)
    assert run is not None and run["status"] == "done"
    await orch.stop()


def test_delete_session_cleans_all_related_tables(tmp_path: Path) -> None:
    """删除会话必须清掉 messages/turns/task_runs/execution_events，不污染其它会话。"""
    from agent.providers import Message

    store = SessionStore(tmp_path / "s.db")
    store.create_session("s1", "default", "要删的")
    store.create_session("s2", "default", "保留的")
    turn = store.create_turn("s1", "干活", ev.TURN_RUNNING)
    store.finish_turn(turn, ev.TURN_COMPLETED, final_text="完成")
    store.create_task(1, "s1", turn, "任务", "p", "m", "F", ev.TASK_DONE)
    store.add_event("s1", ev.TURN_STARTED, 1.0, turn_id=turn, actor=ev.ACTOR_MAIN)
    store.replace("s1", [Message(role="user", content="要删的消息")])
    store.replace("s2", [Message(role="user", content="要保留的消息")])

    store.delete_session("s1")

    assert store.get_task("s1", 1) is None
    assert store.list_turns("s1") == []
    assert store.list_events("s1") == []
    assert store.load("s1") == []
    assert [m.content for m in store.load("s2")] == ["要保留的消息"]
    assert [s["id"] for s in store.list_sessions()] == ["s2"]
    store.close()


def test_task_runs_progress_columns_migration(tmp_path: Path) -> None:
    """旧库 task_runs 无进度列：打开时向后兼容补列，旧数据保留且可写新字段。"""
    import sqlite3

    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, project_id TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
        CREATE TABLE messages (idx INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', tool_calls TEXT,
            tool_call_id TEXT);
        CREATE TABLE task_runs (id INTEGER NOT NULL, session_id TEXT NOT NULL,
            parent_turn_id INTEGER, title TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, created_at REAL NOT NULL,
            started_at REAL, completed_at REAL, final_output TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (session_id, id));
        INSERT INTO task_runs (id, session_id, title, status, created_at)
            VALUES (1, 's1', '旧任务', 'done', 1.0);
        """
    )
    conn.commit()
    conn.close()

    store = SessionStore(db)
    task = store.get_task("s1", 1)
    assert task["title"] == "旧任务"
    assert task["steps_used"] == 0 and task["max_steps"] == 0
    assert task["last_action"] == "" and task["stop_reason"] == ""

    store.update_task(
        1, "s1", ev.TASK_DONE, steps_used=5, last_action="run_shell pytest", stop_reason="max_steps"
    )
    task = store.get_task("s1", 1)
    assert (task["steps_used"], task["last_action"], task["stop_reason"]) == (
        5, "run_shell pytest", "max_steps"
    )
    assert store.list_tasks("s1")[0]["steps_used"] == 5
    store.close()


def test_actions_used_increment_and_migration(tmp_path: Path) -> None:
    """任务动作数（子代理工具调用次数）走 DB 单一来源，可增量累加。"""
    store = SessionStore(tmp_path / "s.db")
    store.create_task(1, "s1", None, "任务", "p", "m", "F", ev.TASK_QUEUED, max_steps=100)
    assert store.get_task("s1", 1)["actions_used"] == 0
    store.update_task(1, "s1", ev.TASK_RUNNING, started=True, actions_used_delta=1)
    store.update_task(1, "s1", ev.TASK_RUNNING, last_action="ls .", actions_used_delta=1)
    task = store.get_task("s1", 1)
    assert task["actions_used"] == 2
    assert task["steps_used"] == 0  # 动作与步数分开
    assert store.list_tasks("s1")[0]["actions_used"] == 2
    store.close()
