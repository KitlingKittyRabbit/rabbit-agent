"""CLI 测试：向导、命令解析、事件渲染、自动拉起（input/getpass/websocket/subprocess 全 mock）。"""

import asyncio
import builtins
import getpass
import json

import agent.cli.__main__ as cli
from agent.core.presets import PRESETS

SESSION_LIST = {"type": "session_list", "sessions": [{"id": "s1", "title": "default"}]}


class FakeWS:
    def __init__(self, events: list[dict] | None = None) -> None:
        self.sent: list[dict] = []
        self._events = events or []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def __aenter__(self) -> "FakeWS":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield json.dumps(event)


class FakeConnect:
    """可 await 可 async with 的 connect 假件（对齐 websockets.connect 的双重用法）。"""

    def __init__(self, ws: FakeWS) -> None:
        self._ws = ws

    def __await__(self):
        async def _():
            return self._ws

        return _().__await__()

    async def __aenter__(self) -> FakeWS:
        return self._ws

    async def __aexit__(self, *args) -> bool:
        return False


def run_inputs(monkeypatch, inputs: list[str], key: str = "sk-x") -> None:
    it = iter(inputs)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": key)


def preset_index(name: str) -> str:
    return str(list(PRESETS).index(name) + 1)


# ---------- 连接向导 ----------


async def test_wizard_preset_path(monkeypatch) -> None:
    ws = FakeWS()
    run_inputs(monkeypatch, ["1", preset_index("deepseek"), ""])
    await cli._connect_wizard(ws, asyncio.get_running_loop())
    assert ws.sent == [
        {
            "type": "connect_provider",
            "role": "main",
            "protocol": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key": "sk-x",
        }
    ]


async def test_wizard_custom_path(monkeypatch) -> None:
    ws = FakeWS()
    custom = str(len(PRESETS) + 1)
    run_inputs(monkeypatch, ["2", custom, "openai", "http://x/v1", "my-model"], key="sk-y")
    await cli._connect_wizard(ws, asyncio.get_running_loop())
    assert ws.sent == [
        {
            "type": "connect_provider",
            "role": "executor",
            "protocol": "openai",
            "base_url": "http://x/v1",
            "model": "my-model",
            "api_key": "sk-y",
        }
    ]


async def test_wizard_invalid_choice_aborts(monkeypatch) -> None:
    ws = FakeWS()
    run_inputs(monkeypatch, ["99"])
    await cli._connect_wizard(ws, asyncio.get_running_loop())
    assert ws.sent == []


async def test_wizard_no_key_preset_skips_getpass(monkeypatch) -> None:
    ws = FakeWS()
    it = iter(["2", preset_index("ollama"), ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))
    monkeypatch.setattr(
        getpass, "getpass", lambda prompt="": (_ for _ in ()).throw(AssertionError("不应问 key"))
    )
    await cli._connect_wizard(ws, asyncio.get_running_loop())
    assert ws.sent[0]["api_key"] == "unused"
    assert ws.sent[0]["role"] == "executor"


async def test_wizard_empty_key_aborts(monkeypatch) -> None:
    ws = FakeWS()
    run_inputs(monkeypatch, ["1", preset_index("deepseek"), ""], key="   ")
    await cli._connect_wizard(ws, asyncio.get_running_loop())
    assert ws.sent == []


# ---------- 主循环命令 ----------


async def run_main(monkeypatch, inputs: list[str], events: list[dict] | None = None) -> FakeWS:
    ws = FakeWS(events=[SESSION_LIST, *(events or [])])
    run_inputs(monkeypatch, inputs)
    monkeypatch.setattr(cli.websockets, "connect", lambda url: FakeConnect(ws))
    await cli.main()
    return ws


async def test_main_loop_parses_commands(monkeypatch) -> None:
    ws = await run_main(monkeypatch, ["/plan on", "你好", "/quit"])
    assert {"type": "set_plan_mode", "on": True} in ws.sent
    assert {"type": "user", "session": "s1", "text": "你好"} in ws.sent


async def test_main_loop_session_commands(monkeypatch) -> None:
    events = [{"type": "session_created", "session": "s2", "title": "新会话"}]
    ws = await run_main(monkeypatch, ["/new", "/switch s1", "/stop", "/quit"], events)
    assert {"type": "new_session", "title": ""} in ws.sent
    assert {"type": "stop", "session": "s1"} in ws.sent


async def test_main_loop_confirm_commands(monkeypatch) -> None:
    ws = await run_main(monkeypatch, ["/allow abc123", "/deny def456", "/quit"])
    assert {"type": "confirm_response", "id": "abc123", "allow": True} in ws.sent
    assert {"type": "confirm_response", "id": "def456", "allow": False} in ws.sent


async def test_multiline_continuation(monkeypatch) -> None:
    ws = await run_main(monkeypatch, ["第一行\\", "第二行", "/quit"])
    assert {"type": "user", "session": "s1", "text": "第一行\n第二行"} in ws.sent


# ---------- 事件渲染 ----------


async def test_receive_renders_events(capsys) -> None:
    state = cli.CliState()
    state.current = "s1"
    events = [
        {"type": "text_delta", "session": "s1", "text": "你好"},
        {"type": "text_delta", "session": "other", "text": "不应显示"},
        {"type": "task_update", "session": "s1", "id": 3, "status": "done", "output": "产出"},
        {"type": "task_question", "session": "s1", "id": 3, "question": "用哪个？"},
        {"type": "provider_result", "role": "main", "ok": True, "message": "连接成功"},
        {"type": "confirm_request", "session": "s1", "id": "abc", "command": "rm -rf x"},
        {"type": "usage", "session": "s1", "input": 10, "output": 5},
        {"type": "context", "session": "s1", "chars": 1000, "threshold": 200000},
        {"type": "compacted", "session": "s1", "before": 5000, "after": 800},
        {"type": "stopped", "session": "s1"},
        {"type": "plan_mode", "on": True},
        {"type": "error", "session": "s1", "message": "炸了"},
        {"type": "turn_end", "session": "s1"},
    ]
    await cli._receive(FakeWS(events), state)
    out = capsys.readouterr().out
    assert "你好" in out
    assert "不应显示" not in out
    assert "[任务 #3 done]" in out and "产出" in out
    assert "用哪个？" in out
    assert "✓" in out
    assert "rm -rf x" in out and "/allow abc" in out
    assert "in 10 out 5" in out and "累计" in out
    assert "上下文 1%" in out or "上下文 0%" in out or "上下文 1" in out
    assert "5000 → 800" in out
    assert "已中断" in out
    assert "plan 模式 开" in out
    assert "炸了" in out


# ---------- 自动拉起 ----------


async def test_connect_with_spawn_when_server_down(monkeypatch) -> None:
    spawned = []

    class FakeProc:
        def terminate(self):
            pass

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: spawned.append(1) or FakeProc())
    attempts = {"n": 0}
    ws = FakeWS()

    def fake_connect(url):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("refused")
        return FakeConnect(ws)

    monkeypatch.setattr(cli.websockets, "connect", fake_connect)
    monkeypatch.setattr(cli.asyncio, "sleep", lambda t: asyncio.sleep(0))

    result_ws, proc = await cli._connect_with_spawn()
    assert result_ws is ws
    assert spawned == [1]  # 拉起了一次
    assert proc is not None


async def test_connect_directly_when_server_running(monkeypatch) -> None:
    spawned = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: spawned.append(1) or None)
    ws = FakeWS()
    monkeypatch.setattr(cli.websockets, "connect", lambda url: FakeConnect(ws))

    result_ws, proc = await cli._connect_with_spawn()
    assert result_ws is ws
    assert spawned == []  # 未拉起
    assert proc is None
