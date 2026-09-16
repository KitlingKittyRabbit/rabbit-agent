"""CLI 测试：向导、命令解析、事件渲染、自动拉起（input/getpass/websocket/subprocess 全 mock）。"""

import asyncio
import builtins
import getpass
import json
from pathlib import Path

import pytest

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
    monkeypatch.setattr(cli.websockets, "connect", lambda url, **kw: FakeConnect(ws))
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: None)
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
        def poll(self):
            return None  # 仍在运行

        def terminate(self):
            pass

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: spawned.append(1) or FakeProc())
    attempts = {"n": 0}
    ws = FakeWS()

    def fake_connect(url, **kw):
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
    monkeypatch.setattr(cli.websockets, "connect", lambda url, **kw: FakeConnect(ws))

    result_ws, proc = await cli._connect_with_spawn()
    assert result_ws is ws
    assert spawned == []  # 未拉起
    assert proc is None


async def test_connect_with_spawn_port_occupied_by_other_service(monkeypatch) -> None:
    """用户报错回归：持续握手失败 → 明确 SystemExit、不拉起（重试计数打零加速）。"""
    import websockets.exceptions

    spawned = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: spawned.append(1) or None)
    monkeypatch.setattr(cli, "_OCCUPIED_RETRIES", 2)
    monkeypatch.setattr(cli, "_RETRY_INTERVAL", 0)

    def fake_connect(url, **kw):
        raise websockets.exceptions.InvalidMessage("did not receive a valid HTTP response")

    monkeypatch.setattr(cli.websockets, "connect", fake_connect)
    with pytest.raises(SystemExit) as exc_info:
        await cli._connect_with_spawn()
    assert "持续被占用" in str(exc_info.value)
    assert spawned == []  # 端口被占时不应拉起（拉起也绑不上）


async def test_transient_handshake_failure_recovers(monkeypatch) -> None:
    """瞬态握手失败（进程死亡窗口）应重试恢复，不误报占用。"""
    import websockets.exceptions

    monkeypatch.setattr(cli, "_RETRY_INTERVAL", 0)
    ws = FakeWS()
    attempts = {"n": 0}

    def fake_connect(url, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise websockets.exceptions.InvalidMessage("transient")
        return FakeConnect(ws)

    spawned = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: spawned.append(1) or None)
    monkeypatch.setattr(cli.websockets, "connect", fake_connect)

    result_ws, proc = await cli._connect_with_spawn()
    assert result_ws is ws
    assert spawned == []  # 瞬态恢复，未拉起


class _Response403:
    status_code = 403


async def test_spawn_uses_fresh_token_after_rewrite(tmp_path, monkeypatch) -> None:
    """缺陷回归：自动拉起后 server 重写 ~/.agent_token，CLI 必须用新 token 才连得上。"""
    import websockets.exceptions

    token_file = tmp_path / "token"
    token_file.write_text("old-token", encoding="utf-8")
    monkeypatch.setattr(cli, "TOKEN_PATH", str(token_file))
    monkeypatch.setattr(cli, "_RETRY_INTERVAL", 0)

    class FakeProc:
        def poll(self):
            return None  # 仍在运行

        def terminate(self):
            pass

    def fake_popen(*args, **kwargs):
        token_file.write_text("new-token", encoding="utf-8")  # 模拟 server 启动重写 token
        return FakeProc()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    urls: list[str] = []
    ws = FakeWS()

    def fake_connect(url, **kw):
        urls.append(url)
        if len(urls) == 1:
            raise OSError("refused")  # 拉起前无人监听
        if "token=new-token" not in url:
            raise websockets.exceptions.InvalidStatus(_Response403())  # 旧 token 403
        return FakeConnect(ws)

    monkeypatch.setattr(cli.websockets, "connect", fake_connect)

    result_ws, proc = await cli._connect_with_spawn("ws://127.0.0.1:8471/ws?token=old-token")
    assert result_ws is ws
    assert proc is not None
    assert urls[-1] == "ws://127.0.0.1:8471/ws?token=new-token"


async def test_stale_token_403_bounded(tmp_path, monkeypatch) -> None:
    """旧 token 持续 403：必须有界重试并给出占用提示，不得无限重试/误拉 server。"""
    import websockets.exceptions

    token_file = tmp_path / "token"
    token_file.write_text("stale-token", encoding="utf-8")
    monkeypatch.setattr(cli, "TOKEN_PATH", str(token_file))
    monkeypatch.setattr(cli, "_OCCUPIED_RETRIES", 2)
    monkeypatch.setattr(cli, "_RETRY_INTERVAL", 0)

    spawned = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: spawned.append(1) or None)
    calls = {"n": 0}

    def fake_connect(url, **kw):
        calls["n"] += 1
        raise websockets.exceptions.InvalidStatus(_Response403())

    monkeypatch.setattr(cli.websockets, "connect", fake_connect)

    with pytest.raises(SystemExit) as exc_info:
        await cli._connect_with_spawn()
    assert "持续被占用" in str(exc_info.value)
    assert calls["n"] == 3  # 1 次初连 + _OCCUPIED_RETRIES，有界
    assert spawned == []


def _write_config(tmp_path: Path, monkeypatch, port: int = 8471) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[agent]\nport = {port}\n\n[main]\nprotocol = "openai"\nmodel = "m"\n\n'
        '[executor]\nprotocol = "openai"\nmodel = "m"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_CONFIG", str(config))


def test_server_url_reads_config_port(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "TOKEN_PATH", str(tmp_path / "no-token"))
    assert cli._server_url() == "ws://127.0.0.1:8471/ws"


def test_server_url_appends_token_when_present(tmp_path: Path, monkeypatch) -> None:
    """有 ~/.agent_token 时必须带上（否则被 Phase F 鉴权拒掉）。"""
    _write_config(tmp_path, monkeypatch)
    token_file = tmp_path / "token"
    token_file.write_text("abc123\n", encoding="utf-8")
    monkeypatch.setattr(cli, "TOKEN_PATH", str(token_file))
    assert cli._server_url() == "ws://127.0.0.1:8471/ws?token=abc123"


def test_read_token_missing_unreadable_empty(tmp_path: Path, monkeypatch) -> None:
    """token 尚未生成、读失败（目录）、空内容、非法编码都视为无 token，不抛异常。"""
    monkeypatch.setattr(cli, "TOKEN_PATH", str(tmp_path / "missing"))
    assert cli._read_token() is None
    directory = tmp_path / "dir"
    directory.mkdir()
    monkeypatch.setattr(cli, "TOKEN_PATH", str(directory))
    assert cli._read_token() is None
    empty = tmp_path / "empty"
    empty.write_text("  \n", encoding="utf-8")
    monkeypatch.setattr(cli, "TOKEN_PATH", str(empty))
    assert cli._read_token() is None
    garbage = tmp_path / "garbage"
    garbage.write_bytes(b"\xff\xfe\x00garbage")
    monkeypatch.setattr(cli, "TOKEN_PATH", str(garbage))
    assert cli._read_token() is None


def test_connect_url_rereads_token(tmp_path: Path, monkeypatch) -> None:
    """每次调用都重读 token（自动拉起后 server 会换 token）。"""
    token_file = tmp_path / "token"
    token_file.write_text("one", encoding="utf-8")
    monkeypatch.setattr(cli, "TOKEN_PATH", str(token_file))
    assert cli._connect_url("ws://127.0.0.1:8471/ws") == "ws://127.0.0.1:8471/ws?token=one"
    token_file.write_text("two", encoding="utf-8")
    assert cli._connect_url("ws://127.0.0.1:8471/ws?token=one") == (
        "ws://127.0.0.1:8471/ws?token=two"
    )


async def test_spawned_server_exits_immediately_reports(monkeypatch) -> None:
    """拉起的 server 启动即死 → 明确报'已退出'，不与外人占用混淆（审核缺口补测）。"""
    spawned = []

    class DyingProc:
        returncode = 1

        def poll(self):
            return 1  # 已退出

        def terminate(self):
            pass

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: spawned.append(1) or DyingProc())

    def fake_connect(url, **kw):
        raise OSError("refused")

    monkeypatch.setattr(cli.websockets, "connect", fake_connect)
    with pytest.raises(SystemExit) as exc_info:
        await cli._connect_with_spawn()
    assert "已退出" in str(exc_info.value)
    assert spawned == [1]


def test_server_url_fallback_default(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CONFIG", "/nonexistent/config.toml")
    assert cli._server_url() == cli.DEFAULT_SERVER_URL


async def test_wizard_custom_allows_empty_key(monkeypatch) -> None:
    """自定义端点 key 可留空（本地/无鉴权），由后端判定；不再直接取消。"""
    ws = FakeWS()
    custom = str(len(PRESETS) + 1)
    run_inputs(monkeypatch, ["2", custom, "openai", "http://127.0.0.1:9/v1", "m"], key="")
    await cli._connect_wizard(ws, asyncio.get_running_loop())
    assert ws.sent[0]["api_key"] == ""
    assert ws.sent[0]["base_url"] == "http://127.0.0.1:9/v1"
