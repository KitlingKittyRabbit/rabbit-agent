"""CLI 客户端：uv run python -m agent.cli

未检测到 server 时自动拉起（退出时关闭自己拉起的进程）。
命令：
  /connect_provider   连接服务商向导
  /plan on|off        plan 模式（全局只读）
  /new [标题]         开新会话并切换
  /sessions           会话列表
  /switch <id>        切换会话
  /stop               中断当前会话的 turn 与全部 subagent
  /allow|/deny <id>   危险命令确认
  /quit               退出
行尾 \\ 续行可多行输入。
"""

import asyncio
import atexit
import getpass
import json
import os
import readline
import subprocess
import sys
from pathlib import Path

import websockets
from websockets.exceptions import WebSocketException

from ..core.config import load_config
from ..core.presets import PRESETS

DEFAULT_SERVER_URL = "ws://127.0.0.1:8000/ws"
HISTORY_PATH = Path(".agent_cli_history")
HELP = """命令：
  /connect_provider   连接服务商向导
  /plan on|off        plan 模式（全局只读）
  /new [标题]         开新会话并切换
  /sessions           会话列表
  /switch <id>        切换会话
  /stop               中断当前会话
  /allow|/deny <id>   危险命令确认
  /quit               退出
行尾 \\ 续行可多行输入。"""


class CliState:
    def __init__(self) -> None:
        self.current: str | None = None
        self.sessions: list[dict] = []
        self.session_list_ready = asyncio.Event()
        self.want_new = False
        self.tokens_in = 0
        self.tokens_out = 0


async def _receive(ws, state: CliState) -> None:
    async for raw in ws:
        event = json.loads(raw)
        kind = event.get("type")
        if kind == "session_list":
            state.sessions = event["sessions"]
            state.session_list_ready.set()
            continue
        if kind == "session_created":
            state.sessions.append({"id": event["session"], "title": event["title"]})
            if state.want_new:
                state.want_new = False
                state.current = event["session"]
                print(f"\n[已切换新会话 {event['session']}]")
            continue
        session = event.get("session")
        if session is not None and state.current is not None and session != state.current:
            continue  # 其他会话的事件不显示
        if kind == "text_delta":
            print(event["text"], end="", flush=True)
        elif kind == "task_update":
            print(f"\n[任务 #{event['id']} {event['status']}]\n{event.get('output', '')}")
        elif kind == "task_question":
            print(f"\n[任务 #{event['id']} 提问] {event['question']}")
        elif kind == "plan_mode":
            print(f"[plan 模式 {'开' if event['on'] else '关'}]")
        elif kind == "provider_result":
            mark = "✓" if event["ok"] else "✗"
            print(f"\n[{mark}] {event['role']}: {event['message']}")
        elif kind == "confirm_request":
            print(f"\n[危险命令] {event['command']}")
            print(f"  /allow {event['id']} 允许 | /deny {event['id']} 拒绝（120 秒不答自动拒绝）")
        elif kind == "usage":
            state.tokens_in += event["input"]
            state.tokens_out += event["output"]
            print(
                f"\n[tokens: in {event['input']} out {event['output']}"
                f" | 累计 in {state.tokens_in} out {state.tokens_out}]"
            )
        elif kind == "context":
            pct = round(event["chars"] / max(1, event["threshold"]) * 100)
            print(f"[上下文 {pct}%]", end=" ")
        elif kind == "compacted":
            print(f"\n[上下文已压缩: {event['before']} → {event['after']} 字符]")
        elif kind == "stopped":
            print("\n[已中断]")
        elif kind == "turn_end":
            print()
        elif kind == "error":
            print(f"\n[错误] {event['message']}")


def _server_url() -> str:
    """从 config.toml 读端口（与 server 同源）；读不到回落 8000。"""
    try:
        config = load_config(os.environ.get("AGENT_CONFIG", "config.toml"))
        return f"ws://127.0.0.1:{config.port}/ws"
    except Exception:
        return DEFAULT_SERVER_URL


async def _connect_with_spawn(url: str = DEFAULT_SERVER_URL):
    """先直连；连不上则自动拉起 server 并重试。返回 (ws, 子进程或 None)。

    OSError（无人监听）→ 自动拉起；
    WebSocketException（端口被非本 agent 的服务占用）→ 明确报错而非栈 trace。
    """
    try:
        return await websockets.connect(url), None
    except OSError:
        pass
    except WebSocketException as e:
        raise SystemExit(
            f"端口被占用且不是本 agent 的 server（{type(e).__name__}）。请关闭占用程序后重试。"
        ) from e
    log = open("agent_server.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent.server"], stdout=log, stderr=subprocess.STDOUT
    )
    log.close()  # 子进程已继承 fd，父进程副本即刻关闭
    for _ in range(60):
        try:
            ws = await websockets.connect(url)
            print("（已自动拉起 server）")
            return ws, proc
        except OSError:
            await asyncio.sleep(0.25)
        except WebSocketException as e:
            proc.terminate()
            raise SystemExit(f"端口被占用且不是本 agent 的 server（{type(e).__name__}）") from e
    proc.terminate()
    raise SystemExit("server 自动拉起失败，请查看 agent_server.log")


def _setup_readline() -> None:
    try:
        if HISTORY_PATH.exists():
            readline.read_history_file(str(HISTORY_PATH))
        atexit.register(lambda: readline.write_history_file(str(HISTORY_PATH)))
    except OSError:
        pass


async def _ask(loop, prompt: str) -> str:
    return (await loop.run_in_executor(None, input, prompt)).strip()


async def _pick(loop, options: list[str]) -> str | None:
    raw = await _ask(loop, "> ")
    if not raw.isdigit() or not (1 <= int(raw) <= len(options)):
        print("无效选择，已取消。")
        return None
    return options[int(raw) - 1]


async def _connect_wizard(ws, loop) -> None:
    print("角色？")
    roles = ["main", "executor"]
    for i, name in enumerate(roles, 1):
        print(f"  {i}) {name}")
    role = await _pick(loop, roles)
    if role is None:
        return

    names = list(PRESETS) + ["custom"]
    print("服务商？")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {name}")
    choice = await _pick(loop, names)
    if choice is None:
        return

    if choice == "custom":
        protocol = await _ask(loop, "protocol（openai/anthropic）: ")
        base_url = (await _ask(loop, "base_url: ")) or None
        model = await _ask(loop, "model: ")
        needs_key = True
    else:
        preset = PRESETS[choice]
        protocol = preset["protocol"]
        base_url = preset["base_url"]
        model = (await _ask(loop, f"model [{preset['model']}]: ")) or preset["model"]
        needs_key = preset["needs_key"]

    if needs_key:
        api_key = (await loop.run_in_executor(None, getpass.getpass, "API key（不回显）: ")).strip()
        if not api_key:
            print("key 为空，已取消。")
            return
    else:
        api_key = "unused"

    message = {
        "type": "connect_provider",
        "role": role,
        "protocol": protocol,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
    }
    await ws.send(json.dumps(message))


async def _input_line(loop) -> str:
    text = (await loop.run_in_executor(None, input, "> ")).rstrip("\n")
    while text.endswith("\\"):
        more = await loop.run_in_executor(None, input, "| ")
        text = text[:-1] + "\n" + more
    return text.strip()


async def main() -> None:
    _setup_readline()
    try:
        ws, spawned = await _connect_with_spawn(_server_url())
    except OSError as e:
        print(f"无法连接 server（{e}）。")
        return
    try:
        async with ws:
            await _chat_loop(ws)
    finally:
        # /quit、Ctrl+C 都要关闭自己拉起的 server（不碰别人已在跑的）
        if spawned is not None:
            spawned.terminate()
            print("（已关闭自动拉起的 server）")


async def _chat_loop(ws) -> None:
    state = CliState()
    receiver = asyncio.create_task(_receive(ws, state))
    loop = asyncio.get_running_loop()
    await ws.send(json.dumps({"type": "list_sessions"}))
    await state.session_list_ready.wait()
    if state.sessions:
        state.current = state.sessions[0]["id"]
    print(f"已连接（会话 {state.current}）。/help 查看命令。")
    try:
        while True:
            text = await _input_line(loop)
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/help":
                print(HELP)
            elif text == "/connect_provider":
                await _connect_wizard(ws, loop)
            elif text.startswith("/plan"):
                parts = text.split()
                on = len(parts) > 1 and parts[1] == "on"
                await ws.send(json.dumps({"type": "set_plan_mode", "on": on}))
            elif text.startswith("/new"):
                state.want_new = True
                await ws.send(json.dumps({"type": "new_session", "title": text[4:].strip()}))
            elif text == "/sessions":
                state.session_list_ready.clear()
                await ws.send(json.dumps({"type": "list_sessions"}))
                await state.session_list_ready.wait()
                for s in state.sessions:
                    mark = "*" if s["id"] == state.current else " "
                    print(f"{mark} {s['id']}  {s['title']}")
            elif text.startswith("/switch"):
                target = text[7:].strip()
                if any(s["id"] == target for s in state.sessions):
                    state.current = target
                    print(f"[已切换到 {target}]")
                else:
                    print(f"会话不存在: {target}（/sessions 查看）")
            elif text == "/stop":
                await ws.send(json.dumps({"type": "stop", "session": state.current}))
            elif text.startswith(("/allow", "/deny")):
                parts = text.split()
                if len(parts) == 2:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "confirm_response",
                                "id": parts[1],
                                "allow": parts[0] == "/allow",
                            }
                        )
                    )
                else:
                    print("用法: /allow <id> 或 /deny <id>")
            else:
                await ws.send(json.dumps({"type": "user", "session": state.current, "text": text}))
    finally:
        receiver.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
