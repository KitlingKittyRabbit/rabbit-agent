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
import webbrowser
from pathlib import Path

import websockets
from websockets.exceptions import WebSocketException

from ..core.config import load_config
from ..core.presets import PRESETS

DEFAULT_SERVER_URL = "ws://127.0.0.1:8000/ws"
TOKEN_PATH = os.path.expanduser("~/.agent_token")
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


def _web_url(ws_url: str) -> str:
    return ws_url.replace("ws://", "http://").rsplit("/ws", 1)[0]


def _open_browser(url: str) -> None:
    """尽力打开浏览器；无显示环境（SSH/服务器）静默跳过。"""
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _read_token() -> str | None:
    """读 ~/.agent_token；尚未生成/读失败/空内容都返回 None。"""
    try:
        token = Path(TOKEN_PATH).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return token or None


def _with_token(url: str, token: str | None) -> str:
    base = url.split("?", 1)[0]
    return f"{base}?token={token}" if token else base


def _connect_url(url: str) -> str:
    """每次连接前重读 token：自动拉起的 server 启动时会重写 ~/.agent_token。"""
    return _with_token(url, _read_token())


def _server_url() -> str:
    """从 config.toml 读端口（与 server 同源），附带 ~/.agent_token；读不到回落 8000。"""
    try:
        config = load_config(os.environ.get("AGENT_CONFIG", "config.toml"))
        base = f"ws://127.0.0.1:{config.port}/ws"
    except Exception:
        return DEFAULT_SERVER_URL
    return _with_token(base, _read_token())


_OCCUPIED_RETRIES = 40  # 握手失败重试次数（约 10 秒窗口，覆盖瞬态/慢启动/代理抽风）
_RETRY_INTERVAL = 0.25


async def _try_connect(url: str):
    """返回 (ws, transient)：成功 (ws, False)；无人监听 (None, False)；握手失败 (None, True)。

    proxy=None：显式禁用 websockets 的系统代理探测，本机服务必须直连。
    """
    try:
        return await websockets.connect(url, proxy=None), False
    except OSError:
        return None, False
    except WebSocketException:
        return None, True


async def _connect_with_spawn(url: str = DEFAULT_SERVER_URL):
    """先直连；连不上则自动拉起 server 并重试。返回 (ws, 子进程或 None)。

    每次尝试都重读 ~/.agent_token：自动拉起的 server 启动时会重写 token，
    继续用旧 token 会被 403 拒绝（本函数按时间窗有界重试）。
    """
    base = url.split("?", 1)[0]
    ws, transient = await _try_connect(_connect_url(base))
    if transient:
        # 端口上有东西但握手失败：瞬态（死亡窗口/慢启动/代理/换 token）与真占用用时间窗区分
        for _ in range(_OCCUPIED_RETRIES):
            await asyncio.sleep(_RETRY_INTERVAL)
            ws, transient = await _try_connect(_connect_url(base))
            if not transient:
                break
        if transient:
            port = base.split(":")[2].split("/")[0]
            raise SystemExit(
                f"端口 {port} 持续被占用且握手失败（非本 agent 的 server）。"
                f"请运行 ss -tlnp | grep :{port} 确认占用者后重试。"
            )
    if ws is not None:
        return ws, None
    log = open("agent_server.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent.server"], stdout=log, stderr=subprocess.STDOUT
    )
    log.close()  # 子进程已继承 fd，父进程副本即刻关闭
    for _ in range(60):
        if proc.poll() is not None:
            raise SystemExit(
                f"自动拉起的 server 已退出（code {proc.returncode}），请查看 agent_server.log"
            )
        ws, _ = await _try_connect(_connect_url(base))
        if ws is not None:
            print("（已自动拉起 server）")
            return ws, proc
        await asyncio.sleep(_RETRY_INTERVAL)
    proc.terminate()
    raise SystemExit("server 自动拉起失败，请查看 agent_server.log")


_HISTORY_LIMIT = 500


def _setup_readline() -> None:
    try:
        readline.set_history_length(_HISTORY_LIMIT)  # 上限防历史文件病态膨胀
        if HISTORY_PATH.exists():
            readline.read_history_file(str(HISTORY_PATH))
        atexit.register(lambda: readline.write_history_file(str(HISTORY_PATH)))
    except OSError:
        pass


async def _ask(loop, prompt: str) -> str:
    try:
        return (await loop.run_in_executor(None, input, prompt)).strip()
    except EOFError:
        return ""


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
        needs_key = False
        optional_key = True  # 自定义端点 key 可留空（本地/无鉴权），由后端判定
    else:
        preset = PRESETS[choice]
        protocol = preset["protocol"]
        base_url = preset["base_url"]
        model = (await _ask(loop, f"model [{preset['model']}]: ")) or preset["model"]
        needs_key = preset["needs_key"]
        optional_key = False

    if needs_key or optional_key:
        api_key = (await loop.run_in_executor(None, getpass.getpass, "API key（不回显）: ")).strip()
        if not api_key and needs_key:
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
    try:
        text = (await loop.run_in_executor(None, input, "> ")).rstrip("\n")
    except EOFError:
        return "/quit"  # stdin 关闭（Ctrl+D / 管道结束）视为退出
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
    web_url = _web_url(_server_url())
    print(f"web 界面: {web_url}（正在打开浏览器…）")
    _open_browser(web_url)
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
