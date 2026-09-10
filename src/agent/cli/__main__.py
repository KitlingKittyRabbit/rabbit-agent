"""CLI 客户端：uv run python -m agent.cli

朴素输入输出循环。/connect_provider 连接服务商向导，/plan on|off 切换 plan 模式，/quit 退出。
"""

import asyncio
import getpass
import json

import websockets

from ..core.presets import PRESETS

SERVER_URL = "ws://127.0.0.1:8000/ws"


async def _receive(ws) -> None:
    async for raw in ws:
        event = json.loads(raw)
        kind = event.get("type")
        if kind == "text_delta":
            print(event["text"], end="", flush=True)
        elif kind == "task_update":
            print(f"\n[任务 #{event['id']} {event['status']}]\n{event.get('output', '')}")
        elif kind == "plan_mode":
            print(f"[plan 模式 {'开' if event['on'] else '关'}]")
        elif kind == "provider_result":
            mark = "✓" if event["ok"] else "✗"
            print(f"\n[{mark}] {event['role']}: {event['message']}")
        elif kind == "turn_end":
            print()
        elif kind == "error":
            print(f"\n[错误] {event['message']}")


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


async def main() -> None:
    try:
        async with websockets.connect(SERVER_URL) as ws:
            print("已连接 server。/connect_provider 连服务商，/plan on|off 切 plan，/quit 退出。")
            receiver = asyncio.create_task(_receive(ws))
            loop = asyncio.get_running_loop()
            try:
                while True:
                    text = (await loop.run_in_executor(None, input, "> ")).strip()
                    if not text:
                        continue
                    if text == "/quit":
                        break
                    if text == "/connect_provider":
                        await _connect_wizard(ws, loop)
                    elif text.startswith("/plan"):
                        parts = text.split()
                        on = len(parts) > 1 and parts[1] == "on"
                        await ws.send(json.dumps({"type": "set_plan_mode", "on": on}))
                    else:
                        await ws.send(json.dumps({"type": "user", "text": text}))
            finally:
                receiver.cancel()
    except OSError as e:
        print(f"无法连接 server（{e}）。请先运行: uv run python -m agent.server")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
