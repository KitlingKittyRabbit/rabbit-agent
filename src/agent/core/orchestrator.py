"""编排器：主 agent + 派发器 + 事件驱动循环，是 server 的唯一对话对象。

输入模型：用户消息与任务完成事件进同一个 inbox 队列；driver 逐批处理并运行主 agent loop。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path

from ..providers import Message, Provider
from ..tools import build_registry
from .config import make_provider
from .context import COMPACT_PROMPT, SUMMARY_PREFIX, estimate_chars, serialize_for_summary
from .dispatch import Dispatcher, SubtaskEvent
from .loop import AgentLoop
from .prompts import SUBAGENT_SYSTEM, build_main_system
from .provider_store import save_provider
from .session import SessionStore


class Orchestrator:
    def __init__(
        self,
        *,
        main_provider: Provider | None,
        executor_provider: Provider | None,
        root: str | Path,
        session: SessionStore | None = None,
        plan_mode: bool = False,
        max_steps_main: int = 50,
        max_steps_executor: int = 30,
        store_path: str | Path | None = None,
        provider_factory: Callable[..., Provider] = make_provider,
        compact_threshold: int = 200_000,
    ) -> None:
        self._main_provider = main_provider
        self._executor_provider = executor_provider
        self._root = Path(root)
        self._session = session
        self.plan_mode = plan_mode
        self._max_steps_main = max_steps_main
        self._max_steps_executor = max_steps_executor
        self._store_path = Path(store_path) if store_path is not None else None
        self._provider_factory = provider_factory
        self._compact_threshold = compact_threshold

        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()
        self._dispatcher = Dispatcher(spawn=self._spawn_subagent, on_event=self._on_subtask_event)

        # 主 agent：只读工具 + 派发工具。写工具的物理缺席即"写操作强制走 subagent"。
        self._main_tools = build_registry(self._root, write=False, shell=False)
        self._main_tools.add(self._dispatcher.make_tool())

        self._messages: list[Message] = session.load() if session else []
        self._system = build_main_system(self._load_discipline())
        self._driver_task: asyncio.Task | None = None

    def _load_discipline(self) -> str | None:
        path = self._root / "AGENTS.md"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    async def start(self) -> None:
        self._driver_task = asyncio.create_task(self._driver())

    async def stop(self) -> None:
        if self._driver_task is not None:
            self._driver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._driver_task
        if self._session is not None:
            self._session.close()

    def handle_client_message(self, data: dict) -> None:
        msg_type = data.get("type")
        if msg_type == "user":
            text = (data.get("text") or "").strip()
            if text:
                self._inbox.put_nowait(Message(role="user", content=text))
        elif msg_type == "set_plan_mode":
            self.plan_mode = bool(data.get("on"))
            self.outbox.put_nowait({"type": "plan_mode", "on": self.plan_mode})
        elif msg_type == "connect_provider":
            asyncio.get_running_loop().create_task(self._connect(data))

    async def _connect(self, data: dict) -> None:
        result = await self.connect_provider(
            str(data.get("role", "")),
            protocol=str(data.get("protocol", "")),
            base_url=data.get("base_url"),
            model=str(data.get("model", "")),
            api_key=str(data.get("api_key", "")),
        )
        self.outbox.put_nowait(result)

    async def connect_provider(
        self,
        role: str,
        *,
        protocol: str,
        base_url: str | None,
        model: str,
        api_key: str,
    ) -> dict:
        """验证连接成功后才持久化并热切换；失败不保存、原样报错。"""
        if role not in ("main", "executor"):
            return {
                "type": "provider_result",
                "role": role,
                "ok": False,
                "message": f"未知角色: {role}（可选: main / executor）",
            }
        try:
            candidate = self._provider_factory(
                protocol=protocol, base_url=base_url, model=model, api_key=api_key
            )
        except Exception as e:
            return {
                "type": "provider_result",
                "role": role,
                "ok": False,
                "message": f"配置无效（未保存）: {e}",
            }
        try:
            await candidate.chat([Message(role="user", content="ping")])
        except Exception as e:
            return {
                "type": "provider_result",
                "role": role,
                "ok": False,
                "message": f"连接失败（未保存）: {type(e).__name__}: {e}",
            }
        if self._store_path is not None:
            save_provider(
                self._store_path,
                role,
                {"protocol": protocol, "base_url": base_url, "model": model, "api_key": api_key},
            )
        if role == "main":
            self._main_provider = candidate
        else:
            self._executor_provider = candidate
        return {
            "type": "provider_result",
            "role": role,
            "ok": True,
            "message": "连接成功，已切换并保存",
        }

    async def _driver(self) -> None:
        while True:
            first = await self._inbox.get()
            batch = [first]
            while True:
                try:
                    batch.append(self._inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                await self._run_turn(batch)
            except Exception as e:
                self.outbox.put_nowait({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                self.outbox.put_nowait({"type": "turn_end"})

    async def _run_turn(self, batch: list[Message]) -> None:
        if self._main_provider is None:
            self.outbox.put_nowait(
                {"type": "error", "message": "主 agent 未连接，请发送 /connect_provider 配置"}
            )
            return
        working = [Message(role="system", content=self._system), *self._messages, *batch]
        loop = AgentLoop(
            self._main_provider,
            self._main_tools,
            max_steps=self._max_steps_main,
            compactor=self._compact,
        )

        def on_text(text: str) -> None:
            self.outbox.put_nowait({"type": "text_delta", "text": text})

        await loop.run(working, on_text=on_text, event_source=self._inbox)
        # 压缩/截断可能已改写 working：整段替换（去掉 system），持久化同步整段重写
        self._messages = working[1:]
        if self._session is not None:
            self._session.replace(self._messages)

    async def _compact(self, messages: list[Message]) -> None:
        """B（默认）：接近阈值时把旧历史压成 [前情摘要] 替换。只在 user 边界切割。"""
        if self._main_provider is None or estimate_chars(messages) < self._compact_threshold:
            return
        keep_head = 1 if messages and messages[0].role == "system" else 0
        body = messages[keep_head:]
        # 最近一段原样保留（约阈值 1/4），从后往前数
        budget = self._compact_threshold // 4
        cut = len(body)
        size = 0
        while cut > 1 and size < budget:
            m = body[cut - 1]
            size += len(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls or [])
            if size <= budget:
                cut -= 1
        # recent 不得以 tool 结果开头（其调用在 old 里，压缩后会成孤链）
        while cut < len(body) and body[cut].role == "tool":
            cut += 1
        old, recent = body[:cut], body[cut:]
        if not old:
            return
        serialized = serialize_for_summary(old, self._compact_threshold)
        result = await self._main_provider.chat(
            [Message(role="user", content=f"{COMPACT_PROMPT}\n\n{serialized}")]
        )
        messages[keep_head:] = [
            Message(role="user", content=f"{SUMMARY_PREFIX}\n{result.text}"),
            *recent,
        ]

    def _on_subtask_event(self, event: SubtaskEvent) -> None:
        status_text = "完成" if event.status == "done" else "出错"
        self.outbox.put_nowait(
            {
                "type": "task_update",
                "id": event.id,
                "status": event.status,
                "output": event.output,
            }
        )
        self._inbox.put_nowait(
            Message(role="user", content=f"[任务 #{event.id} {status_text}]\n{event.output}")
        )

    def _spawn_subagent(self, prompt: str):
        """构造 subagent 的一次执行。plan 模式下写与 shell 工具物理缺席。"""
        if self._executor_provider is None:

            async def noop() -> str:
                return "[执行 subagent 未连接，请 /connect_provider 配置]"

            return noop()
        tools = build_registry(self._root, write=not self.plan_mode, shell=not self.plan_mode)
        loop = AgentLoop(self._executor_provider, tools, max_steps=self._max_steps_executor)
        messages = [
            Message(role="system", content=SUBAGENT_SYSTEM),
            Message(role="user", content=prompt),
        ]

        async def go() -> str:
            result = await loop.run(messages)
            if result.stop_reason == "max_steps":
                return f"[已达最大步数上限，任务可能未完成]\n{result.text}"
            return result.text

        return go()
