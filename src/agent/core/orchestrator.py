"""编排器（注册表）：共享 provider/配置/审计/事件总线 + 会话注册表。

事件模型：总线广播，事件带 session 字段；每个 WS 连接各自订阅，互不干扰。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from ..providers import Message, Provider
from .audit import AuditLogger
from .config import make_provider
from .conversation import Conversation
from .presets import PRESETS
from .project_store import save_project
from .provider_store import save_provider
from .session import SessionStore


class _EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subs.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subs.discard(queue)

    def publish(self, event: dict) -> None:
        for queue in self._subs:
            queue.put_nowait(event)


class Orchestrator:
    def __init__(
        self,
        *,
        main_provider: Provider | None,
        executor_provider: Provider | None,
        root: str | Path,
        store: SessionStore | None = None,
        plan_mode: bool = False,
        max_steps_main: int = 50,
        max_steps_executor: int = 30,
        store_path: str | Path | None = None,
        provider_factory: Callable[..., Provider] = make_provider,
        compact_threshold: int = 200_000,
        audit_log_path: str | Path | None = None,
        projects: dict[str, dict] | None = None,
        projects_path: str | Path | None = None,
    ) -> None:
        self.main_provider = main_provider
        self.executor_provider = executor_provider
        self.root = Path(root)
        self.store = store
        self.plan_mode = plan_mode
        self.max_steps_main = max_steps_main
        self.max_steps_executor = max_steps_executor
        self.compact_threshold = compact_threshold
        self._store_path = Path(store_path) if store_path is not None else None
        self._provider_factory = provider_factory
        self.audit = AuditLogger(audit_log_path)

        self.projects: dict[str, dict] = dict(projects) if projects else {}
        if not self.projects:
            resolved = self.root.resolve()
            self.projects["default"] = {"name": resolved.name, "path": str(resolved)}
        self._projects_path = Path(projects_path) if projects_path is not None else None

        self._bus = _EventBus()
        self.conversations: dict[str, Conversation] = {}
        self._started = False
        self._restore()

    def project_root(self, project_id: str) -> Path | None:
        entry = self.projects.get(project_id)
        return Path(entry["path"]) if entry else None

    # ---------- 会话注册表 ----------

    def _restore(self) -> None:
        default_project = next(iter(self.projects))
        stored = self.store.list_sessions() if self.store else []
        if not stored:
            conv = self._add_conversation(
                session_id=uuid4().hex[:8], project_id=default_project, title="default"
            )
            if self.store is not None:
                self.store.create_session(conv.id, conv.project_id, conv.title)
            return
        for s in stored:
            project_id = s.get("project_id") or default_project
            self._add_conversation(session_id=s["id"], project_id=project_id, title=s["title"])

    def _add_conversation(self, *, session_id: str, project_id: str, title: str) -> Conversation:
        if self.project_root(project_id) is None:
            # 项目已被删除：回退到首个项目，绝不让 server 起不来
            fallback = next(iter(self.projects))
            self.emit(
                {
                    "type": "error",
                    "message": f"会话 {session_id} 的项目 {project_id} 不存在，已回退到 {fallback}",
                }
            )
            project_id = fallback
        conv = Conversation(
            session_id=session_id, title=title, project_id=project_id, registry=self
        )
        self.conversations[session_id] = conv
        if self._started:
            conv.start()
        return conv

    def create_session(self, project_id: str | None = None, title: str = "") -> Conversation | None:
        pid = project_id or next(iter(self.projects))
        if self.project_root(pid) is None:
            self.emit({"type": "error", "message": f"项目不存在: {pid}"})
            return None
        conv = self._add_conversation(
            session_id=uuid4().hex[:8], project_id=pid, title=title or "新会话"
        )
        if self.store is not None:
            self.store.create_session(conv.id, conv.project_id, conv.title)
        self.emit(
            {"type": "session_created", "session": conv.id, "project": pid, "title": conv.title}
        )
        return conv

    def rename_session(self, session_id: str, title: str) -> None:
        conv = self.conversations.get(session_id)
        if conv is None:
            self.emit({"type": "error", "message": f"会话不存在: {session_id}"})
            return
        conv.title = title
        if self.store is not None:
            self.store.set_title(session_id, title)
        self.emit({"type": "session_updated", "session": session_id, "title": title})

    async def delete_session(self, session_id: str) -> None:
        conv = self.conversations.pop(session_id, None)
        if conv is None:
            self.emit({"type": "error", "message": f"会话不存在: {session_id}"})
            return
        await conv.shutdown()
        if self.store is not None:
            self.store.delete_session(session_id)
        self.emit({"type": "session_deleted", "session": session_id})

    def create_project(self, name: str, path: str) -> dict:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return {"type": "project_result", "ok": False, "message": f"目录不存在: {path}"}
        project_id = uuid4().hex[:8]
        self.projects[project_id] = {"name": name or root.name, "path": str(root)}
        if self._projects_path is not None:
            save_project(self._projects_path, project_id, self.projects[project_id])
        self.emit(
            {
                "type": "project_created",
                "project": project_id,
                "name": self.projects[project_id]["name"],
            }
        )
        return {
            "type": "project_result",
            "ok": True,
            "message": f"已添加项目 {name}",
            "project": project_id,
        }

    # ---------- 事件总线 ----------

    def subscribe(self) -> asyncio.Queue:
        return self._bus.subscribe()

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._bus.unsubscribe(queue)

    def emit(self, event: dict) -> None:
        self._bus.publish(event)

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        self._started = True
        for conv in self.conversations.values():
            conv.start()

    async def stop(self) -> None:
        for conv in self.conversations.values():
            await conv.shutdown()
        if self.store is not None:
            self.store.close()

    # ---------- 客户端消息路由 ----------

    def handle_client_message(self, data: dict) -> None:
        msg_type = data.get("type")
        if msg_type == "user":
            conv = self.conversations.get(str(data.get("session", "")))
            text = (data.get("text") or "").strip()
            if conv is None:
                self.emit({"type": "error", "message": f"会话不存在: {data.get('session')}"})
            elif text:
                conv.enqueue_user(text)
        elif msg_type == "new_session":
            self.create_session(
                project_id=data.get("project") or None, title=str(data.get("title") or "")
            )
        elif msg_type == "rename_session":
            self.rename_session(str(data.get("session", "")), str(data.get("title", "")))
        elif msg_type == "delete_session":
            asyncio.get_running_loop().create_task(
                self.delete_session(str(data.get("session", "")))
            )
        elif msg_type == "create_project":
            result = self.create_project(
                name=str(data.get("name") or ""), path=str(data.get("path") or "")
            )
            self.emit(result)
        elif msg_type == "list_projects":
            self.emit(
                {
                    "type": "project_list",
                    "projects": [
                        {"id": pid, "name": p["name"], "path": p["path"]}
                        for pid, p in self.projects.items()
                    ],
                }
            )
        elif msg_type == "list_sessions":
            self.emit(
                {
                    "type": "session_list",
                    "sessions": [
                        {"id": c.id, "title": c.title, "project": c.project_id}
                        for c in self.conversations.values()
                    ],
                }
            )
        elif msg_type == "list_tasks":
            conv = self.conversations.get(str(data.get("session", "")))
            if conv is not None:
                self.emit(
                    {
                        "type": "task_list",
                        "session": conv.id,
                        "tasks": [
                            {"id": tid, "status": status}
                            for tid, status in conv._dispatcher.tasks.items()
                        ],
                    }
                )
        elif msg_type == "stop":
            conv = self.conversations.get(str(data.get("session", "")))
            if conv is not None:
                conv.stop()
        elif msg_type == "set_plan_mode":
            self.plan_mode = bool(data.get("on"))
            self.emit({"type": "plan_mode", "on": self.plan_mode})
        elif msg_type == "confirm_response":
            allow = bool(data.get("allow"))
            for conv in self.conversations.values():
                conv.resolve_confirm(str(data.get("id", "")), allow)
        elif msg_type == "connect_provider":
            asyncio.get_running_loop().create_task(self._connect(data))

    # ---------- provider 连接 ----------

    async def _connect(self, data: dict) -> None:
        preset_name = str(data.get("preset") or "")
        preset = PRESETS.get(preset_name) if preset_name else None
        protocol = str(data.get("protocol") or (preset["protocol"] if preset else ""))
        base_url = data.get("base_url") or (preset["base_url"] if preset else None)
        model = str(data.get("model") or (preset["model"] if preset else ""))
        api_key = str(data.get("api_key") or "")
        if preset and not preset["needs_key"] and not api_key:
            api_key = "unused"
        result = await self.connect_provider(
            str(data.get("role", "")),
            protocol=protocol,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )
        self.emit(result)

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
            self.main_provider = candidate
        else:
            self.executor_provider = candidate
        return {
            "type": "provider_result",
            "role": role,
            "ok": True,
            "message": "连接成功，已切换并保存",
        }
