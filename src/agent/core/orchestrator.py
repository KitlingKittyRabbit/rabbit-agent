"""编排器（注册表）：共享 provider/配置/审计/事件总线 + 会话注册表。

事件模型：总线广播，事件带 session 字段；每个 WS 连接各自订阅，互不干扰。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from ..providers import Provider
from .audit import AuditLogger
from .codex_import import import_conversation
from .config import make_provider
from .conversation import Conversation
from .presets import PRESETS
from .project_store import save_project
from .provider_manager import ProviderManager
from .provider_store import derive_provider_id
from .session import SessionStore
from .skills import find_skill, list_skills, skill_prompt


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
        max_steps_executor: int = 100,
        store_path: str | Path | None = None,
        keys_path: str | Path | None = None,
        provider_factory: Callable[..., Provider] = make_provider,
        catalog_fetcher: Callable[[], object] | None = None,
        audit_log_path: str | Path | None = None,
        projects: dict[str, dict] | None = None,
        projects_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.store = store
        self.plan_mode = plan_mode
        self.max_steps_main = max_steps_main
        self.max_steps_executor = max_steps_executor
        self.audit = AuditLogger(audit_log_path)

        self.projects: dict[str, dict] = dict(projects) if projects else {}
        if not self.projects:
            resolved = self.root.resolve()
            self.projects["default"] = {"name": resolved.name, "path": str(resolved)}
        self._projects_path = Path(projects_path) if projects_path is not None else None

        # provider 运行时实现已全部迁到 ProviderManager；以下接口保持兼容
        self.providers = ProviderManager(
            store_path=store_path, keys_path=keys_path,
            provider_factory=provider_factory, catalog_fetcher=catalog_fetcher,
            main_provider=main_provider, executor_provider=executor_provider,
        )
        self._bus = _EventBus()
        self.conversations: dict[str, Conversation] = {}
        self.writer_locks: dict[str, asyncio.Lock] = {}  # 项目 → writer 锁（writer 串行）
        self._started = False
        self._restore()

    def __getattr__(self, name):
        """provider 相关实现委托 ProviderManager；保持旧 API/测试兼容。"""
        providers = self.__dict__.get("providers")
        if providers is not None and hasattr(providers, name):
            return getattr(providers, name)
        raise AttributeError(name)

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
            if self.store is not None:
                self.store.reconcile_stale_tasks(s["id"])

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
                if text.startswith("/") and self._handle_command(conv, text):
                    return
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
            created = (
                {row["id"]: row["created_at"] for row in self.store.list_sessions()}
                if self.store
                else {}
            )
            self.emit(
                {
                    "type": "session_list",
                    "sessions": [
                        {
                            "id": c.id,
                            "title": c.title,
                            "project": c.project_id,
                            "created_at": created.get(c.id),
                        }
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
        elif msg_type == "executor_message":
            conv = self.conversations.get(str(data.get("session") or ""))
            text = str(data.get("text") or "").strip()
            if conv is not None and text:
                conv.executor_message(text)
        elif msg_type == "executor_stop":
            conv = self.conversations.get(str(data.get("session") or ""))
            if conv is not None:
                conv.executor_stop()
        elif msg_type == "undo_message":
            asyncio.get_running_loop().create_task(self._undo_message(data))
        elif msg_type == "codex_login":
            asyncio.get_running_loop().create_task(self._codex_login())
        elif msg_type == "codex_logout":
            asyncio.get_running_loop().create_task(self._codex_logout())
        elif msg_type == "import_codex":
            conv = self.conversations.get(str(data.get("session") or ""))
            codex_id = str(data.get("codex_session") or "").strip()
            project_id = str(data.get("project") or "").strip()
            if not project_id and conv is not None:
                project_id = conv.project_id
            if not project_id:
                self.emit({"type": "error", "message": "缺少项目，无法导入"})
            elif self.project_root(project_id) is None:
                self.emit({"type": "error", "message": f"项目不存在: {project_id}"})
            else:
                try:
                    info = self.import_codex_session(project_id, codex_id)
                except (FileNotFoundError, ValueError, RuntimeError) as e:
                    self.emit({"type": "error", "message": str(e)})
                else:
                    self.emit({"type": "notice", "message": (
                        f"已导入 Codex 会话为 {info['session']}"
                        f"（{info['turns']} 轮对话）")})
        elif msg_type == "set_executor_report":
            conv = self.conversations.get(str(data.get("session") or ""))
            if conv is not None:
                conv.set_executor_report(bool(data.get("on")))
                self.emit({
                    "type": "executor_report", "session": conv.id,
                    "on": conv.executor_report(),
                })
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
        elif msg_type == "disconnect_provider":
            asyncio.get_running_loop().create_task(self._disconnect(data))
        elif msg_type == "get_provider_status":
            self.emit(self.provider_status())
        elif msg_type == "set_model":
            asyncio.get_running_loop().create_task(self._set_model(data))
        elif msg_type == "set_reasoning_effort":
            asyncio.get_running_loop().create_task(self._set_effort(data))
        elif msg_type == "list_models":
            asyncio.get_running_loop().create_task(self._list_models(data))
        elif msg_type == "set_model_capability":
            provider_id = str(data.get("provider") or "")
            model = str(data.get("model") or "")
            override = data.get("override") if isinstance(data.get("override"), dict) else {}
            self.emit(self.set_model_capability(provider_id, model, override))
            self.emit(self.provider_status())
            self.emit(self.model_catalog())
        elif msg_type == "set_context_window":
            role = str(data.get("role") or "")
            window = data.get("window")
            window = int(window) if isinstance(window, int) else None
            self.emit(self.set_context_window(role, window))
            self.emit(self.provider_status())

    # ---------- provider 注册表（多 provider） ----------















    # ---------- 上下文能力（窗口/思考强度/预算） ----------
















    async def _set_model(self, data: dict) -> None:
        result = await self.set_model(
            str(data.get("role") or ""),
            str(data.get("provider_id") or ""),
            str(data.get("model") or "").strip(),
        )
        self.emit(result)
        self.emit(self.provider_status())
        self.emit(self.model_catalog())

    async def _set_effort(self, data: dict) -> None:
        result = self.set_reasoning_effort(
            str(data.get("role") or ""), str(data.get("effort") or "off")
        )
        self.emit(result)
        self.emit(self.provider_status())

    async def _list_models(self, data: dict) -> None:
        provider_id = data.get("provider")
        result = await self.list_models(
            str(provider_id) if provider_id else None,
            refresh=bool(data.get("refresh")),
        )
        self.emit(result)
        self.emit(self.provider_status())





    # ---------- provider 状态 ----------

    def provider_status(self) -> dict:
        """角色当前生效 provider/model/能力 + 全部已配置 provider 摘要（绝不含 api_key）。"""
        return {
            "type": "provider_status",
            "roles": {
                role: self._role_status(role)
                for role in ("main", "executor")
            },
            "providers": [
                {
                    "id": pid,
                    "name": entry["name"],
                    "protocol": entry["protocol"],
                    "base_url": entry["base_url"],
                    "configured": self._provider_configured(pid),
                    "has_key": self._provider_has_key(pid),
                    "model_count": len(entry["models"]),
                    "error": entry["error"],
                }
                for pid, entry in self._providers.items()
            ],
            "presets": self._presets_status(),
            "codex_login": self._codex_login_status(),
            "max_steps": {"main": self.max_steps_main, "executor": self.max_steps_executor},
            "plan_mode": self.plan_mode,
        }



    # ---------- provider 连接 ----------

    async def _disconnect(self, data: dict) -> None:
        result = self.disconnect_provider(str(data.get("provider_id") or ""))
        self.emit(result)
        self.emit(self.provider_status())
        self.emit(self.model_catalog())

    async def _undo_message(self, data: dict) -> None:
        conv = self.conversations.get(str(data.get("session") or ""))
        try:
            turn_id = int(data.get("turn") or 0)
        except (TypeError, ValueError):
            turn_id = 0
        if conv is None:
            self.emit({"type": "error", "message": "会话不存在，无法撤销"})
            return
        supported = conv.undo_supported(turn_id)
        if supported is None:
            self.emit({"type": "error", "message": "要撤销的消息不存在（可能已被撤销）"})
            return
        if not supported:
            self.emit({"type": "error", "message": (
                "这条消息来自旧数据（缺少消息边界记录），不支持撤销；之后的新消息不受影响")})
            return
        await conv.stop_and_wait()     # 停止并等取消落定，再回滚
        try:
            text = conv.undo_turn(turn_id)
        except ValueError as e:
            self.emit({"type": "error", "message": str(e)})
            return
        if text is None:
            self.emit({"type": "error", "message": "要撤销的消息不存在（可能已被撤销）"})

    @staticmethod
    def _codex_login_status() -> dict:
        from ..providers.codex_auth import load_tokens

        tokens = load_tokens()
        if not tokens:
            return {"logged_in": False}
        account = str(tokens.get("accountId") or "")
        return {"logged_in": True, "account": (account[:8] + "…") if account else ""}

    async def _codex_login(self) -> None:
        """ChatGPT 会员登录：浏览器 OAuth（阻塞流程放线程里跑）。"""
        from ..providers import codex_auth

        loop = asyncio.get_running_loop()

        def announce(url: str) -> None:
            loop.call_soon_threadsafe(
                self.emit, {"type": "codex_login", "state": "pending", "url": url})

        try:
            tokens = await asyncio.to_thread(codex_auth.login, on_url=announce)
        except Exception as e:
            self.emit({"type": "error", "message": f"ChatGPT 登录失败：{e}"})
            self.emit(self.provider_status())
            return
        account = str(tokens.get("accountId") or "")
        self.emit({"type": "notice", "message": (
            f"ChatGPT 登录成功（账号 {account[:8] or '未知'}…）")})
        preset = PRESETS["chatgpt"]
        result = await self.connect_provider(
            "", protocol=preset["protocol"], base_url=preset["base_url"],
            model="", api_key="", preset="chatgpt",
        )
        self.emit(result)
        self.emit(self.provider_status())
        self.emit(self.model_catalog())

    async def _codex_logout(self) -> None:
        from ..providers import codex_auth

        codex_auth.delete_tokens()
        preset = PRESETS["chatgpt"]
        pid = derive_provider_id(preset["protocol"], preset["base_url"])
        self.disconnect_provider(pid)      # 没有可断凭据时返回失败，忽略
        self.emit({"type": "notice", "message": "已退出 ChatGPT 登录"})
        self.emit(self.provider_status())
        self.emit(self.model_catalog())

    def import_codex_session(self, project_id: str, session_id: str) -> dict:
        """导入 Codex 会话到指定项目，注册成新会话并广播。"""
        if self.store is None:
            raise RuntimeError("无会话存储，无法导入")
        info = import_conversation(
            self.store, session_id=session_id, project_id=project_id,
        )
        conv = self._add_conversation(
            session_id=info["session"], project_id=project_id,
            title=f"codex · {session_id[:8]}",
        )
        self.emit({"type": "session_created", "session": conv.id,
                   "title": conv.title, "project": project_id, "focus": True})
        return info

    def _handle_command(self, conv, text: str) -> bool:
        """斜杠命令：技能唤起/列表（普通消息返回 False 走常规流程）。"""
        parts = text.split(maxsplit=2)
        command = parts[0]
        root = self.project_root(conv.project_id)
        if command in ("/skills", "/skill") and len(parts) == 1:
            skills = list_skills(root)
            if not skills:
                self.emit({"type": "notice", "message": (
                    "还没有技能：在 <项目>/.agent/skills/<名称>/SKILL.md "
                    "或 ~/.agent/skills/ 下创建")})
            else:
                lines = [f"/skill {s['name']} — {s['description'] or '（无描述）'}（{s['scope']}）"
                         for s in skills]
                self.emit({"type": "notice", "message": "可用技能：\n" + "\n".join(lines)})
            return True
        if command == "/skill":
            name = parts[1] if len(parts) > 1 else ""
            spec = find_skill(root, name)
            if spec is None:
                self.emit({"type": "error",
                           "message": f"未找到技能 {name or '（空）'}：/skills 查看可用技能"})
                return True
            rest = parts[2] if len(parts) > 2 else ""
            conv.enqueue_user(skill_prompt(spec, rest))
            return True
        return False

    async def _connect(self, data: dict) -> None:
        """页面/WS 连接入口：密钥需求判断统一由 connect_provider 负责，这里不改写 key。"""
        preset_name = str(data.get("preset") or "")
        preset = PRESETS.get(preset_name) if preset_name else None
        protocol = str(data.get("protocol") or (preset["protocol"] if preset else ""))
        base_url = data.get("base_url") or (preset["base_url"] if preset else None)
        model = str(data.get("model") or "")  # 留空 = 连上后自动选默认模型
        api_key = str(data.get("api_key") or "")  # 原样交给 connect_provider 判断
        effort = str(data.get("reasoning_effort") or "off")
        result = await self.connect_provider(
            str(data.get("role", "")),
            protocol=protocol,
            base_url=base_url,
            model=model,
            api_key=api_key,
            preset=preset_name,
            reasoning_effort=effort,
        )
        self.emit(result)
        self.emit(self.provider_status())  # 成功失败都刷新界面状态
        self.emit(self.model_catalog())    # 目录/角色模型下拉同步刷新







