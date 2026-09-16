"""FastAPI 服务：WebSocket 薄壳 + web 静态页 + 只读端点。

安全模型（localhost trust）：
- server 启动生成随机 token 写 ~/.agent_token（600 权限）；页面注入、CLI 从文件读
- WS 握手校验 ?token=；Origin 仅接受本机来源（缺失视为非浏览器客户端放行）
- HTTP 只读端点同样要求 ?token=
每个 WS 连接各自订阅事件总线（多客户端互不偷事件）。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from ..tools import build_registry
from ..tools.base import ToolError

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
_TOKEN_PLACEHOLDER = "__AGENT_TOKEN__"
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def _origin_ok(origin: str | None) -> bool:
    """仅接受本机来源；缺失视为非浏览器客户端（token 仍是主闸）。"""
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and parsed.hostname in _LOCAL_HOSTS


def create_app(orchestrator, token: str = "") -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await orchestrator.start()
        yield
        await orchestrator.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.orchestrator = orchestrator

    def _token_ok(value: str | None) -> bool:
        return not token or value == token

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        if not _token_ok(websocket.query_params.get("token")):
            await websocket.close(code=1008)
            return
        origin = websocket.headers.get("origin")
        if not _origin_ok(origin):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        queue = orchestrator.subscribe()
        pump = asyncio.create_task(_pump(websocket, queue))
        try:
            while True:
                data = await websocket.receive_json()
                orchestrator.handle_client_message(data)
        except WebSocketDisconnect:
            pass
        finally:
            orchestrator.unsubscribe(queue)
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace(_TOKEN_PLACEHOLDER, token))

    @app.get("/marked.min.js")
    async def marked_js() -> FileResponse:
        return FileResponse(WEB_DIR / "marked.min.js", media_type="text/javascript")

    @app.get("/styles.css")
    async def styles_css() -> FileResponse:
        return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

    @app.get("/app.js")
    async def app_js() -> Response:
        # token 注入：占位符在 app.js，页面脚本从这里拿实时 token
        script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
        return Response(script.replace(_TOKEN_PLACEHOLDER, token), media_type="text/javascript")

    @app.get("/timeline_logic.mjs")
    async def timeline_logic_js() -> FileResponse:
        return FileResponse(WEB_DIR / "timeline_logic.mjs", media_type="text/javascript")

    @app.get("/api/ls")
    async def api_ls(
        project: str = Query(...), path: str = Query("."), token: str = Query("")
    ) -> JSONResponse:
        if not _token_ok(token):
            return JSONResponse({"error": "未授权"}, status_code=401)
        return await _call_read_tool(orchestrator, project, "ls", {"path": path})

    @app.get("/api/read")
    async def api_read(
        project: str = Query(...),
        path: str = Query(...),
        offset: int = Query(1),
        limit: int | None = Query(None),
        token: str = Query(""),
    ) -> JSONResponse:
        if not _token_ok(token):
            return JSONResponse({"error": "未授权"}, status_code=401)
        args: dict = {"path": path}
        if limit is not None:
            args["offset"] = offset
            args["limit"] = limit
        return await _call_read_tool(orchestrator, project, "read_file", args)

    @app.get("/api/browse")
    async def api_browse(path: str = Query(None), token: str = Query("")) -> JSONResponse:
        """目录选择器用：列指定路径的子目录（仅目录，含 .. 父级）。默认家目录。"""
        if not _token_ok(token):
            return JSONResponse({"error": "未授权"}, status_code=401)
        base = Path(path).expanduser() if path else Path.home()
        try:
            base = base.resolve()
            if not base.is_dir():
                return JSONResponse({"error": f"目录不存在: {base}"}, status_code=400)
            dirs = sorted(
                (p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")),
                key=str.lower,
            )
        except PermissionError:
            return JSONResponse({"error": f"无权限访问: {base}"}, status_code=400)
        except OSError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"path": str(base), "parent": str(base.parent), "dirs": dirs})

    # ---------- 历史 API（UI timeline 数据源，与 LLM messages 分离） ----------

    @app.get("/api/timeline")
    async def api_timeline(session: str = Query(...), token: str = Query("")) -> JSONResponse:
        """会话完整可视历史：turns（含各自执行事件）+ tasks + context 估算。"""
        if not _token_ok(token):
            return JSONResponse({"error": "未授权"}, status_code=401)
        store = orchestrator.store
        if store is None:
            return JSONResponse({"turns": [], "tasks": [], "context": None})
        turns = store.list_turns(session)
        for turn in turns:
            turn["events"] = store.list_events(session, turn_id=turn["id"])
        tasks = store.list_tasks(session)
        return JSONResponse(
            {"turns": turns, "tasks": tasks, "context": _session_context(orchestrator, session)}
        )

    @app.get("/api/task")
    async def api_task(
        session: str = Query(...), task: int = Query(...), token: str = Query("")
    ) -> JSONResponse:
        """单个 TaskRun 详情 + 全部执行事件（inspector 数据源）。"""
        if not _token_ok(token):
            return JSONResponse({"error": "未授权"}, status_code=401)
        store = orchestrator.store
        if store is None:
            return JSONResponse({"error": "无存储"}, status_code=404)
        record = store.get_task(session, task)
        if record is None:
            return JSONResponse({"error": f"任务不存在: #{task}"}, status_code=404)
        events = store.list_events(session, task_id=task)
        return JSONResponse({"task": record, "events": events})

    return app


def _session_context(orchestrator, session: str) -> dict | None:
    """上下文环数据（与 WS context 事件同一来源）。"""
    conv = orchestrator.conversations.get(session)
    if conv is None:
        return None
    return conv.context_payload()


async def _call_read_tool(orchestrator, project: str, tool: str, args: dict) -> JSONResponse:
    root = orchestrator.project_root(project)
    if root is None:
        return JSONResponse({"error": f"项目不存在: {project}"}, status_code=404)
    registry = build_registry(root, write=False, shell=False)
    try:
        result = await registry.call(tool, args)
    except ToolError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"result": result})


async def _pump(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        event = await queue.get()
        await websocket.send_json(event)
