"""FastAPI 服务：WebSocket 薄壳 + web 静态页 + 只读文件端点。

每个 WS 连接各自订阅事件总线（多客户端互不偷事件）。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..tools import build_registry
from ..tools.base import ToolError

WEB_DIR = (
    Path(__file__).resolve().parents[3] / "web"
)  # 仓库根/web（src/agent/server/app.py 上溯三级）


def create_app(orchestrator) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await orchestrator.start()
        yield
        await orchestrator.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.orchestrator = orchestrator

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
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
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/marked.min.js")
    async def marked_js() -> FileResponse:
        return FileResponse(WEB_DIR / "marked.min.js", media_type="text/javascript")

    @app.get("/api/ls")
    async def api_ls(project: str = Query(...), path: str = Query(".")) -> JSONResponse:
        return await _call_read_tool(orchestrator, project, "ls", {"path": path})

    @app.get("/api/read")
    async def api_read(
        project: str = Query(...),
        path: str = Query(...),
        offset: int = Query(1),
        limit: int | None = Query(None),
    ) -> JSONResponse:
        args: dict = {"path": path}
        if limit is not None:
            args["offset"] = offset
            args["limit"] = limit
        return await _call_read_tool(orchestrator, project, "read_file", args)

    return app


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
