"""FastAPI 服务：WebSocket 薄壳，所有逻辑在 Orchestrator。"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


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
        pump = asyncio.create_task(_pump(websocket, orchestrator))
        try:
            while True:
                data = await websocket.receive_json()
                orchestrator.handle_client_message(data)
        except WebSocketDisconnect:
            pass
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    return app


async def _pump(websocket: WebSocket, orchestrator) -> None:
    while True:
        event = await orchestrator.outbox.get()
        await websocket.send_json(event)
