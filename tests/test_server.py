"""server WebSocket 管道测试：stub orchestrator 验证收发接线（逻辑由 orchestrator 测试覆盖）。"""

import asyncio

from fastapi.testclient import TestClient

from agent.server.app import create_app


class StubOrchestrator:
    def __init__(self) -> None:
        self.received: list[dict] = []
        self.outbox: asyncio.Queue[dict] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def handle_client_message(self, data: dict) -> None:
        self.received.append(data)
        self.outbox.put_nowait({"type": "text_delta", "text": "回显:" + data.get("text", "")})


def test_ws_roundtrip() -> None:
    stub = StubOrchestrator()
    app = create_app(stub)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "user", "text": "hi"})
            assert ws.receive_json() == {"type": "text_delta", "text": "回显:hi"}
            ws.send_json({"type": "set_plan_mode", "on": True})
            assert ws.receive_json() == {"type": "text_delta", "text": "回显:"}
    assert stub.received == [
        {"type": "user", "text": "hi"},
        {"type": "set_plan_mode", "on": True},
    ]
