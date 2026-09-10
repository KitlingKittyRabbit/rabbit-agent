"""connect_provider 功能测试：验证连接→持久化→热切换全流程、失败路径、无连接容错。"""

import asyncio
from pathlib import Path

from agent.core.orchestrator import Orchestrator
from agent.providers import AuthError, ChatResult, FakeProvider


def make_orchestrator(tmp_path: Path, factory, store: bool = True) -> Orchestrator:
    return Orchestrator(
        main_provider=None,
        executor_provider=None,
        root=tmp_path,
        store_path=(tmp_path / ".providers.toml") if store else None,
        provider_factory=factory,
    )


async def _until(queue: asyncio.Queue, pred, timeout: float = 5.0):
    async def _wait():
        while True:
            event = await queue.get()
            if pred(event):
                return event

    return await asyncio.wait_for(_wait(), timeout)


CONNECT = dict(protocol="openai", base_url=None, model="m", api_key="sk-live")


async def test_connect_success_persists_and_hot_swaps(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)

    result = await orch.connect_provider("main", **CONNECT)

    assert result["ok"] is True
    assert orch._main_provider is fake
    from agent.core.provider_store import load_providers

    stored = load_providers(tmp_path / ".providers.toml")
    assert stored["main"]["api_key"] == "sk-live"
    assert stored["main"]["protocol"] == "openai"
    assert "base_url" not in stored["main"]


async def test_connect_executor_role(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider("executor", **CONNECT)
    assert result["ok"] is True
    assert orch._executor_provider is fake
    assert orch._main_provider is None


async def test_connect_failure_keeps_nothing(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([AuthError("bad key")]))

    result = await orch.connect_provider("main", **CONNECT)

    assert result["ok"] is False
    assert "连接失败" in result["message"]
    assert not (tmp_path / ".providers.toml").exists()
    assert orch._main_provider is None


async def test_connect_invalid_config_not_saved(tmp_path: Path) -> None:
    def factory(**kw):
        raise ValueError("未知协议")

    orch = make_orchestrator(tmp_path, factory)
    result = await orch.connect_provider("main", **CONNECT)

    assert result["ok"] is False
    assert "配置无效" in result["message"]
    assert not (tmp_path / ".providers.toml").exists()


async def test_connect_unknown_role(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    result = await orch.connect_provider("bogus", **CONNECT)
    assert result["ok"] is False
    assert "未知角色" in result["message"]


async def test_connect_without_store_path_is_runtime_only(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake, store=False)
    result = await orch.connect_provider("main", **CONNECT)
    assert result["ok"] is True
    assert orch._main_provider is fake
    assert not (tmp_path / ".providers.toml").exists()


async def test_chat_without_provider_gets_friendly_error(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    await orch.start()
    try:
        orch.handle_client_message({"type": "user", "text": "你好"})
        error = await _until(orch.outbox, lambda e: e.get("type") == "error")
        assert "未连接" in error["message"]
        await _until(orch.outbox, lambda e: e.get("type") == "turn_end")
    finally:
        await orch.stop()


async def test_spawn_subagent_without_executor_returns_hint(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    output = await orch._spawn_subagent("干活")
    assert "未连接" in output


async def test_connect_message_flows_to_provider_result(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.handle_client_message(
        {"type": "connect_provider", "role": "main", **CONNECT},
    )
    result = await _until(orch.outbox, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    assert result["role"] == "main"
    assert orch._main_provider is fake
