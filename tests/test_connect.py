"""connect_provider 功能测试：验证连接→持久化→热切换全流程、失败路径、无连接容错。"""

import asyncio
import json
from pathlib import Path

from agent.core.keystore import load_keys, save_key
from agent.core.orchestrator import Orchestrator
from agent.core.presets import PRESETS
from agent.core.provider_store import derive_provider_id, load_registry, save_registry
from agent.providers import AuthError, ChatResult, FakeProvider
from agent.providers.catalog import save_catalog


def make_orchestrator(tmp_path: Path, factory, store: bool = True) -> Orchestrator:
    return Orchestrator(
        main_provider=None,
        executor_provider=None,
        root=tmp_path,
        store_path=(tmp_path / ".providers.toml") if store else None,
        keys_path=tmp_path / "keys.json",
        provider_factory=factory,
    )


def first_session(orch: Orchestrator) -> str:
    return next(iter(orch.conversations))


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
    assert orch.main_provider is fake
    from agent.core.keystore import load_keys
    from agent.core.provider_store import derive_provider_id, load_registry

    pid = derive_provider_id("openai", None)
    registry = load_registry(tmp_path / ".providers.toml")
    assert registry["roles"]["main"]["provider_id"] == pid
    assert registry["roles"]["main"]["model"] == "m"
    text = (tmp_path / ".providers.toml").read_text(encoding="utf-8")
    assert "api_key" not in text and "sk-live" not in text  # 密钥不落配置
    assert load_keys(tmp_path / "keys.json")[pid] == "sk-live"
    assert registry["providers"][pid]["protocol"] == "openai"


async def test_connect_executor_role(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider("executor", **CONNECT)
    assert result["ok"] is True
    assert orch.executor_provider is fake
    assert orch.main_provider is None


async def test_connect_failure_keeps_nothing(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([AuthError("bad key")]))
    result = await orch.connect_provider("main", **CONNECT)
    assert result["ok"] is False
    assert "连接失败" in result["message"]
    assert not (tmp_path / ".providers.toml").exists()
    assert orch.main_provider is None


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
    assert orch.main_provider is fake
    assert not (tmp_path / ".providers.toml").exists()


async def test_chat_without_provider_gets_friendly_error(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    queue = orch.subscribe()
    await orch.start()
    try:
        orch.handle_client_message({"type": "user", "session": first_session(orch), "text": "你好"})
        error = await _until(queue, lambda e: e.get("type") == "error")
        assert "未连接" in error["message"]
        await _until(queue, lambda e: e.get("type") == "turn_end")
    finally:
        await orch.stop()


async def test_spawn_subagent_without_executor_raises(tmp_path: Path) -> None:
    import pytest

    from agent.core.dispatch import ExecutorUnconfigured

    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    conv = orch.conversations[first_session(orch)]
    with pytest.raises(ExecutorUnconfigured):
        conv._spawn_subagent(1, "干活", [])


async def test_connect_message_flows_to_provider_result(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    queue = orch.subscribe()
    orch.handle_client_message({"type": "connect_provider", "role": "main", **CONNECT})
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    assert result["role"] == "main"
    assert orch.main_provider is fake


async def test_provider_status_unconfigured(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    queue = orch.subscribe()
    orch.handle_client_message({"type": "get_provider_status"})
    status = await _until(queue, lambda e: e.get("type") == "provider_status")
    assert status["roles"]["main"]["configured"] is False
    assert status["roles"]["executor"]["configured"] is False
    assert status["roles"]["main"]["provider_id"] == ""
    assert status["presets"]["kimi-coding"]["model"] == "kimi-for-coding"
    assert status["max_steps"] == {"main": 50, "executor": 100}
    assert status["plan_mode"] is False
    assert status["providers"] == []


def test_provider_status_reflects_plan_mode(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    orch.handle_client_message({"type": "set_plan_mode", "on": True})
    assert orch.plan_mode is True
    assert orch.provider_status()["plan_mode"] is True


def test_provider_status_static_config(tmp_path: Path) -> None:
    """静态配置的 provider 注册为 provider 条目并绑定角色。"""
    import httpx2

    from agent.providers import OpenAICompatProvider

    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1", api_key="sk-static-secret", model="deepseek-chat",
        http_client=httpx2.AsyncClient(trust_env=False),
    )
    orch = Orchestrator(
        main_provider=provider, executor_provider=None, root=tmp_path,
        store_path=tmp_path / ".providers.toml",
    )
    status = orch.provider_status()
    main = status["roles"]["main"]
    assert main["configured"] is True
    assert main["model"] == "deepseek-chat"
    assert main["protocol"] == "openai"
    assert main["provider_id"].startswith("p-")
    assert "sk-static-secret" not in json.dumps(status)
    assert status["providers"][0]["id"] == main["provider_id"]


async def test_provider_status_after_preset_connect(tmp_path: Path) -> None:
    """连接 preset 后状态返回 provider/model/window，且绝不包含 api_key。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    queue = orch.subscribe()
    orch.handle_client_message(
        {
            "type": "connect_provider", "role": "main", "preset": "kimi-coding",
            "model": "", "api_key": "sk-secret-xyz",
        }
    )
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    status = await _until(queue, lambda e: e.get("type") == "provider_status")
    main = status["roles"]["main"]
    assert main["configured"] is True
    assert main["model"] == "kimi-for-coding"
    assert main["base_url"] == "https://api.kimi.com/coding/"
    assert "sk-secret-xyz" not in json.dumps(status)


async def test_provider_status_refreshes_after_failure(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([AuthError("bad")]))
    queue = orch.subscribe()
    orch.handle_client_message(
        {"type": "connect_provider", "role": "executor", "preset": "deepseek",
         "model": "", "api_key": "sk-x"}
    )
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is False
    status = await _until(queue, lambda e: e.get("type") == "provider_status")
    assert status["roles"]["executor"]["configured"] is False
    assert orch.executor_provider is None


def _pid(orch: Orchestrator, role: str = "main") -> str:
    return orch.provider_status()["roles"][role]["provider_id"]


async def test_connect_takes_window_from_model_catalog(tmp_path: Path) -> None:
    """连接成功后按模型目录条目设置窗口（provider 元数据来源）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{
        "id": "m", "display_name": "模型 M",
        "capability": {
            "window": 123_456, "reasoning_returned": False, "reasoning_mode": "none",
            "levels": None, "max_output": 4_096, "tools": True, "source": "provider",
        },
    }]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider("main", **CONNECT)
    assert result["ok"] is True
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 123_456 and status["window_source"] == "provider"
    assert status["reasoning_mode"] == "none"
    catalog = await orch.list_models()
    provider = catalog["providers"][0]
    assert provider["status"] == "ok"
    assert [m["id"] for m in provider["models"]] == ["m"]


async def test_connect_unknown_model_window_none(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])  # models 默认空 → 拉取失败但不断连
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider("main", **CONNECT)
    assert result["ok"] is True
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] is None and status["window_source"] == "unknown"
    assert status["reasoning_mode"] == "unknown"
    assert orch.context_budget("main") is None
    catalog = await orch.list_models()
    assert catalog["providers"][0]["status"] == "error"
    assert catalog["providers"][0]["models"] == []  # 失败不伪造列表


async def test_list_models_reports_failure_without_fallback(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = RuntimeError("接口 500")
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    catalog = await orch.list_models(refresh=True)
    assert catalog["providers"][0]["status"] == "error"
    assert "无法获取模型列表" in catalog["providers"][0]["error"]
    assert catalog["providers"][0]["models"] == []


async def test_list_models_groups_capabilities_and_caches(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [
        {"id": "a", "display_name": "A", "capability": {
            "window": 1000, "reasoning_returned": True, "reasoning_mode": "adjustable",
            "levels": ["off", "low"], "max_output": None, "tools": True, "source": "provider"}},
        {"id": "b", "display_name": "B", "capability": {
            "window": None, "reasoning_returned": None, "reasoning_mode": "unknown",
            "levels": None, "max_output": None, "tools": None, "source": "provider"}},
    ]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    first = await orch.list_models(refresh=True)
    provider = first["providers"][0]
    assert provider["status"] == "ok" and len(provider["models"]) == 2
    assert provider["models"][0]["provider_id"] == provider["id"]
    assert provider["models"][1]["capability"]["window"] is None  # 未知保持未知
    calls_after_refresh = fake.models_calls
    second = await orch.list_models()  # 命中缓存，不再调用 provider
    assert second["providers"][0]["models"]
    assert fake.models_calls == calls_after_refresh


async def test_set_model_hot_swaps_and_reuses_key(tmp_path: Path) -> None:
    made: list[dict] = []

    def factory(**kw):
        made.append(kw)
        return FakeProvider([ChatResult(text="pong")])

    orch = make_orchestrator(tmp_path, factory)
    assert (await orch.connect_provider("main", **CONNECT))["ok"] is True
    pid = _pid(orch)
    result = await orch.set_model("main", pid, "new-model")
    assert result["ok"] is True
    assert made[-1]["model"] == "new-model"
    assert made[-1]["api_key"] == "sk-live"  # 复用已存密钥
    assert orch.provider_status()["roles"]["main"]["model"] == "new-model"
    assert "sk-live" not in (tmp_path / ".providers.toml").read_text(encoding="utf-8")


async def test_set_model_unknown_provider_fails(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([]))
    result = await orch.set_model("main", "p-nope", "m")
    assert result["ok"] is False and "不存在" in result["message"]


async def test_connect_custom_endpoint_allows_empty_key(tmp_path: Path) -> None:
    """显式自定义本地端点：空 key 连接成功（内部使用占位）。"""
    made: list[dict] = []

    def factory(**kw):
        made.append(kw)
        return FakeProvider([ChatResult(text="pong")])

    orch = make_orchestrator(tmp_path, factory)
    result = await orch.connect_provider(
        "main", protocol="openai", base_url="http://127.0.0.1:9/v1", model="m", api_key=""
    )
    assert result["ok"] is True
    assert made[-1]["api_key"] == "unused"


async def test_set_reasoning_effort_four_states(tmp_path: Path) -> None:
    """可调/固定/不支持/未知 四种状态（能力来自 provider 元数据或用户覆盖）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "display_name": "M", "capability": {
        "window": 100_000, "reasoning_returned": True, "reasoning_mode": "adjustable",
        "levels": ["off", "low", "medium", "high"], "max_output": None,
        "tools": True, "source": "provider"}}]
    # 各状态用独立目录：connect 失败重连会保留旧缓存，不能跨用例共享 provider 状态
    d1 = tmp_path / "adjustable"
    d1.mkdir()
    orch = make_orchestrator(d1, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    assert orch.provider_status()["roles"]["main"]["efforts"] == [
        "off", "low", "medium", "high"]
    result = orch.set_reasoning_effort("main", "high")
    assert result["ok"] is True
    assert orch.main_provider._reasoning_effort == "high"
    assert orch.provider_status()["roles"]["main"]["effort"] == "high"

    fixed = FakeProvider([ChatResult(text="pong")])
    fixed.models = [{"id": "m", "capability": {
        "window": 1000, "reasoning_returned": True, "reasoning_mode": "fixed",
        "levels": None, "max_output": None, "tools": None, "source": "provider"}}]
    d2 = tmp_path / "fixed"
    d2.mkdir()
    orch2 = make_orchestrator(d2, lambda **kw: fixed)
    await orch2.connect_provider("main", **CONNECT)
    assert "固定" in orch2.set_reasoning_effort("main", "high")["message"]

    none = FakeProvider([ChatResult(text="pong")])
    none.models = [{"id": "m", "capability": {
        "window": 1000, "reasoning_returned": False, "reasoning_mode": "none",
        "levels": None, "max_output": None, "tools": None, "source": "provider"}}]
    d3 = tmp_path / "none"
    d3.mkdir()
    orch3 = make_orchestrator(d3, lambda **kw: none)
    await orch3.connect_provider("main", **CONNECT)
    assert "不支持推理" in orch3.set_reasoning_effort("main", "high")["message"]

    d4 = tmp_path / "unknown"
    d4.mkdir()
    orch4 = make_orchestrator(d4, lambda **kw: FakeProvider([ChatResult(text="pong")]))
    await orch4.connect_provider("main", **CONNECT)
    assert "能力未知" in orch4.set_reasoning_effort("main", "high")["message"]


async def test_first_connect_keeps_requested_effort(tmp_path: Path) -> None:
    """首连选择 medium：factory 最终参数/运行参数/状态/持久化/下一请求全部保留。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "display_name": "M", "capability": {
        "window": 128_000, "reasoning_returned": True, "reasoning_mode": "adjustable",
        "levels": ["low", "medium", "high"], "max_output": None,
        "tools": True, "source": "provider"}}]
    made: list[dict] = []

    def factory(**kw):
        made.append(kw)
        return fake

    orch = make_orchestrator(tmp_path, factory)
    result = await orch.connect_provider(
        "main", protocol="openai", base_url=None, model="m", api_key="sk-x",
        reasoning_effort="medium",
    )
    assert result["ok"] is True
    assert made[-1]["reasoning_effort"] == "medium"  # factory 最终参数
    assert fake._reasoning_effort == "medium"        # 运行参数
    assert orch.provider_status()["roles"]["main"]["effort"] == "medium"  # 状态
    from agent.core.provider_store import load_registry
    assert load_registry(tmp_path / ".providers.toml")["roles"]["main"][
        "reasoning_effort"] == "medium"               # 持久化
    # 协议参数进入真实请求由 provider 层测试覆盖：
    # tests/test_providers_reasoning.py::test_openai_reasoning_effort_kwarg_gated_by_value
    # tests/test_providers_reasoning.py::test_anthropic_thinking_budget_reaches_request


async def test_first_connect_unsupported_effort_fails_loudly(tmp_path: Path) -> None:
    """首连请求了不支持的强度：明确失败，禁止静默降为 off。"""
    fake = FakeProvider([ChatResult(text="pong")])  # 无能力元数据 → unknown
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider(
        "main", protocol="openai", base_url=None, model="m", api_key="sk-x",
        reasoning_effort="medium",
    )
    assert result["ok"] is False
    assert "能力未知" in result["message"] or "无法设置思考强度" in result["message"]
    assert orch.main_provider is None


async def test_set_effort_static_provider_with_capability(tmp_path: Path) -> None:
    """静态 provider：用户声明能力后思考强度可用。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = Orchestrator(main_provider=fake, executor_provider=None, root=tmp_path)
    pid = _pid(orch)
    model = orch.provider_status()["roles"]["main"]["model"]
    assert orch.set_model_capability(pid, model, {
        "reasoning_mode": "adjustable", "levels": ["off", "low", "medium", "high"],
    })["ok"] is True
    result = orch.set_reasoning_effort("main", "high")
    assert result["ok"] is True
    assert fake._reasoning_effort == "high"


async def test_set_model_capability_override_and_clear(tmp_path: Path) -> None:
    """能力覆盖绑定 provider_id+model_id，可持久化、可清除。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    pid = _pid(orch)
    assert orch.set_model_capability(pid, "m", {
        "window": 99_000, "reasoning_mode": "adjustable", "levels": ["low", "high"],
        "tools": False,
    })["ok"] is True
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 99_000 and status["window_source"] == "user"
    assert status["tools"] is False  # False 不被吃掉

    from agent.core.provider_store import load_registry
    override = load_registry(tmp_path / ".providers.toml")["providers"][pid][
        "model_overrides"]["m"]
    assert override["window"] == 99_000 and override["tools"] is False

    assert orch.set_model_capability(pid, "m", {"window": None})["ok"] is True
    assert orch.provider_status()["roles"]["main"]["window"] is None  # 清除后回到未知


async def test_context_window_override_persists_across_restart(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    assert orch.set_context_window("main", 77_000)["ok"] is True

    reopened = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="pong")]),
        executor_provider=None,
        root=tmp_path,
        store_path=tmp_path / ".providers.toml",
        keys_path=tmp_path / "keys.json",
        provider_factory=lambda **kw: FakeProvider([ChatResult(text="pong")]),
    )
    assert reopened.context_window("main") == (77_000, "user")


async def test_metadata_window_not_saved_as_user_override(tmp_path: Path) -> None:
    """provider 元数据窗口不得被误存为用户覆盖。"""

    def factory(**kw):
        fake = FakeProvider([ChatResult(text="pong")])
        fake.models = [{"id": "m", "display_name": "M", "capability": {
            "window": 123_456, "reasoning_returned": None, "reasoning_mode": "unknown",
            "levels": None, "max_output": None, "tools": None, "source": "provider"}}]
        return fake

    orch = make_orchestrator(tmp_path, factory)
    await orch.connect_provider("main", **CONNECT)
    pid = _pid(orch)
    result = await orch.set_model("main", pid, "m")
    assert result["ok"] is True
    assert "model_overrides" not in (tmp_path / ".providers.toml").read_text(encoding="utf-8")
    assert orch.context_window("main") == (123_456, "provider")


async def test_user_window_survives_later_saves(tmp_path: Path) -> None:
    """回归：切换模型等保存操作不得清除其它模型上的用户窗口覆盖。"""
    fakes: list[FakeProvider] = []

    def factory(**kw):
        fake = FakeProvider([ChatResult(text="pong")])
        fakes.append(fake)
        return fake

    orch = make_orchestrator(tmp_path, factory)
    await orch.connect_provider("main", **CONNECT)
    pid = _pid(orch)
    assert orch.set_context_window("main", 77_000)["ok"] is True
    assert (await orch.set_model("main", pid, "m2"))["ok"] is True
    assert (await orch.set_model("main", pid, "m"))["ok"] is True

    reopened = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="pong")]),
        executor_provider=None, root=tmp_path,
        store_path=tmp_path / ".providers.toml", keys_path=tmp_path / "keys.json",
        provider_factory=lambda **kw: FakeProvider([ChatResult(text="pong")]),
    )
    assert reopened.context_window("main") == (77_000, "user")


async def test_multi_provider_catalog_isolates_failures(tmp_path: Path) -> None:
    """两个 provider：一个失败不影响另一个的模型；同名模型按 provider_id 区分。"""
    good = FakeProvider([ChatResult(text="pong")])
    good.models = [{"id": "same", "display_name": "好模型", "capability": {
        "window": 1000, "reasoning_returned": None, "reasoning_mode": "unknown",
        "levels": None, "max_output": None, "tools": None, "source": "provider"}}]
    bad = FakeProvider([ChatResult(text="pong")])
    bad.models = RuntimeError("挂了")

    def factory(**kw):
        return bad if kw.get("base_url") == "http://bad/v1" else good

    orch = make_orchestrator(tmp_path, factory)
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url="http://good/v1", model="same", api_key="k",
    ))["ok"] is True
    assert (await orch.connect_provider(
        "executor", protocol="openai", base_url="http://bad/v1", model="same", api_key="k",
    ))["ok"] is True
    catalog = await orch.list_models()
    good_entry = next(p for p in catalog["providers"] if p["base_url"] == "http://good/v1")
    bad_entry = next(p for p in catalog["providers"] if p["base_url"] == "http://bad/v1")
    assert good_entry["status"] == "ok" and bad_entry["status"] == "error"
    assert good_entry["models"][0]["id"] == "same"
    # 同名模型可区分
    assert good_entry["id"] != bad_entry["id"]
    assert good_entry["models"][0]["provider_id"] == good_entry["id"]


async def test_ws_list_models_and_capability_override_events(tmp_path: Path) -> None:
    """WS 数据流：list_models 聚合事件 + set_model_capability 覆盖后能力立即可用。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "display_name": "M", "capability": {}}]  # 无能力字段
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    queue = orch.subscribe()
    orch.handle_client_message({"type": "connect_provider", "role": "main", **CONNECT})
    await _until(queue, lambda e: e.get("type") == "provider_result")

    orch.handle_client_message({"type": "list_models", "refresh": True})
    catalog = await _until(queue, lambda e: e.get("type") == "model_catalog")
    pid = catalog["providers"][0]["id"]
    assert catalog["providers"][0]["status"] == "ok"
    assert orch.provider_status()["roles"]["main"]["reasoning_mode"] == "unknown"

    orch.handle_client_message({
        "type": "set_model_capability", "provider": pid, "model": "m",
        "override": {"reasoning_mode": "adjustable", "levels": ["off", "medium"],
                     "window": 50_000},
    })
    result = await _until(
        queue, lambda e: e.get("type") == "provider_result" and e.get("ok") is True
    )
    assert "能力设置" in result["message"]
    status = await _until(queue, lambda e: e.get("type") == "provider_status")
    assert status["roles"]["main"]["reasoning_mode"] == "adjustable"
    assert status["roles"]["main"]["efforts"] == ["off", "medium"]
    assert status["roles"]["main"]["window"] == 50_000
    assert "medium" in [
        level for level in status["roles"]["main"]["efforts"]
    ]
    assert orch.set_reasoning_effort("main", "medium")["ok"] is True


async def test_registry_bindings_active_after_restart(tmp_path: Path) -> None:
    """回归：重启后注册表里的角色绑定必须被激活（provider/窗口/强度全部生效）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "display_name": "M", "capability": {
        "window": 10_000, "reasoning_returned": True, "reasoning_mode": "adjustable",
        "levels": ["off", "medium"], "max_output": None, "tools": None,
        "source": "provider"}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url=None, model="m", api_key="sk-x",
        reasoning_effort="medium",
    ))["ok"] is True
    assert orch.set_context_window("main", 50_000)["ok"] is True

    provider_holder: list[FakeProvider] = []

    def factory(**kw):
        fresh = FakeProvider([ChatResult(text="pong")])
        provider_holder.append(fresh)
        return fresh

    reopened = Orchestrator(
        main_provider=None, executor_provider=None, root=tmp_path,
        store_path=tmp_path / ".providers.toml", keys_path=tmp_path / "keys.json",
        provider_factory=factory,
    )
    status = reopened.provider_status()["roles"]["main"]
    assert status["configured"] is True
    assert reopened.main_provider is not None
    assert status["model"] == "m"
    assert status["window"] == 50_000 and status["window_source"] == "user"
    assert status["effort"] == "medium" and status["efforts"] == ["off", "medium"]


# ---------- 角色作用域能力（问题 1/2：main/executor 互不污染，未知语义显式） ----------


def _adjustable_entry(model_id: str, levels=("off", "low", "medium", "high"), tools=None):
    return {"id": model_id, "display_name": model_id, "capability": {
        "window": 100_000, "reasoning_returned": True, "reasoning_mode": "adjustable",
        "levels": list(levels), "max_output": None, "tools": tools, "source": "provider"}}


async def test_role_status_keeps_explicit_efforts_none_vs_empty(tmp_path: Path) -> None:
    """未知档位必须暴露为 None（不是 []）；固定不支持时才是 []。"""
    fake = FakeProvider([ChatResult(text="pong"), ChatResult(text="pong")])
    fake.models = [
        _adjustable_entry("m", tools=False),
        {"id": "m-none", "display_name": "m-none", "capability": {
            "window": 1000, "reasoning_returned": False, "reasoning_mode": "none",
            "levels": None, "max_output": None, "tools": None, "source": "provider"}},
    ]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url=None, model="m", api_key="k"))["ok"] is True
    assert (await orch.connect_provider(
        "executor", protocol="openai", base_url=None, model="m-none", api_key="k"))["ok"] is True
    status = orch.provider_status()["roles"]
    assert status["main"]["efforts"] == ["off", "low", "medium", "high"]
    assert status["main"]["tools"] is False  # False 不得变 None
    assert status["executor"]["efforts"] == []  # none=明确为空
    assert status["executor"]["reasoning_mode"] == "none"
    assert status["executor"]["tools"] is None


async def test_role_unknown_capability_exposes_none_not_empty(tmp_path: Path) -> None:
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)  # models 空 → 未知能力
    status = orch.provider_status()["roles"]["main"]
    assert status["efforts"] is None  # 未知 ≠ 空数组
    assert status["tools"] is None and status["window"] is None
    assert status["window_source"] == "unknown"


async def test_capability_resave_keeps_levels_and_false_across_restart(tmp_path: Path) -> None:
    """打开设置不改动直接保存：档位、tools=False、effort 都不得丢失（跨重启）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [_adjustable_entry("m", tools=True)]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url=None, model="m", api_key="k",
        reasoning_effort="high"))["ok"] is True
    pid = _pid(orch)
    status = orch.provider_status()["roles"]["main"]
    # 与前端 capabilityOverrideFromForm 不改动直接保存等价的 override
    override = {
        "window": status["window"], "max_output": None, "reasoning_mode": "adjustable",
        "levels": list(status["efforts"]), "tools": False,
    }
    assert orch.set_model_capability(pid, "m", override)["ok"] is True
    saved = orch.provider_status()["roles"]["main"]
    assert saved["efforts"] == ["off", "low", "medium", "high"]
    assert saved["tools"] is False and saved["effort"] == "high"

    # 再存一次同样内容（第二次“打开设置直接保存”）不得丢档位
    assert orch.set_model_capability(pid, "m", dict(override))["ok"] is True
    again = orch.provider_status()["roles"]["main"]
    assert again["efforts"] == ["off", "low", "medium", "high"]
    assert again["tools"] is False and again["effort"] == "high"

    reopened = Orchestrator(
        main_provider=None, executor_provider=None, root=tmp_path,
        store_path=tmp_path / ".providers.toml", keys_path=tmp_path / "keys.json",
        provider_factory=lambda **kw: FakeProvider([ChatResult(text="pong")]),
    )
    restarted = reopened.provider_status()["roles"]["main"]
    assert restarted["efforts"] == ["off", "low", "medium", "high"]
    assert restarted["tools"] is False
    assert restarted["effort"] == "high" and restarted["configured"] is True
    assert reopened.main_provider._reasoning_effort == "high"


async def test_capability_clear_only_affects_target_model(tmp_path: Path) -> None:
    """清除能力按 provider+model 作用，不得串到同 provider 的另一个角色模型。"""
    fake = FakeProvider([ChatResult(text="pong"), ChatResult(text="pong")])
    fake.models = [_adjustable_entry("m-main"), _adjustable_entry("m-exec")]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", protocol="openai", base_url=None,
                                model="m-main", api_key="k")
    await orch.connect_provider("executor", protocol="openai", base_url=None,
                                model="m-exec", api_key="k")
    pid = _pid(orch)
    orch.set_model_capability(pid, "m-main", {"reasoning_mode": "adjustable",
                                              "levels": ["off", "low"]})
    clearing = {"window": None, "max_output": None, "reasoning_mode": None,
                "levels": None, "tools": None, "reasoning_returned": None}
    assert orch.set_model_capability(pid, "m-exec", clearing)["ok"] is True
    status = orch.provider_status()["roles"]
    assert status["main"]["efforts"] == ["off", "low"]      # main 覆盖未被动
    assert status["executor"]["efforts"] == ["off", "low", "medium", "high"]  # 回到 provider


# ---------- 缺密钥的官方端点不可用（问题 3） ----------


def _registry_for(tmp_path: Path, protocol: str, base_url, model: str, effort: str = "off"):
    from agent.core.provider_store import derive_provider_id, save_registry

    pid = derive_provider_id(protocol, base_url)
    save_registry(tmp_path / ".providers.toml", {
        "roles": {"main": {"provider_id": pid, "model": model, "reasoning_effort": effort}},
        "providers": {pid: {"name": f"{protocol} · x", "protocol": protocol,
                            "base_url": base_url, "model_overrides": {}, "models": [],
                            "models_fetched_at": 0.0}},
    })
    return pid


async def test_official_endpoint_without_key_stays_unconfigured(tmp_path: Path) -> None:
    """官方端点缺密钥：启动不建实例、不算已连接、目录报缺凭据，且不调用 factory。"""
    _registry_for(tmp_path, "openai", None, "m")
    calls: list[dict] = []

    def factory(**kw):
        calls.append(dict(kw))
        return FakeProvider([ChatResult(text="pong")])

    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=factory)
    assert orch.main_provider is None
    assert orch.provider_status()["roles"]["main"]["configured"] is False
    catalog = await orch.list_models()
    assert catalog["providers"][0]["configured"] is False
    assert "缺少凭据" in catalog["providers"][0]["error"]
    assert calls == []  # 启动/列举都不得构造空 key 实例

    orch2 = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                         store_path=tmp_path / ".providers.toml",
                         keys_path=tmp_path / "keys.json", provider_factory=factory)
    assert orch2.set_reasoning_effort("main", "low")["ok"] is False
    assert calls == []


async def test_explicit_official_host_without_key_stays_unconfigured(tmp_path: Path) -> None:
    """显式官方主机（PRESETS 中 needs_key 的地址）缺密钥同样不可用。"""
    _registry_for(tmp_path, "openai", "https://api.deepseek.com/v1", "deepseek-chat")
    calls: list[dict] = []
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: calls.append(kw) or FakeProvider([]))
    assert orch.main_provider is None
    assert orch.provider_status()["roles"]["main"]["configured"] is False
    assert calls == []


async def test_local_endpoint_without_key_still_usable(tmp_path: Path) -> None:
    """显式本地/自定义端点：无密钥仍可启动为可用实例（保留既有允许规则）。"""
    _registry_for(tmp_path, "openai", "http://127.0.0.1:9/v1", "m")
    calls: list[dict] = []

    def factory(**kw):
        calls.append(dict(kw))
        return FakeProvider([ChatResult(text="pong")])

    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=factory)
    assert orch.main_provider is not None
    assert orch.provider_status()["roles"]["main"]["configured"] is True
    assert calls and calls[-1]["api_key"] == "unused"


# ---------- 共用 provider 的角色强度隔离（问题 4） ----------


async def test_shared_provider_roles_keep_distinct_efforts(tmp_path: Path) -> None:
    """main=m-main/low 与 executor=m-exec/high 共用 provider 时互不污染。"""
    def make_factory(records):
        def factory(**kw):
            records.append(dict(kw))
            fake = FakeProvider([ChatResult(text="pong")])
            fake.models = [_adjustable_entry("m-main", ("off", "low")),
                           _adjustable_entry("m-exec", ("off", "high"))]
            return fake
        return factory

    records: list[dict] = []
    orch = make_orchestrator(tmp_path, make_factory(records))
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url=None, model="m-main", api_key="k",
        reasoning_effort="low"))["ok"] is True
    assert (await orch.connect_provider(
        "executor", protocol="openai", base_url=None, model="m-exec", api_key="k",
        reasoning_effort="high"))["ok"] is True
    assert orch.main_provider is not orch.executor_provider
    assert orch.main_provider._reasoning_effort == "low"
    assert orch.executor_provider._reasoning_effort == "high"
    assert [r["reasoning_effort"] for r in records if r["model"] == "m-main"][-1] == "low"
    assert [r["reasoning_effort"] for r in records if r["model"] == "m-exec"][-1] == "high"

    records2: list[dict] = []
    reopened = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=make_factory(records2))
    assert reopened.main_provider._reasoning_effort == "low"
    assert reopened.executor_provider._reasoning_effort == "high"
    assert [r["reasoning_effort"] for r in records2 if r["model"] == "m-main"] == ["low"]
    assert [r["reasoning_effort"] for r in records2 if r["model"] == "m-exec"] == ["high"]


async def test_list_models_instance_never_inherits_role_effort(tmp_path: Path) -> None:
    """模型目录实例固定 effort=off、不带角色窗口。"""
    records: list[dict] = []

    def factory(**kw):
        records.append(dict(kw))
        fake = FakeProvider([ChatResult(text="pong")])
        fake.models = [_adjustable_entry("m-main", ("off", "low"))]
        return fake

    orch = make_orchestrator(tmp_path, factory)
    await orch.connect_provider("main", protocol="openai", base_url=None,
                                model="m-main", api_key="k", reasoning_effort="low")
    await orch.list_models(refresh=True)
    listing = [r for r in records if r["model"] == "__listing__"]
    assert listing, records
    assert listing[-1]["reasoning_effort"] == "off"
    assert listing[-1]["context_window"] is None


async def test_shared_provider_request_bodies_keep_their_efforts(tmp_path: Path) -> None:
    """真实 provider 请求体：共用 provider 时 main/executor 各自带自己的 effort。"""
    from types import SimpleNamespace

    import httpx2

    from agent.providers import Message
    from agent.providers.openai_compat import OpenAICompatProvider

    payloads: list[dict] = []

    def _chunk(content=None, finish=None):
        delta = SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
                               usage=None)

    class _Stream:
        def __aiter__(self):
            async def gen():
                yield _chunk(content="x", finish="stop")
            return gen()

    def factory(**kw):
        provider = OpenAICompatProvider(
            base_url=kw["base_url"], api_key=kw["api_key"], model=kw["model"],
            context_window=kw.get("context_window"),
            reasoning_effort=kw.get("reasoning_effort"),
            http_client=httpx2.AsyncClient(trust_env=False),
        )

        async def _list():
            return [_adjustable_entry("m-main", ("off", "low")),
                    _adjustable_entry("m-exec", ("off", "high"))]

        provider.list_models = _list

        class _Completions:
            async def create(self, **request):
                payloads.append({"model": request.get("model"),
                                 "reasoning_effort": request.get("reasoning_effort")})
                return _Stream()

        provider._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        return provider

    orch = make_orchestrator(tmp_path, factory)
    await orch.connect_provider("main", protocol="openai", base_url=None,
                                model="m-main", api_key="k", reasoning_effort="low")
    await orch.connect_provider("executor", protocol="openai", base_url=None,
                                model="m-exec", api_key="k", reasoning_effort="high")
    await orch.main_provider.chat([Message(role="user", content="hi")])
    await orch.executor_provider.chat([Message(role="user", content="hi")])
    main_payloads = [p for p in payloads if p["model"] == "m-main"]
    exec_payloads = [p for p in payloads if p["model"] == "m-exec"]
    assert main_payloads[-1]["reasoning_effort"] == "low"
    assert exec_payloads[-1]["reasoning_effort"] == "high"


async def test_missing_key_connect_returns_explicit_error(tmp_path: Path) -> None:
    """官方端点未提供 key：连接必须显式失败，不得构造空 key 实例。"""
    calls: list[dict] = []
    orch = make_orchestrator(tmp_path, lambda **kw: calls.append(kw) or FakeProvider([]))
    result = await orch.connect_provider(
        "main", protocol="openai", base_url=None, model="m", api_key="")
    assert result["ok"] is False
    assert "缺少 API key" in result["message"]
    assert calls == []


async def test_list_models_keeps_metadata_and_override_separate(tmp_path: Path) -> None:
    """回归：目录缓存只存 provider 原始元数据，用户覆盖不得被洗成 provider 来源。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "display_name": "M", "capability": {
        "window": 8_000, "reasoning_returned": True, "reasoning_mode": "adjustable",
        "levels": ["off", "low"], "max_output": None, "tools": True, "source": "provider"}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", protocol="openai", base_url=None, model="m",
                                api_key="k")
    pid = _pid(orch)
    assert orch.set_model_capability(pid, "m", {
        "window": 50_000, "reasoning_mode": "adjustable",
        "levels": ["off", "medium"], "tools": False})["ok"] is True
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 50_000 and status["window_source"] == "user"
    assert status["tools"] is False

    await orch.list_models(refresh=True)  # 刷新目录不得污染来源/覆盖
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 50_000 and status["window_source"] == "user"
    assert status["tools"] is False and status["efforts"] == ["off", "medium"]

    from agent.core.provider_store import load_registry
    cached = load_registry(tmp_path / ".providers.toml")["providers"][pid]["models"]
    raw = next(m for m in cached if m["id"] == "m")["capability"]
    assert raw["window"] == 8_000 and raw["tools"] is True  # 缓存是原始元数据

    reopened = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=lambda **kw: FakeProvider([ChatResult(text="pong")]))
    restarted = reopened.provider_status()["roles"]["main"]
    assert restarted["window"] == 50_000 and restarted["window_source"] == "user"
    assert restarted["tools"] is False

    clearing = {"window": None, "max_output": None, "reasoning_mode": None,
                "levels": None, "tools": None, "reasoning_returned": None}
    reopened.set_model_capability(pid, "m", clearing)
    cleared = reopened.provider_status()["roles"]["main"]
    assert cleared["window"] == 8_000 and cleared["window_source"] == "provider"
    assert cleared["tools"] is True


async def test_static_env_key_provider_is_configured_and_catalog_usable(tmp_path: Path) -> None:
    """静态配置（config.toml/env）带入的 key：状态/目录必须判为已配置，且不落盘。"""
    import httpx2

    from agent.providers import OpenAICompatProvider

    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1", api_key="sk-static-env",
        model="deepseek-chat", http_client=httpx2.AsyncClient(trust_env=False),
    )

    async def canned():
        return [_adjustable_entry("deepseek-chat")]

    provider.list_models = canned
    listing_calls: list[dict] = []

    def factory(**kw):
        listing_calls.append(dict(kw))
        return provider

    orch = Orchestrator(main_provider=provider, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=factory)
    status = orch.provider_status()
    assert status["roles"]["main"]["configured"] is True
    assert status["providers"][0]["configured"] is True
    catalog = await orch.list_models(refresh=True)
    assert catalog["providers"][0]["configured"] is True
    assert catalog["providers"][0]["error"] is None
    assert catalog["providers"][0]["status"] == "ok"
    assert listing_calls and listing_calls[-1]["api_key"] == "sk-static-env"
    assert not (tmp_path / "keys.json").exists()  # 静态 key 只在内存，不落盘


async def test_explicit_openai_official_host_without_key_not_configured(tmp_path: Path) -> None:
    """显式官方 OpenAI 主机无 key：不可用，且目录不得用占位 key 外呼。"""
    _registry_for(tmp_path, "openai", "https://api.openai.com/v1", "gpt-5")
    calls: list[dict] = []
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: calls.append(kw) or FakeProvider([]))
    assert orch.main_provider is None
    assert orch.provider_status()["roles"]["main"]["configured"] is False
    catalog = await orch.list_models(refresh=True)
    assert "缺少凭据" in catalog["providers"][0]["error"]
    assert calls == []


async def test_adjustable_unknown_levels_effort_rejected_with_hint(tmp_path: Path) -> None:
    """可调但档位未声明：设置强度必须明确报错，不得静默。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    pid = _pid(orch)
    orch.set_model_capability(pid, "m", {"reasoning_mode": "adjustable"})
    status = orch.provider_status()["roles"]["main"]
    assert status["reasoning_mode"] == "adjustable" and status["efforts"] is None
    result = orch.set_reasoning_effort("main", "low")
    assert result["ok"] is False and "档位未知" in result["message"]


async def test_key_rotation_via_connect_beats_static_env_key(tmp_path: Path) -> None:
    """回归（审核 D1-r）：运行时 /connect 轮换后的 keys.json 必须压过静态 env key。"""
    import httpx2

    from agent.providers import OpenAICompatProvider

    static = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1", api_key="sk-old-env",
        model="deepseek-chat", http_client=httpx2.AsyncClient(trust_env=False),
    )
    orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: FakeProvider([ChatResult(text="pong")]))
    pid = derive_provider_id("openai", "https://api.deepseek.com/v1")
    assert orch._usable_key(pid) == "sk-old-env"
    result = await orch.connect_provider(
        "main", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="deepseek-chat", api_key="sk-new-rotated")
    assert result["ok"] is True
    assert orch._usable_key(pid) == "sk-new-rotated"

    reopened = Orchestrator(
        main_provider=OpenAICompatProvider(
            base_url="https://api.deepseek.com/v1", api_key="sk-old-env",
            model="deepseek-chat", http_client=httpx2.AsyncClient(trust_env=False)),
        executor_provider=None, root=tmp_path,
        store_path=tmp_path / ".providers.toml", keys_path=tmp_path / "keys.json",
        provider_factory=lambda **kw: FakeProvider([ChatResult(text="pong")]))
    assert reopened._usable_key(pid) == "sk-new-rotated"


async def test_adjustable_empty_levels_effort_rejected(tmp_path: Path) -> None:
    """明确声明为空档位：设置强度同样明确报错。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    pid = _pid(orch)
    orch.set_model_capability(pid, "m", {"reasoning_mode": "adjustable", "levels": []})
    result = orch.set_reasoning_effort("main", "low")
    assert result["ok"] is False and "档位未知或为空" in result["message"]


# ---------- 页面入口 _connect：密钥判定唯一权威（问题一） ----------


async def test_connect_message_official_custom_endpoint_empty_key_fails_before_factory(
    tmp_path: Path,
) -> None:
    """页面入口 custom+官方地址+空 key：创建 provider 前失败，零 factory，零落盘。"""
    calls: list[dict] = []
    orch = make_orchestrator(tmp_path, lambda **kw: calls.append(kw) or FakeProvider([]))
    queue = orch.subscribe()
    await orch._connect({
        "role": "main", "protocol": "openai",
        "base_url": "https://api.openai.com/v1", "model": "gpt-x", "api_key": "",
    })
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is False
    assert "缺少 API key" in result["message"]
    assert calls == []
    assert orch.main_provider is None
    assert orch.provider_status()["roles"]["main"]["configured"] is False
    assert not (tmp_path / ".providers.toml").exists()
    assert not (tmp_path / "keys.json").exists()


async def test_connect_message_local_endpoint_empty_key_still_allowed(tmp_path: Path) -> None:
    """页面入口 127.0.0.1 空 key：仍允许占位密钥并连接成功。"""
    calls: list[dict] = []

    def factory(**kw):
        calls.append(dict(kw))
        return FakeProvider([ChatResult(text="pong")])

    orch = make_orchestrator(tmp_path, factory)
    queue = orch.subscribe()
    await orch._connect({
        "role": "main", "protocol": "openai",
        "base_url": "http://127.0.0.1:9/v1", "model": "m", "api_key": "",
    })
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    assert calls and all(call["api_key"] == "unused" for call in calls)
    assert orch.main_provider is not None
    assert orch.provider_status()["roles"]["main"]["configured"] is True


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(dict(kw))
        return FakeProvider([])


async def test_connect_message_official_empty_key_rejected(tmp_path: Path) -> None:
    """页面入口：preset 官方服务商 / 显式官方地址 / 官方默认端点，空 key 全部拒绝。"""
    cases = [
        {"preset": "deepseek", "model": ""},
        {"protocol": "openai", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
        {"preset": "openai", "model": ""},
        {"preset": "anthropic", "model": ""},
    ]
    for i, case in enumerate(cases):
        d = tmp_path / f"case{i}"
        d.mkdir()
        recorder = _RecordingFactory()
        orch = make_orchestrator(d, recorder)
        queue = orch.subscribe()
        await orch._connect({"role": "main", "api_key": "", **case})
        result = await _until(queue, lambda e: e.get("type") == "provider_result")
        assert result["ok"] is False, case
        assert "缺少 API key" in result["message"], case
        assert recorder.calls == [], case
        assert orch.main_provider is None, case
        assert not (d / ".providers.toml").exists(), case
        assert not (d / "keys.json").exists(), case


async def test_connect_message_ollama_preset_empty_key_allowed(tmp_path: Path) -> None:
    """页面入口 Ollama preset（本地端点，needs_key=False）空 key 仍可用。"""
    calls: list[dict] = []
    orch = make_orchestrator(tmp_path, lambda **kw: calls.append(kw) or FakeProvider(
        [ChatResult(text="pong")]))
    queue = orch.subscribe()
    await orch._connect({"role": "main", "preset": "ollama", "model": "", "api_key": ""})
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    assert calls and calls[-1]["api_key"] == "unused"


# ---------- 重启时静态 provider 不得覆盖运行时密钥实例（问题二） ----------


def _static_openai(base_url: str, api_key: str, model: str):
    import httpx2

    from agent.providers import OpenAICompatProvider

    return OpenAICompatProvider(base_url=base_url, api_key=api_key, model=model,
                                http_client=httpx2.AsyncClient(trust_env=False))


async def test_restart_runtime_key_beats_static_provider(tmp_path: Path) -> None:
    """keys.json 有凭据时：active provider 必须是运行时实例（新 key），不是静态实例。"""
    base_url = "http://127.0.0.1:9107/v1"
    pid = derive_provider_id("openai", base_url)
    save_registry(tmp_path / ".providers.toml", {
        "roles": {"main": {"provider_id": pid, "model": "m-static", "reasoning_effort": "off"}},
        "providers": {pid: {"name": "x", "protocol": "openai", "base_url": base_url,
                            "model_overrides": {}, "models": [], "models_fetched_at": 0.0}},
    })
    save_key(pid, "sk-new-test", tmp_path / "keys.json")
    static = _static_openai(base_url, "sk-old-test", "m-static")
    made: list[dict] = []

    def factory(**kw):
        made.append(dict(kw))
        return _static_openai(kw["base_url"], kw["api_key"], kw["model"])

    orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=factory)
    assert orch.main_provider is not static
    assert orch.main_provider._api_key == "sk-new-test"
    assert orch.main_provider._client.api_key == "sk-new-test"  # SDK 实际鉴权凭据
    assert made and all(call["api_key"] == "sk-new-test" for call in made)
    assert orch._usable_key(pid) == "sk-new-test"
    assert "sk-old-test" not in (tmp_path / "keys.json").read_text(encoding="utf-8")


async def test_static_only_env_key_still_usable(tmp_path: Path) -> None:
    """只有静态 env 密钥、没有 keys.json：静态 provider 正常作为 active provider。"""
    base_url = "http://127.0.0.1:9107/v1"
    static = _static_openai(base_url, "sk-env", "m-static")
    orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: FakeProvider([]))
    pid = derive_provider_id("openai", base_url)
    assert orch.main_provider is static
    assert orch.provider_status()["roles"]["main"]["configured"] is True
    assert orch._usable_key(pid) == "sk-env"
    assert not (tmp_path / "keys.json").exists()


async def test_registry_binding_static_fallback_when_keys_missing(tmp_path: Path) -> None:
    """有注册表绑定但 keys.json 缺失：静态 provider 作为对应绑定的兜底。"""
    base_url = "http://127.0.0.1:9107/v1"
    pid = derive_provider_id("openai", base_url)
    save_registry(tmp_path / ".providers.toml", {
        "roles": {"main": {"provider_id": pid, "model": "m-static", "reasoning_effort": "off"}},
        "providers": {pid: {"name": "x", "protocol": "openai", "base_url": base_url,
                            "model_overrides": {}, "models": [], "models_fetched_at": 0.0}},
    })
    static = _static_openai(base_url, "sk-env", "m-static")
    orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: FakeProvider([]))
    assert orch.main_provider is static
    assert orch.provider_status()["roles"]["main"]["configured"] is True
    assert orch._usable_key(pid) == "sk-env"
    assert not (tmp_path / "keys.json").exists()


async def test_shared_provider_restart_keeps_new_key_and_role_efforts(tmp_path: Path) -> None:
    """main/executor 共用 provider 不同模型：重启后都用新 key，强度各自独立。"""
    base_url = "http://127.0.0.1:9107/v1"

    def make_factory(records):
        def factory(**kw):
            records.append(dict(kw))
            fake = FakeProvider([ChatResult(text="pong")])
            fake.models = [_adjustable_entry("m-main", ("off", "low")),
                           _adjustable_entry("m-exec", ("off", "high"))]
            return fake
        return factory

    seed: list[dict] = []
    orch = make_orchestrator(tmp_path, make_factory(seed))
    await orch.connect_provider("main", protocol="openai", base_url=base_url,
                                model="m-main", api_key="sk-k", reasoning_effort="low")
    await orch.connect_provider("executor", protocol="openai", base_url=base_url,
                                model="m-exec", api_key="sk-k", reasoning_effort="high")
    pid = _pid(orch)
    assert load_keys(tmp_path / "keys.json")[pid] == "sk-k"

    static = _static_openai(base_url, "sk-old-env", "m-main")  # 静态指向 main 的模型
    restarted: list[dict] = []

    def factory2(**kw):
        restarted.append(dict(kw))
        fake = FakeProvider([ChatResult(text="pong")])
        fake._api_key = kw["api_key"]
        return fake

    reopened = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json", provider_factory=factory2)
    assert reopened.main_provider is not static
    assert reopened.main_provider._api_key == "sk-k"
    assert reopened.executor_provider._api_key == "sk-k"
    assert all(call["api_key"] == "sk-k" for call in restarted)
    assert reopened.main_provider._reasoning_effort == "low"
    assert reopened.executor_provider._reasoning_effort == "high"
    assert reopened.main_provider is not reopened.executor_provider


async def test_rotation_active_provider_and_restart_use_latest_key(tmp_path: Path) -> None:
    """运行时轮换后：当前 active provider 与重启后的实例都必须用最新 key。"""
    base_url = "http://127.0.0.1:9107/v1"
    static = _static_openai(base_url, "sk-old-env", "m-static")

    def factory(**kw):
        fake = FakeProvider([ChatResult(text="pong")])
        fake._api_key = kw["api_key"]
        return fake

    orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=factory)
    pid = derive_provider_id("openai", base_url)
    assert orch.main_provider._api_key == "sk-old-env"
    result = await orch.connect_provider("main", protocol="openai", base_url=base_url,
                                         model="m-static", api_key="sk-rotated")
    assert result["ok"] is True
    assert orch.main_provider._api_key == "sk-rotated"
    assert orch._usable_key(pid) == "sk-rotated"

    reopened = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json", provider_factory=factory)
    assert reopened.main_provider is not static
    assert reopened.main_provider._api_key == "sk-rotated"


async def test_legacy_unused_key_is_ignored_for_official_endpoint(tmp_path: Path) -> None:
    """回归（审核残留）：旧 bug 写入的 keys.json "unused" 不得被官方端点当凭据。"""
    _registry_for(tmp_path, "openai", "https://api.deepseek.com/v1", "deepseek-chat")
    pid = derive_provider_id("openai", "https://api.deepseek.com/v1")
    save_key(pid, "unused", tmp_path / "keys.json")
    calls: list[dict] = []
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: calls.append(kw) or FakeProvider([]))
    assert orch._usable_key(pid) is None
    assert orch.main_provider is None
    assert orch.provider_status()["roles"]["main"]["configured"] is False
    catalog = await orch.list_models(refresh=True)
    assert "缺少凭据" in catalog["providers"][0]["error"]
    assert calls == []


async def test_legacy_unused_key_still_placeholder_for_local_endpoint(tmp_path: Path) -> None:
    """本地端点：历史 "unused" 仍等同占位密钥，不影响可用性。"""
    _registry_for(tmp_path, "openai", "http://127.0.0.1:9/v1", "m")
    pid = derive_provider_id("openai", "http://127.0.0.1:9/v1")
    save_key(pid, "unused", tmp_path / "keys.json")
    calls: list[dict] = []
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: calls.append(kw) or FakeProvider([]))
    assert orch._usable_key(pid) == "unused"
    assert orch.main_provider is not None
    assert orch.provider_status()["roles"]["main"]["configured"] is True


async def test_static_env_key_survives_legacy_unused_in_keys_json(tmp_path: Path) -> None:
    """静态 env key + 遗留 keys.json "unused"：静态 key 必须生效（不得被短路挤掉）。"""
    base_url = "https://api.deepseek.com/v1"
    pid = derive_provider_id("openai", base_url)
    save_key(pid, "unused", tmp_path / "keys.json")
    static = _static_openai(base_url, "sk-env-real", "m-env")

    async def canned():
        return [_adjustable_entry("m-env")]

    static.list_models = canned
    made: list[dict] = []

    def factory(**kw):
        made.append(dict(kw))
        return static

    orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=factory)
    assert orch._usable_key(pid) == "sk-env-real"
    assert orch.provider_status()["roles"]["main"]["configured"] is True
    catalog = await orch.list_models(refresh=True)
    assert catalog["providers"][0]["configured"] is True
    assert catalog["providers"][0]["error"] is None
    assert made and all(call["api_key"] == "sk-env-real" for call in made)


async def test_legacy_unused_does_not_block_static_fallback(tmp_path: Path) -> None:
    """注册表绑定 + 遗留 keys.json "unused"：静态 env key 必须作为兜底生效。"""
    base_url = "https://api.deepseek.com/v1"
    pid = derive_provider_id("openai", base_url)
    save_registry(tmp_path / ".providers.toml", {
        "roles": {"main": {"provider_id": pid, "model": "m-env", "reasoning_effort": "off"}},
        "providers": {pid: {"name": "x", "protocol": "openai", "base_url": base_url,
                            "model_overrides": {}, "models": [], "models_fetched_at": 0.0}},
    })
    save_key(pid, "unused", tmp_path / "keys.json")
    static = _static_openai(base_url, "sk-env-real", "m-env")
    calls: list[dict] = []
    orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: calls.append(kw) or FakeProvider([]))
    assert orch.main_provider is static
    assert orch.provider_status()["roles"]["main"]["configured"] is True
    assert orch._usable_key(pid) == "sk-env-real"
    assert calls == []  # 无需重建，静态实例即兜底


# ---------- 问题一：历史 "unused" 不得绕过官方端点密钥检查 ----------


def _seed_registry(tmp_path: Path, providers: dict, roles: dict) -> None:
    save_registry(tmp_path / ".providers.toml", {"roles": roles, "providers": providers})


def _provider_entry(name: str, protocol: str, base_url, models=None, fetched_at=0.0,
                    overrides=None) -> dict:
    return {"name": name, "protocol": protocol, "base_url": base_url,
            "model_overrides": overrides or {}, "models": models or [],
            "models_fetched_at": fetched_at}


def _model(model_id: str, window=None, mode="unknown", levels=None, tools=None) -> dict:
    returned = True if mode == "adjustable" else None
    return {"id": model_id, "display_name": model_id, "provider_id": "x", "provider": "x",
            "capability": {"window": window, "reasoning_returned": returned,
                           "reasoning_mode": mode, "levels": levels, "max_output": None,
                           "tools": tools, "source": "provider"}}


    def __call__(self, **kw):
        self.calls.append(dict(kw))
        return FakeProvider([])


async def test_connect_message_legacy_unused_official_empty_key_fails(tmp_path: Path) -> None:
    """页面入口：keys.json 历史 "unused" + 官方地址 + 空 key → 失败、factory=0、未配置。"""
    base = "https://api.openai.com/v1"
    pid = derive_provider_id("openai", base)
    _seed_registry(tmp_path,
                   {pid: _provider_entry("openai", "openai", base)},
                   {"main": {"provider_id": pid, "model": "gpt-x", "reasoning_effort": "off"}})
    save_key(pid, "unused", tmp_path / "keys.json")
    recorder = _RecordingFactory()
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=recorder)
    queue = orch.subscribe()
    await orch._connect({"role": "main", "protocol": "openai",
                         "base_url": base, "model": "gpt-x", "api_key": ""})
    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is False and "缺少 API key" in result["message"]
    assert recorder.calls == []
    assert orch.main_provider is None
    assert orch.provider_status()["roles"]["main"]["configured"] is False
    assert load_keys(tmp_path / "keys.json").get(pid) is None  # 启动即清理遗留占位


async def test_direct_connect_provider_legacy_unused_fails(tmp_path: Path) -> None:
    """直接调用 connect_provider：启动后写入的历史 "unused" 也不得当凭据。"""
    base = "https://api.openai.com/v1"
    pid = derive_provider_id("openai", base)
    recorder = _RecordingFactory()
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=recorder)
    save_key(pid, "unused", tmp_path / "keys.json")  # 启动后残留（模拟旧版运行中写入）
    result = await orch.connect_provider(
        "main", protocol="openai", base_url=base, model="gpt-x", api_key="")
    assert result["ok"] is False and "缺少 API key" in result["message"]
    assert recorder.calls == []
    assert orch.main_provider is None
    assert not (tmp_path / ".providers.toml").exists()


async def test_connect_provider_user_typed_unused_rejected(tmp_path: Path) -> None:
    """用户在官方端点 key 输入框直接输入 "unused"：同样视为无凭据。"""
    recorder = _RecordingFactory()
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", provider_factory=recorder)
    result = await orch.connect_provider(
        "main", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="deepseek-chat", api_key="unused")
    assert result["ok"] is False and "缺少 API key" in result["message"]
    assert recorder.calls == []
    assert orch.main_provider is None


async def test_static_provider_unused_key_official_not_configured(tmp_path: Path) -> None:
    """静态 provider 携带 _api_key="unused" + 官方端点：不得视为已配置。"""
    for base in ("https://api.openai.com/v1", "https://api.deepseek.com/v1"):
        static = _static_openai(base, "unused", "m")
        orch = Orchestrator(main_provider=static, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=lambda **kw: FakeProvider([]))
        assert orch.main_provider is None, base
        assert orch.provider_status()["roles"]["main"]["configured"] is False, base


async def test_startup_cleans_legacy_unused_only_for_official(tmp_path: Path) -> None:
    """启动清理：官方 provider 的 "unused" 删除；真实密钥与未知条目保留。"""
    official = derive_provider_id("openai", "https://api.deepseek.com/v1")
    local = derive_provider_id("openai", "http://127.0.0.1:9/v1")
    unknown = "p-unknown"
    _seed_registry(tmp_path, {
        official: _provider_entry("official", "openai", "https://api.deepseek.com/v1"),
        local: _provider_entry("local", "openai", "http://127.0.0.1:9/v1"),
    }, {
        "main": {"provider_id": official, "model": "m", "reasoning_effort": "off"},
        "executor": {"provider_id": local, "model": "m", "reasoning_effort": "off"},
    })
    save_key(official, "unused", tmp_path / "keys.json")
    save_key(local, "sk-real-local", tmp_path / "keys.json")
    save_key(unknown, "unused", tmp_path / "keys.json")
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json",
                        provider_factory=lambda **kw: FakeProvider([]))
    keys = load_keys(tmp_path / "keys.json")
    assert official not in keys                      # 官方遗留占位被清理
    assert keys[local] == "sk-real-local"            # 本地真实密钥保留
    assert keys[unknown] == "unused"                 # 未知 provider 不误删
    assert orch.provider_status()["roles"]["main"]["configured"] is False
    assert orch.provider_status()["roles"]["executor"]["configured"] is True


# ---------- 问题二：模型目录刷新必须持久化（原始元数据 + fetched_at） ----------


async def test_list_models_refresh_persists_and_restart_recovers(tmp_path: Path) -> None:
    """初始无目录：刷新成功即落盘；重启命中缓存，无需再请求 provider。"""
    base = "https://api.deepseek.com/v1"
    pid = _registry_for(tmp_path, "openai", base, "m1")
    save_key(pid, "sk-k", tmp_path / "keys.json")
    fake = FakeProvider([])
    fake.models = [_model("m1", window=77_777, mode="adjustable", levels=["off", "low"])]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    catalog = await orch.list_models(refresh=True)
    entry = catalog["providers"][0]
    assert entry["status"] == "ok" and entry["error"] is None
    assert entry["models"][0]["capability"]["window"] == 77_777

    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert saved["models"][0]["id"] == "m1"
    assert saved["models"][0]["capability"]["window"] == 77_777
    assert saved["models_fetched_at"] > 0
    assert "sk-k" not in (tmp_path / ".providers.toml").read_text(encoding="utf-8")

    fake2 = FakeProvider([])
    fake2.models = RuntimeError("不应被调用")
    reopened = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=lambda **kw: fake2)
    assert reopened.context_window("main") == (77_777, "provider")
    catalog2 = await reopened.list_models()  # 命中持久化缓存
    assert fake2.models_calls == 0
    assert catalog2["providers"][0]["models"][0]["capability"]["window"] == 77_777


async def test_list_models_refresh_replaces_catalog_on_disk(tmp_path: Path) -> None:
    """已有旧目录：刷新后文件变成新原始元数据，重启后仍是新目录。"""
    base = "https://api.deepseek.com/v1"
    pid = _registry_for(tmp_path, "openai", base, "m1")
    save_key(pid, "sk-k", tmp_path / "keys.json")
    old = [_model("m1", window=1_000, mode="adjustable", levels=["off", "low"])]
    _seed_registry(tmp_path,
                   {pid: _provider_entry("x", "openai", base, models=old, fetched_at=1.0)},
                   {"main": {"provider_id": pid, "model": "m1", "reasoning_effort": "off"}})
    fake = FakeProvider([])
    fake.models = [_model("m1", window=2_000, mode="adjustable", levels=["off", "medium"])]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.list_models(refresh=True)
    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert saved["models"][0]["capability"]["window"] == 2_000
    reopened = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=lambda **kw: FakeProvider([]))
    assert reopened.context_window("main") == (2_000, "provider")
    assert reopened.provider_status()["roles"]["main"]["efforts"] == ["off", "medium"]


async def test_list_models_refresh_persists_raw_not_merged_override(tmp_path: Path) -> None:
    """有用户覆盖：刷新落盘的是原始元数据；重启最终能力仍为用户覆盖，清除后回新元数据。"""
    base = "https://api.deepseek.com/v1"
    pid = _registry_for(tmp_path, "openai", base, "m1")
    save_key(pid, "sk-k", tmp_path / "keys.json")
    override = {"window": 50_000, "reasoning_mode": "adjustable", "levels": ["off", "low"]}
    _seed_registry(tmp_path,
                   {pid: _provider_entry("x", "openai", base, overrides={"m1": override})},
                   {"main": {"provider_id": pid, "model": "m1", "reasoning_effort": "off"}})
    fake = FakeProvider([])
    fake.models = [_model("m1", window=2_000, mode="adjustable", levels=["off", "high"])]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.list_models(refresh=True)
    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert saved["models"][0]["capability"]["window"] == 2_000  # 原始值，未被覆盖 50000 污染

    reopened = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=lambda **kw: FakeProvider([]))
    status = reopened.provider_status()["roles"]["main"]
    assert status["window"] == 50_000 and status["window_source"] == "user"
    assert status["efforts"] == ["off", "low"]

    clearing = {"window": None, "max_output": None, "reasoning_mode": None,
                "levels": None, "tools": None, "reasoning_returned": None}
    reopened.set_model_capability(pid, "m1", clearing)
    cleared = reopened.provider_status()["roles"]["main"]
    assert cleared["window"] == 2_000 and cleared["window_source"] == "provider"
    assert cleared["efforts"] == ["off", "high"]


async def test_list_models_failure_keeps_last_catalog(tmp_path: Path) -> None:
    """刷新失败：明确报错、内存与磁盘旧缓存都不丢，重启后旧缓存仍在。"""
    base = "https://api.deepseek.com/v1"
    pid = _registry_for(tmp_path, "openai", base, "m-old")
    save_key(pid, "sk-k", tmp_path / "keys.json")
    old = [_model("m-old", window=1_234, mode="adjustable", levels=["off", "low"])]
    _seed_registry(tmp_path,
                   {pid: _provider_entry("x", "openai", base, models=old, fetched_at=123.0)},
                   {"main": {"provider_id": pid, "model": "m-old", "reasoning_effort": "off"}})
    fake = FakeProvider([])
    fake.models = RuntimeError("接口 500")
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    catalog = await orch.list_models(refresh=True)
    entry = catalog["providers"][0]
    assert entry["status"] == "error" and "无法获取模型列表" in entry["error"]
    assert [m["id"] for m in entry["models"]] == ["m-old"]  # 内存缓存保留

    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert [m["id"] for m in saved["models"]] == ["m-old"]
    assert saved["models_fetched_at"] == 123.0
    text = (tmp_path / ".providers.toml").read_text(encoding="utf-8")
    assert "sk-k" not in text and "接口 500" not in text  # 错误正文与密钥不落盘

    reopened = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=lambda **kw: FakeProvider([]))
    assert reopened.context_window("main") == (1_234, "provider")
    recatalog = await reopened.list_models(refresh=True)
    assert recatalog["providers"][0]["models"][0]["id"] == "m-old"


async def test_multi_provider_refresh_isolated_persistence(tmp_path: Path) -> None:
    """多 provider：A 成功落盘新目录；B 失败保留旧缓存；互不覆盖。"""
    base_a = "https://api.deepseek.com/v1"
    base_b = "http://127.0.0.1:9/v1"
    pid_a = derive_provider_id("openai", base_a)
    pid_b = derive_provider_id("openai", base_b)
    old_b = [_model("b-old", window=500, mode="adjustable", levels=["off", "low"])]
    _seed_registry(tmp_path, {
        pid_a: _provider_entry("A", "openai", base_a, models=[], fetched_at=0.0),
        pid_b: _provider_entry("B", "openai", base_b, models=old_b, fetched_at=55.0),
    }, {
        "main": {"provider_id": pid_a, "model": "a1", "reasoning_effort": "off"},
        "executor": {"provider_id": pid_b, "model": "b-old", "reasoning_effort": "off"},
    })
    save_key(pid_a, "sk-a", tmp_path / "keys.json")
    fake_a = FakeProvider([])
    fake_a.models = [_model("a1", window=1_000)]
    fake_b = FakeProvider([])
    fake_b.models = RuntimeError("B 挂了")

    def factory(**kw):
        return fake_a if kw.get("base_url") == base_a else fake_b

    orch = make_orchestrator(tmp_path, factory)
    catalog = await orch.list_models(refresh=True)
    by_id = {p["id"]: p for p in catalog["providers"]}
    assert by_id[pid_a]["status"] == "ok"
    assert by_id[pid_b]["status"] == "error"
    assert [m["id"] for m in by_id[pid_b]["models"]] == ["b-old"]

    saved = load_registry(tmp_path / ".providers.toml")["providers"]
    assert saved[pid_a]["models"][0]["id"] == "a1"
    assert saved[pid_a]["models_fetched_at"] > 0
    assert [m["id"] for m in saved[pid_b]["models"]] == ["b-old"]
    assert saved[pid_b]["models_fetched_at"] == 55.0


async def test_list_models_cache_hit_no_write_and_stale_refreshes(tmp_path: Path) -> None:
    """缓存命中不写盘；fetched_at 过期才刷新并更新落盘时间。"""
    import time as _time

    base = "https://api.deepseek.com/v1"
    pid = _registry_for(tmp_path, "openai", base, "m1")
    save_key(pid, "sk-k", tmp_path / "keys.json")
    fresh = [_model("m1", window=900)]
    _seed_registry(tmp_path,
                   {pid: _provider_entry("x", "openai", base, models=fresh,
                                         fetched_at=_time.time())},
                   {"main": {"provider_id": pid, "model": "m1", "reasoning_effort": "off"}})
    fake = FakeProvider([])
    fake.models = [_model("m1", window=999)]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    before = (tmp_path / ".providers.toml").read_bytes()
    catalog = await orch.list_models()  # 命中有效缓存
    assert fake.models_calls == 0
    assert (tmp_path / ".providers.toml").read_bytes() == before  # 无意义写盘
    assert catalog["providers"][0]["models"][0]["capability"]["window"] == 900

    stale = [_model("m1", window=900)]
    _seed_registry(tmp_path,
                   {pid: _provider_entry("x", "openai", base, models=stale, fetched_at=0.0)},
                   {"main": {"provider_id": pid, "model": "m1", "reasoning_effort": "off"}})
    fake2 = FakeProvider([])
    fake2.models = [_model("m1", window=999)]
    orch2 = make_orchestrator(tmp_path, lambda **kw: fake2)
    await orch2.list_models()
    assert fake2.models_calls == 1
    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert saved["models"][0]["capability"]["window"] == 999
    assert saved["models_fetched_at"] > 0


async def test_reconnect_listing_failure_keeps_catalog(tmp_path: Path) -> None:
    """回归（审核同族漏洞）：重连时目录拉取失败不得清空/覆盖已持久化目录。"""
    base = "https://api.deepseek.com/v1"
    fake = FakeProvider([ChatResult(text="pong"), ChatResult(text="pong"),
                         ChatResult(text="pong"), ChatResult(text="pong")])
    fake.models = [_model("m1", window=12_345, mode="adjustable", levels=["off", "low"])]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url=base, model="m1", api_key="sk-k"))["ok"] is True
    pid = _pid(orch)
    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert saved["models"][0]["capability"]["window"] == 12_345
    fetched_before = saved["models_fetched_at"]

    fake.models = RuntimeError("临时 500")  # 重连时目录拉取失败
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url=base, model="m1", api_key="sk-k"))["ok"] is True
    catalog = orch.model_catalog()
    entry = catalog["providers"][0]
    assert entry["status"] == "error" and "无法获取模型列表" in entry["error"]
    assert [m["id"] for m in entry["models"]] == ["m1"]  # 内存保留

    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert [m["id"] for m in saved["models"]] == ["m1"]  # 磁盘保留
    assert saved["models_fetched_at"] == fetched_before
    text = (tmp_path / ".providers.toml").read_text(encoding="utf-8")
    assert "sk-k" not in text and "临时 500" not in text

    reopened = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                            store_path=tmp_path / ".providers.toml",
                            keys_path=tmp_path / "keys.json",
                            provider_factory=lambda **kw: FakeProvider([]))
    assert reopened.context_window("main") == (12_345, "provider")


async def test_reconnect_empty_listing_keeps_catalog(tmp_path: Path) -> None:
    """回归：重连返回空列表同样只报错、不清空旧目录。"""
    base = "https://api.deepseek.com/v1"
    fake = FakeProvider([ChatResult(text="pong"), ChatResult(text="pong")])
    fake.models = [_model("m1", window=6_000)]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", protocol="openai", base_url=base,
                                model="m1", api_key="sk-k")
    pid = _pid(orch)
    fake.models = []  # 空列表
    await orch.connect_provider("main", protocol="openai", base_url=base,
                                model="m1", api_key="sk-k")
    saved = load_registry(tmp_path / ".providers.toml")["providers"][pid]
    assert [m["id"] for m in saved["models"]] == ["m1"]
    assert "无法获取模型列表" in orch.model_catalog()["providers"][0]["error"]


# ---------- 旧角色名凭据迁移（真实用户回归：重启后 active provider 必须用真实 key） ----------

_REAL = "sk-real-deepseek-0000000000000001"   # 模拟真实 key 长度
_SHORT = "sk-x"                                # 旧版残留/测试占位


def _bind_deepseek(tmp_path: Path, model: str = "deepseek-chat") -> str:
    base = "https://api.deepseek.com/v1"
    pid = derive_provider_id("openai", base)
    _seed_registry(tmp_path,
                   {pid: _provider_entry("deepseek", "openai", base)},
                   {"main": {"provider_id": pid, "model": model, "reasoning_effort": "off"},
                    "executor": {"provider_id": pid, "model": model, "reasoning_effort": "off"}})
    return pid


def test_legacy_role_keys_migrate_when_pid_missing(tmp_path: Path) -> None:
    pid = _bind_deepseek(tmp_path)
    save_key("main", _REAL, tmp_path / "keys.json")
    save_key("executor", _REAL, tmp_path / "keys.json")
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json")
    keys = load_keys(tmp_path / "keys.json")
    assert keys[pid] == _REAL
    assert "main" not in keys and "executor" not in keys
    assert orch.main_provider._api_key == _REAL
    assert orch.executor_provider._api_key == _REAL


def test_legacy_role_key_migrates_over_placeholder_pid(tmp_path: Path) -> None:
    pid = _bind_deepseek(tmp_path)
    save_key("main", _REAL, tmp_path / "keys.json")
    save_key(pid, _SHORT, tmp_path / "keys.json")   # 旧迁移写入的残留短值
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json")
    keys = load_keys(tmp_path / "keys.json")
    assert keys[pid] == _REAL
    assert "main" not in keys
    assert orch._usable_key(pid) == _REAL
    assert orch.main_provider._api_key == _REAL


def test_legacy_role_key_migrates_over_unused_pid(tmp_path: Path) -> None:
    pid = _bind_deepseek(tmp_path)
    save_key("main", _REAL, tmp_path / "keys.json")
    save_key(pid, "unused", tmp_path / "keys.json")
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json")
    keys = load_keys(tmp_path / "keys.json")
    assert keys[pid] == _REAL
    assert orch.main_provider._api_key == _REAL


def test_legacy_role_key_does_not_override_real_pid_key(tmp_path: Path) -> None:
    pid = _bind_deepseek(tmp_path)
    other = "sk-newer-rotated-0000000000000002"
    save_key("main", _REAL, tmp_path / "keys.json")
    save_key(pid, other, tmp_path / "keys.json")
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json")
    keys = load_keys(tmp_path / "keys.json")
    assert keys[pid] == other            # 运行时新凭据优先
    assert keys["main"] == _REAL        # 旧值保留，不丢失
    assert orch.main_provider._api_key == other


def test_legacy_role_key_short_not_migrated(tmp_path: Path) -> None:
    pid = _bind_deepseek(tmp_path)
    save_key("main", _SHORT, tmp_path / "keys.json")
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json")
    keys = load_keys(tmp_path / "keys.json")
    assert pid not in keys
    assert keys["main"] == _SHORT
    assert orch.main_provider is None    # 无可信用凭据 → 未配置（不会假装已连接）


def test_legacy_role_keys_unused_not_migrated(tmp_path: Path) -> None:
    pid = _bind_deepseek(tmp_path)
    save_key("main", "unused", tmp_path / "keys.json")
    Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                 store_path=tmp_path / ".providers.toml",
                 keys_path=tmp_path / "keys.json")
    keys = load_keys(tmp_path / "keys.json")
    assert pid not in keys and keys["main"] == "unused"


def test_legacy_role_keys_shared_pid_keeps_first_deterministic(tmp_path: Path) -> None:
    """两角色同 pid 且旧角色键不同：迁移后 pid 取 main 的 key，另一角色键保留不丢。"""
    pid = _bind_deepseek(tmp_path)
    main_key = "sk-main-0000000000000000000001"
    exec_key = "sk-exec-0000000000000000000002"
    save_key("main", main_key, tmp_path / "keys.json")
    save_key("executor", exec_key, tmp_path / "keys.json")
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json")
    keys = load_keys(tmp_path / "keys.json")
    assert keys[pid] == main_key          # 先迁移 main，executor 不覆盖
    assert keys["executor"] == exec_key   # 不同值的旧凭据保留不丢
    assert "main" not in keys
    assert orch.main_provider._api_key == main_key
    assert orch.executor_provider._api_key == main_key


# ---------- 服务商级连接：凭据全局共享 + 自动默认模型（对齐 opencode 体验） ----------


def _entry(model_id: str, levels=None) -> dict:
    return {"id": model_id, "display_name": model_id, "capability": {
        "window": None, "reasoning_returned": None, "reasoning_mode": "unknown",
        "levels": levels, "max_output": None, "tools": None, "source": "provider"}}


async def test_provider_connect_binds_unconfigured_roles_with_default(tmp_path: Path) -> None:
    """role=""：一次连接把未配置的 main/executor 都绑到默认模型，凭据只保存一份。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [_entry("other-model"), _entry("deepseek-flash")]

    def factory(**kw):
        fake._api_key = kw["api_key"]
        return fake

    orch = make_orchestrator(tmp_path, factory)
    result = await orch.connect_provider(
        "", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="", api_key="sk-shared", preset="deepseek")
    assert result["ok"] is True, result
    assert "main" in result["message"] and "executor" in result["message"]
    status = orch.provider_status()
    assert status["roles"]["main"]["model"] == "deepseek-flash"   # 预设默认，且在列表内
    assert status["roles"]["executor"]["model"] == "deepseek-flash"
    pid = status["roles"]["main"]["provider_id"]
    assert pid == status["roles"]["executor"]["provider_id"]
    assert status["presets"]["deepseek"]["configured"] is True
    keys = load_keys(tmp_path / "keys.json")
    assert list(keys.values()) == ["sk-shared"]                  # 凭据只有一份
    assert orch.main_provider._api_key == "sk-shared"
    assert orch.executor_provider._api_key == "sk-shared"


async def test_provider_connect_keeps_existing_bindings(tmp_path: Path) -> None:
    """role=""：已绑定的角色不被改动，只补未绑定的角色。"""
    fake_a = FakeProvider([ChatResult(text="pong")])
    fake_a.models = [_entry("a1")]
    fake_b = FakeProvider([ChatResult(text="pong")])
    fake_b.models = [_entry("b1")]

    def factory(**kw):
        return fake_b if kw.get("base_url") == "http://127.0.0.1:9/v1" else fake_a

    orch = make_orchestrator(tmp_path, factory)
    assert (await orch.connect_provider(
        "main", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="a1", api_key="sk-a"))["ok"] is True
    result = await orch.connect_provider(
        "", protocol="openai", base_url="http://127.0.0.1:9/v1",
        model="", api_key="")
    assert result["ok"] is True, result
    status = orch.provider_status()
    assert status["roles"]["main"]["model"] == "a1"   # 未被动
    assert status["roles"]["executor"]["model"] == "b1"
    assert "main" not in result["message"]


async def test_provider_connect_falls_back_to_first_listed(tmp_path: Path) -> None:
    """预设默认模型不在真实列表中 → 选列表第一个。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [_entry("deepseek-v4-pro"), _entry("deepseek-v3.2")]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider(
        "", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="", api_key="sk-x", preset="deepseek")
    assert result["ok"] is True
    assert orch.provider_status()["roles"]["main"]["model"] == "deepseek-v4-pro"


async def test_custom_endpoint_connect_without_model_fails_when_no_listing(tmp_path: Path) -> None:
    """自定义端点：目录为空且无预设默认模型 → 明确报错，不静默。"""
    fake = FakeProvider([ChatResult(text="pong")])  # models 默认 []
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider(
        "", protocol="openai", base_url="http://127.0.0.1:9/v1", model="", api_key="")
    assert result["ok"] is False
    assert "无法确定使用的模型" in result["message"]
    assert orch.main_provider is None and orch.executor_provider is None


async def test_custom_endpoint_connect_with_manual_model_binds_roles(tmp_path: Path) -> None:
    """高级自定义端点：显式给模型时按该模型绑定两个角色。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider(
        "", protocol="openai", base_url="http://127.0.0.1:9/v1",
        model="my-model", api_key="")
    assert result["ok"] is True
    status = orch.provider_status()
    assert status["roles"]["main"]["model"] == "my-model"
    assert status["roles"]["executor"]["model"] == "my-model"
    assert status["roles"]["main"]["configured"] is True


async def test_shared_credentials_reused_across_roles_and_models(tmp_path: Path) -> None:
    """共享凭据：换模型不需要重新输入 key，两个角色的请求都用同一份。"""
    made: list[dict] = []

    def factory(**kw):
        made.append(dict(kw))
        fake = FakeProvider([ChatResult(text="pong")])
        fake.models = [_entry("m1"), _entry("m2")]
        fake._api_key = kw["api_key"]
        return fake

    orch = make_orchestrator(tmp_path, factory)
    await orch.connect_provider("", protocol="openai", base_url="https://api.deepseek.com/v1",
                                model="", api_key="sk-shared", preset="deepseek")
    pid = derive_provider_id("openai", "https://api.deepseek.com/v1")
    assert (await orch.set_model("executor", pid, "m2"))["ok"] is True
    assert all(call["api_key"] == "sk-shared" for call in made)
    assert orch.main_provider._api_key == "sk-shared"
    assert orch.executor_provider._api_key == "sk-shared"


async def test_provider_reconnect_refreshes_active_credentials(tmp_path: Path) -> None:
    """回归：服务商级换 key 重连后，已绑定角色的 active 实例必须立即用新 key（无需重启）。"""
    made: list[dict] = []

    def factory(**kw):
        made.append(dict(kw))
        fake = FakeProvider([ChatResult(text="pong")])
        fake.models = [_entry("m1")]
        fake._api_key = kw["api_key"]
        return fake

    orch = make_orchestrator(tmp_path, factory)
    base = "https://api.deepseek.com/v1"
    assert (await orch.connect_provider(
        "", protocol="openai", base_url=base, model="m1", api_key="sk-old"))["ok"] is True
    assert orch.main_provider._api_key == "sk-old"
    assert (await orch.connect_provider(
        "", protocol="openai", base_url=base, model="m1", api_key="sk-new"))["ok"] is True
    assert orch.main_provider._api_key == "sk-new"
    assert orch.executor_provider._api_key == "sk-new"
    assert made[-1]["api_key"] == "sk-new"
    status = orch.provider_status()
    assert status["roles"]["main"]["model"] == "m1"  # 已绑定模型不被默认值覆盖
    assert status["roles"]["executor"]["model"] == "m1"


async def test_provider_connect_takes_over_unusable_role_binding(tmp_path: Path) -> None:
    """角色绑定指向缺凭据的官方 provider 时，新连接把它接管为可用绑定。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [_entry("m1")]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    official_pid = derive_provider_id("openai", None)
    orch._providers[official_pid] = {
        "name": "openai（默认端点）", "protocol": "openai", "base_url": None,
        "model_overrides": {}, "models": [], "error": None, "fetched_at": 0.0}
    orch._roles["main"] = {"provider_id": official_pid, "model": "gpt-x",
                           "reasoning_effort": "off"}
    result = await orch.connect_provider(
        "", protocol="openai", base_url="http://127.0.0.1:9/v1", model="m1", api_key="")
    assert result["ok"] is True, result
    status = orch.provider_status()
    assert status["roles"]["main"]["configured"] is True
    assert status["roles"]["main"]["model"] == "m1"     # 被接管
    assert status["roles"]["executor"]["model"] == "m1"


async def test_custom_anthropic_local_endpoint_allows_empty_key(tmp_path: Path) -> None:
    """显式本地 anthropic 端点：空 key 允许占位（与 UI 提示一致）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [_entry("claude-x")]
    made: list[dict] = []

    def factory(**kw):
        made.append(dict(kw))
        return fake

    orch = make_orchestrator(tmp_path, factory)
    result = await orch.connect_provider(
        "", protocol="anthropic", base_url="http://127.0.0.1:9", model="claude-x", api_key="")
    assert result["ok"] is True, result
    assert made[-1]["api_key"] == "unused"
    assert orch.main_provider is not None and orch.executor_provider is not None


async def test_official_anthropic_host_still_requires_key(tmp_path: Path) -> None:
    """官方 anthropic 主机（含显式 api.anthropic.com）空 key 仍拒绝。"""
    cases = (None, "https://api.anthropic.com", "https://api.anthropic.com.",
             "api.anthropic.com")
    for i, base in enumerate(cases):
        d = tmp_path / f"case{i}"
        d.mkdir()
        recorder = _RecordingFactory()
        orch = make_orchestrator(d, recorder)
        result = await orch.connect_provider(
            "", protocol="anthropic", base_url=base, model="claude-x", api_key="")
        assert result["ok"] is False, base
        assert "缺少 API key" in result["message"], base
        assert recorder.calls == [], base


async def test_malformed_base_url_does_not_crash(tmp_path: Path) -> None:
    """畸形 URL（如截断 IPv6）：不得抛异常，空 key 走“缺少 API key”拒绝。"""
    recorder = _RecordingFactory()
    orch = make_orchestrator(tmp_path, recorder)
    result = await orch.connect_provider(
        "", protocol="openai", base_url="https://[::1", model="m", api_key="")
    assert result["ok"] is False
    assert "缺少 API key" in result["message"]
    assert recorder.calls == []


# ---------- models.dev 公共目录补齐能力（离线夹具 + 注入 fetcher） ----------

_CATALOG_FIXTURE = {
    "deepseek": {"models": {
        "ds-a": {"reasoning": True, "tool_call": True,
                 "reasoning_options": [{"type": "toggle"},
                                       {"type": "effort", "values": ["low", "high", "max"]}],
                 "interleaved": {"field": "reasoning_content"},
                 "limit": {"context": 1_000_000, "output": 384_000}},
    }},
}


def _catalog_orch(tmp_path, models, *, catalog=(), fresh_cache=False):
    """构建带夹具 catalog 的 Orchestrator。"""
    providers = _CATALOG_FIXTURE if "deepseek" in catalog else {}

    async def fetcher():
        return providers

    orch = make_orchestrator(tmp_path, lambda **kw: None, store=False)
    orch.providers._catalog_fetcher = fetcher
    return orch


async def test_catalog_fills_unknown_capability_after_connect(tmp_path: Path) -> None:
    """连接后：provider 未返回能力时由公共目录补齐（窗口/档位/最大输出/tools）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "ds-a", "display_name": "A", "capability": {}}]  # 无能力字段

    async def fetcher():
        return _CATALOG_FIXTURE

    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider(
        "", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="ds-a", api_key="sk-shared", preset="deepseek")
    assert result["ok"] is True, result
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 1_000_000 and status["window_source"] == "catalog"
    assert status["reasoning_mode"] == "adjustable"
    assert status["efforts"] == ["off", "low", "high", "max"]
    assert status["tools"] is True
    # 目录里的模型行也带能力（展示用）
    catalog = await orch.list_models()
    models = catalog["providers"][0]["models"]
    assert models[0]["capability"]["window"] == 1_000_000
    # 注册表仍只存原始 provider 元数据（不把目录数据写进去）
    saved = load_registry(tmp_path / ".providers.toml")["providers"]
    pid = derive_provider_id("openai", "https://api.deepseek.com/v1")
    raw_cap = saved[pid]["models"][0]["capability"]
    assert raw_cap == {"source": "provider"} or all(
        v is None for k, v in raw_cap.items() if k != "source")

    assert orch.set_reasoning_effort("main", "max")["ok"] is True
    assert orch.main_provider._reasoning_effort == "max"


async def test_provider_metadata_beats_catalog_and_user_beats_all(tmp_path: Path) -> None:
    """优先级：provider 元数据 > 公共目录；用户覆盖 > 一切。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "ds-a", "capability": {
        "window": 777, "reasoning_returned": None, "reasoning_mode": "unknown",
        "levels": None, "max_output": None, "tools": None, "source": "provider"}}]

    async def fetcher():
        return _CATALOG_FIXTURE

    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    await orch.connect_provider("", protocol="openai", base_url="https://api.deepseek.com/v1",
                                model="ds-a", api_key="sk-k", preset="deepseek")
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 777 and status["window_source"] == "provider"  # provider 胜出
    assert status["efforts"] == ["off", "low", "high", "max"]               # 档位补自目录

    pid = derive_provider_id("openai", "https://api.deepseek.com/v1")
    orch.set_model_capability(pid, "ds-a", {"window": 42_000})
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 42_000 and status["window_source"] == "user"


async def test_catalog_fetch_failure_silently_unknown(tmp_path: Path) -> None:
    async def fetcher():
        raise RuntimeError("网络挂了")

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "ds-a", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider(
        "", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="ds-a", api_key="sk-k", preset="deepseek")
    assert result["ok"] is True  # 目录拉取失败不影响连接
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] is None and status["reasoning_mode"] == "unknown"


async def test_catalog_cache_fresh_skips_fetch(tmp_path: Path) -> None:
    """缓存新鲜时连接不重拉（fetcher 不应被调用）。"""
    cache = tmp_path / "models-dev.json"  # keys_path 同目录（缓存跟随 keys_path）
    calls = 0

    async def fetcher():
        nonlocal calls
        calls += 1
        return _CATALOG_FIXTURE

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "ds-a", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    # 先写一份新鲜缓存（conftest 已把 KEYS_DIR 隔离到 tmp）
    save_catalog(cache, _CATALOG_FIXTURE)
    await orch.connect_provider("", protocol="openai", base_url="https://api.deepseek.com/v1",
                                model="ds-a", api_key="sk-k", preset="deepseek")
    assert calls == 0
    assert orch.provider_status()["roles"]["main"]["window"] == 1_000_000


async def test_custom_endpoint_without_catalog_mapping_stays_unknown(tmp_path: Path) -> None:
    """自定义端点不匹配任何预设目录 → 保持未知（不猜测）。"""
    async def fetcher():
        return _CATALOG_FIXTURE

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "ds-a", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    await orch.connect_provider("", protocol="openai", base_url="http://127.0.0.1:9/v1",
                                model="ds-a", api_key="")
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] is None and status["reasoning_mode"] == "unknown"


async def test_catalog_disk_cache_applies_at_startup_without_fetch(tmp_path: Path) -> None:
    """启动只读磁盘缓存（零外呼）：注册表绑定的无能力模型由公共目录补齐。"""
    pid = derive_provider_id("openai", "http://127.0.0.1:9/v1")
    _seed_registry(tmp_path,
                   {pid: {"name": "x", "protocol": "openai",
                          "base_url": "http://127.0.0.1:9/v1", "catalog": "deepseek",
                          "model_overrides": {},
                          "models": [_model("ds-a")], "models_fetched_at": 1.0}},
                   {"main": {"provider_id": pid, "model": "ds-a", "reasoning_effort": "off"}})
    save_key(pid, "sk-k", tmp_path / "keys.json")

    save_catalog(tmp_path / "models-dev.json", _CATALOG_FIXTURE)

    async def boom():
        raise AssertionError("启动不得拉取公共目录")

    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "keys.json", catalog_fetcher=boom)
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 1_000_000 and status["window_source"] == "catalog"
    assert status["efforts"] == ["off", "low", "high", "max"]


async def test_no_catalog_fetch_when_no_provider_mapped(tmp_path: Path) -> None:
    """没有任何映射到公共目录的 provider 时：连接/刷新不得拉取（零外呼）。"""
    calls = 0

    async def fetcher():
        nonlocal calls
        calls += 1
        return _CATALOG_FIXTURE

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    await orch.connect_provider("", protocol="openai", base_url="http://127.0.0.1:9/v1",
                                model="m", api_key="")
    await orch.list_models(refresh=True)
    assert calls == 0


async def test_anthropic_catalog_never_produces_unsupported_effort(tmp_path: Path) -> None:
    """漏洞 A：kimi（anthropic 协议）经目录获得档位后，强度请求必须落到已知预算档。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "kimi-for-coding", "capability": {}}]
    fixture = {"kimi-code-plan-cn": {"models": {"kimi-for-coding": {
        "reasoning": True, "tool_call": True,
        "reasoning_options": [{"type": "toggle"},
                              {"type": "effort", "values": ["low", "high", "max"]}],
        "limit": {"context": 262_144, "output": 262_144}}}}}

    async def fetcher():
        return fixture

    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider(
        "", protocol="anthropic", base_url="https://api.kimi.com/coding/",
        model="", api_key="sk-k", preset="kimi-coding")
    assert result["ok"] is True, result
    status = orch.provider_status()["roles"]["main"]
    assert status["efforts"] == ["off", "low", "medium", "high"]  # 无 max
    assert orch.set_reasoning_effort("main", "high")["ok"] is True


async def test_custom_official_host_matches_preset_catalog(tmp_path: Path) -> None:
    """自定义填官方主机（https://api.anthropic.com）也命中预设目录（兜底匹配主机）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "claude-x", "capability": {}}]
    fixture = {"anthropic": {"models": {"claude-x": {
        "reasoning": True, "tool_call": True,
        "reasoning_options": [{"type": "budget_tokens"}],
        "limit": {"context": 400_000, "output": 64_000}}}}}

    async def fetcher():
        return fixture

    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider(
        "", protocol="anthropic", base_url="https://api.anthropic.com",
        model="claude-x", api_key="sk-k")
    assert result["ok"] is True, result
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 400_000 and status["window_source"] == "catalog"


def test_catalog_path_follows_keys_path(tmp_path: Path) -> None:
    orch = Orchestrator(main_provider=None, executor_provider=None, root=tmp_path,
                        store_path=tmp_path / ".providers.toml",
                        keys_path=tmp_path / "mykeys.json")
    assert orch._catalog_path() == tmp_path / "models-dev.json"


def _anthropic_fake(reasoning_effort="off"):
    fake = FakeProvider([ChatResult(text="pong")])
    fake.reasoning_reserve = lambda effort: {"off": 0, "low": 2048, "medium": 8192,
                                             "high": 16384}.get(effort, 0)
    fake._reasoning_effort = reasoning_effort
    return fake


async def test_anthropic_unknown_declared_effort_rejected_at_use(tmp_path: Path) -> None:
    """用户声明协议不认识的档位（如 anthropic 的 max）：设置强度必须明确报错，不得静默。"""
    fake = _anthropic_fake()
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    result = await orch.connect_provider(
        "", protocol="anthropic", base_url="http://127.0.0.1:9",
        model="claude-x", api_key="")
    assert result["ok"] is True, result
    pid = derive_provider_id("anthropic", "http://127.0.0.1:9")
    orch.set_model_capability(pid, "claude-x",
                              {"reasoning_mode": "adjustable", "levels": ["off", "max"]})
    result = orch.set_reasoning_effort("main", "max")
    assert result["ok"] is False and "不识别" in result["message"]
    assert orch.main_provider._reasoning_effort == "off"
    assert orch.set_reasoning_effort("main", "off")["ok"] is True


async def test_connect_with_unmappable_effort_fails_loudly(tmp_path: Path) -> None:
    """连接时选了协议不认识的档位：明确失败，不得静默省略 thinking。"""
    fake = _anthropic_fake()
    fake.models = [{"id": "claude-x", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("", protocol="anthropic", base_url="http://127.0.0.1:9",
                                model="claude-x", api_key="")
    pid = derive_provider_id("anthropic", "http://127.0.0.1:9")
    orch.set_model_capability(pid, "claude-x",
                              {"reasoning_mode": "adjustable", "levels": ["off", "max"]})
    result = await orch.connect_provider(
        "", protocol="anthropic", base_url="http://127.0.0.1:9",
        model="claude-x", api_key="", reasoning_effort="max")
    assert result["ok"] is False and "不识别" in result["message"]


async def test_catalog_matches_by_api_host_when_provider_renamed(tmp_path: Path) -> None:
    """改名免疫：预设名与显式名都对不上时，按目录条目的 api 主机一致来匹配。"""
    fixture = {"deepseek-renamed-2026": {
        "api": "https://api.deepseek.com/v1",
        "models": {"ds-a": {
            "reasoning": True, "tool_call": True,
            "reasoning_options": [{"type": "toggle"},
                                  {"type": "effort", "values": ["low", "high"]}],
            "limit": {"context": 555_000, "output": 32_000}}},
    }}

    async def fetcher():
        return fixture

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "ds-a", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider(
        "", protocol="openai", base_url="https://api.deepseek.com/v1",
        model="ds-a", api_key="sk-k", preset="deepseek")
    assert result["ok"] is True, result
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 555_000 and status["window_source"] == "catalog"
    assert status["efforts"] == ["off", "low", "high"]


async def test_catalog_alias_fallback_when_primary_name_missing(tmp_path: Path) -> None:
    """候选名兜底：主名对不上时用旧名别名命中（条目无 api 字段，排除地址匹配）。"""
    fixture = {"kimi-for-coding": {"models": {
        "kimi-for-coding": {
            "reasoning": True, "tool_call": True,
            "reasoning_options": [{"type": "toggle"},
                                  {"type": "effort", "values": ["low", "high"]}],
            "limit": {"context": 262_144, "output": 32_768}}},
    }}

    async def fetcher():
        return fixture

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "kimi-for-coding", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider(
        "", protocol="anthropic", base_url="https://api.kimi.com/coding/",
        model="", api_key="sk-k", preset="kimi-coding")
    assert result["ok"] is True, result
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] == 262_144 and status["window_source"] == "catalog"
    assert status["efforts"] == ["off", "low", "medium", "high"]


async def test_catalog_endpoint_match_uses_path_not_just_host(tmp_path: Path) -> None:
    """回归（审核 V1）：同一主机不同路径按端点指纹匹配，不得只按主机串台。"""
    base = "https://api.example.com/v1"
    fixture = {
        "wrong-path": {"api": "https://api.example.com/v2",
                       "models": {"m": {"limit": {"context": 111}, "reasoning": False}}},
        "right-path": {"api": base,
                       "models": {"m": {"limit": {"context": 222}, "reasoning": False}}},
    }

    async def fetcher():
        return fixture

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_data = {"providers": fixture}  # 模拟目录已加载
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider("", protocol="openai", base_url=base,
                                         model="m", api_key="sk-k")
    assert result["ok"] is True, result
    assert orch.provider_status()["roles"]["main"]["window"] == 222


async def test_activate_rebuild_passes_interleaved(tmp_path: Path) -> None:
    """重建路径（_build_instance）也必须把目录 interleaved 传给 provider。"""
    base = "https://api.example.com/v1"
    fixture = {
        "thinking-api": {"api": base, "models": {"m": {
            "reasoning": True, "interleaved": {"field": "reasoning_content"},
            "limit": {"context": 1_000}}}},
    }

    async def fetcher():
        return fixture

    seen: list[dict] = []

    def factory(**kw):
        seen.append(kw)
        fake = FakeProvider([ChatResult(text="pong")])
        fake.models = [{"id": "m", "capability": {}}]
        return fake

    orch = make_orchestrator(tmp_path, factory)
    orch.providers._catalog_data = {"providers": fixture}
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider("main", protocol="openai", base_url=base,
                                         model="m", api_key="sk-k")
    assert result["ok"] is True, result
    orch.providers._instances.clear()  # 模拟实例重建路径
    seen.clear()
    orch.providers._activate("main")
    assert seen and seen[-1].get("echo_reasoning_field") == "reasoning_content"


async def test_static_instance_reuse_fills_interleaved(tmp_path: Path) -> None:
    """静态实例复用：目录能力必须补进已存在实例的 _echo_reasoning_field。"""
    base = "https://api.example.com/v1"
    fixture = {"thinking-api": {"api": base, "models": {"m": {
        "reasoning": True, "interleaved": {"field": "reasoning_content"},
        "limit": {"context": 1_000}}}}}
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_data = {"providers": fixture}
    pid = derive_provider_id("openai", base)
    orch.providers._providers[pid] = {
        "name": "x", "protocol": "openai", "base_url": base, "catalog": "",
        "model_overrides": {}, "models": [], "error": None, "fetched_at": 0.0,
    }
    orch.providers._roles["main"] = {
        "provider_id": pid, "model": "m", "reasoning_effort": "off"}
    orch.providers._instances[("main", pid, "m")] = fake  # 已存在的静态实例
    orch.providers._activate("main")
    assert fake._echo_reasoning_field == "reasoning_content"


async def test_static_instance_reuse_keeps_existing_echo_field(tmp_path: Path) -> None:
    """已有非空回传字段不得被目录能力覆盖（显式配置/自愈所得优先）。"""
    base = "https://api.example.com/v1"
    fixture = {"thinking-api": {"api": base, "models": {"m": {
        "reasoning": True, "interleaved": {"field": "reasoning_content"},
        "limit": {"context": 1_000}}}}}
    fake = FakeProvider([ChatResult(text="pong")])
    fake._echo_reasoning_field = "reasoning_details"  # 已有值（显式配置或自愈所得）
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_data = {"providers": fixture}
    pid = derive_provider_id("openai", base)
    orch.providers._providers[pid] = {
        "name": "x", "protocol": "openai", "base_url": base, "catalog": "",
        "model_overrides": {}, "models": [], "error": None, "fetched_at": 0.0,
    }
    orch.providers._roles["main"] = {
        "provider_id": pid, "model": "m", "reasoning_effort": "off"}
    orch.providers._instances[("main", pid, "m")] = fake
    orch.providers._activate("main")
    assert fake._echo_reasoning_field == "reasoning_details"


async def test_catalog_interleaved_reaches_provider_factory(tmp_path: Path) -> None:
    """目录声明 interleaved 的模型：字段名必须传到 provider（否则下一轮 400）。"""
    base = "https://api.example.com/v1"
    fixture = {
        "thinking-api": {"api": base, "models": {"m": {
            "reasoning": True, "interleaved": {"field": "reasoning_content"},
            "limit": {"context": 1_000}}}},
    }

    async def fetcher():
        return fixture

    seen: list[dict] = []

    def factory(**kw):
        seen.append(kw)
        fake = FakeProvider([ChatResult(text="pong")])
        fake.models = [{"id": "m", "capability": {}}]
        return fake

    orch = make_orchestrator(tmp_path, factory)
    orch.providers._catalog_data = {"providers": fixture}
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider("main", protocol="openai", base_url=base,
                                         model="m", api_key="sk-k")
    assert result["ok"] is True, result
    assert seen and seen[-1].get("echo_reasoning_field") == "reasoning_content"


async def test_catalog_capability_skips_candidate_without_model(tmp_path: Path) -> None:
    """回归（审核 V2）：同端点多候选时按"谁真含该模型"回溯，首个命中不得遮蔽。"""
    base = "https://api.example.com/v1"
    fixture = {
        "alpha": {"api": base, "models": {"other-model": {"limit": {"context": 1}}}},
        "beta": {"api": base, "models": {"m": {"limit": {"context": 333},
                                               "reasoning": False}}},
    }

    async def fetcher():
        return fixture

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_data = {"providers": fixture}  # 模拟目录已加载
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider("", protocol="openai", base_url=base,
                                         model="m", api_key="sk-k")
    assert result["ok"] is True, result
    assert orch.provider_status()["roles"]["main"]["window"] == 333


async def test_catalog_local_hosts_never_matched_by_address(tmp_path: Path) -> None:
    """回归（审核 V1）：本地/私网端点不参与目录地址匹配（不同端口不得串台）。"""
    base = "http://127.0.0.1:11434/v1"
    fixture = {
        "wrong-local": {"api": "http://127.0.0.1:1337/v1",
                        "models": {"qwen3": {"limit": {"context": 111}}}},
        "right-local": {"api": base,
                        "models": {"qwen3": {"limit": {"context": 222}}}},
    }

    async def fetcher():
        return fixture

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "qwen3", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_data = {"providers": fixture}
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider("", protocol="openai", base_url=base,
                                         model="qwen3", api_key="")
    assert result["ok"] is True, result
    status = orch.provider_status()["roles"]["main"]
    assert status["window"] is None and status["window_source"] == "unknown"


async def test_catalog_path_prefix_must_be_version_segment(tmp_path: Path) -> None:
    """回归（审核残留）：/v1 与 /v1beta 不得视为同一端点。"""
    fixture = {"beta-api": {"api": "https://api.example.com/v1beta",
                            "models": {"m": {"limit": {"context": 111}}}},
               "v1-api": {"api": "https://api.example.com/v1",
                          "models": {"m": {"limit": {"context": 222}}}}}

    async def fetcher():
        return fixture

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_data = {"providers": fixture}
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider("", protocol="openai",
                                         base_url="https://api.example.com/v1",
                                         model="m", api_key="sk-k")
    assert result["ok"] is True
    # 只允许 /v1 精确命中；/v1beta 不得串入
    assert orch.provider_status()["roles"]["main"]["window"] == 222


async def test_catalog_dot_local_hosts_excluded(tmp_path: Path) -> None:
    """回归（审核残留）：.local 等私网后缀不参与地址匹配。"""
    base = "http://myhost.local:1234/v1"
    fixture = {"local-ish": {"api": base,
                             "models": {"m": {"limit": {"context": 111}}}}}

    async def fetcher():
        return fixture

    fake = FakeProvider([ChatResult(text="pong")])
    fake.models = [{"id": "m", "capability": {}}]
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    orch.providers._catalog_data = {"providers": fixture}
    orch.providers._catalog_fetcher = fetcher
    result = await orch.connect_provider("", protocol="openai", base_url=base,
                                         model="m", api_key="")
    assert result["ok"] is True
    assert orch.provider_status()["roles"]["main"]["window"] is None


def test_opencode_go_preset_present() -> None:
    """OpenCode Go：订阅网关，OpenAI 兼容，UI 有连接按钮（自己填 key 连）。"""
    spec = PRESETS["opencode-go"]
    assert spec["label"] == "OpenCode Go"
    assert spec["protocol"] == "openai"
    assert spec["base_url"] == "https://opencode.ai/zen/go/v1"
    assert spec["model"] == "deepseek-v4.1-flash"
    assert spec["needs_key"] is True


async def test_disconnect_deletes_key_unbinds_and_keeps_row(tmp_path: Path) -> None:
    """断开：删凭据、解绑角色、条目保留；换 key 可直接重连。"""
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([ChatResult(text="pong")]))
    await orch.connect_provider("main", **CONNECT)
    pid = derive_provider_id("openai", None)
    assert load_keys(tmp_path / "keys.json")[pid] == "sk-live"

    result = orch.disconnect_provider(pid)

    assert result["ok"] is True
    assert pid not in load_keys(tmp_path / "keys.json")
    assert orch.main_provider is None                       # 角色已解绑
    status = orch.provider_status()
    entry = next(p for p in status["providers"] if p["id"] == pid)
    assert entry["configured"] is False and entry["has_key"] is False
    assert status["roles"]["main"]["configured"] is False
    again = await orch.connect_provider("main", **{**CONNECT, "api_key": "sk-new"})
    assert again["ok"] is True
    assert load_keys(tmp_path / "keys.json")[pid] == "sk-new"   # 换 key 重连成功


async def test_disconnect_without_key_is_rejected(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, lambda **kw: FakeProvider([ChatResult(text="pong")]))
    result = orch.disconnect_provider(derive_provider_id("openai", None))
    assert result["ok"] is False and "凭据" in result["message"]


async def test_disconnect_message_flow_emits_status(tmp_path: Path) -> None:
    """WS 断开消息：provider_result + provider_status（has_key=False）都要广播。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    await orch.connect_provider("main", **CONNECT)
    queue = orch.subscribe()
    pid = derive_provider_id("openai", None)

    orch.handle_client_message({"type": "disconnect_provider", "provider_id": pid})

    result = await _until(queue, lambda e: e.get("type") == "provider_result")
    assert result["ok"] is True
    status = await _until(
        queue,
        lambda e: e.get("type") == "provider_status"
        and all(p["has_key"] is False for p in e["providers"] if p["id"] == pid),
    )
    assert status["roles"]["main"]["configured"] is False


async def test_disconnect_local_endpoint_keeps_role_usable(tmp_path: Path) -> None:
    """本地端点：删显式 key 后仍可用占位运行，角色保持可用（只是 has_key 变 false）。"""
    fake = FakeProvider([ChatResult(text="pong")])
    orch = make_orchestrator(tmp_path, lambda **kw: fake)
    key = dict(protocol="openai", base_url="http://127.0.0.1:9/v1", model="m",
               api_key="sk-local")
    await orch.connect_provider("main", **key)
    pid = derive_provider_id("openai", "http://127.0.0.1:9/v1")
    assert load_keys(tmp_path / "keys.json")[pid] == "sk-local"

    result = orch.disconnect_provider(pid)

    assert result["ok"] is True
    assert pid not in load_keys(tmp_path / "keys.json")
    assert orch.main_provider is fake                     # 本地端点仍可用
    entry = next(p for p in orch.provider_status()["providers"] if p["id"] == pid)
    assert entry["has_key"] is False and entry["configured"] is True


def test_chatgpt_login_preset() -> None:
    """ChatGPT 会员登录预设：Codex Responses 协议、不要 key、带登录标记。"""
    spec = PRESETS["chatgpt"]
    assert spec["protocol"] == "openai-responses"
    assert spec["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert spec["needs_key"] is False
    assert spec["login"] == "codex"


def test_codex_endpoint_needs_no_api_key() -> None:
    from agent.core.provider_manager import ProviderManager

    fake = ProviderManager.__new__(ProviderManager)
    assert fake._endpoint_needs_key(
        {"base_url": "https://chatgpt.com/backend-api/codex"}) is False
    assert fake._endpoint_needs_key({"base_url": "https://api.openai.com/v1"}) is True


async def test_connect_chatgpt_without_login_is_explicit(tmp_path: Path) -> None:
    """未登录时连接：报"尚未登录"而不是"缺少 API key"。"""
    from agent.core.config import make_provider

    orch = make_orchestrator(tmp_path, make_provider)
    result = await orch.connect_provider(
        "", protocol="openai-responses",
        base_url="https://chatgpt.com/backend-api/codex", model="gpt-5.2-codex",
        api_key="", preset="chatgpt",
    )
    assert result["ok"] is False
    assert "尚未登录" in result["message"] and "缺少 API key" not in result["message"]
