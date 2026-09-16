"""多 provider 注册表测试：新结构读写、能力覆盖、旧结构安全迁移、密钥按 provider_id。"""

import json
from pathlib import Path

from agent.core.keystore import load_keys
from agent.core.provider_store import (
    clean_override,
    derive_provider_id,
    load_registry,
    migrate_legacy,
    provider_label,
    save_registry,
)


def test_derive_provider_id_stable_and_dedup() -> None:
    a = derive_provider_id("openai", "https://x/v1")
    assert a == derive_provider_id("openai", "https://x/v1/")  # 尾斜杠归一
    assert a != derive_provider_id("openai", "https://y/v1")
    assert a != derive_provider_id("anthropic", "https://x/v1")


def test_registry_roundtrip_with_overrides(tmp_path: Path) -> None:
    path = tmp_path / ".providers.toml"
    registry = {
        "roles": {"main": {"provider_id": "p-1", "model": "m1", "reasoning_effort": "medium"}},
        "providers": {
            "p-1": {
                "name": "openai · x",
                "protocol": "openai",
                "base_url": "https://x/v1",
                "model_overrides": {
                    "m1": {"window": 128_000, "reasoning_mode": "adjustable",
                           "levels": ["low", "medium"], "max_output": 8192, "tools": False},
                },
            },
        },
    }
    save_registry(path, registry)
    loaded = load_registry(path)
    assert loaded["roles"]["main"]["provider_id"] == "p-1"
    assert loaded["roles"]["main"]["reasoning_effort"] == "medium"
    override = loaded["providers"]["p-1"]["model_overrides"]["m1"]
    assert override["window"] == 128_000
    assert override["levels"] == ["low", "medium"]
    assert override["tools"] is False  # 布尔 False 必须保留
    assert path.stat().st_mode & 0o777 == 0o600


def test_clean_override_drops_invalid() -> None:
    assert clean_override({"window": 0, "tools": "yes", "reasoning_mode": "bogus"}) == {}
    assert clean_override({"window": 10, "tools": False}) == {"window": 10, "tools": False}
    assert clean_override({"levels": "off, low ,"}) == {"levels": ["off", "low"]}


def test_migrate_legacy_roles_and_keys(tmp_path: Path) -> None:
    """旧 [main]/[executor] + api_key → 新注册表 + keys.json(provider_id)，不丢配置。"""
    store = tmp_path / ".providers.toml"
    keys = tmp_path / "keys.json"
    store.write_text(
        '[main]\nprotocol = "openai"\nbase_url = "https://x/v1"\nmodel = "m1"\n'
        'api_key = "sk-legacy"\ncontext_window = 12345\n\n'
        '[executor]\nprotocol = "openai"\nbase_url = "https://x/v1"\nmodel = "m2"\n'
        'api_key = "sk-legacy"\n',
        encoding="utf-8",
    )
    assert migrate_legacy(store, keys) is True
    registry = load_registry(store)
    pid = registry["roles"]["main"]["provider_id"]
    assert registry["roles"]["executor"]["provider_id"] == pid  # 同地址去重
    assert registry["roles"]["main"]["model"] == "m1"
    assert registry["providers"][pid]["base_url"] == "https://x/v1"
    assert registry["providers"][pid]["model_overrides"]["m1"]["window"] == 12345
    assert "sk-legacy" not in store.read_text(encoding="utf-8")  # 密钥已剥离
    assert load_keys(keys)[pid] == "sk-legacy"
    assert migrate_legacy(store, keys) is False  # 幂等


def test_load_registry_bad_file_degrades(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    store.write_text("this is not toml [", encoding="utf-8")
    assert load_registry(store) == {"roles": {}, "providers": {}}


def test_load_registry_invalid_override_fields_dropped(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    store.write_text(
        '[roles]\nmain_provider = "p-1"\nmain_model = "m"\n\n'
        '[providers."p-1"]\nname = "x"\nprotocol = "openai"\n\n'
        '[providers."p-1".model_overrides."m"]\nwindow = -5\n'
        'levels = "off,medium"\ntools = true\n',
        encoding="utf-8",
    )
    registry = load_registry(store)
    override = registry["providers"]["p-1"]["model_overrides"]["m"]
    assert "window" not in override  # 非正数不合法
    assert override["levels"] == ["off", "medium"]
    assert override["tools"] is True


def test_provider_label() -> None:
    assert provider_label("openai", "https://api.x.com/v1") == "openai · api.x.com/v1"
    assert provider_label("anthropic", None) == "anthropic（默认端点）"


def test_keystore_keys_keyed_by_provider_id(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    keys = tmp_path / "keys.json"
    registry = {
        "roles": {"main": {"provider_id": "p-a", "model": "m", "reasoning_effort": "off"}},
        "providers": {"p-a": {"name": "a", "protocol": "openai",
                              "base_url": "https://x", "model_overrides": {}}},
    }
    save_registry(store, registry)
    keys.write_text(json.dumps({"p-a": "sk-a"}), encoding="utf-8")
    assert load_keys(keys)["p-a"] == "sk-a"


def test_reasoning_returned_roundtrips_for_metadata_and_override(tmp_path):
    """审核回归：reasoning_returned=False 不得在持久化后丢失。"""
    path = tmp_path / ".providers.toml"
    registry = {
        "roles": {"main": {"provider_id": "p-a", "model": "m", "reasoning_effort": "off"}},
        "providers": {
            "p-a": {
                "name": "n", "protocol": "openai", "base_url": "http://x/v1",
                "model_overrides": {"m": {"reasoning_returned": False, "tools": False}},
                "models": [{
                    "id": "m", "display_name": "M", "provider_id": "p-a", "provider": "n",
                    "capability": {"reasoning_returned": False, "window": 1000},
                }],
                "models_fetched_at": 123.5,
            }
        },
    }
    save_registry(path, registry)
    loaded = load_registry(path)
    provider = loaded["providers"]["p-a"]
    assert provider["model_overrides"]["m"]["reasoning_returned"] is False
    assert provider["model_overrides"]["m"]["tools"] is False
    cached = provider["models"][0]["capability"]
    assert cached["reasoning_returned"] is False and cached["window"] == 1000
    assert provider["models_fetched_at"] == 123.5
