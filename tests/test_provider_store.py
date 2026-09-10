"""provider_store 测试：.providers.toml 读写、权限、转义。"""

import stat
from pathlib import Path

from agent.core.provider_store import load_providers, save_provider


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    save_provider(store, "main", {"protocol": "anthropic", "model": "m1", "api_key": "sk-a"})
    save_provider(
        store,
        "executor",
        {"protocol": "openai", "base_url": "http://x/v1", "model": "m2", "api_key": "sk-b"},
    )

    loaded = load_providers(store)
    assert loaded["main"] == {"protocol": "anthropic", "model": "m1", "api_key": "sk-a"}
    assert loaded["executor"]["base_url"] == "http://x/v1"


def test_update_one_role_keeps_other(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    save_provider(store, "main", {"protocol": "openai", "model": "m1", "api_key": "k1"})
    save_provider(store, "executor", {"protocol": "openai", "model": "m2", "api_key": "k2"})
    save_provider(store, "main", {"protocol": "openai", "model": "m3", "api_key": "k3"})

    loaded = load_providers(store)
    assert loaded["main"]["model"] == "m3"
    assert loaded["executor"]["model"] == "m2"


def test_none_fields_dropped(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    save_provider(
        store, "main", {"protocol": "openai", "base_url": None, "model": "m", "api_key": "k"}
    )
    assert "base_url" not in load_providers(store)["main"]


def test_file_permission_is_600(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    save_provider(store, "main", {"protocol": "openai", "model": "m", "api_key": "k"})
    assert stat.S_IMODE(store.stat().st_mode) == 0o600


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert load_providers(tmp_path / "nope.toml") == {}


def test_special_characters_roundtrip(tmp_path: Path) -> None:
    store = tmp_path / ".providers.toml"
    save_provider(store, "main", {"protocol": "openai", "model": 'm"x', "api_key": 'k"\\1'})
    assert load_providers(store)["main"]["api_key"] == 'k"\\1'
