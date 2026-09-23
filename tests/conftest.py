"""测试全局隔离：默认密钥路径重定向到临时目录，并守卫真实 keys.json 不被改写。"""

from pathlib import Path

import pytest

import agent.core.keystore as keystore

REAL_KEYS = Path.home() / ".rabbit-agent" / "keys.json"


@pytest.fixture(autouse=True)
def _isolate_default_keystore(tmp_path_factory, monkeypatch):
    """任何使用默认路径（keys_path=None）的测试都只能写到临时目录。"""
    base = tmp_path_factory.mktemp("keystore")
    monkeypatch.setattr(keystore, "KEYS_DIR", base)
    monkeypatch.setattr(keystore, "KEYS_PATH", base / "keys.json")


@pytest.fixture(scope="session", autouse=True)
def _guard_real_keystore():
    """安全网：整个测试会话内真实 ~/.rabbit-agent/keys.json 不得被改写。"""
    before = REAL_KEYS.stat().st_mtime_ns if REAL_KEYS.exists() else None
    yield
    after = REAL_KEYS.stat().st_mtime_ns if REAL_KEYS.exists() else None
    assert after == before, (
        f"测试改写了真实密钥文件 {REAL_KEYS}！"
        "请为相关测试显式传入 keys_path，或依赖 conftest 的默认路径隔离。"
    )


@pytest.fixture(autouse=True)
def _no_real_catalog_fetch(monkeypatch):
    """默认套件零网络：测试中禁止真实拉取 models.dev 目录。"""
    async def _boom():
        raise AssertionError("测试不得拉取 models.dev（外部网络）")
    monkeypatch.setattr("agent.core.provider_manager.fetch_catalog", _boom)


@pytest.fixture(autouse=True)
def _disable_real_keyring(monkeypatch):
    """每个测试都强制文件模式（用 monkeypatch，结束后恢复；不会永久丢掉开关）。"""
    monkeypatch.setenv("RABBIT_AGENT_KEYRING", "0")


def _real_keyring_values():
    """真实系统钥匙串里本项目条目的快照（读失败返回 None）。"""
    import json

    import keyring
    try:
        names = list(json.loads(REAL_KEYS.read_text(encoding="utf-8")).keys())
        return {name: keyring.get_password("rabbit-agent", name) for name in names}
    except Exception:
        return None


@pytest.fixture(scope="session", autouse=True)
def _guard_real_keyring():
    """安全网：整个测试会话内真实系统钥匙串不得被改写。"""
    before = _real_keyring_values()
    yield
    after = _real_keyring_values()
    if before is not None and after is not None:
        assert after == before, "测试改写了真实系统钥匙串（rabbit-agent/*）！"
