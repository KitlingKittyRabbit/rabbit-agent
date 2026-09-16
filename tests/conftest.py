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
