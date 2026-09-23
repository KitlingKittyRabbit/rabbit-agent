"""钥匙串存储：迁移、读写、回退（用假后端，绝不碰真实系统钥匙串）。"""

from pathlib import Path

import pytest

import agent.core.keystore as keystore


class FakeKeyring:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}
        self.deleted: list[str] = []

    def get_password(self, service: str, name: str):
        return self.data.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.data[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        self.deleted.append(name)
        self.data.pop((service, name), None)


@pytest.fixture()
def fake_kr(monkeypatch):
    monkeypatch.delenv("RABBIT_AGENT_KEYRING", raising=False)   # 允许注入后端
    backend = FakeKeyring()
    monkeypatch.setattr(keystore, "_OVERRIDE_BACKEND", backend)
    yield backend
    monkeypatch.setattr(keystore, "_OVERRIDE_BACKEND", None)


def test_save_and_load_via_keyring(tmp_path: Path, fake_kr) -> None:
    path = tmp_path / "keys.json"
    keystore.save_key("p-abc", "sk-secret", path)

    assert keystore.load_keys(path) == {"p-abc": "sk-secret"}
    assert fake_kr.data[("rabbit-agent", "p-abc")] == "sk-secret"
    text = path.read_text(encoding="utf-8")
    assert "sk-secret" not in text            # 明文不落盘
    assert "p-abc" in text                    # 文件只留名字索引


def test_delete_removes_from_keyring_and_index(tmp_path: Path, fake_kr) -> None:
    path = tmp_path / "keys.json"
    keystore.save_key("p-abc", "sk-secret", path)
    keystore.delete_key("p-abc", path)
    assert keystore.load_keys(path) == {}
    assert fake_kr.data == {}
    assert "p-abc" in fake_kr.deleted


def test_migrate_moves_plaintext_and_strips_file(tmp_path: Path, fake_kr) -> None:
    path = tmp_path / "keys.json"
    path.write_text('{"p-old": "sk-plain", "p-empty": ""}', encoding="utf-8")

    assert keystore.migrate_keys(path) == 1

    assert fake_kr.data[("rabbit-agent", "p-old")] == "sk-plain"
    text = path.read_text(encoding="utf-8")
    assert "sk-plain" not in text
    assert keystore.load_keys(path) == {"p-old": "sk-plain"}
    assert keystore.migrate_keys(path) == 0    # 幂等


def test_fallback_to_file_when_keyring_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RABBIT_AGENT_KEYRING", "0")
    monkeypatch.setattr(keystore, "_OVERRIDE_BACKEND", None)
    path = tmp_path / "keys.json"

    keystore.save_key("p-abc", "sk-file", path)

    assert keystore.load_keys(path) == {"p-abc": "sk-file"}
    assert "sk-file" in path.read_text(encoding="utf-8")   # 回退模式仍写文件


def test_keyring_write_failure_falls_back(tmp_path: Path, monkeypatch) -> None:
    class Broken(FakeKeyring):
        def set_password(self, service: str, name: str, value: str) -> None:
            raise RuntimeError("keyring 坏了")

    monkeypatch.delenv("RABBIT_AGENT_KEYRING", raising=False)
    monkeypatch.setattr(keystore, "_OVERRIDE_BACKEND", Broken())
    path = tmp_path / "keys.json"

    keystore.save_key("p-abc", "sk-x", path)

    assert keystore.load_keys(path) == {"p-abc": "sk-x"}   # 回退文件仍可用
