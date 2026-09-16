"""回归：测试的默认 keystore 路径必须是临时目录，绝不指向真实 ~/.rabbit-agent。"""

from pathlib import Path

import agent.core.keystore as keystore


def test_default_keystore_never_points_to_real_home() -> None:
    real = Path.home() / ".rabbit-agent" / "keys.json"
    assert Path(keystore.KEYS_PATH) != real
    assert "pytest" in str(keystore.KEYS_PATH) or "/tmp/" in str(keystore.KEYS_PATH)


def test_default_path_roundtrip_goes_to_temp(monkeypatch) -> None:
    keystore.save_key("probe", "sk-probe-00000000000000000001")
    assert keystore.load_keys().get("probe") == "sk-probe-00000000000000000001"
    keystore.delete_key("probe")
    assert "probe" not in keystore.load_keys()
