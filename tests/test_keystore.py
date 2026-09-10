"""keystore 测试：.env 启动加载。"""

from pathlib import Path

from agent.core.keystore import load_env_file


def test_load_roundtrip(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("KEY_A=1\nKEY_B=2\n", encoding="utf-8")
    assert load_env_file(env) == {"KEY_A": "1", "KEY_B": "2"}


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "nope.env") == {}


def test_load_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# 注释\n\nKEY=v\n", encoding="utf-8")
    assert load_env_file(env) == {"KEY": "v"}
