"""配置加载与 provider 构建测试。"""

from pathlib import Path

import pytest

from agent.core.config import ConfigError, build_provider, load_config
from agent.providers import AnthropicCompatProvider, OpenAICompatProvider

TOML = """
[agent]
working_dir = "."
plan_mode = true
max_steps_main = 7
max_steps_executor = 3
session_db = "test.db"
compact_threshold_chars = 5000

[main]
protocol = "anthropic"
base_url = "https://api.example.com/coding/"
model = "main-model"
api_key_env = "TEST_MAIN_KEY"

[executor]
protocol = "openai"
base_url = "http://127.0.0.1:11434/v1"
model = "local-model"
"""


def write_config(tmp_path: Path, content: str = TOML) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    assert config.main.protocol == "anthropic"
    assert config.main.model == "main-model"
    assert config.main.api_key_env == "TEST_MAIN_KEY"
    assert config.executor.protocol == "openai"
    assert config.executor.api_key_env is None
    assert config.plan_mode is True
    assert config.max_steps_main == 7
    assert config.max_steps_executor == 3
    assert config.session_db == "test.db"
    assert config.compact_threshold_chars == 5000


def test_compact_threshold_default(tmp_path: Path) -> None:
    toml = (
        '[main]\nprotocol = "openai"\nmodel = "m"\n\n[executor]\nprotocol = "openai"\nmodel = "m"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    assert config.compact_threshold_chars == 200_000
    assert config.port == 8000  # 缺省端口


def test_custom_port(tmp_path: Path) -> None:
    toml = (
        "[agent]\nport = 8471\n\n"
        '[main]\nprotocol = "openai"\nmodel = "m"\n\n'
        '[executor]\nprotocol = "openai"\nmodel = "m"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    assert config.port == 8471


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="不存在"):
        load_config("/nonexistent/config.toml")


def test_missing_section_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="main"):
        load_config(write_config(tmp_path, "[executor]\nprotocol='openai'\nmodel='m'\n"))


def test_build_openai_provider_without_key(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    provider = build_provider(config.executor)
    assert isinstance(provider, OpenAICompatProvider)


def test_build_openai_provider_with_default_base_url(tmp_path: Path) -> None:
    toml = (
        '[main]\nprotocol = "openai"\nmodel = "m"\n\n[executor]\nprotocol = "openai"\nmodel = "m"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    provider = build_provider(config.executor)
    assert isinstance(provider, OpenAICompatProvider)


def test_build_anthropic_provider_reads_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_MAIN_KEY", "sk-test")
    config = load_config(write_config(tmp_path))
    provider = build_provider(config.main)
    assert isinstance(provider, AnthropicCompatProvider)


def test_missing_env_key_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TEST_MAIN_KEY", raising=False)
    config = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="TEST_MAIN_KEY"):
        build_provider(config.main)


def test_unknown_protocol_raises(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    config.main.protocol = "bogus"
    with pytest.raises(ConfigError, match="未知协议"):
        build_provider(config.main)
