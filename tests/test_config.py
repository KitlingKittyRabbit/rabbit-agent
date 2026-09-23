"""配置加载与 provider 构建测试。"""

from pathlib import Path

import pytest

import agent.core.config as config_mod
from agent.core.config import ConfigError, RoleConfig, build_provider, load_config
from agent.providers import AnthropicCompatProvider, OpenAICompatProvider

TOML = """
[agent]
working_dir = "."
plan_mode = true
max_steps_main = 7
max_steps_executor = 3
session_db = "test.db"

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


def test_compact_threshold_default(tmp_path: Path) -> None:
    toml = (
        '[main]\nprotocol = "openai"\nmodel = "m"\n\n[executor]\nprotocol = "openai"\nmodel = "m"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    assert not hasattr(config, "compact_threshold_chars")  # 死配置已删除
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


def test_build_openai_provider_without_key_local_base_url(tmp_path: Path) -> None:
    """显式本地 base_url（Ollama 等）无 key：允许创建。"""
    config = load_config(write_config(tmp_path))
    provider = build_provider(config.executor)
    assert isinstance(provider, OpenAICompatProvider)


def test_official_openai_without_key_or_base_url_raises(tmp_path: Path, monkeypatch) -> None:
    """无 key env 且无 base_url 的官方 openai：配置错误，factory 零调用（即零外呼）。"""
    toml = (
        '[main]\nprotocol = "openai"\nmodel = "m"\n\n[executor]\nprotocol = "openai"\nmodel = "m"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    calls: list = []
    monkeypatch.setattr(config_mod, "make_provider", lambda **kw: calls.append(kw))
    with pytest.raises(ConfigError, match="base_url"):
        build_provider(config.main)
    assert calls == []  # 未创建 provider → 零网络


def test_official_anthropic_without_key_or_base_url_raises(tmp_path: Path, monkeypatch) -> None:
    toml = (
        '[main]\nprotocol = "anthropic"\nmodel = "m"\n\n'
        '[executor]\nprotocol = "openai"\nmodel = "m"\nbase_url = "http://127.0.0.1:1/v1"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    calls: list = []
    monkeypatch.setattr(config_mod, "make_provider", lambda **kw: calls.append(kw))
    with pytest.raises(ConfigError, match="base_url"):
        build_provider(config.main)
    assert calls == []


def test_custom_base_url_without_key_allowed(tmp_path: Path) -> None:
    """显式自定义兼容端点无 key：不误伤（端点由用户明确指定）。"""
    toml = (
        '[main]\nprotocol = "openai"\nbase_url = "http://127.0.0.1:9100/v1"\nmodel = "m"\n\n'
        '[executor]\nprotocol = "openai"\nmodel = "m"\nbase_url = "http://127.0.0.1:1/v1"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    assert isinstance(build_provider(config.main), OpenAICompatProvider)


def test_custom_anthropic_base_url_without_key_allowed() -> None:
    role = RoleConfig(
        protocol="anthropic", model="m", base_url="http://127.0.0.1:1", api_key_env=None
    )
    assert isinstance(build_provider(role), AnthropicCompatProvider)


def test_try_build_returns_none_for_invalid_static_config() -> None:
    """server 启动容错：无效静态配置只记 None，不阻断启动、不构建 provider。"""
    from agent.server.__main__ import _try_build

    role = RoleConfig(protocol="openai", model="m", base_url=None, api_key_env=None)
    assert _try_build(role, "main") is None


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


def test_default_executor_steps_at_least_100(tmp_path: Path) -> None:
    """executor 默认步数上限 ≥100（不再轻易撞旧 30 步限制）。"""
    toml = (
        '[main]\nprotocol="openai"\nmodel="m"\n\n'
        '[executor]\nprotocol="openai"\nmodel="m"\n'
    )
    config = load_config(write_config(tmp_path, toml))
    assert config.max_steps_executor >= 100


def test_repo_config_executor_steps_is_100() -> None:
    """真实仓库配置必须给出 ≥100 的 executor 步数（不是仅代码默认）。"""
    repo_config = Path(__file__).resolve().parents[1] / "config.toml"
    config = load_config(repo_config)
    assert config.max_steps_executor >= 100


def test_no_startup_model_fetch() -> None:
    """启动不再拉取模型元数据（未配置 provider 不得发起外部请求）。"""
    from agent.server import __main__ as server_main

    assert not hasattr(server_main, "_startup_window")
