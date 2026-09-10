"""配置加载：config.toml → AppConfig；api key 走环境变量，永不入库。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..providers import AnthropicCompatProvider, OpenAICompatProvider, Provider


class ConfigError(Exception):
    pass


@dataclass
class RoleConfig:
    protocol: str  # "openai" | "anthropic"
    model: str
    base_url: str | None
    api_key_env: str | None


@dataclass
class AppConfig:
    main: RoleConfig
    executor: RoleConfig
    working_dir: Path
    plan_mode: bool = False
    max_steps_main: int = 50
    max_steps_executor: int = 30
    session_db: str = ".agent_sessions.db"
    compact_threshold_chars: int = 200_000


def _parse_role(name: str, data: dict) -> RoleConfig:
    try:
        return RoleConfig(
            protocol=data["protocol"],
            model=data["model"],
            base_url=data.get("base_url"),
            api_key_env=data.get("api_key_env"),
        )
    except KeyError as e:
        raise ConfigError(f"[{name}] 缺少必需键: {e}") from e


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    for section in ("main", "executor"):
        if section not in data:
            raise ConfigError(f"配置缺少 [{section}] 节")
    agent = data.get("agent", {})
    return AppConfig(
        main=_parse_role("main", data["main"]),
        executor=_parse_role("executor", data["executor"]),
        working_dir=Path(agent.get("working_dir", ".")),
        plan_mode=bool(agent.get("plan_mode", False)),
        max_steps_main=int(agent.get("max_steps_main", 50)),
        max_steps_executor=int(agent.get("max_steps_executor", 30)),
        session_db=str(agent.get("session_db", ".agent_sessions.db")),
        compact_threshold_chars=int(agent.get("compact_threshold_chars", 200_000)),
    )


def build_provider(role: RoleConfig) -> Provider:
    if role.protocol not in ("openai", "anthropic"):
        raise ConfigError(f"未知协议: {role.protocol}")
    api_key = "unused"
    if role.api_key_env:
        api_key = os.environ.get(role.api_key_env, "")
        if not api_key:
            raise ConfigError(f"环境变量 {role.api_key_env} 未设置（该角色的 api key 来源）")
    return make_provider(
        protocol=role.protocol, base_url=role.base_url, model=role.model, api_key=api_key
    )


def make_provider(
    *, protocol: str, model: str, api_key: str, base_url: str | None = None
) -> Provider:
    """按显式参数构建 provider（运行时连接用，不经环境变量）。"""
    if protocol == "openai":
        return OpenAICompatProvider(base_url=base_url, api_key=api_key, model=model)
    if protocol == "anthropic":
        return AnthropicCompatProvider(api_key=api_key, model=model, base_url=base_url)
    raise ConfigError(f"未知协议: {protocol}")
