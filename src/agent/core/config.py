"""配置加载：config.toml → AppConfig；api key 走环境变量，永不入库。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..providers import (
    AnthropicCompatProvider,
    CodexResponsesProvider,
    OpenAICompatProvider,
    Provider,
)


class ConfigError(Exception):
    pass


@dataclass
class RoleConfig:
    protocol: str  # "openai" | "anthropic" | "openai-responses"
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
    max_steps_executor: int = 100
    session_db: str = ".agent_sessions.db"
    audit_log: str = ".agent_audit.log"
    port: int = 8000


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
        max_steps_executor=int(agent.get("max_steps_executor", 100)),
        session_db=str(agent.get("session_db", ".agent_sessions.db")),
        audit_log=str(agent.get("audit_log", ".agent_audit.log")),
        port=int(agent.get("port", 8000)),
    )


def build_provider(role: RoleConfig) -> Provider:
    """静态配置构建 provider；无效组合直接报错，绝不猜测端点或补占位 key 外呼。"""
    if role.protocol not in ("openai", "anthropic", "openai-responses"):
        raise ConfigError(f"未知协议: {role.protocol}")
    if role.protocol == "openai-responses":
        # ChatGPT 会员登录：凭据走 OAuth，不需要 api_key
        return CodexResponsesProvider(base_url=role.base_url, model=role.model)
    if role.api_key_env:
        api_key = os.environ.get(role.api_key_env, "")
        if not api_key:
            raise ConfigError(f"环境变量 {role.api_key_env} 未设置（该角色的 api key 来源）")
    else:
        if role.base_url is None:
            raise ConfigError(
                f"[{role.protocol}] 未配置 api_key_env 且未提供 base_url："
                "会默认访问官方端点，已拒绝（避免无 key 外呼）。"
                "请设置 api_key_env，或显式提供 base_url（本地/自定义兼容端点）"
            )
        api_key = "unused"  # 显式 base_url（Ollama/自定义端点）：调用路径已明确端点，允许无 key
    return make_provider(
        protocol=role.protocol, base_url=role.base_url, model=role.model, api_key=api_key
    )


def make_provider(
    *,
    protocol: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    context_window: int | None = None,
    reasoning_effort: str | None = None,
    echo_reasoning_field: str | None = None,
) -> Provider:
    """按显式参数构建 provider（运行时连接用，不经环境变量）。

    echo_reasoning_field：目录声明 interleaved 的模型，下一轮按该字段回传思考。
    """
    if protocol == "openai-responses":
        return CodexResponsesProvider(
            base_url=base_url, api_key=api_key, model=model,
            context_window=context_window, reasoning_effort=reasoning_effort,
        )
    if protocol == "openai":
        return OpenAICompatProvider(
            base_url=base_url, api_key=api_key, model=model,
            context_window=context_window, reasoning_effort=reasoning_effort,
            echo_reasoning_field=echo_reasoning_field,
        )
    if protocol == "anthropic":
        return AnthropicCompatProvider(
            api_key=api_key, model=model, base_url=base_url,
            context_window=context_window, reasoning_effort=reasoning_effort,
        )
    raise ConfigError(f"未知协议: {protocol}")
