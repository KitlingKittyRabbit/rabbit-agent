"""服务入口：uv run python -m agent.server"""

import os

import uvicorn

from ..core.config import ConfigError, RoleConfig, build_provider, load_config, make_provider
from ..core.keystore import load_env_file
from ..core.orchestrator import Orchestrator
from ..core.provider_store import load_providers
from ..core.session import SessionStore
from .app import create_app

ENV_PATH = ".env"
STORE_PATH = ".providers.toml"


def _try_build(role: RoleConfig, role_name: str):
    """无 key 也允许启动：构建失败返回 None，之后可用 /connect_provider 补配。"""
    try:
        return build_provider(role)
    except ConfigError as e:
        print(f"警告: {role_name} 未配置（{e}）。启动后可用 /connect_provider 设置。")
        return None


def _resolve(stored: dict[str, dict], role_name: str, role_config: RoleConfig):
    """运行时状态（.providers.toml）优先；缺则静态配置；再缺则 None。"""
    settings = stored.get(role_name)
    if settings:
        try:
            return make_provider(**settings)
        except ConfigError as e:
            print(f"警告: {role_name} 的运行时配置无效（{e}），回退静态配置。")
    return _try_build(role_config, role_name)


def main() -> None:
    config = load_config(os.environ.get("AGENT_CONFIG", "config.toml"))
    for name, value in load_env_file(ENV_PATH).items():
        os.environ.setdefault(name, value)
    stored = load_providers(STORE_PATH)
    orchestrator = Orchestrator(
        main_provider=_resolve(stored, "main", config.main),
        executor_provider=_resolve(stored, "executor", config.executor),
        root=config.working_dir,
        session=SessionStore(config.session_db),
        plan_mode=config.plan_mode,
        max_steps_main=config.max_steps_main,
        max_steps_executor=config.max_steps_executor,
        store_path=STORE_PATH,
        compact_threshold=config.compact_threshold_chars,
    )
    uvicorn.run(create_app(orchestrator), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
