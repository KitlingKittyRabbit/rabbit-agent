"""服务入口：uv run python -m agent.server"""

import logging
import os

import uvicorn

from ..core.config import ConfigError, RoleConfig, build_provider, load_config, make_provider
from ..core.keystore import load_env_file
from ..core.orchestrator import Orchestrator
from ..core.project_store import load_projects
from ..core.provider_store import load_providers
from ..core.session import SessionStore
from .app import create_app

ENV_PATH = ".env"
STORE_PATH = ".providers.toml"
PROJECTS_PATH = ".projects.toml"
LOG_PATH = "agent_server.log"

logger = logging.getLogger("agent")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def _try_build(role: RoleConfig, role_name: str):
    """无 key 也允许启动：构建失败返回 None，之后可用 /connect_provider 补配。"""
    try:
        return build_provider(role)
    except ConfigError as e:
        logger.warning("%s 未配置（%s）。启动后可用 /connect_provider 设置。", role_name, e)
        return None


def _resolve(stored: dict[str, dict], role_name: str, role_config: RoleConfig):
    """运行时状态（.providers.toml）优先；缺则静态配置；再缺则 None。"""
    settings = stored.get(role_name)
    if settings:
        try:
            return make_provider(**settings)
        except ConfigError as e:
            logger.warning("%s 的运行时配置无效（%s），回退静态配置。", role_name, e)
    return _try_build(role_config, role_name)


def main() -> None:
    _setup_logging()
    config = load_config(os.environ.get("AGENT_CONFIG", "config.toml"))
    for name, value in load_env_file(ENV_PATH).items():
        os.environ.setdefault(name, value)
    stored = load_providers(STORE_PATH)
    projects = load_projects(PROJECTS_PATH)
    orchestrator = Orchestrator(
        main_provider=_resolve(stored, "main", config.main),
        executor_provider=_resolve(stored, "executor", config.executor),
        root=config.working_dir,
        store=SessionStore(config.session_db),
        plan_mode=config.plan_mode,
        max_steps_main=config.max_steps_main,
        max_steps_executor=config.max_steps_executor,
        store_path=STORE_PATH,
        compact_threshold=config.compact_threshold_chars,
        audit_log_path=config.audit_log,
        projects=projects,
        projects_path=PROJECTS_PATH,
    )
    logger.info("server 启动，工作目录 %s，端口 %s", config.working_dir, config.port)
    uvicorn.run(create_app(orchestrator), host="127.0.0.1", port=config.port)


if __name__ == "__main__":
    main()
