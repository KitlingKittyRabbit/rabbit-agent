"""服务入口：uv run python -m agent.server"""

import logging
import os
import secrets

import uvicorn

from ..core.config import ConfigError, RoleConfig, build_provider, load_config
from ..core.keystore import load_env_file
from ..core.orchestrator import Orchestrator
from ..core.project_store import load_projects
from ..core.session import SessionStore
from .app import create_app

ENV_PATH = ".env"
STORE_PATH = ".providers.toml"
PROJECTS_PATH = ".projects.toml"
LOG_PATH = "agent_server.log"
TOKEN_PATH = os.path.expanduser("~/.agent_token")

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


def _write_token() -> str:
    """生成并落盘 localhost trust token（600），返回明文供 create_app 使用。"""
    token = secrets.token_hex(16)
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(token)
    os.chmod(TOKEN_PATH, 0o600)
    return token


def main() -> None:
    _setup_logging()
    config = load_config(os.environ.get("AGENT_CONFIG", "config.toml"))
    for name, value in load_env_file(ENV_PATH).items():
        os.environ.setdefault(name, value)
    token = _write_token()
    projects = load_projects(PROJECTS_PATH)
    # 运行时多 provider 注册表由 Orchestrator 负责；这里只提供静态配置兜底
    main_provider = _try_build(config.main, "main")
    executor_provider = _try_build(config.executor, "executor")
    orchestrator = Orchestrator(
        main_provider=main_provider,
        executor_provider=executor_provider,
        root=config.working_dir,
        store=SessionStore(config.session_db),
        plan_mode=config.plan_mode,
        max_steps_main=config.max_steps_main,
        max_steps_executor=config.max_steps_executor,
        store_path=STORE_PATH,
        keys_path=None,
        audit_log_path=config.audit_log,
        projects=projects,
        projects_path=PROJECTS_PATH,
    )
    logger.info("server 启动，工作目录 %s，端口 %s", config.working_dir, config.port)
    uvicorn.run(create_app(orchestrator, token=token), host="127.0.0.1", port=config.port)


if __name__ == "__main__":
    main()
