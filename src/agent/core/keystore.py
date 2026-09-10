"""启动时加载 .env 到环境变量（静态配置的 key 来源）。

运行时连接的服务商状态存于 provider_store（.providers.toml）。
"""

from __future__ import annotations

from pathlib import Path


def load_env_file(path: str | Path) -> dict[str, str]:
    """读取 .env 全部键值；文件不存在返回空。"""
    p = Path(path)
    if not p.is_file():
        return {}
    pairs: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            pairs[name.strip()] = value.strip()
    return pairs
