"""密钥存储：静态配置走环境变量（.env），运行时连接走独立密钥文件。

密钥文件位于 ~/.rabbit-agent/keys.json（600 权限，仓库外），
.providers.toml 只保存非敏感配置，绝不落盘 api_key。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

KEYS_DIR = Path(os.path.expanduser("~/.rabbit-agent"))
KEYS_PATH = KEYS_DIR / "keys.json"


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


def load_keys(path: str | Path | None = None) -> dict[str, str]:
    p = Path(path) if path is not None else KEYS_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str) and v}


def save_key(role: str, key: str, path: str | Path | None = None) -> None:
    p = Path(path) if path is not None else KEYS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_keys(p)
    if key:
        data[role] = key
    else:
        data.pop(role, None)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(p, 0o600)


def delete_key(role: str, path: str | Path | None = None) -> None:
    save_key(role, "", path)
