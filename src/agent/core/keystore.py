"""密钥存储：优先系统钥匙串（keyring），回退到独立密钥文件。

- 钥匙串可用：密钥存系统 keyring（service=rabbit-agent，username=provider_id），
  keys.json 只保留名字索引（值为空串），明文不落盘。
- 钥匙串不可用：退回 ~/.rabbit-agent/keys.json（600，仓库外）。
- 环境变量 RABBIT_AGENT_KEYRING=0 强制文件模式（测试隔离用）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

KEYS_DIR = Path(os.path.expanduser("~/.rabbit-agent"))
KEYS_PATH = KEYS_DIR / "keys.json"

_SERVICE = "rabbit-agent"
_DISABLE_ENV = "RABBIT_AGENT_KEYRING"

logger = logging.getLogger(__name__)

_OVERRIDE_BACKEND: object | None = None   # 测试注入：需提供 get/set/delete_password


def _probe(backend: object) -> bool:
    try:
        backend.get_password(_SERVICE, "__probe__")   # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def _backend() -> object | None:
    """钥匙串后端；不可用返回 None（回退文件）。"""
    if _OVERRIDE_BACKEND is not None:
        return _OVERRIDE_BACKEND
    if os.environ.get(_DISABLE_ENV) == "0":
        return None
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get(_DISABLE_ENV) != "1":
        return None          # 测试默认走文件模式；确需真实钥匙串请显式设 RABBIT_AGENT_KEYRING=1
    try:
        import keyring
    except Exception:
        return None
    return keyring if _probe(keyring) else None


def _kr_get(name: str) -> str | None:
    backend = _backend()
    if backend is None:
        return None
    try:
        value = backend.get_password(_SERVICE, name)   # type: ignore[attr-defined]
    except Exception:
        return None
    return value or None


def _kr_set(name: str, key: str) -> bool:
    backend = _backend()
    if backend is None:
        return False
    try:
        backend.set_password(_SERVICE, name, key)      # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("钥匙串写入失败，回退文件: %s", e)
        return False
    return True


def _kr_delete(name: str) -> None:
    backend = _backend()
    if backend is None:
        return
    try:
        backend.delete_password(_SERVICE, name)        # type: ignore[attr-defined]
    except Exception:
        pass


def _read_file(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _write_index(path: Path, names: dict[str, str]) -> None:
    """写文件：钥匙串条目值写空串，未迁移的明文原样保留。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(path, 0o600)


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
    """返回 name → 真实密钥（钥匙串优先；文件里的明文向后兼容）。"""
    p = Path(path) if path is not None else KEYS_PATH
    raw = _read_file(p)
    out: dict[str, str] = {}
    for name, value in raw.items():
        if value:
            out[name] = value          # 迁移前的明文条目
            continue
        secret = _kr_get(name)
        if secret:
            out[name] = secret
    return out


def save_key(role: str, key: str, path: str | Path | None = None) -> None:
    p = Path(path) if path is not None else KEYS_PATH
    raw = _read_file(p)
    if key:
        if _kr_set(role, key):
            raw[role] = ""             # 钥匙串接管：文件只留名字
        else:
            raw[role] = key            # 回退：明文落文件（600）
    else:
        _kr_delete(role)
        raw.pop(role, None)
    _write_index(p, raw)


def delete_key(role: str, path: str | Path | None = None) -> None:
    save_key(role, "", path)


def migrate_keys(path: str | Path | None = None) -> int:
    """把文件里的明文密钥迁进钥匙串并清空明文（不可用则原样保留）。返回迁移条数。"""
    p = Path(path) if path is not None else KEYS_PATH
    raw = _read_file(p)
    moved = 0
    for name, value in list(raw.items()):
        if not value:
            continue
        if _kr_set(name, value):
            raw[name] = ""
            moved += 1
    if moved:
        _write_index(p, raw)
        logger.info("已迁移 %d 条密钥到系统钥匙串，明文已删除", moved)
    return moved
