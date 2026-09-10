"""运行时服务商状态：.providers.toml 的读写。

由 /connect_provider 成功连接后写入（600 权限、gitignore 覆盖）；
启动时若存在则优先于 config.toml 的静态角色配置。
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

_FIELDS = ("protocol", "base_url", "model", "api_key")


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def save_provider(path: str | Path, role: str, settings: dict) -> None:
    """写入/更新一个角色的连接配置（含 api_key），保留其他角色；权限 600。"""
    data = load_providers(path)
    data[role] = {k: v for k, v in settings.items() if k in _FIELDS and v is not None}
    lines: list[str] = []
    for name, section in data.items():
        lines.append(f"[{name}]")
        lines.extend(f'{key} = "{_escape(str(value))}"' for key, value in section.items())
        lines.append("")
    p = Path(path)
    p.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(p, 0o600)


def load_providers(path: str | Path) -> dict[str, dict]:
    """读取全部角色的连接配置；文件不存在返回空。"""
    p = Path(path)
    if not p.is_file():
        return {}
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    return {role: section for role, section in data.items() if isinstance(section, dict)}
