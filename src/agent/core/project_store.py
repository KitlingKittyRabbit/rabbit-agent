"""多项目运行时状态：.projects.toml 读写。会话依附于项目。"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

_FIELDS = ("name", "path")


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def save_project(path: str | Path, project_id: str, settings: dict) -> None:
    """写入/更新一个项目，保留其他项目；权限 600。"""
    data = load_projects(path)
    data[project_id] = {k: str(v) for k, v in settings.items() if k in _FIELDS and v is not None}
    lines: list[str] = []
    for pid, section in data.items():
        lines.append(f"[{pid}]")
        lines.extend(f'{key} = "{_escape(value)}"' for key, value in section.items())
        lines.append("")
    p = Path(path)
    p.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(p, 0o600)


def remove_project(path: str | Path, project_id: str) -> None:
    data = load_projects(path)
    if project_id in data:
        del data[project_id]
        lines: list[str] = []
        for pid, section in data.items():
            lines.append(f"[{pid}]")
            lines.extend(f'{key} = "{_escape(value)}"' for key, value in section.items())
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")


def load_projects(path: str | Path) -> dict[str, dict]:
    """读取全部项目（id → {name, path}）；文件不存在返回空。"""
    p = Path(path)
    if not p.is_file():
        return {}
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    return {pid: section for pid, section in data.items() if isinstance(section, dict)}
