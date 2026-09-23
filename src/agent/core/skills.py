"""Skill：项目/全局目录下的 SKILL.md，按命令唤起（不自动进上下文）。

查找：项目级 <root>/.agent/skills/<名称>/SKILL.md 优先，其次 ~/.agent/skills/。
唤起：用户发 `/skill <名称> [指令]`，把 SKILL.md 正文作为本轮提示注入（仅指挥者）。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_SKILL_DIR = Path(".agent") / "skills"
GLOBAL_SKILL_DIR = Path(os.path.expanduser("~/.agent/skills"))
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _describe(text: str) -> str:
    """描述：front-matter 的 description:，否则第一条非空非标题行。"""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.lower().startswith("description:"):
                return line.split(":", 1)[1].strip()[:120]
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:120]
    return ""


def _read(path: Path) -> tuple[str, str] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text, _describe(text)


def list_skills(root: str | Path | None) -> list[dict]:
    """可用技能列表（项目级覆盖同名的全局技能），按名称排序。"""
    found: dict[str, dict] = {}
    bases: list[tuple[str, Path]] = [("global", GLOBAL_SKILL_DIR)]
    if root is not None:
        bases.append(("project", Path(root) / PROJECT_SKILL_DIR))
    for scope, base in bases:
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or not _NAME_RE.match(entry.name):
                continue
            read = _read(entry / "SKILL.md")
            if read is None:
                continue
            _text, description = read
            found[entry.name] = {
                "name": entry.name, "description": description,
                "scope": scope, "path": str(entry / "SKILL.md"),
            }
    return [found[name] for name in sorted(found)]


def find_skill(root: str | Path | None, name: str) -> dict | None:
    if not _NAME_RE.match(name or ""):
        return None
    for spec in list_skills(root):
        if spec["name"] == name:
            return spec
    return None


def skill_prompt(spec: dict, user_text: str = "") -> str:
    """把技能正文与用户指令拼成本轮提示。"""
    try:
        body = Path(spec["path"]).read_text(encoding="utf-8").strip()
    except OSError:
        body = ""
    parts = [f"[技能 {spec['name']}]", body]
    if user_text.strip():
        parts += ["", "[用户指令]", user_text.strip()]
    return "\n".join(parts)
