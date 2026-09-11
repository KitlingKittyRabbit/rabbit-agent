"""文件工具：读组（ls/read_file/grep）与写组（write_file/edit_file），根目录沙箱。"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from ..providers import ToolSpec
from .base import Tool, ToolError

_IGNORE_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache"}
_MAX_READ_LINES = 2000
_MAX_GREP_MATCHES = 100
_MAX_FILE_SIZE = 1024 * 1024


def _safe_path(root: Path, path: str) -> Path:
    """解析相对路径并禁止越出工作目录（含绝对路径与 .. 逃逸）。"""
    resolved_root = root.resolve()
    target = (resolved_root / path).resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ToolError(f"路径越界（只允许访问工作目录内）: {path}")
    return target


def _walk_files(base: Path):
    if base.is_file():
        yield base
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for name in filenames:
            file = Path(dirpath) / name
            try:
                if file.stat().st_size > _MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            yield file


def make_read_tools(root: Path) -> list[Tool]:
    resolved_root = root.resolve()

    async def ls(args: dict) -> str:
        path = args.get("path") or "."
        target = _safe_path(root, path)
        if not target.is_dir():
            raise ToolError(f"目录不存在: {path}")
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        return "\n".join(p.name + ("/" if p.is_dir() else "") for p in entries) or "(空目录)"

    async def read_file(args: dict) -> str:
        target = _safe_path(root, args["path"])
        if not target.is_file():
            raise ToolError(f"文件不存在: {args['path']}")
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        limit = args.get("limit")
        if limit is not None:
            offset = max(1, int(args.get("offset", 1)))
            window = lines[offset - 1 : offset - 1 + int(limit)]
            return "\n".join(window) + f"\n（第 {offset} 行起，共 {len(lines)} 行）"
        if len(lines) > _MAX_READ_LINES:
            head = "\n".join(lines[:_MAX_READ_LINES])
            return f"{head}\n...（截断：共 {len(lines)} 行，仅显示前 {_MAX_READ_LINES} 行）"
        return text

    async def glob(args: dict) -> str:
        pattern = args["pattern"]
        base = _safe_path(root, args.get("path") or ".")
        matches: list[str] = []
        for file in _walk_files(base):
            rel = str(file.relative_to(resolved_root))
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(file.name, pattern):
                matches.append(rel)
                if len(matches) >= 200:
                    matches.append("...（匹配过多，已截断）")
                    break
        return "\n".join(sorted(matches)) if matches else "(无匹配)"

    async def grep(args: dict) -> str:
        base = _safe_path(root, args.get("path") or ".")
        try:
            regex = re.compile(args["pattern"])
        except re.error as e:
            raise ToolError(f"正则无效: {e}") from e
        matches: list[str] = []
        for file in _walk_files(base):
            rel = file.relative_to(resolved_root)
            for lineno, line in enumerate(
                file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(matches) >= _MAX_GREP_MATCHES:
                        return "\n".join(matches) + "\n...（匹配过多，已截断）"
        return "\n".join(matches) if matches else "(无匹配)"

    return [
        Tool(
            ToolSpec(
                name="ls",
                description="列出目录内容，目录名带 / 后缀",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的路径，默认 ."}
                    },
                },
            ),
            ls,
        ),
        Tool(
            ToolSpec(
                name="read_file",
                description="读取文件内容（可用 offset/limit 取行范围；超长截断）",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的文件路径"},
                        "offset": {
                            "type": "integer",
                            "description": "起始行号（1 起），配合 limit 使用",
                        },
                        "limit": {"type": "integer", "description": "读取行数"},
                    },
                    "required": ["path"],
                },
            ),
            read_file,
        ),
        Tool(
            ToolSpec(
                name="glob",
                description="按通配符找文件（如 **/*.py），返回相对路径列表",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "通配符模式"},
                        "path": {"type": "string", "description": "搜索范围，默认整个工作目录"},
                    },
                    "required": ["pattern"],
                },
            ),
            glob,
        ),
        Tool(
            ToolSpec(
                name="grep",
                description="在工作目录内按正则搜索文件内容，返回 路径:行号: 内容",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "正则表达式"},
                        "path": {"type": "string", "description": "搜索范围，默认整个工作目录"},
                    },
                    "required": ["pattern"],
                },
            ),
            grep,
        ),
    ]


def make_write_tools(root: Path) -> list[Tool]:
    async def write_file(args: dict) -> str:
        target = _safe_path(root, args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        return f"已写入 {args['path']}（{len(args['content'])} 字符）"

    async def edit_file(args: dict) -> str:
        target = _safe_path(root, args["path"])
        if not target.is_file():
            raise ToolError(f"文件不存在: {args['path']}")
        text = target.read_text(encoding="utf-8")
        count = text.count(args["old"])
        if count == 0:
            raise ToolError("未找到要替换的原文（old 必须与文件内容完全一致）")
        if count > 1:
            raise ToolError(f"原文出现 {count} 处，无法确定替换位置，请提供更多上下文")
        target.write_text(text.replace(args["old"], args["new"], 1), encoding="utf-8")
        return f"已修改 {args['path']}"

    return [
        Tool(
            ToolSpec(
                name="write_file",
                description="写入整个文件（不存在则创建，含父目录）",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的文件路径"},
                        "content": {"type": "string", "description": "完整文件内容"},
                    },
                    "required": ["path", "content"],
                },
            ),
            write_file,
        ),
        Tool(
            ToolSpec(
                name="edit_file",
                description="精准修改：把文件中唯一匹配的一段原文替换为新内容",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的文件路径"},
                        "old": {
                            "type": "string",
                            "description": "被替换的原文，必须在文件中唯一出现",
                        },
                        "new": {"type": "string", "description": "替换后的内容"},
                    },
                    "required": ["path", "old", "new"],
                },
            ),
            edit_file,
        ),
    ]
