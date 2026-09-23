"""文件改动行数统计：write_file/edit_file 的纯函数（供任务 diff 汇总）。"""

from __future__ import annotations


def count_lines(text: str) -> int:
    """行数 = 换行数 + 末行（非空且不以换行结尾时）。"""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def file_change(tool: str, args: dict) -> dict | None:
    """把一次文件工具调用折算成 {path, added, removed}；非文件工具返回 None。"""
    path = str(args.get("path") or "")
    if not path:
        return None
    if tool == "write_file":
        return {"path": path, "added": count_lines(str(args.get("content") or "")),
                "removed": 0}
    if tool == "edit_file":
        return {"path": path, "added": count_lines(str(args.get("new") or "")),
                "removed": count_lines(str(args.get("old") or ""))}
    return None


def accrue(diff: dict[str, list[int]], change: dict) -> None:
    """把单次改动累加到 path → [added, removed]。"""
    entry = diff.setdefault(change["path"], [0, 0])
    entry[0] += change["added"]
    entry[1] += change["removed"]


def diff_payload(diff: dict[str, list[int]]) -> dict | None:
    """按路径排序生成事件载荷；无改动返回 None。"""
    if not diff:
        return None
    return {"files": [
        {"path": path, "added": added, "removed": removed}
        for path, (added, removed) in sorted(diff.items())
    ]}
