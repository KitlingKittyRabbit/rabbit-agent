"""工具层测试：tmp 目录沙箱，覆盖路径安全与各工具行为。"""

from pathlib import Path

import pytest

from agent.tools import ToolError, build_registry
from agent.tools.file_tools import make_read_tools, make_write_tools
from agent.tools.shell_tool import make_shell_tool


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.txt").write_text("hello world\n第二行 hello\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# 标题\n", encoding="utf-8")
    return tmp_path


def read_registry(root: Path):
    from agent.tools import ToolRegistry

    return ToolRegistry(make_read_tools(root))


def write_registry(root: Path):
    from agent.tools import ToolRegistry

    return ToolRegistry(make_write_tools(root))


async def test_ls_lists_dirs_with_suffix(root: Path) -> None:
    output = await read_registry(root).call("ls", {})
    assert "src/" in output
    assert "b.md" in output


async def test_read_file_returns_content(root: Path) -> None:
    output = await read_registry(root).call("read_file", {"path": "src/a.txt"})
    assert output == "hello world\n第二行 hello\n"


async def test_read_file_missing_raises(root: Path) -> None:
    with pytest.raises(ToolError, match="不存在"):
        await read_registry(root).call("read_file", {"path": "nope.txt"})


async def test_read_file_truncates_long_files(root: Path) -> None:
    (root / "big.txt").write_text("\n".join(f"行{i}" for i in range(2005)), encoding="utf-8")
    output = await read_registry(root).call("read_file", {"path": "big.txt"})
    assert "截断" in output
    assert "行1999" in output
    assert "行2004" not in output


async def test_grep_finds_matches_with_location(root: Path) -> None:
    output = await read_registry(root).call("grep", {"pattern": "hello"})
    assert "src/a.txt:1: hello world" in output
    assert "src/a.txt:2: 第二行 hello" in output


async def test_grep_no_match_and_invalid_regex(root: Path) -> None:
    assert await read_registry(root).call("grep", {"pattern": "zzzzz"}) == "(无匹配)"
    with pytest.raises(ToolError, match="正则无效"):
        await read_registry(root).call("grep", {"pattern": "["})


async def test_grep_ignores_git_directory(root: Path) -> None:
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("hello secret", encoding="utf-8")
    output = await read_registry(root).call("grep", {"pattern": "hello"})
    assert ".git" not in output


async def test_write_file_creates_parents(root: Path) -> None:
    output = await write_registry(root).call(
        "write_file", {"path": "deep/dir/c.txt", "content": "内容"}
    )
    assert "已写入" in output
    assert (root / "deep" / "dir" / "c.txt").read_text(encoding="utf-8") == "内容"


async def test_edit_file_replaces_unique_match(root: Path) -> None:
    output = await write_registry(root).call(
        "edit_file", {"path": "src/a.txt", "old": "hello world", "new": "bye world"}
    )
    assert "已修改" in output
    assert (root / "src" / "a.txt").read_text(encoding="utf-8").startswith("bye world")


async def test_edit_file_not_found_and_multiple_matches(root: Path) -> None:
    registry = write_registry(root)
    with pytest.raises(ToolError, match="未找到"):
        await registry.call("edit_file", {"path": "src/a.txt", "old": "不存在", "new": "x"})
    with pytest.raises(ToolError, match="2 处"):
        await registry.call("edit_file", {"path": "src/a.txt", "old": "hello", "new": "x"})


@pytest.mark.parametrize(
    "tool,args",
    [
        ("read_file", {"path": "../outside.txt"}),
        ("read_file", {"path": "/etc/passwd"}),
        ("write_file", {"path": "../escape.txt", "content": "x"}),
        ("edit_file", {"path": "/etc/passwd", "old": "a", "new": "b"}),
    ],
)
async def test_path_escape_rejected(root: Path, tool: str, args: dict) -> None:
    registry = build_registry(root, write=True, shell=True)
    with pytest.raises(ToolError, match="越界"):
        await registry.call(tool, args)


async def test_shell_echo_and_exit_code(root: Path) -> None:
    registry = build_registry(root, write=False, shell=True)
    output = await registry.call("run_shell", {"command": "echo 你好"})
    assert "exit code: 0" in output
    assert "你好" in output
    output = await registry.call("run_shell", {"command": "exit 3"})
    assert "exit code: 3" in output


async def test_shell_runs_in_working_dir(root: Path) -> None:
    output = await make_shell_tool(root).handler({"command": "pwd"})
    assert str(root) in output


async def test_shell_timeout_kills_process(root: Path) -> None:
    output = await make_shell_tool(root).handler({"command": "sleep 5", "timeout": 0.2})
    assert "超时" in output


async def test_call_safe_turns_errors_into_text(root: Path) -> None:
    registry = build_registry(root, write=False, shell=False)
    output = await registry.call_safe("read_file", {"path": "../x"})
    assert output.startswith("错误: ")
    output = await registry.call_safe("不存在的工具", {})
    assert "未知工具" in output


async def test_build_registry_role_assembly(root: Path) -> None:
    read_only = build_registry(root, write=False, shell=False)
    assert read_only.names() == ["grep", "ls", "read_file"]
    full = build_registry(root, write=True, shell=True)
    assert full.names() == ["edit_file", "grep", "ls", "read_file", "run_shell", "write_file"]
