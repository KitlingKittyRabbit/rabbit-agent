"""宿主侧 GitHub/Git 工具测试：零网络、零真实凭据、不碰真实 ~/.config/gh 与 active 状态。

所有用例都用临时 fake gh/git 可执行文件（或真实 git 的纯本地操作）验证边界：
- 守护 fixture 让测试进程永远解析不到真实 gh/git（防误碰真实配置与网络）；
- HOME 重定向到临时目录（沙箱遮蔽的是这个假 HOME，绝不动真实 HOME）；
- fake 可执行文件与日志都放在"项目根"内（沙箱里项目根是唯一可写目录）。
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import tempfile
import time
from pathlib import Path

import pytest

import agent.core.executor_session as executor_session
import agent.tools.host_github as host_github
from agent.core.audit import AuditLogger
from agent.core.orchestrator import Orchestrator
from agent.providers import ChatResult, FakeProvider
from agent.tools import ToolError, ToolRegistry, build_registry
from agent.tools.base import scrub_secrets
from agent.tools.host_github import (
    HostCredentialRunner,
    _masked_in_sandbox,
    _Sandbox,
    _under_home,
    make_gh_accounts_tool,
    make_gh_command_tool,
    make_git_remote_tool,
    parse_auth_status,
    sandbox_mount_argv,
    tool_mount_argv,
)

_REAL_WHICH = shutil.which
REAL_GIT = shutil.which("git")     # 收集期取一次（守护 fixture 生效后就解析不到了）
REAL_BWRAP = shutil.which("bwrap")

TOKEN_ALICE = "tok-alice-0001"
TOKEN_BOB = "tok-bob-0002"
FAKE_STATUS_TOKEN = "gho_FAKEFAKEFAKE0FAKEFAKEFAKE0FAKEFAKE0"
HOME_SECRET = "PRIVATE-KEY-SENTINEL-XYZ"

requires_bwrap = pytest.mark.skipif(
    REAL_BWRAP is None, reason="bwrap 不可用：文件系统沙箱用例无法执行"
)

_STATUS_TEXT = f"""github.com
  ✓ Logged in to github.com account alice (keyring)
  - Active account: true
  - Git operations protocol: https
  - Token: {FAKE_STATUS_TOKEN}
  - Token scopes: 'gist', 'read:org', 'repo', 'workflow'

  ✓ Logged in to github.com account bob (keyring)
  - Active account: false
  - Token: {FAKE_STATUS_TOKEN}
  - Token scopes: 'repo', 'workflow'
"""

_NOT_LOGGED_IN = "You are not logged into any GitHub hosts. To log in, run: gh auth login"


@pytest.fixture(autouse=True)
def _fake_home(tmp_path, monkeypatch):
    """HOME 指向临时目录：沙箱遮蔽它，真实 HOME 与 ~/.config/gh 全程不参与。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _never_resolve_real_gh_git(monkeypatch):
    """守护：测试进程解析不到真实 gh/git，绝不可能执行宿主二进制或碰真实配置。"""

    def guarded(name: str):
        if name in ("gh", "git"):
            return None  # 只有显式传入的 fake 路径才会被使用
        return _REAL_WHICH(name)  # bwrap 等仍按真实环境解析

    monkeypatch.setattr(host_github.shutil, "which", guarded)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """项目根（HOME 之外）：fake 可执行文件与日志都放这里，沙箱内可读写。"""
    proj = tmp_path / "proj"
    proj.mkdir()
    return proj


def _write_script(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o700)
    return str(path)


_GH_SCRIPT = """#!/bin/sh
LOG=__LOG__
printf 'CALL | cwd=%s | config_dir=%s | config_entries=%s | token=%s | argv=%s\\n' \\
  "$PWD" "${GH_CONFIG_DIR-<unset>}" \\
  "$(ls -A "${GH_CONFIG_DIR:-/nonexistent}" 2>/dev/null | wc -l)" \\
  "$([ -n "${GH_TOKEN:-}" ] && echo yes || echo no)" "$*" >> "$LOG"
if [ "$1 $2" = "auth switch" ]; then printf 'AUTH_SWITCH_CALLED\\n' >> "$LOG"; exit 97; fi
case "$1" in
  auth)
    case "$2" in
      status)
        cat __STATUS__ >&2
        exit __STATUS_CODE__
        ;;
      token)
__TOKENS__
        ;;
      *)
        printf 'AUTH_OTHER=%s\\n' "$*" >> "$LOG"
        exit 97
        ;;
    esac
    ;;
__COMMANDS__
  *)
    printf 'UNEXPECTED\\n' >&2
    exit 98
    ;;
esac
exit 0
"""


def _token_block(pairs: dict[str, str]) -> str:
    lines = ['        case "$*" in']
    for user, token in pairs.items():
        lines.append(f'          *"--user {user}"*) printf \'%s\\n\' {token!r} ;;')
    lines.append("          *) printf 'no oauth token found\\n' >&2; exit 1 ;;")
    lines.append("        esac")
    return "\n".join(lines)


def make_fake_gh(
    project: Path,
    *,
    log: Path | None = None,
    status_text: str = _STATUS_TEXT,
    status_code: int = 0,
    commands: str = "",
    tokens: dict[str, str] | str | None = None,
) -> str:
    """生成 fake gh：所有调用写日志，认证相关行为可控（放在项目根内）。"""
    bindir = project / ".fakebin"
    bindir.mkdir(exist_ok=True)
    log = log or (project / ".fake.log")
    status_file = bindir / "gh-status.txt"
    status_file.write_text(status_text, encoding="utf-8")
    if isinstance(tokens, str):
        block = tokens
    else:
        default_tokens = {"alice": TOKEN_ALICE, "bob": TOKEN_BOB}
        block = _token_block(tokens if tokens is not None else default_tokens)
    script = (
        _GH_SCRIPT.replace("__COMMANDS__", commands)
        .replace("__TOKENS__", block)
        .replace("__STATUS__", shlex.quote(str(status_file)))
        .replace("__STATUS_CODE__", str(status_code))
        .replace("__LOG__", shlex.quote(str(log)))
    )
    return _write_script(bindir / "gh", script)


_GIT_SCRIPT = """#!/bin/sh
if [ "$1" = "--exec-path" ]; then exit 1; fi
LOG=__LOG__
FULL="$*"
HOOKS=""
for arg in "$@"; do
  case "$arg" in core.hooksPath=*) HOOKS="${arg#core.hooksPath=}" ;; esac
done
while [ "$1" = "-c" ]; do shift 2; done
SUB="${1:-}"
CREDS="$([ -n "${GIT_ASKPASS:-}" ] && echo yes || echo no)"
TOK="$([ -n "${RABBIT_AGENT_GIT_TOKEN:-}" ] && echo yes || echo no)"
HOK="$([ -n "$HOOKS" ] && [ -d "$HOOKS" ] && echo yes || echo no)"
HEN="$(ls -A "$HOOKS" 2>/dev/null | tr '\\n' ',')"
WD="$(dirname "${GIT_ASKPASS:-/nonexistent}")"
printf 'CALL | sub=%s | creds=%s | token=%s | hk=%s | hok=%s | hen=%s' \\
  "$SUB" "$CREDS" "$TOK" "$HOOKS" "$HOK" "$HEN" >> "$LOG"
printf ' | wd=%s | cwd=%s | argv=%s\\n' "$WD" "$PWD" "$FULL" >> "$LOG"
case "$SUB" in
  rev-parse)
__REVPARSE__
    ;;
  remote)
    if [ "$2" = "get-url" ]; then
      case "$3" in
__URLS__
        *) printf 'error: No such remote\\n' >&2; exit 2 ;;
      esac
    else
      printf 'origin\\n'
    fi
    ;;
  fetch)
__FETCH__
    ;;
  merge)
__MERGE__
    ;;
  push)
__PUSH__
    ;;
  *)
    printf 'UNEXPECTED: %s\\n' "$FULL" >&2
    exit 98
    ;;
esac
exit 0
"""


def make_fake_git(
    project: Path,
    *,
    log: Path | None = None,
    toplevel: Path | None = None,
    urls: dict[str, str] | None = None,
    revparse: str = 'printf \'%s\\n\' "__TOPLEVEL__"',
    fetch: str = 'printf \'FETCH %s\\n\' "$*"',
    merge: str = 'printf \'MERGE %s\\n\' "$*"',
    push: str = 'printf \'PUSH %s\\n\' "$*"',
) -> str:
    """生成 fake git：每个调用一行结构化日志（argv/cwd/凭据/空 hooks 目录）。"""
    bindir = project / ".fakebin"
    bindir.mkdir(exist_ok=True)
    log = log or (project / ".fake.log")
    urls = urls if urls is not None else {"origin": "https://github.com/owner/repo.git"}
    url_cases = "\n".join(
        f"        {name}) printf '%s\\n' {shlex.quote(url)} ;;" for name, url in urls.items()
    )
    script = (
        _GIT_SCRIPT.replace("__URLS__", url_cases)
        .replace("__FETCH__", fetch)
        .replace("__MERGE__", merge)
        .replace("__PUSH__", push)
        .replace("__REVPARSE__", revparse)
        .replace("__TOPLEVEL__", str(toplevel if toplevel is not None else project))
        .replace("__LOG__", shlex.quote(str(log)))
    )
    return _write_script(bindir / "git", script)


class Fake:
    """一对 fake 可执行文件 + 日志读取。"""

    def __init__(self, project: Path, *, gh: str, git: str, log: Path) -> None:
        self.project = project
        self.gh = gh
        self.git = git
        self.log = log

    def text(self) -> str:
        return self.log.read_text(encoding="utf-8") if self.log.exists() else ""

    def gh_calls(self) -> list[str]:
        """gh 侧的调用 argv（git 侧另有 sub= 行，用 parse_git_calls 读）。"""
        return [
            line.split(" | argv=")[1] for line in self.text().splitlines()
            if " | config_dir=" in line
        ]


def make_fake(project: Path, *, git: dict | None = None, **kwargs) -> Fake:
    """一对共享同一日志的 fake gh/git（git 行为可用 git={...} 定制）。"""
    log = project / ".fake.log"
    return Fake(
        project,
        gh=make_fake_gh(project, log=log, **kwargs),
        git=make_fake_git(project, log=log, **(git or {})),
        log=log,
    )


def make_runner(fake: Fake, **kwargs) -> HostCredentialRunner:
    return HostCredentialRunner(gh_path=fake.gh, git_path=fake.git, **kwargs)


def parse_git_calls(text: str) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("CALL | ") or " | sub=" not in line:
            continue
        fields = line[len("CALL | "):].split(" | ")
        calls.append(dict(field.partition("=")[::2] for field in fields))
    return calls


def _no_bwrap(monkeypatch) -> None:
    monkeypatch.setattr(host_github, "_bwrap_path", lambda: None)


def _sandbox_dirs() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("rabbit-sbx-*")}


# ---------- 沙箱构造（argv，不经 shell） ----------

def test_sandbox_argv_is_argv_not_shell(project: Path) -> None:
    prefix = sandbox_mount_argv(project)
    assert prefix[0].endswith("bwrap")
    assert "sh" not in prefix and "-c" not in prefix  # 挂载参数里没有 shell
    sandbox = _Sandbox(project, ["/usr/bin/gh"])
    try:
        target = ["/usr/bin/gh", "issue", "list", "--limit", "5"]
        argv = sandbox.argv_for(target)
        assert argv[:len(prefix)] == prefix
        # 目标 argv 原样追加，未被拼接成字符串
        assert argv[-len(target):] == target
        assert all(isinstance(part, str) for part in argv)
        assert "sh -c" not in " ".join(argv)
    finally:
        sandbox.close()


def test_sandbox_masks_whole_home_before_project_bind(project: Path) -> None:
    """顺序锁定：HOME 整体遮蔽必须早于把项目根 bind 回来，否则项目根会被一并遮掉。"""
    home = Path(os.environ["HOME"]).resolve()
    argv = sandbox_mount_argv(project)
    home_at = argv.index(str(home))
    project_at = argv.index(str(project))
    assert argv[home_at - 1] == "--tmpfs"        # 整个 HOME 被遮蔽
    assert home_at < project_at                  # 遮蔽在前、项目根 bind 在后
    assert "--ro-bind" in argv and argv[argv.index("--ro-bind") + 1] == "/"
    assert "--tmpfs" in argv and "/tmp" in argv  # 临时 /tmp
    assert "--unshare-pid" in argv               # 宿主进程列表/环境不可见
    assert "--unshare-net" not in argv           # 网络保持可用（要访问 GitHub）


def test_sandbox_fails_closed_without_bwrap(project: Path, monkeypatch) -> None:
    _no_bwrap(monkeypatch)
    with pytest.raises(ToolError, match="未找到 bwrap"):
        sandbox_mount_argv(project)
    fake = make_fake(project)
    with pytest.raises(ToolError, match="未找到 bwrap"):
        asyncio.run(make_runner(fake).run_gh(["pr", "list"], user="alice", cwd=project))
    with pytest.raises(ToolError, match="未找到 bwrap"):
        asyncio.run(make_runner(fake).git_remote(
            action="fetch", remote="origin", user="alice", cwd=project
        ))
    assert fake.text() == ""  # 绝不启动目标 gh/git（连凭据查询都没发生）


async def test_sandbox_failure_is_reported_and_never_falls_back(project: Path, monkeypatch) -> None:
    """bwrap 启动失败：明确报错且不执行目标命令，绝不回退到未沙箱宿主执行。"""
    broken = project / ".fakebin" / "bwrap"
    broken.parent.mkdir(exist_ok=True)
    _write_script(broken, "#!/bin/sh\nprintf 'bwrap: 沙箱启动失败\\n' >&2\nexit 1\n")
    monkeypatch.setattr(host_github, "_bwrap_path", lambda: str(broken))
    fake = make_fake(project, commands='  pr)\n    printf "SHOULD_NOT_RUN\\n"\n    ;;\n')
    output = await make_runner(fake).run_gh(["pr", "list"], user="alice", cwd=project)
    assert "沙箱启动失败" in output
    assert "不会回退到未沙箱" in output
    assert "SHOULD_NOT_RUN" not in output
    assert "pr list" not in fake.text()  # 目标 gh 从未运行


@pytest.mark.parametrize(
    "home_override,reason",
    [
        ("SAME_AS_PROJECT", "等于宿主 HOME"),
        ("PARENT_OF_PROJECT", "包含宿主 HOME"),
        ("/", "文件系统根"),
    ],
)
def test_sandbox_rejects_unsafe_boundaries(
    project: Path, monkeypatch, home_override: str, reason: str
) -> None:
    if home_override == "SAME_AS_PROJECT":
        monkeypatch.setenv("HOME", str(project))
    elif home_override == "PARENT_OF_PROJECT":
        monkeypatch.setenv("HOME", str(project / "inner"))
    else:
        monkeypatch.setenv("HOME", home_override)
    with pytest.raises(ToolError, match=reason):
        sandbox_mount_argv(project)


def test_sandbox_rejects_missing_home_env(project: Path, monkeypatch) -> None:
    monkeypatch.delenv("HOME", raising=False)
    with pytest.raises(ToolError, match="无法确定宿主 HOME"):
        sandbox_mount_argv(project)


def test_sandbox_rejects_missing_project(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="项目根不存在"):
        sandbox_mount_argv(tmp_path / "nope")


def test_sandbox_rejects_project_containing_work_mount(monkeypatch) -> None:
    """项目根若包含沙箱临时挂载点，临时空间会遮住项目内容：必须拒绝。

    HOME 指到 /tmp 之外，确保命中的是这条检查而不是"项目包含 HOME"那条。
    """
    monkeypatch.setenv("HOME", "/var/empty-rabbit-test")
    with pytest.raises(ToolError, match="沙箱临时挂载点"):
        sandbox_mount_argv(Path("/tmp"))


def test_tool_mount_argv_rebinds_masked_executables(project: Path) -> None:
    """HOME 内/ /tmp 内的 gh 会被遮蔽，必须只读挂回；系统路径无需重复挂载。"""
    home = Path(os.environ["HOME"]).resolve()
    home_gh = home / ".local" / "bin" / "gh"
    home_gh.parent.mkdir(parents=True)
    home_gh.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tmp_gh = project / ".fakebin" / "gh"
    tmp_gh.parent.mkdir(exist_ok=True)
    tmp_gh.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    assert _under_home(home_gh, home) and _masked_in_sandbox(home_gh, home)
    assert _masked_in_sandbox(tmp_gh, home)          # 项目在 /tmp 下时程序也被遮蔽
    assert not _masked_in_sandbox(Path("/usr/bin/git"), home)

    argv = tool_mount_argv([str(home_gh), "/usr/bin/git"], home)
    assert argv == ["--ro-bind", str(home_gh), str(home_gh)]
    assert str(home_gh) in sandbox_mount_argv(project, executables=[str(home_gh)])
    # git 运行文件目录（HOME 内安装时）同样挂回
    exec_path = home / ".local" / "libexec" / "git-core"
    exec_path.mkdir(parents=True)
    assert tool_mount_argv([str(exec_path)], home) == [
        "--ro-bind", str(exec_path), str(exec_path),
    ]


@requires_bwrap
async def test_sandbox_masks_home_installed_binary_dir_contents(project: Path) -> None:
    """HOME 内二进制只挂回程序本身：同目录其它文件与用户配置仍不可见。"""
    home = Path(os.environ["HOME"]).resolve()
    bindir = home / ".local" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "gh").write_text("#!/bin/sh\necho FAKE\n", encoding="utf-8")
    (bindir / "gh").chmod(0o700)
    (bindir / "notes.txt").write_text("HOME-NOTES", encoding="utf-8")
    sandbox = _Sandbox(project, [str(bindir / "gh")])
    try:
        argv = sandbox.argv_for(["sh", "-c", f"ls -A {bindir}; cat {bindir}/notes.txt 2>&1"])
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(project), env=dict(os.environ),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out = (await proc.communicate())[0].decode()
        assert "gh" in out and "HOME-NOTES" not in out
    finally:
        sandbox.close()


@requires_bwrap
async def test_real_bwrap_failure_is_annotated(project: Path) -> None:
    """真实 bwrap 报错（目标程序不存在）必须被标注为沙箱失败，绝不静默当普通错误。"""
    sandbox = _Sandbox(project, ["/nonexistent/gh-xyz"])
    try:
        runner = make_runner(make_fake(project))
        result = await runner._exec(
            ["/nonexistent/gh-xyz", "pr", "list"], cwd=project, env=sandbox.git_env(),
            timeout=5, sandbox=sandbox,
        )
        rendered = runner._render(result, sandboxed=True)
        assert "bwrap:" in rendered
        assert "不会回退到未沙箱" in rendered
    finally:
        sandbox.close()


def test_render_marks_sandbox_startup_failure() -> None:
    from agent.tools.host_github import _Exec

    runner = HostCredentialRunner(gh_path="/x", git_path="/y")
    rendered = runner._render(_Exec(1, "", "bwrap: execvp: No such file"), sandboxed=True)
    assert "沙箱启动失败" in rendered and "不会回退到未沙箱" in rendered
    # 普通失败不加这句（避免误导）
    plain = runner._render(_Exec(4, "boom", ""), sandboxed=True)
    assert "沙箱启动失败" not in plain
    assert plain == "exit code: 4\nboom"


@requires_bwrap
async def test_git_exec_path_probe_runs_inside_sandbox(project: Path, _fake_home) -> None:
    """HOME 内 git 的运行文件目录探测本身也是 git 命令：必须进 bwrap，不能在宿主直跑。

    用"记录调用参数"的 fake git 证明：探测发生在沙箱内（cwd 是项目根、日志写在项目内——
    只有沙箱内可写），且它报告的 HOME 内 git-core 目录会被并入后续挂载参数。
    """
    bindir = _fake_home / "bindir"
    bindir.mkdir()
    exec_path = _fake_home / ".local" / "libexec" / "git-core"
    exec_path.mkdir(parents=True)
    log = project / ".probe.log"
    fake_git = bindir / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        f"printf 'PROBE | cwd=%s | argv=%s\\n' \"$PWD\" \"$*\" >> {log}\n"
        f"printf '%s\\n' {exec_path}\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    sandbox = _Sandbox(project, [str(fake_git)])
    try:
        # 探测已发生且在沙箱内：cwd 是项目根、日志落在项目内
        assert log.exists(), "探测未运行"
        assert f"cwd={project}" in log.read_text(encoding="utf-8")
        # 探测报告的 HOME 内 git-core 目录被挂回沙箱（否则 HOME 内安装的 git 会缺运行文件）
        assert str(exec_path) in sandbox.argv_for([])
    finally:
        sandbox.close()


@requires_bwrap
@pytest.mark.skipif(REAL_GIT is None, reason="需要本机 git")
async def test_git_exec_path_probe_never_touches_host_when_bwrap_missing(
    project: Path, _fake_home, monkeypatch
) -> None:
    """bwrap 不可用时探测直接放弃（返回空），绝不在宿主上跑 git。"""
    bindir = _fake_home / "bindir"
    bindir.mkdir()
    marker = project / ".probe-ran"
    fake_git = bindir / "git"
    fake_git.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    fake_git.chmod(0o700)
    _no_bwrap(monkeypatch)
    assert host_github._git_exec_path(str(fake_git), _fake_home, project) == ""
    assert not marker.exists(), "bwrap 不可用时探测不得在宿主执行 git"


# ---------- 账号枚举（只读，无需沙箱） ----------

async def test_gh_accounts_lists_accounts_active_and_scopes(project: Path) -> None:
    fake = make_fake(project)
    output = await make_gh_accounts_tool(make_runner(fake)).handler({})
    assert "github.com / alice（active）" in output
    assert "github.com / bob" in output
    assert "bob（active）" not in output
    assert "scopes: gist, read:org, repo, workflow" in output
    assert FAKE_STATUS_TOKEN not in output
    assert "Token" not in output
    assert "keyring" not in output
    assert fake.gh_calls() == ["auth status"]  # 只读命令，未加 --show-token


async def test_gh_accounts_works_without_bwrap(project: Path, monkeypatch) -> None:
    """只读账号查询是固定命令、输出受控：不依赖 bwrap。"""
    _no_bwrap(monkeypatch)
    fake = make_fake(project)
    output = await make_gh_accounts_tool(make_runner(fake)).handler({})
    assert "github.com / alice（active）" in output


async def test_gh_accounts_reports_none_when_not_logged_in(project: Path) -> None:
    fake = make_fake(project, status_text=_NOT_LOGGED_IN, status_code=1)
    output = await make_gh_accounts_tool(make_runner(fake)).handler({})
    assert "没有发现已登录的 GitHub 账号" in output
    assert "auth login" not in output
    assert "auth switch" not in fake.text()


def test_parse_auth_status_old_format_takes_first_account_as_active() -> None:
    text = (
        "github.com\n"
        "  ✓ Logged in to github.com as alice (/home/u/.config/gh/hosts.yml)\n"
        "  ✓ Token: *******************\n"
        "  ✓ Logged in to github.com as bob (/home/u/.config/gh/hosts.yml)\n"
    )
    accounts = parse_auth_status(text)
    assert [(a.user, a.active) for a in accounts] == [("alice", True), ("bob", False)]


def test_parse_auth_status_reports_failed_accounts() -> None:
    text = (
        "github.com\n"
        "  X Failed to log in to github.com account bob (/home/u/.config/gh/hosts.yml)\n"
        "  - Active account: true\n"
    )
    accounts = parse_auth_status(text)
    assert [(a.user, a.ok, a.active) for a in accounts] == [("bob", False, True)]


async def test_missing_gh_is_reported_cleanly(project: Path) -> None:
    runner = HostCredentialRunner(gh_path="", git_path="", timeout=5)
    with pytest.raises(ToolError, match="宿主机未找到 gh"):
        await runner.accounts()


async def test_unstartable_gh_is_reported_cleanly(project: Path) -> None:
    runner = HostCredentialRunner(gh_path="/nonexistent/gh-xyz", git_path="/nonexistent/git-xyz")
    with pytest.raises(ToolError, match="无法启动 gh"):
        await runner.accounts()


# ---------- gh 子命令：身份、凭据、沙箱 ----------

@requires_bwrap
async def test_gh_command_uses_specified_account_in_sandbox(project: Path) -> None:
    commands = """
  pr)
    if [ "$GH_TOKEN" = '__ALICE_TOKEN__' ]; then
      printf 'IDENTITY=alice\\n'
    else
      printf 'IDENTITY=other\\n'
    fi
    printf 'config_dir=%s\\n' "$GH_CONFIG_DIR"
    printf 'config_entries=%s\\n' "$(ls -A "$GH_CONFIG_DIR" | wc -l)"
    printf 'cwd=%s\\n' "$PWD"
    printf 'home=%s\\n' "$HOME"
    ;;
"""
    fake = make_fake(project, commands=commands.replace("__ALICE_TOKEN__", TOKEN_ALICE))
    before = _sandbox_dirs()
    output = await make_gh_command_tool(make_runner(fake), project).handler(
        {"user": "alice", "args": ["pr", "list"]}
    )
    assert "IDENTITY=alice" in output
    assert "exit code: 0" in output
    log = fake.text()
    assert fake.gh_calls() == ["auth token --hostname github.com --user alice", "pr list"]
    assert "auth switch" not in log and "AUTH_SWITCH_CALLED" not in log
    assert TOKEN_ALICE not in log  # token 从不进 argv
    # GH_CONFIG_DIR 在沙箱临时空间，且为空（读不到宿主 hosts.yml）
    assert "config_dir=/tmp/.rabbit-agent/gh-config" in output
    assert "config_entries=0" in output
    assert "cwd=" + str(project) in output
    assert f"home={Path(os.environ['HOME']).resolve()}" in output
    assert _sandbox_dirs() == before  # 沙箱临时空间结束即销毁
    # 取凭据仍读宿主配置（无 GH_CONFIG_DIR），目标进程才有沙箱内配置目录
    assert "config_dir=<unset>" in log


async def test_gh_command_tool_has_no_cwd_parameter(project: Path) -> None:
    tool = make_gh_command_tool(make_runner(make_fake(project)), project)
    props = tool.spec.parameters["properties"]
    assert "cwd" not in props and set(tool.spec.parameters["required"]) == {"user", "args"}


async def test_gh_command_requires_string_array(project: Path) -> None:
    runner = make_runner(make_fake(project))
    for argv in ("pr list", [], None, ["pr", 7], ["pr", None], {"command": "pr"}):
        with pytest.raises(ToolError, match="args 必须是|字符串|前导选项|命令名"):
            await runner.run_gh(argv, user="alice", cwd=project)


@pytest.mark.parametrize(
    "argv",
    [
        ["auth", "token"],
        ["auth", "switch", "--user", "bob"],
        ["auth", "status", "--show-token"],
        ["config", "get", "git_protocol"],
        ["alias", "set", "x", "!sh -c whoami"],
        ["extension", "exec", "whoami"],
        ["ext", "list"],
        ["copilot", "suggest", "-t", "shell"],
        ["--hostname", "evil.example", "pr", "list"],
        ["pr", "list", "--hostname=evil.example"],
        ["pr", "list", "--show-token"],
    ],
)
async def test_gh_command_rejects_auth_config_alias_extension(
    project: Path, argv: list[str]
) -> None:
    fake = make_fake(project)
    with pytest.raises(ToolError):
        await make_runner(fake).run_gh(argv, user="alice", cwd=project)
    assert fake.gh_calls() == []  # 拒绝发生在创建子进程之前


async def test_gh_command_rejects_bad_user(project: Path) -> None:
    fake = make_fake(project)
    for user in ("", "alice; rm -rf /", "-x", "a b"):
        with pytest.raises(ToolError, match="user 必须是"):
            await make_runner(fake).run_gh(["pr", "list"], user=user, cwd=project)
    assert fake.gh_calls() == []


async def test_gh_command_unknown_account_is_reported(project: Path) -> None:
    fake = make_fake(project)
    with pytest.raises(ToolError, match="无法取得账号 carol"):
        await make_runner(fake).run_gh(["pr", "list"], user="carol", cwd=project)
    assert fake.gh_calls() == ["auth token --hostname github.com --user carol"]


async def test_only_fixed_credential_queries_skip_the_sandbox(project: Path, monkeypatch) -> None:
    """结构锁定：只有两条固定凭据查询可以在宿主直跑，其余 gh/git 命令必须带沙箱。"""
    seen: list[tuple[list[str], bool]] = []
    original = HostCredentialRunner._exec

    async def spy(self, argv, **kwargs):
        seen.append((list(argv), kwargs.get("sandbox") is not None))
        return await original(self, argv, **kwargs)

    monkeypatch.setattr(HostCredentialRunner, "_exec", spy)
    fake = make_fake(project)
    runner = make_runner(fake)
    await runner.accounts()
    await runner.run_gh(["pr", "list"], user="alice", cwd=project)
    await runner.git_remote(action="fetch", remote="origin", user="alice", cwd=project)

    unsandboxed = [argv for argv, sandboxed in seen if not sandboxed]
    # 只允许固定凭据查询：auth status 与 auth token（每次目标调用各取一次凭据）
    assert unsandboxed, "应当有固定凭据查询在宿主直跑"
    for argv in unsandboxed:
        assert argv[1] == "auth" and argv[2] in ("status", "token"), argv
    assert [argv[2] for argv in unsandboxed].count("status") == 1
    assert [argv[2] for argv in unsandboxed].count("token") == 2
    # 其它一律沙箱内：目标 gh 子命令、git rev-parse/remote/fetch
    sandboxed_calls = [argv for argv, sandboxed in seen if sandboxed]
    assert any(argv[-2:] == ["pr", "list"] for argv in sandboxed_calls)
    assert any("--show-toplevel" in argv for argv in sandboxed_calls)
    assert any("get-url" in argv for argv in sandboxed_calls)
    assert any("fetch" in argv for argv in sandboxed_calls)
    assert len(sandboxed_calls) >= 4


@requires_bwrap
async def test_gh_command_args_are_not_shell_interpreted(project: Path) -> None:
    pwned = project.parent / "PWNED"
    commands = """
  pr)
    printf 'args=%s\\n' "$*"
    ;;
"""
    fake = make_fake(project, commands=commands)
    payload = f"x; touch {pwned}"
    output = await make_gh_command_tool(make_runner(fake), project).handler(
        {"user": "alice", "args": ["pr", "list", "--search", payload]}
    )
    assert payload in output  # 原样作为单个参数
    assert fake.gh_calls()[-1] == f"pr list --search {payload}"
    assert not pwned.exists()  # 没有经过 shell，也没有越过沙箱


@requires_bwrap
async def test_host_tools_never_use_a_shell(project: Path, monkeypatch) -> None:
    """结构锁定：gh/git 都必须直接创建子进程（create_subprocess_shell 一律不可用）。"""

    async def boom(*args, **kwargs):
        raise AssertionError("宿主工具不得使用 shell")

    monkeypatch.setattr(host_github.asyncio, "create_subprocess_shell", boom)
    fake = make_fake(project, commands="""
  pr)
    printf 'PR LIST OK\\n'
    ;;
""")
    runner = make_runner(fake)
    assert "PR LIST OK" in await runner.run_gh(["pr", "list"], user="alice", cwd=project)
    assert "FETCH" in await runner.git_remote(
        action="fetch", remote="origin", user="alice", cwd=project
    )


async def test_gh_command_passes_explicit_hostname_to_config_lookup(project: Path) -> None:
    fake = make_fake(project)
    await make_runner(fake).run_gh(
        ["pr", "list"], user="alice", cwd=project, hostname="ghe.example"
    )
    assert fake.gh_calls()[0] == "auth token --hostname ghe.example --user alice"


@requires_bwrap
async def test_gh_command_timeout_kills_process(project: Path) -> None:
    commands = """
  pr)
    sleep 30
    ;;
"""
    fake = make_fake(project, commands=commands)
    start = time.monotonic()
    output = await make_gh_command_tool(make_runner(fake), project).handler(
        {"user": "alice", "args": ["pr", "list"], "timeout": 0.3}
    )
    assert "超时" in output
    assert time.monotonic() - start < 3


async def test_gh_command_timeout_argument_is_validated(project: Path) -> None:
    runner = make_runner(make_fake(project))
    for timeout in ("soon", -1, 10_000):
        with pytest.raises(ToolError, match="timeout"):
            await runner.run_gh(["pr", "list"], user="alice", cwd=project, timeout=timeout)


@requires_bwrap
@pytest.mark.parametrize(
    "emit,code",
    [
        ('printf \'%s\\n\' "$GH_TOKEN"', 0),        # 成功路径的 stdout 回显
        ('printf \'%s\\n\' "$GH_TOKEN" >&2', 5),    # 失败路径的 stderr 回显
    ],
)
async def test_gh_command_scrubs_token_in_success_and_failure(
    project: Path, emit: str, code: int
) -> None:
    commands = f"""
  pr)
    {emit}
    exit {code}
    ;;
"""
    fake = make_fake(project, commands=commands)
    output = await make_gh_command_tool(make_runner(fake), project).handler(
        {"user": "alice", "args": ["pr", "list"]}
    )
    assert f"exit code: {code}" in output
    assert TOKEN_ALICE not in output
    assert "***" in output


async def test_token_lookup_failure_scrubs_token_shapes(project: Path) -> None:
    tokens = """
        case "$*" in
          *) printf 'bad credential gho_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\\n' >&2; exit 1 ;;
        esac
"""
    fake = make_fake(project, tokens=tokens)
    with pytest.raises(ToolError) as err:
        await make_runner(fake).run_gh(["pr", "list"], user="alice", cwd=project)
    message = str(err.value)
    assert "无法取得账号 alice" in message
    assert "gho_" not in message and "***" in message


# ---------- 文件系统隔离（核心缺口） ----------

@requires_bwrap
async def test_gh_command_cannot_read_home_secrets(project: Path, _fake_home) -> None:
    """--body-file 之类的绝对路径指向 HOME 敏感文件：必须被 OS 沙箱拒绝。"""
    (_fake_home / ".ssh").mkdir()
    secret = _fake_home / ".ssh" / "id_rsa"
    secret.write_text(HOME_SECRET, encoding="utf-8")
    (_fake_home / ".config" / "gh").mkdir(parents=True)
    (_fake_home / ".config" / "gh" / "hosts.yml").write_text(
        "oauth_token: HOSTS_YML_SENTINEL\n", encoding="utf-8"
    )
    commands = """
  pr)
    printf 'home_listing=[%s]\\n' "$(ls -A "$HOME" | tr '\\n' ',')"
    printf 'secret_exists=%s\\n' "$([ -e "$HOME/.ssh/id_rsa" ] && echo yes || echo no)"
    printf 'hosts_exists=%s\\n' "$([ -e "$HOME/.config/gh/hosts.yml" ] && echo yes || echo no)"
    for a in "$@"; do
      [ -f "$a" ] && printf 'READ(%s)=%s\\n' "$a" "$(cat "$a")"
    done
    ;;
"""
    fake = make_fake(project, commands=commands)
    output = await make_gh_command_tool(make_runner(fake), project).handler(
        {"user": "alice", "args": ["pr", "list", "--body-file", str(secret)]}
    )
    assert HOME_SECRET not in output                      # 密钥内容不可见
    assert "HOSTS_YML_SENTINEL" not in output             # gh 宿主配置不可见
    assert f"READ({secret})" not in output                # 参数引用的敏感文件读不到
    assert "home_listing=[]" in output                    # HOME 整体为空
    assert "secret_exists=no" in output and "hosts_exists=no" in output


@requires_bwrap
async def test_gh_command_reads_project_files(project: Path, tmp_path: Path) -> None:
    """项目根内由参数引用的文件必须可读；项目外（含 /tmp 兄弟目录）不可读。"""
    body = project / "body.md"
    body.write_text("PROJECT-BODY", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SENTINEL", encoding="utf-8")
    commands = """
  issue)
    for a in "$@"; do
      [ -f "$a" ] && printf 'READ(%s)=%s\\n' "$a" "$(cat "$a")"
    done
    printf 'outside_exists=%s\\n' "$([ -e __OUTSIDE__ ] && echo yes || echo no)"
    ;;
"""
    fake = make_fake(project, commands=commands.replace("__OUTSIDE__", str(outside)))
    tool = make_gh_command_tool(make_runner(fake), project)
    output = await tool.handler(
        {"user": "alice", "args": ["issue", "create", "--body-file", str(body)]}
    )
    assert f"READ({body})=PROJECT-BODY" in output
    assert "OUTSIDE-SENTINEL" not in output
    assert "outside_exists=no" in output


@requires_bwrap
async def test_gh_command_writes_only_inside_project(project: Path, _fake_home) -> None:
    """写项目外路径必须被 OS 拒绝；项目内可写；系统路径与宿主 HOME 不被写脏。

    注意 /tmp 在沙箱内是临时 tmpfs：写它不会报错，但绝不在宿主上留下痕迹。
    """
    commands = """
  repo)
    touch __INSIDE__ && printf 'inside_write=OK\\n' || printf 'inside_write=FAIL\\n'
    mkdir -p __EPHEMERAL__ 2>&1 && printf 'ephemeral=OK\\n' || printf 'ephemeral=FAIL\\n'
    mkdir -p __SYSTEM__ 2>&1 && printf 'system_write=OK\\n' || printf 'system_write=FAIL\\n'
    touch /etc/rabbit_probe 2>&1 && printf 'etc_write=OK\\n' || printf 'etc_write=FAIL\\n'
    touch /usr/rabbit_probe 2>&1 && printf 'usr_write=OK\\n' || printf 'usr_write=FAIL\\n'
    printf 'home_exists=%s\\n' "$([ -e __HOME_PROBE__ ] && echo yes || echo no)"
    ;;
"""
    inside = project / "cloned"
    ephemeral = project.parent / "outside_repo"
    system = Path("/var/tmp/rabbit_outside_probe")
    home_probe = _fake_home / "home_probe"
    fake = make_fake(
        project,
        commands=commands.replace("__INSIDE__", str(inside))
        .replace("__EPHEMERAL__", str(ephemeral))
        .replace("__SYSTEM__", str(system))
        .replace("__HOME_PROBE__", str(home_probe)),
    )
    output = await make_gh_command_tool(make_runner(fake), project).handler(
        {"user": "alice", "args": ["repo", "clone", "owner/repo", str(ephemeral)]}
    )
    assert "inside_write=OK" in output
    assert "system_write=FAIL" in output                 # /var/tmp 只读
    assert "etc_write=FAIL" in output and "usr_write=FAIL" in output
    assert inside.exists()                               # 项目内落盘（宿主可见）
    assert not ephemeral.exists()                        # 项目外写不落宿主
    assert not system.exists()
    assert not home_probe.exists() and not Path("/etc/rabbit_probe").exists()


@requires_bwrap
async def test_project_inside_home_is_supported(project: Path, _fake_home) -> None:
    """项目位于 HOME 子目录：遮蔽 HOME 后项目根仍可读写，其余 HOME 内容不可见。"""
    inner = _fake_home / "work" / "proj"
    inner.mkdir(parents=True)
    (_fake_home / "secret.txt").write_text("HOME-SECRET-SENTINEL", encoding="utf-8")
    (inner / "body.txt").write_text("INNER-BODY", encoding="utf-8")
    commands = """
  issue)
    printf 'body=[%s]\\n' "$(cat body.txt)"
    printf 'write=%s\\n' "$(touch written.txt && echo OK || echo FAIL)"
    printf 'home_sibling=[%s]\\n' "$(cat "$HOME/secret.txt" 2>&1 | head -c 60)"
    ;;
"""
    fake = make_fake(inner, commands=commands)
    output = await make_gh_command_tool(make_runner(fake), inner).handler(
        {"user": "alice", "args": ["issue", "create", "--body-file", str(inner / "body.txt")]}
    )
    assert "body=[INNER-BODY]" in output
    assert "write=OK" in output
    assert "HOME-SECRET-SENTINEL" not in output
    assert (inner / "written.txt").exists()      # 项目内写落在宿主项目根
    assert fake.log.exists()


@requires_bwrap
async def test_fake_executables_run_from_project_home_and_tmp(
    project: Path, tmp_path: Path, _fake_home
) -> None:
    """fake gh/git 位于项目、HOME、/tmp 时都必须按设计运行（不被父挂载顺序意外遮蔽）。

    项目在 HOME 之外时，HOME 与 /tmp 都会被 tmpfs 遮蔽：程序本身必须被只读挂回，
    否则 gh/git 会 "not found"，这是挂载顺序最容易被写错的一处。
    """
    log = project / ".fake.log"
    commands = """
  pr)
    printf 'IDENTITY=%s\\n' "$([ "$GH_TOKEN" = '__TOKEN__' ] && echo alice || echo other)"
    ;;
""".replace("__TOKEN__", TOKEN_ALICE)
    home_bindir = Path(os.environ["HOME"]).resolve() / "bindir"
    home_bindir.mkdir(parents=True)
    tmp_tools = tmp_path / "tools"
    tmp_tools.mkdir()
    for label, ghdir in (
        ("项目内", project),
        ("HOME 内", home_bindir),
        ("/tmp 内", tmp_tools),
    ):
        gh = make_fake_gh(ghdir, log=log, commands=commands)
        git = make_fake_git(ghdir, log=log, toplevel=project)
        runner = HostCredentialRunner(gh_path=gh, git_path=git)
        gh_out = await runner.run_gh(["pr", "list"], user="alice", cwd=project)
        assert "IDENTITY=alice" in gh_out, f"{label} fake gh 未按设计运行: {gh_out}"
        git_out = await runner.git_remote(
            action="fetch", remote="origin", user="alice", cwd=project
        )
        assert "exit code: 0" in git_out, f"{label} fake git 未按设计运行: {git_out}"


@requires_bwrap
async def test_fake_executables_run_when_project_inside_home(_fake_home) -> None:
    """项目与 fake 都位于 HOME 内时同样可运行（遮蔽 HOME 后按顺序挂回程序与项目根）。"""
    inner = _fake_home / "work" / "proj"
    inner.mkdir(parents=True)
    bindir = _fake_home / "bindir"
    bindir.mkdir()
    log = inner / ".fake.log"
    gh = make_fake_gh(bindir, log=log, commands='  pr)\n    printf "GH-RAN\\n"\n    ;;\n')
    git = make_fake_git(bindir, log=log, toplevel=inner)
    runner = HostCredentialRunner(gh_path=gh, git_path=git)
    assert "GH-RAN" in await runner.run_gh(["pr", "list"], user="alice", cwd=inner)
    assert "exit code: 0" in await runner.git_remote(
        action="fetch", remote="origin", user="alice", cwd=inner
    )


@requires_bwrap
async def test_gh_command_home_missing_still_masks(project: Path, monkeypatch) -> None:
    """HOME 目录不存在也要能建边界（tmpfs 遮蔽一个空路径）。"""
    monkeypatch.setenv("HOME", str(project.parent / "no-such-home"))
    fake = make_fake(project, commands='  pr)\n    printf "OK\\n"\n    ;;\n')
    output = await make_gh_command_tool(make_runner(fake), project).handler(
        {"user": "alice", "args": ["pr", "list"]}
    )
    assert "OK" in output


@requires_bwrap
async def test_gh_command_cancel_kills_process_and_cleans_up(project: Path, monkeypatch) -> None:
    commands = """
  pr)
    sleep 30
    ;;
"""
    fake = make_fake(project, commands=commands)
    kills: list[tuple] = []
    original = host_github.os.killpg

    def spy(pid, sig):
        kills.append((pid, sig))
        return original(pid, sig)

    monkeypatch.setattr(host_github.os, "killpg", spy)
    before = _sandbox_dirs()
    start = time.monotonic()
    task = asyncio.create_task(
        make_runner(fake).run_gh(["pr", "list"], user="alice", cwd=project)
    )
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - start < 3
    assert kills, "取消路径必须杀进程组"
    assert _sandbox_dirs() == before  # 取消路径也要销毁沙箱临时空间


# ---------- 受限 Git 远程（全部在沙箱内） ----------

@requires_bwrap
async def test_git_fetch_uses_credentials_and_disabled_hooks(project: Path) -> None:
    fake = make_fake(project)
    before = _sandbox_dirs()
    output = await make_git_remote_tool(make_runner(fake), project).handler(
        {"action": "fetch", "remote": "origin", "user": "alice", "branch": "main"}
    )
    assert "exit code: 0" in output and "FETCH" in output
    log = fake.text()
    assert fake.gh_calls()[0] == "auth token --hostname github.com --user alice"
    assert TOKEN_ALICE not in log
    calls = parse_git_calls(log)
    assert [call["sub"] for call in calls] == ["rev-parse", "remote", "fetch"]
    fetch = calls[-1]
    assert fetch["creds"] == "yes" and fetch["token"] == "yes"
    assert fetch["wd"] == "/tmp/.rabbit-agent"           # askpass 在沙箱临时空间
    assert fetch["hk"] == "/tmp/.rabbit-agent/hooks"     # hooks 指向沙箱内空目录
    assert fetch["hok"] == "yes" and fetch["hen"] == ""
    assert fetch["cwd"] == str(project)
    assert "credential.helper=" in fetch["argv"]
    assert "fetch --no-recurse-submodules origin main" in fetch["argv"]
    assert _sandbox_dirs() == before


@requires_bwrap
async def test_git_push_uses_credentials_without_force(project: Path) -> None:
    fake = make_fake(project)
    output = await make_git_remote_tool(make_runner(fake), project).handler(
        {"action": "push", "remote": "origin", "user": "bob", "branch": "main"}
    )
    assert "PUSH" in output
    push = parse_git_calls(fake.text())[-1]
    assert push["sub"] == "push" and push["creds"] == "yes"
    assert "push --recurse-submodules=no origin main" in push["argv"]
    assert "--force" not in push["argv"]


@requires_bwrap
async def test_git_pull_splits_credentialed_fetch_and_plain_merge(project: Path) -> None:
    fake = make_fake(project)
    output = await make_git_remote_tool(make_runner(fake), project).handler(
        {"action": "pull", "remote": "origin", "user": "alice"}
    )
    assert "整合阶段" in output and "MERGE" in output
    calls = parse_git_calls(fake.text())
    assert [call["sub"] for call in calls] == ["rev-parse", "remote", "fetch", "merge"]
    fetch, merge = calls[-2], calls[-1]
    assert fetch["creds"] == "yes" and fetch["token"] == "yes"
    assert merge["creds"] == "no" and merge["token"] == "no"  # 整合阶段不带凭据
    assert "credential.helper=" in merge["argv"]              # hooks 仍禁用
    assert "merge --ff-only FETCH_HEAD" in merge["argv"]
    assert "auth switch" not in fake.text()


@requires_bwrap
async def test_git_pull_stops_when_fetch_fails(project: Path) -> None:
    fake = make_fake(
        project,
        git={"fetch": 'printf \'fatal: could not read from remote\\n\' >&2; exit 128'},
    )
    output = await make_git_remote_tool(make_runner(fake), project).handler(
        {"action": "pull", "remote": "origin", "user": "alice"}
    )
    assert "exit code: 128" in output
    assert [call["sub"] for call in parse_git_calls(fake.text())][-1] == "fetch"


@requires_bwrap
async def test_git_remote_cannot_touch_home_or_outside(project: Path, _fake_home) -> None:
    """git 同样受文件系统沙箱约束：HOME 敏感文件与项目外路径不可读，项目外不可写。"""
    (_fake_home / ".git-credentials").write_text("HOME-GIT-CRED-SENTINEL", encoding="utf-8")
    outside = project.parent / "outside_git.txt"
    outside.write_text("OUTSIDE-GIT-SENTINEL", encoding="utf-8")
    probe = Path("/var/tmp/rabbit_git_probe")
    fake = make_fake(
        project,
        git={
            "fetch": (
                "printf 'home_exists=%s\\n' "
                "\"$([ -e \"$HOME/.git-credentials\" ] && echo yes || echo no)\"; "
                f"printf 'outside_exists=%s\\n' \"$([ -e {outside} ] && echo yes || echo no)\"; "
                f"touch {probe} 2>&1 || printf 'outside_write=FAIL\\n'; "
                "printf 'inside=%s\\n' \"$(touch inside_git.txt && echo OK)\""
            )
        },
    )
    output = await make_git_remote_tool(make_runner(fake), project).handler(
        {"action": "fetch", "remote": "origin", "user": "alice"}
    )
    assert "HOME-GIT-CRED-SENTINEL" not in output
    assert "OUTSIDE-GIT-SENTINEL" not in output
    assert "home_exists=no" in output and "outside_exists=no" in output
    assert "outside_write=FAIL" in output
    assert "inside=OK" in output
    assert not probe.exists()
    assert (project / "inside_git.txt").exists()


async def test_git_unknown_remote_lists_configured_remotes(project: Path) -> None:
    fake = make_fake(project)
    with pytest.raises(ToolError, match="已配置 remote：origin"):
        await make_runner(fake).git_remote(
            action="fetch", remote="upstream", user="alice", cwd=project
        )
    subs = [call["sub"] for call in parse_git_calls(fake.text())]
    assert "fetch" not in subs and "push" not in subs


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:owner/repo.git",
        "ssh://git@github.com/owner/repo.git",
        "http://github.com/owner/repo.git",
        "file:///srv/repo.git",
        "/srv/repo.git",
    ],
)
async def test_git_rejects_non_https_remote(project: Path, url: str) -> None:
    fake = make_fake(project, git={"urls": {"origin": url}})
    with pytest.raises(ToolError, match="不是 HTTPS"):
        await make_runner(fake).git_remote(
            action="fetch", remote="origin", user="alice", cwd=project
        )
    assert "fetch" not in [call["sub"] for call in parse_git_calls(fake.text())]


async def test_git_rejects_remote_on_other_host(project: Path) -> None:
    fake = make_fake(
        project, git={"urls": {"origin": "https://gitlab.example/owner/repo.git"}}
    )
    with pytest.raises(ToolError, match="不一致"):
        await make_runner(fake).git_remote(
            action="push", remote="origin", user="alice", cwd=project
        )
    assert "push" not in [call["sub"] for call in parse_git_calls(fake.text())]


async def test_git_rejects_remote_with_embedded_credentials(project: Path) -> None:
    """URL 内嵌口令会绕过本次指定账号：必须拒绝，且错误信息不回显口令。"""
    fake = make_fake(
        project,
        git={"urls": {"origin": "https://someone:sup3rsecret@github.com/owner/repo.git"}},
    )
    with pytest.raises(ToolError, match="内嵌了口令") as err:
        await make_runner(fake).git_remote(
            action="fetch", remote="origin", user="alice", cwd=project
        )
    assert "sup3rsecret" not in str(err.value)
    assert "fetch" not in [call["sub"] for call in parse_git_calls(fake.text())]


@requires_bwrap
async def test_git_trusts_explicit_hostname(project: Path) -> None:
    fake = make_fake(project, git={"urls": {"origin": "https://ghe.example/owner/repo.git"}})
    output = await make_runner(fake).git_remote(
        action="fetch", remote="origin", user="alice", cwd=project, hostname="ghe.example"
    )
    assert "FETCH" in output


@pytest.mark.parametrize(
    "branch",
    [
        "--upload-pack=touch /tmp/pwned",
        "-D",
        "main..evil",
        "main/",
        "main.lock",
        "@{0}",
        "main:evil",
    ],
)
async def test_git_rejects_branch_injection(project: Path, branch: str) -> None:
    fake = make_fake(project)
    with pytest.raises(ToolError, match="分支名不合法"):
        await make_runner(fake).git_remote(
            action="push", remote="origin", user="alice", cwd=project, branch=branch
        )
    assert fake.gh_calls() == []  # 参数校验在创建子进程之前


@pytest.mark.parametrize("remote", ["--all", "-x", "origin; rm -rf /", "../origin", "a b"])
async def test_git_rejects_remote_injection(project: Path, remote: str) -> None:
    fake = make_fake(project)
    with pytest.raises(ToolError, match="remote 名不合法"):
        await make_runner(fake).git_remote(
            action="fetch", remote=remote, user="alice", cwd=project
        )
    assert fake.gh_calls() == []


@pytest.mark.parametrize("action", ["", "clone", "pull --rebase", "config"])
async def test_git_rejects_unknown_action(project: Path, action: str) -> None:
    fake = make_fake(project)
    with pytest.raises(ToolError, match="action 只支持"):
        await make_runner(fake).git_remote(
            action=action, remote="origin", user="alice", cwd=project
        )
    assert fake.gh_calls() == []


async def test_git_rejects_foreign_repo_root(project: Path) -> None:
    fake = make_fake(project, git={"toplevel": project.parent})
    with pytest.raises(ToolError, match="项目根不是仓库根"):
        await make_runner(fake).git_remote(
            action="fetch", remote="origin", user="alice", cwd=project
        )


async def test_git_reports_non_repository(project: Path) -> None:
    fake = make_fake(
        project,
        git={"revparse": "printf 'fatal: not a git repository\\n' >&2; exit 128"},
    )
    with pytest.raises(ToolError, match="不是 Git 仓库"):
        await make_runner(fake).git_remote(
            action="fetch", remote="origin", user="alice", cwd=project
        )


@requires_bwrap
async def test_git_scrubs_token_from_output(project: Path) -> None:
    fake = make_fake(
        project,
        git={"fetch": 'printf \'leak %s\\n\' "$RABBIT_AGENT_GIT_TOKEN"; exit 6'},
    )
    output = await make_git_remote_tool(make_runner(fake), project).handler(
        {"action": "fetch", "remote": "origin", "user": "alice"}
    )
    assert "exit code: 6" in output
    assert TOKEN_ALICE not in output and "***" in output


async def test_git_never_writes_credential_config(project: Path) -> None:
    fake = make_fake(project)
    runner = make_runner(fake)
    await runner.git_remote(action="fetch", remote="origin", user="alice", cwd=project)
    log = fake.text()
    assert "config" not in [call["sub"] for call in parse_git_calls(log)]
    assert "credential.helper=store" not in log and "--global" not in log
    assert "remote set-url" not in log and "auth switch" not in log


# ---------- 真实 git 的纯本地验证（零网络，沙箱内） ----------

async def _run(argv: list[str], cwd: Path, env: dict, stdin: bytes = b"") -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd), env=env, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate(stdin)
    return proc.returncode, out.decode("utf-8", errors="replace")


@requires_bwrap
@pytest.mark.skipif(REAL_GIT is None, reason="需要本机 git")
async def test_real_git_in_sandbox_askpass_helpers_and_hooks(project: Path, _fake_home) -> None:
    """真实 git 在沙箱内的行为锁定：askpass 供凭据、helper 清空、hooks 禁用、HOME 遮蔽。"""
    home_config = _fake_home / ".gitconfig"
    home_config.write_text("[user]\n\tname = HostUser\n", encoding="utf-8")
    repo_config_probe = project / ".git"
    sandbox = _Sandbox(project, [REAL_GIT])
    try:
        identity = ["-c", "user.email=t@example", "-c", "user.name=t"]
        code, out = await _run(
            sandbox.argv_for([REAL_GIT, *sandbox.git_argv_prefix(), "init", "-q", "."]),
            project, sandbox.git_env(),
        )
        assert code == 0, out

        # 1) 宿主全局 gitconfig 不可见（HOME 被遮蔽）
        code, out = await _run(
            sandbox.argv_for([REAL_GIT, "config", "--global", "user.name"]),
            project, sandbox.git_env(),
        )
        assert "HostUser" not in out

        # 2) 仓库级 helper 被清空：凭据只来自临时 askpass
        code, _ = await _run(
            [REAL_GIT, "config", "credential.helper", "store"], project, sandbox.git_env()
        )
        assert code == 0
        (project / ".git-credentials").write_text(
            "https://stored-user:stored-pass@github.com\n", encoding="utf-8"
        )
        code, out = await _run(
            sandbox.argv_for([REAL_GIT, "credential", "fill"]), project, sandbox.git_env(),
            stdin=b"protocol=https\nhost=github.com\n\n",
        )
        assert "stored-user" not in out or code != 0  # 仓库 helper 不被采用
        code, out = await _run(
            sandbox.argv_for([REAL_GIT, *sandbox.git_argv_prefix(), "credential", "fill"]),
            project, sandbox.git_env("tok-sandbox-secret"),
            stdin=b"protocol=https\nhost=github.com\n\n",
        )
        assert code == 0
        assert "username=x-access-token" in out and "password=tok-sandbox-secret" in out

        # 3) hooks：仓库自带 pre-commit 不执行（对照组证明它本来会执行）
        hook = repo_config_probe / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\ntouch HOOK_RAN\n", encoding="utf-8")
        hook.chmod(0o700)
        code, out = await _run(
            sandbox.argv_for([REAL_GIT, *identity, "commit", "-q", "--allow-empty", "-m", "x"]),
            project, sandbox.git_env(),
        )
        assert code == 0, out
        assert (project / "HOOK_RAN").exists()
        (project / "HOOK_RAN").unlink()
        code, out = await _run(
            sandbox.argv_for(
                [REAL_GIT, *sandbox.git_argv_prefix(), *identity,
                 "commit", "-q", "--allow-empty", "-m", "y"]
            ),
            project, sandbox.git_env(),
        )
        assert code == 0, out
        assert not (project / "HOOK_RAN").exists()
    finally:
        sandbox.close()


@requires_bwrap
@pytest.mark.skipif(REAL_GIT is None, reason="需要本机 git")
async def test_real_git_cannot_write_outside_project(project: Path) -> None:
    """真实 git 的写操作同样越不出项目根（系统路径只读）。"""
    outside = Path("/var/tmp/rabbit_git_outside_probe")
    sandbox = _Sandbox(project, [REAL_GIT])
    try:
        code, out = await _run(
            sandbox.argv_for([REAL_GIT, "init", "-q", str(outside)]),
            project, sandbox.git_env(),
        )
        assert code != 0, out
        assert not outside.exists()
        code, out = await _run(
            sandbox.argv_for([REAL_GIT, "init", "-q", "inner_repo"]),
            project, sandbox.git_env(),
        )
        assert code == 0, out
        assert (project / "inner_repo").exists()
    finally:
        sandbox.close()


# ---------- 角色权限 ----------

def _fake_runner(project: Path) -> HostCredentialRunner:
    return make_runner(make_fake(project))


def test_build_registry_host_tools_are_opt_in(project: Path) -> None:
    assert build_registry(project, write=True, shell=True).names() == [
        "edit_file", "glob", "grep", "ls", "read_file", "run_shell", "write_file",
    ]
    read_only = build_registry(
        project, write=False, shell=False, host_github="read", host_runner=_fake_runner(project)
    )
    assert read_only.names() == ["gh_accounts", "glob", "grep", "ls", "read_file"]
    full = build_registry(
        project, write=True, shell=True, host_github="write", host_git=True,
        host_runner=_fake_runner(project),
    )
    assert set(full.names()) >= {"gh_accounts", "gh_command", "git_remote", "run_shell"}
    without_git = build_registry(
        project, write=True, shell=True, host_github="write",
        host_runner=_fake_runner(project),
    )
    assert "git_remote" not in without_git.names()


def test_build_registry_rejects_unknown_host_github_mode(project: Path) -> None:
    with pytest.raises(ValueError, match="off/read/write"):
        build_registry(
            project, write=False, shell=False, host_github="root",
            host_runner=_fake_runner(project),
        )


async def test_main_agent_gets_only_readonly_host_tools(tmp_path: Path) -> None:
    orch = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=FakeProvider([ChatResult(text="y")]),
        root=tmp_path,
    )
    conv = next(iter(orch.conversations.values()))
    names = conv._main_tools.names()
    assert "gh_accounts" in names
    assert "gh_command" not in names and "git_remote" not in names


@pytest.mark.parametrize(
    "plan_mode,github,git_tools",
    [(False, "write", True), (True, "off", False)],
)
async def test_executor_registry_follows_plan_mode(
    tmp_path: Path, monkeypatch, plan_mode: bool, github: str, git_tools: bool
) -> None:
    captured: list[tuple[dict, list[str]]] = []
    original = executor_session.build_registry

    def spy(root, **kwargs):
        registry = original(root, **kwargs)
        captured.append((kwargs, registry.names()))
        return registry

    monkeypatch.setattr(executor_session, "build_registry", spy)
    orch = Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=FakeProvider([ChatResult(text="完成")]),
        root=tmp_path,
        plan_mode=plan_mode,
    )
    conv = next(iter(orch.conversations.values()))
    await orch.start()
    try:
        conv._dispatcher.dispatch("任务")
        for _ in range(500):
            if conv._dispatcher.tasks.get(1) == "done":
                break
            await asyncio.sleep(0.01)
        assert conv._dispatcher.tasks.get(1) == "done"
    finally:
        await orch.stop()

    kwargs, names = captured[-1]
    assert kwargs["host_github"] == github
    assert kwargs["host_git"] is git_tools
    assert kwargs["write"] is (not plan_mode) and kwargs["shell"] is (not plan_mode)
    if plan_mode:
        # plan 模式工具集保持原样：纯本地只读，不挂任何 GitHub/Git 能力
        assert names == ["glob", "grep", "ls", "read_file"]
    else:
        assert "gh_command" in names and "git_remote" in names
        assert "gh_accounts" in names


# ---------- 审计不泄密 ----------

@requires_bwrap
async def test_audit_logs_host_write_tools_without_token(project: Path) -> None:
    log_path = project.parent / "audit.log"
    audit = AuditLogger(log_path)
    commands = """
  pr)
    printf 'out %s\\n' "$GH_TOKEN"
    ;;
"""
    fake = make_fake(project, commands=commands)
    runner = make_runner(fake)

    def on_call(tool: str, args: dict, phase: str, payload: str | None) -> None:
        audit.log(
            session="s1", tool=tool, phase=phase,
            status="started" if phase == "started" else "success",
            args=args, result=payload or "",
        )

    registry = ToolRegistry(
        [make_gh_command_tool(runner, project), make_git_remote_tool(runner, project)],
        on_call=on_call,
    )
    output = await registry.call("gh_command", {"user": "alice", "args": ["pr", "list"]})
    assert TOKEN_ALICE not in output

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [r["tool"] for r in records] == ["gh_command", "gh_command"]
    assert records[0]["phase"] == "started"
    assert "'user': 'alice'" in records[0]["args"] and "args" in records[0]["args"]
    assert records[-1]["result"].startswith("exit code: 0")
    assert TOKEN_ALICE not in log_path.read_text(encoding="utf-8")
    assert "***" in log_path.read_text(encoding="utf-8")


def test_audit_ignores_readonly_gh_accounts(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    AuditLogger(log_path).log(
        session="s1", tool="gh_accounts", phase="finished", status="success",
        args={}, result="github.com / alice（active）",
    )
    assert not log_path.exists()


def test_audit_scrubs_token_shapes(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    AuditLogger(log_path).log(
        session="s1", tool="run_shell", phase="finished", status="success",
        args={"command": "cat .env"}, result=f"TOKEN={FAKE_STATUS_TOKEN}",
    )
    text = log_path.read_text(encoding="utf-8")
    assert FAKE_STATUS_TOKEN not in text
    assert "***" in text


def test_scrub_secrets_unit() -> None:
    assert scrub_secrets("值 TOK-123456 值", "TOK-123456") == "值 *** 值"
    assert "gho_" not in scrub_secrets("gho_ABCDEFGHIJKLMNOP1234")
    url = scrub_secrets("https://user:pass@github.com/owner/repo.git")
    assert "user:pass" not in url and "github.com" in url


def test_run_shell_side_is_not_relaxed(tmp_path: Path, monkeypatch, project: Path) -> None:
    """边界：宿主能力不是靠放开 run_shell 实现的——shell 侧仍遮蔽 gh 配置、仍不继承凭据变量。"""
    from agent.tools import shell_tool

    fake_home = tmp_path / "home"
    (fake_home / ".config" / "gh").mkdir(parents=True)
    (fake_home / ".config" / "gh" / "hosts.yml").write_text("github.com:\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    mask_args = shell_tool._mask_args()
    assert str(fake_home / ".config") in mask_args  # .config（含 gh 配置）仍被遮蔽

    monkeypatch.setenv("GH_TOKEN", "gho_" + "x" * 36)
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "host-gh"))
    assert "GH_TOKEN" not in shell_tool._sandbox_env()
    assert "HOME" in shell_tool._sandbox_env()  # 其余行为不变

    sandbox = _Sandbox(project, [])
    try:
        env = sandbox.gh_env(host="github.com", token="tok")
        assert env["GH_TOKEN"] == "tok"        # 目标进程环境只在本沙箱内构造
        assert "GH_CONFIG_DIR" in env
        assert "GH_TOKEN" not in sandbox.git_env()          # git 阶段不带 gh token
        assert "RABBIT_AGENT_GIT_TOKEN" not in sandbox.git_env()
    finally:
        sandbox.close()
