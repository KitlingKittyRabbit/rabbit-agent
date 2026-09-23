"""宿主侧 GitHub/Git 能力：按调用方显式指定的账号，在 server 进程内执行宿主 gh/git。

与 run_shell 的边界（本模块不放松任何一条）：
- run_shell 仍在 bwrap 沙箱内运行、仍遮蔽 ~/.config/gh 等路径；本模块不经过 run_shell，
  也不改它的 HOME/配置目录/环境变量——普通 shell 始终拿不到宿主凭据。
- 本模块只执行白名单可执行文件（宿主 gh / 宿主 git），参数结构化并直接创建子进程
  （绝不 sh -c），cwd 固定为项目根。

文件系统隔离（bwrap，fail closed）：
- 固定凭据查询（gh auth status / gh auth token --user）命令固定、输出受控，在宿主直接运行；
- 调用方给出的目标 gh 子命令与全部目标 git 命令一律进入 bwrap 沙箱：系统只读 + 临时 /tmp +
  整个宿主 HOME 遮蔽 + 项目根唯一可写（项目在 HOME 内时先遮蔽 HOME 再把项目根 bind 回来）+
  独立 PID 命名空间（宿主进程列表与环境不可见）+ 网络保持可用；
- HOME 内/ /tmp 内的 gh/git 程序本身（及 HOME 内 git 的运行文件目录）只读挂回，
  其它用户文件仍不可见；
- bwrap 不可用、项目根与 HOME 无法安全划界（相等/包含/HOME 为根）时直接拒绝执行，
  绝不回退到未沙箱的宿主执行；只读 gh_accounts 不依赖 bwrap。

凭据隔离：
- 取凭据只用 `gh auth token --hostname H --user U`（读宿主已登录配置）；绝不调用
  `gh auth switch`，宿主 active 账号始终不变。
- token 只注入单次目标进程：gh 用 GH_TOKEN + 沙箱内临时 GH_CONFIG_DIR；git 用沙箱内
  临时 GIT_ASKPASS 助手（token 走环境变量，不进 argv、不写入脚本），并禁用 hooks、
  清空 credential helper、不读写宿主/仓库的凭据配置。
- token 绝不进入工具结果、异常、日志：对外文本统一经 scrub_secrets 出口清洗。

能力保持通用：本模块不认识"维护者/贡献者"、Issue-first 之类的项目工作流；
账号由调用方每次显式指定，工作流约束写在项目 AGENTS.md 里。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from ..providers import ToolSpec
from .base import Tool, ToolError, cap_output, scrub_secrets
from .shell_tool import base_mount_args, root_bind_args

DEFAULT_HOSTNAME = "github.com"
_DEFAULT_TIMEOUT = 60.0
_GIT_TIMEOUT = 120.0
_MAX_TIMEOUT = 600.0
_MAX_OUTPUT = 8192
_MAX_ACCOUNTS_TEXT = 200_000

# 沙箱内固定挂载点：单次调用的临时工作区（宿主临时目录 bind 到这里）
_SANDBOX_WORK = "/tmp/.rabbit-agent"
_SANDBOX_ASKPASS = f"{_SANDBOX_WORK}/askpass.sh"
_SANDBOX_GH_CONFIG = f"{_SANDBOX_WORK}/gh-config"
_SANDBOX_HOOKS = f"{_SANDBOX_WORK}/hooks"
_SANDBOX_TMP = f"{_SANDBOX_WORK}/tmp"

# 目标进程环境：最小 allowlist（不继承 server 的 GH_TOKEN/GITHUB_TOKEN/GH_CONFIG_DIR 等）
_ENV_ALLOWLIST = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME")

# 禁止的 gh 命令族：可读取/修改认证与配置，或可执行任意宿主代码
_FORBIDDEN_GH_COMMANDS = frozenset({
    "auth",       # 可输出 token、可改 active 账号
    "config",     # 可读写 gh 配置（含认证相关项）
    "alias",      # 可定义 `!shell` 别名 => 任意宿主代码
    "extension",  # 可安装/执行扩展 => 任意宿主代码
    "ext",        # extension 的短写
    "copilot",    # 由扩展提供，可能执行宿主代码
})
# 禁止的参数：改写目标主机（会把指定账号的 token 发给别的主机）/ 打印 token
_FORBIDDEN_GH_FLAGS = ("--hostname", "--show-token")

_GH_COMMAND_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_USER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?$")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
_REMOTE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,200}$")
_GIT_ACTIONS = ("fetch", "pull", "push")

# git 临时凭据助手：git 用 $1 区分用户名/口令提示，token 只从环境变量读取（不进 argv/文件）
_ASKPASS_SCRIPT = """#!/bin/sh
# rabbit-agent 单次调用临时助手，用完即删（不含任何凭据）
case "${1:-}" in
  *sername*) printf '%s\\n' 'x-access-token' ;;
  *) printf '%s\\n' "${RABBIT_AGENT_GIT_TOKEN:-}" ;;
esac
"""


# ---------- 输出清洗 ----------

def scrub(text: str, *secrets: str) -> str:
    """本模块对外文本的唯一清洗入口（实现见 base.scrub_secrets）。"""
    return scrub_secrets(text, *secrets)


# ---------- 文件系统沙箱（目标 gh/git 专用） ----------

def _bwrap_path() -> str | None:
    """每次调用重新解析 bwrap（宿主机可能变化，测试也要能注入）。"""
    return shutil.which("bwrap")


def _host_home() -> Path:
    """宿主 HOME：被遮蔽的那棵树，也是安全划界的参照。"""
    raw = os.environ.get("HOME") or ""
    if not raw:
        raise ToolError("无法确定宿主 HOME，无法建立文件系统沙箱边界，已拒绝执行")
    try:
        return Path(raw).expanduser().resolve()
    except (OSError, RuntimeError):
        raise ToolError(f"宿主 HOME 无法解析（{raw!r}），已拒绝执行") from None


def _under_home(path: Path, home: Path) -> bool:
    return path == home or home in path.parents


def _masked_in_sandbox(path: Path, home: Path) -> bool:
    """该路径在沙箱里会不会被遮蔽：HOME 整棵被 tmpfs 盖住，/tmp 被换成临时 tmpfs。"""
    if _under_home(path, home):
        return True
    tmp = Path("/tmp")
    return path == tmp or tmp in path.parents


def tool_mount_argv(executables: Sequence[str], home: Path) -> list[str]:
    """把被遮蔽的 gh/git 可执行文件（及 git 运行文件目录）只读挂回沙箱。

    HOME 被整体遮蔽后，装在 ~/.local/bin、~/.asdf 之类的 gh/git 会随之不可见；
    /tmp 被换成临时 tmpfs 同理。这里只把"我们要执行的程序本身"挂回来，
    不恢复任何用户配置/数据文件（.ssh、.config/gh、个人文件仍然不可见）。
    """
    parts: list[str] = []
    seen: set[Path] = set()
    for raw in executables:
        if not raw:
            continue
        given = Path(raw)
        try:
            real = given.resolve()
        except (OSError, RuntimeError):
            real = given
        for candidate in (real, given):
            if candidate in seen or not (candidate.is_file() or candidate.is_dir()):
                continue
            if not _masked_in_sandbox(candidate, home):
                continue  # 系统路径本来就只读可见，无需挂载
            seen.add(candidate)
            parts += ["--ro-bind", str(candidate), str(candidate)]
    return parts


def _git_exec_path(git: str, home: Path, root: Path) -> str:
    """固定探测 git 的运行文件目录：在沙箱内运行（argv 固定、无调用方参数、输出不回显）。

    仅当 git 本身落在 HOME 内时才需要（asdf/pyenv 等这类安装的 git-core 也在 HOME 内，
    遮蔽 HOME 后 git 会缺运行文件）。探测本身也是 git 命令，因此同样进 bwrap：
    只挂回 git 二进制、遮蔽 HOME，git-core 不在场时 --exec-path 仍能回答。
    """
    if not _bwrap_path():
        return ""
    env = {"PATH": os.environ.get("PATH") or os.defpath, "HOME": str(home), "TMPDIR": "/tmp"}
    try:
        prefix = sandbox_mount_argv(root, executables=[git])
    except ToolError:
        return ""
    try:
        proc = subprocess.run(
            [*prefix, "--chdir", str(root), git, "--exec-path"],
            capture_output=True, text=True, timeout=10, check=False,
            env=env, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def sandbox_mount_argv(root: Path, *, executables: Sequence[str] = ()) -> list[str]:
    """目标 gh/git 的文件系统沙箱挂载参数（argv 形式，绝不拼 shell 字符串）。

    边界：系统只读、临时 /tmp、整个宿主 HOME 遮蔽、项目根可写、沙箱网络保持可用、
    宿主进程不可见（独立 PID 命名空间）。
    顺序不可调换：遮蔽 HOME 必须早于把项目根 bind 回来，否则项目根会被一并遮蔽。
    fail closed：bwrap 不可用或项目根与 HOME 无法安全划界时直接报错，绝不退回宿主执行。
    """
    root = Path(root).resolve()
    home = _host_home()
    if home == Path(home.anchor):
        raise ToolError(
            f"宿主 HOME 为文件系统根（{home}）：无法在遮蔽 HOME 的同时保留系统可用，"
            "拒绝执行（本工具不会在未沙箱的宿主机上执行目标命令）"
        )
    if home == root:
        raise ToolError(
            "项目根等于宿主 HOME：遮蔽 HOME 与保留项目可写无法同时成立，拒绝执行"
            "（请把项目放在 HOME 之外的独立目录）"
        )
    if root in home.parents:
        raise ToolError(
            f"项目根包含宿主 HOME（{home} 在 {root} 内）：无法只遮蔽 HOME 而保留项目可写，"
            "拒绝执行（请把项目放在 HOME 之外的独立目录）"
        )
    # 项目在 HOME 内是支持的：先遮蔽整个 HOME，再把项目根 bind 回可写（顺序不可颠倒）
    if not root.is_dir():
        raise ToolError(f"项目根不存在或不是目录: {root}")
    work = Path(_SANDBOX_WORK)
    if root == work or work in root.parents or root in work.parents:
        raise ToolError(
            f"项目根与沙箱临时挂载点（{work}）互相包含：临时空间会遮住项目内容，"
            "拒绝执行（请把项目放在其它目录）"
        )
    bwrap = _bwrap_path()
    if not bwrap:
        raise ToolError(
            "宿主文件系统沙箱不可用（未找到 bwrap）：为避免在未沙箱的宿主机上执行可读写宿主"
            "文件系统的 gh/git 命令，已拒绝执行。请安装 bubblewrap（bwrap）后重试；"
            "只读账号查询 gh_accounts 不受影响。"
        )
    return [
        bwrap,
        *base_mount_args(),
        "--unshare-pid",        # 独立 PID 命名空间：宿主进程列表/环境不可见
        "--tmpfs", str(home),   # 整个 HOME 遮蔽（含 .ssh/.config/个人文件）
        *tool_mount_argv(executables, home),  # 只把要执行的 gh/git 程序挂回来
        *root_bind_args(root),  # 再按需把项目根 bind 回可写
    ]


class _Sandbox:
    """单次调用的沙箱运行区：宿主临时目录 + 沙箱内固定挂载点 + 环境构造。

    宿主临时目录只放 askpass 助手与空目录，绝不写入 token；调用结束整体销毁。
    """

    def __init__(self, root: Path, executables: Sequence[str] = ()) -> None:
        self.root = Path(root).resolve()
        home = _host_home()
        mounts = list(executables)
        # git 装在 HOME 内时（asdf/pyenv 等），运行文件目录也在 HOME 内：一并挂回
        git = next((e for e in mounts if Path(e).name.startswith("git")), "")
        if git and _under_home(Path(git).resolve(), home):
            exec_path = _git_exec_path(git, home, self.root)
            if exec_path:
                mounts.append(exec_path)
        self._prefix = sandbox_mount_argv(self.root, executables=mounts)  # fail closed 判定
        self._dir = Path(tempfile.mkdtemp(prefix="rabbit-sbx-"))
        os.chmod(self._dir, 0o700)
        for name in ("gh-config", "hooks", "tmp"):
            (self._dir / name).mkdir(mode=0o700)
        askpass = self._dir / "askpass.sh"
        askpass.write_text(_ASKPASS_SCRIPT, encoding="utf-8")
        os.chmod(askpass, 0o700)
        self._token = ""

    @property
    def host_dir(self) -> Path:
        return self._dir

    def argv_for(self, argv: Sequence[str]) -> list[str]:
        """bwrap + 挂载 + 目标 argv：目标进程参数原样传入，绝不经 shell。"""
        return [
            *self._prefix,
            "--bind", str(self._dir), _SANDBOX_WORK,   # 沙箱内临时空间（含 askpass）
            "--chdir", str(self.root),
            *[str(part) for part in argv],
        ]

    def _base_env(self) -> dict[str, str]:
        """目标进程最小环境：不继承 server 的 GH_TOKEN/GITHUB_TOKEN/GH_CONFIG_DIR 等。"""
        env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
        env["PATH"] = env.get("PATH") or os.defpath
        env.update({
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PAGER": "cat",
            "TMPDIR": _SANDBOX_TMP,   # 宿主 TMPDIR 已被遮蔽，指向沙箱内临时空间
        })
        return env

    def gh_env(self, *, host: str, token: str) -> dict[str, str]:
        return {
            **self._base_env(),
            "GH_TOKEN": token,                 # 只在本次目标进程的环境里
            "GH_HOST": host,
            "GH_CONFIG_DIR": _SANDBOX_GH_CONFIG,  # 沙箱内空目录，读不到宿主配置
            "GH_PROMPT_DISABLED": "1",
            "GH_PAGER": "cat",
            "GH_NO_UPDATE_NOTIFIER": "1",
        }

    def git_env(self, token: str = "") -> dict[str, str]:
        env = {**self._base_env(), "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"}
        if token:
            env["GIT_ASKPASS"] = _SANDBOX_ASKPASS   # 脚本不含凭据，token 只在环境变量里
            env["RABBIT_AGENT_GIT_TOKEN"] = token
        return env

    def git_argv_prefix(self) -> list[str]:
        """写死的行为约束：清空 credential helper + hooks 指向沙箱内空目录。"""
        return ["-c", "credential.helper=", "-c", f"core.hooksPath={_SANDBOX_HOOKS}"]

    def close(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)
        self._token = ""


# ---------- 账号（只读） ----------

@dataclass(frozen=True)
class Account:
    """gh 已登录账号的快照；绝不含 token。"""

    hostname: str
    user: str
    active: bool = False
    scopes: tuple[str, ...] = ()
    ok: bool = True  # False = gh 报告该账号认证检查失败（离线/凭据失效）


_HOST_PART = r"(?P<host>[A-Za-z0-9][A-Za-z0-9.:_-]*)"
_USER_PART = r"(?:account (?P<u1>[^\s]+)|as (?P<u2>[^\s]+))"
_LOGGED_IN_RE = re.compile(r"Logged in to " + _HOST_PART + r" " + _USER_PART)
_FAILED_IN_RE = re.compile(r"Failed to log in to " + _HOST_PART + r" " + _USER_PART)
_ACTIVE_RE = re.compile(r"Active account:\s*(true|false)", re.IGNORECASE)
_SCOPES_RE = re.compile(r"Token scopes:\s*(?P<value>.+)$")


def _split_scopes(value: str) -> tuple[str, ...]:
    parts = [part.strip().strip("'\"").strip() for part in value.split(",")]
    return tuple(part for part in parts if part)


def parse_auth_status(text: str) -> list[Account]:
    """解析 `gh auth status` 文本：只取主机/账号/active/scopes，token 行一律丢弃。

    gh 的新旧两种登录行都兼容；没有显式 active 标记时按 gh 自身约定取该主机首个账号。
    """
    accounts: list[Account] = []
    explicit_active: set[str] = set()
    current_host = ""
    for raw_line in (text or "")[:_MAX_ACCOUNTS_TEXT].splitlines():
        line = raw_line.strip()
        logged_in = _LOGGED_IN_RE.search(line)
        match = logged_in or _FAILED_IN_RE.search(line)
        if match is not None:
            current_host = match.group("host")
            accounts.append(Account(
                hostname=current_host,
                user=match.group("u1") or match.group("u2") or "",
                ok=logged_in is not None,
            ))
            continue
        if not accounts:
            continue
        active = _ACTIVE_RE.search(line)
        if active is not None:
            explicit_active.add(current_host)
            accounts[-1] = replace(accounts[-1], active=active.group(1).lower() == "true")
            continue
        scopes = _SCOPES_RE.search(line)
        if scopes is not None:
            accounts[-1] = replace(accounts[-1], scopes=_split_scopes(scopes.group("value")))
    seen_hosts: set[str] = set()
    for index, account in enumerate(accounts):
        if account.hostname in explicit_active or account.hostname in seen_hosts:
            continue
        seen_hosts.add(account.hostname)
        accounts[index] = replace(account, active=True)
    return [account for account in accounts if account.user]


# ---------- 参数校验 ----------

def _validate_user(user: str) -> str:
    user = str(user or "")
    if not _USER_RE.fullmatch(user):
        raise ToolError(
            "user 必须是登录账号名（用 gh_accounts 查看已登录账号），每次调用都要显式指定"
        )
    return user


def _normalize_hostname(hostname: str) -> str:
    hostname = str(hostname or "").strip() or DEFAULT_HOSTNAME
    if not _HOSTNAME_RE.fullmatch(hostname):
        raise ToolError(f"hostname 不合法: {hostname!r}")
    return hostname.lower()


def _validate_gh_argv(argv: object) -> list[str]:
    """gh 参数必须是字符串数组（直接作为子进程参数，不经 shell）。"""
    if isinstance(argv, str) or not isinstance(argv, list) or not argv:
        raise ToolError(
            'args 必须是 gh 子命令参数的字符串数组，如 ["issue", "list", "--limit", "10"]'
        )
    out: list[str] = []
    for item in argv:
        if not isinstance(item, str):
            raise ToolError("args 中每一项都必须是字符串")
        if "\x00" in item:
            raise ToolError("args 不允许包含 NUL 字符")
        out.append(item)
    command = out[0]
    if not _GH_COMMAND_RE.fullmatch(command):
        raise ToolError(f"gh 子命令必须是命令名（收到 {command!r}）：不接受前导选项")
    if command in _FORBIDDEN_GH_COMMANDS:
        raise ToolError(
            f"禁止通过本工具执行 gh {command}（可读取/修改认证配置或执行扩展）；"
            "账号由本工具每次显式指定，不需要也不允许改认证状态"
        )
    for item in out[1:]:
        if any(item == flag or item.startswith(flag + "=") for flag in _FORBIDDEN_GH_FLAGS):
            raise ToolError(
                f"args 不允许包含 {item!r}：目标主机由 hostname 参数指定（防止把凭据发给别处）"
            )
    return out


def _validate_remote(remote: str) -> str:
    remote = str(remote or "")
    if not _REMOTE_RE.fullmatch(remote):
        raise ToolError(f"remote 名不合法（只接受已配置的 remote 名，如 origin）: {remote!r}")
    return remote


def _validate_branch(branch: str) -> str:
    branch = str(branch or "")
    bad = (
        not _BRANCH_RE.fullmatch(branch)
        or ".." in branch
        or "@{" in branch
        or "//" in branch
        or branch.endswith("/")
        or branch.endswith(".lock")
        or branch.endswith(".")
    )
    if bad:
        raise ToolError(f"分支名不合法（拒绝选项注入/危险引用）: {branch!r}")
    return branch


def _display_url(url: str) -> str:
    """只显示协议+主机+路径，绝不回显 URL 里的内嵌凭据。"""
    parts = urlsplit(url)
    if not parts.scheme:
        return scrub(url)
    return f"{parts.scheme}://{parts.hostname or '?'}{parts.path}"


def _check_remote_url(url: str, hostname: str, remote: str) -> None:
    """remote 必须是 https://<指定 host>：否则凭据可能被发给别处或用别的传输入口。"""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ToolError(
            f"remote {remote} 不是 HTTPS（{_display_url(url)}）：本工具只按指定账号提供临时 "
            "HTTPS 凭据，不接受 SSH/本地路径等其他传输方式"
        )
    if (parts.hostname or "").lower() != hostname.lower():
        raise ToolError(
            f"remote {remote} 指向 {parts.hostname or '未知主机'}，与指定 hostname {hostname} "
            "不一致（拒绝把凭据发给其他主机）"
        )
    if parts.password:
        raise ToolError(
            f"remote {remote} 的 URL 内嵌了口令（{_display_url(url)}）：git 会直接用它而不用本次"
            "指定的账号，请把 remote URL 改成不含口令的 https 地址后重试"
        )


def _kill_tree(proc) -> None:
    """杀整个进程组：单杀子进程会留下占着管道的孤儿。"""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


@dataclass
class _Exec:
    code: int | None  # None = 超时被杀
    out: str
    err: str


class HostCredentialRunner:
    """宿主凭据执行器：唯一接触宿主 gh 凭据的组件，按调用指定账号产出一次性凭据环境。

    工具层只拿到已清洗的文本；token 只在本对象内部流转。
    目标 gh/git 命令一律经 _Sandbox（bwrap）执行；只有固定凭据查询在宿主直接执行。
    """

    def __init__(
        self,
        *,
        gh_path: str | None = None,
        git_path: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_output: int = _MAX_OUTPUT,
    ) -> None:
        # 用绝对路径调用（不受项目内同名文件影响）
        self._gh = gh_path or shutil.which("gh")
        self._git = git_path or shutil.which("git")
        self._timeout = float(timeout)
        self._max_output = int(max_output)

    # ---------- 基础执行 ----------

    async def _exec(
        self,
        argv: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str],
        timeout: float,
        split: bool = False,
        sandbox: _Sandbox | None = None,
    ) -> _Exec:
        """直接创建子进程（绝不 sh -c）；超时/取消杀整个进程组。

        sandbox 非空时以 `bwrap <挂载…> <目标 argv>` 启动：目标命令只看到沙箱文件系统。
        split=True 时 stdout/stderr 分离（仅取 token 用）：
        失败路径只回显 err（stdout 可能是凭据本身，一律丢弃）。
        """
        run_argv = sandbox.argv_for(argv) if sandbox is not None else argv
        try:
            proc = await asyncio.create_subprocess_exec(
                *run_argv,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if split else asyncio.subprocess.STDOUT,
                start_new_session=True,  # 独立进程组，killpg 的前提
            )
        except OSError as e:
            what = "宿主文件系统沙箱（bwrap）" if sandbox is not None else Path(argv[0]).name
            raise ToolError(f"无法启动 {what}（{type(e).__name__}）") from None
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            _kill_tree(proc)
            with contextlib.suppress(Exception):
                await proc.wait()
            return _Exec(None, "", f"超时（{timeout:g} 秒），进程已终止")
        except asyncio.CancelledError:
            _kill_tree(proc)
            with contextlib.suppress(Exception):
                await proc.wait()
            raise
        return _Exec(
            proc.returncode,
            stdout.decode("utf-8", errors="replace"),
            (stderr or b"").decode("utf-8", errors="replace"),
        )

    def _render(self, result: _Exec, *secrets: str, sandboxed: bool = False) -> str:
        """工具可见文本的唯一出口：清洗 + 截断（token 绝不出现）。"""
        if result.code is None:
            return scrub(result.err or "进程已终止", *secrets)
        raw = "\n".join(p for p in (result.out, result.err) if p)
        body = cap_output(scrub(raw.strip(), *secrets), self._max_output)
        if sandboxed and result.code != 0 and "bwrap:" in raw:
            # 沙箱自身失败必须显式说明：绝不静默回退到未沙箱的宿主执行
            body += "\n（宿主文件系统沙箱启动失败；本工具不会回退到未沙箱的宿主执行）"
        return f"exit code: {result.code}" + (f"\n{body}" if body else "")

    @staticmethod
    def _require(path: str | None, name: str) -> str:
        if not path:
            raise ToolError(f"宿主机未找到 {name} 可执行文件（不在 PATH）")
        return path

    @staticmethod
    def _host_env() -> dict[str, str]:
        """宿主 gh 配置查询环境：原样继承——这就是"宿主已登录的 gh CLI"本身。

        只用于固定凭据查询（auth status / auth token）；调用方提供的目标命令一律经沙箱。
        """
        return dict(os.environ)

    def _resolve_timeout(self, timeout: object, *, default: float) -> float:
        try:
            value = float(timeout) if timeout is not None else default  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ToolError(f"timeout 必须是数字（收到 {timeout!r}）") from None
        if not 0 < value <= _MAX_TIMEOUT:
            raise ToolError(f"timeout 必须在 0 到 {_MAX_TIMEOUT:g} 秒之间")
        return value

    # ---------- 只读能力 ----------

    async def accounts(self) -> list[Account]:
        """枚举宿主已登录账号（只读；不用 --show-token，也不回显原始输出）。"""
        gh = self._require(self._gh, "gh")
        result = await self._exec(
            [gh, "auth", "status"], cwd=None, env=self._host_env(),
            timeout=min(self._timeout, 30.0),
        )
        return parse_auth_status(result.out + result.err)

    # ---------- 凭据 ----------

    async def _token_for(self, user: str, hostname: str) -> str:
        """取指定账号 token（内部使用；绝不写入结果/异常/日志）。"""
        gh = self._require(self._gh, "gh")
        result = await self._exec(
            [gh, "auth", "token", "--hostname", hostname, "--user", user],
            cwd=None, env=self._host_env(), timeout=min(self._timeout, 30.0), split=True,
        )
        if result.code != 0:
            # 失败路径只回显 stderr 首行的清洗结果；stdout 一律丢弃（它可能是凭据本身）
            detail = result.err.strip().splitlines()
            hint = scrub(detail[0][:200]) if detail else ""
            raise ToolError(
                f"无法取得账号 {user}（{hostname}）的凭据"
                + (f"：{hint}" if hint else "")
                + "。用 gh_accounts 查看已登录账号，或先完成 gh 登录（本工具不会切换账号）"
            )
        token = result.out.strip()
        if not token or any(ch.isspace() for ch in token):
            raise ToolError(f"账号 {user}（{hostname}）的凭据输出异常，已丢弃")
        return token

    # ---------- gh 子命令 ----------

    async def run_gh(
        self, argv: object, *, user: str, cwd: Path, hostname: str = "", timeout: object = None,
    ) -> str:
        """以指定账号执行 gh 子命令：参数结构化、无 shell、cwd 固定项目根。

        目标进程在 bwrap 沙箱内运行：项目根唯一可写、宿主 HOME 遮蔽、系统只读。
        """
        gh = self._require(self._gh, "gh")
        parsed = _validate_gh_argv(argv)
        account = _validate_user(user)
        host = _normalize_hostname(hostname)
        seconds = self._resolve_timeout(timeout, default=self._timeout)
        sandbox = _Sandbox(Path(cwd), [gh])     # 先判定沙箱可用性（fail closed）
        token = ""
        try:
            token = await self._token_for(account, host)
            env = sandbox.gh_env(host=host, token=token)
            result = await self._exec(
                [gh, *parsed], cwd=Path(cwd), env=env, timeout=seconds, sandbox=sandbox,
            )
        finally:
            sandbox.close()                    # 沙箱临时空间结束即销毁
        return self._render(result, token, sandboxed=True)

    # ---------- 受限 git 远程 ----------

    async def git_remote(
        self,
        *,
        action: str,
        remote: str,
        user: str,
        cwd: Path,
        branch: str = "",
        hostname: str = "",
        timeout: object = None,
    ) -> str:
        """按指定账号对当前项目仓库的已配置 remote 执行 fetch / pull / push。

        所有 git 命令（含仓库探测与 pull 的无凭据整合阶段）都在 bwrap 沙箱内运行。
        """
        git = self._require(self._git, "git")
        action = str(action or "")
        if action not in _GIT_ACTIONS:
            raise ToolError(f"action 只支持 {' / '.join(_GIT_ACTIONS)}")
        remote = _validate_remote(remote)
        branch = _validate_branch(branch) if branch else ""
        account = _validate_user(user)
        host = _normalize_hostname(hostname)
        seconds = self._resolve_timeout(timeout, default=_GIT_TIMEOUT)
        root = Path(cwd)
        sandbox = _Sandbox(root, [git])        # 先判定沙箱可用性（fail closed）
        try:
            await self._require_repo_root(git, root, sandbox)
            url = await self._remote_url(git, root, remote, sandbox)
            _check_remote_url(url, host, remote)
            token = await self._token_for(account, host)
            tail = [remote, *([branch] if branch else [])]
            if action in ("fetch", "push"):
                args = (
                    ["fetch", "--no-recurse-submodules", *tail]
                    if action == "fetch"
                    else ["push", "--recurse-submodules=no", *tail]
                )
                _, text = await self._git_run(
                    git, root, args, sandbox=sandbox, with_credentials=True,
                    timeout=seconds, secret=token,
                )
                return text
            # pull 拆两段：带凭据 fetch → 不带凭据的安全整合（--ff-only，不产生额外提交）
            code, fetched = await self._git_run(
                git, root, ["fetch", "--no-recurse-submodules", *tail],
                sandbox=sandbox, with_credentials=True, timeout=seconds, secret=token,
            )
            if code != 0:
                return fetched
            _, merged = await self._git_run(
                git, root, ["merge", "--ff-only", "FETCH_HEAD"],
                sandbox=sandbox, with_credentials=False, timeout=seconds,
            )
            return f"{fetched}\n[整合阶段：git merge --ff-only FETCH_HEAD（不带凭据）]\n{merged}"
        finally:
            sandbox.close()

    async def _git_run(
        self,
        git: str,
        root: Path,
        args: list[str],
        *,
        sandbox: _Sandbox,
        with_credentials: bool,
        timeout: float,
        secret: str = "",
    ) -> tuple[int | None, str]:
        """跑一条 git 命令：参数结构化；清空 credential helper、hooks 指向沙箱内空目录。"""
        argv = [git, *sandbox.git_argv_prefix(), *args]
        env = sandbox.git_env(secret) if with_credentials else sandbox.git_env()
        result = await self._exec(argv, cwd=root, env=env, timeout=timeout, sandbox=sandbox)
        return result.code, self._render(result, secret, sandboxed=True)

    async def _require_repo_root(self, git: str, root: Path, sandbox: _Sandbox) -> None:
        """限定操作对象：项目根必须就是 Git 仓库根（不碰项目外的仓库）。"""
        result = await self._exec(
            [git, "rev-parse", "--show-toplevel"], cwd=root, env=sandbox.git_env(),
            timeout=min(self._timeout, 30.0), sandbox=sandbox,
        )
        toplevel = result.out.strip()
        if result.code != 0 or not toplevel:
            raise ToolError("当前项目不是 Git 仓库（git rev-parse --show-toplevel 失败）")
        if Path(toplevel).resolve() != root.resolve():
            raise ToolError(
                f"项目根不是仓库根（仓库根：{toplevel}）：git 工具只在项目仓库根上操作"
            )

    async def _remote_url(self, git: str, root: Path, remote: str, sandbox: _Sandbox) -> str:
        result = await self._exec(
            [git, "remote", "get-url", remote], cwd=root, env=sandbox.git_env(),
            timeout=min(self._timeout, 30.0), sandbox=sandbox,
        )
        url = result.out.strip()
        if result.code != 0 or not url:
            listed = await self._exec(
                [git, "remote"], cwd=root, env=sandbox.git_env(),
                timeout=min(self._timeout, 30.0), sandbox=sandbox,
            )
            names = [n for n in listed.out.split() if n]
            hint = (
                f"（已配置 remote：{scrub(', '.join(names))}）" if names
                else "（仓库没有配置 remote）"
            )
            raise ToolError(f"remote {remote!r} 不存在或读取失败{hint}")
        return url


# ---------- 工具定义 ----------

def _account_lines(accounts: list[Account]) -> str:
    lines: list[str] = []
    for account in accounts:
        mark = "（active）" if account.active else ""
        state = "" if account.ok else "（认证检查失败：可能离线或凭据已失效）"
        lines.append(f"- {account.hostname} / {account.user}{mark}{state}")
        if account.scopes:
            lines.append(f"  scopes: {', '.join(account.scopes)}")
    return "\n".join(lines)


def make_gh_accounts_tool(runner: HostCredentialRunner) -> Tool:
    """只读账号查询：看得到账号与 active 标志，看不到 token。"""

    async def gh_accounts(args: dict) -> str:
        accounts = await runner.accounts()
        if not accounts:
            return "没有发现已登录的 GitHub 账号（gh CLI 未登录或不可用）"
        return "宿主机 gh CLI 已登录账号：\n" + _account_lines(accounts)

    return Tool(
        ToolSpec(
            name="gh_accounts",
            description=(
                "列出宿主机已登录的 GitHub 账号、active 标志与 scopes（只读，不返回 token）。"
                "调用 gh_command / git_remote 前先用它确认账号名。"
                "本条为固定命令（gh auth status），不需要文件系统沙箱；"
                "其余两个工具在 bwrap 不可用时会被拒绝。"
            ),
            parameters={"type": "object", "properties": {}},
        ),
        gh_accounts,
    )


def make_gh_command_tool(runner: HostCredentialRunner, root: Path) -> Tool:
    """通用 gh CLI 执行：参数数组、无 shell、按指定账号、临时凭据。"""

    async def gh_command(args: dict) -> str:
        return await runner.run_gh(
            args.get("args"),
            user=str(args.get("user") or ""),
            hostname=str(args.get("hostname") or ""),
            cwd=root,
            timeout=args.get("timeout"),
        )

    return Tool(
        ToolSpec(
            name="gh_command",
            description=(
                "以显式指定的已登录账号执行宿主 gh CLI 子命令（issue/pr/repo/release 等通用能力）。"
                "参数必须是字符串数组，直接作为子进程参数（不经 shell）。"
                "命令在 bwrap 文件系统沙箱内运行：项目根是唯一可写目录，系统只读，宿主 HOME "
                "（含 .ssh/.config/个人文件）不可读，/tmp 为临时目录，网络保持可用；"
                "bwrap 不可用时本工具直接拒绝执行（不会在未沙箱的宿主机上运行）。"
                "禁止 auth/config/alias/extension/copilot 命令族与 --hostname/--show-token；"
                "工作目录固定为项目根。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "user": {"type": "string", "description": "本次调用使用的已登录账号名（必填）"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'gh 子命令参数数组，如 ["issue", "list", "--limit", "10"]',
                    },
                    "hostname": {"type": "string", "description": "GitHub 主机名，默认 github.com"},
                    "timeout": {"type": "number", "description": "超时秒数，默认 60"},
                },
                "required": ["user", "args"],
            },
        ),
        gh_command,
    )


def make_git_remote_tool(runner: HostCredentialRunner, root: Path) -> Tool:
    """受限 git 远程能力：只对项目仓库的已配置 HTTPS remote 操作。"""

    async def git_remote(args: dict) -> str:
        return await runner.git_remote(
            action=str(args.get("action") or ""),
            remote=str(args.get("remote") or ""),
            branch=str(args.get("branch") or ""),
            user=str(args.get("user") or ""),
            hostname=str(args.get("hostname") or ""),
            cwd=root,
            timeout=args.get("timeout"),
        )

    return Tool(
        ToolSpec(
            name="git_remote",
            description=(
                "对当前项目仓库的已配置 remote 执行 fetch / pull / push，按指定账号提供"
                "临时 HTTPS 凭据（不改宿主 gh active，也不写全局/仓库凭据配置）。"
                "remote 必须是 HTTPS 且指向指定 hostname；pull 为「带凭据 fetch + 不带凭据 "
                "ff-only 整合」；hooks 已禁用。所有 git 命令都在 bwrap 文件系统沙箱内运行"
                "（项目根唯一可写、宿主 HOME 不可读、系统只读）；bwrap 不可用时直接拒绝执行。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(_GIT_ACTIONS)},
                    "remote": {"type": "string", "description": "已配置的 remote 名（如 origin）"},
                    "user": {"type": "string", "description": "本次调用使用的已登录账号名（必填）"},
                    "branch": {
                        "type": "string",
                        "description": "可选：分支名（fetch/push 的目标）",
                    },
                    "hostname": {"type": "string", "description": "GitHub 主机名，默认 github.com"},
                    "timeout": {"type": "number", "description": "超时秒数，默认 120"},
                },
                "required": ["action", "remote", "user"],
            },
        ),
        git_remote,
    )


def make_host_tools(
    root: Path,
    *,
    runner: HostCredentialRunner,
    github: str = "off",
    git: bool = False,
) -> list[Tool]:
    """按角色装配宿主 GitHub/Git 工具：off / read（只看账号）/ write（可执行）。"""
    if github not in ("off", "read", "write"):
        raise ValueError(f"github 只支持 off/read/write，收到 {github!r}")
    tools: list[Tool] = []
    if github != "off":
        tools.append(make_gh_accounts_tool(runner))
    if github == "write":
        tools.append(make_gh_command_tool(runner, root))
    if git:
        tools.append(make_git_remote_tool(runner, root))
    return tools
