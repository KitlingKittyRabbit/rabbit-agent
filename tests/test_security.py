"""安全边界测试（Phase F）：WS token/Origin、API token、shell env 隔离、bwrap、symlink。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from agent.core.orchestrator import Orchestrator
from agent.providers import ChatResult, FakeProvider
from agent.server.app import create_app
from agent.tools.shell_tool import make_shell_tool, sandbox_mode


def make_orch(tmp_path: Path) -> Orchestrator:
    return Orchestrator(
        main_provider=FakeProvider([ChatResult(text="x")]),
        executor_provider=FakeProvider([ChatResult(text="y")]),
        root=tmp_path,
    )


# ---------- WebSocket token ----------

def test_ws_rejects_missing_and_wrong_token(tmp_path: Path) -> None:
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws?token=wrong"):
                pass


def test_ws_accepts_correct_token(tmp_path: Path) -> None:
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        with client.websocket_connect("/ws?token=secret123"):
            pass  # 不抛即通过


def test_ws_rejects_foreign_origin(tmp_path: Path) -> None:
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ws?token=secret123", headers={"Origin": "https://evil.example"}
            ):
                pass


def test_ws_accepts_local_origin(tmp_path: Path) -> None:
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        with client.websocket_connect(
            "/ws?token=secret123", headers={"Origin": "http://127.0.0.1:8471"}
        ):
            pass


# ---------- HTTP API token ----------

def test_api_requires_token(tmp_path: Path) -> None:
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        assert client.get("/api/browse").status_code == 401
        assert client.get("/api/browse", params={"token": "wrong"}).status_code == 401
        assert client.get("/api/browse", params={"token": "secret123"}).status_code == 200


def test_timeline_requires_token(tmp_path: Path) -> None:
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        assert client.get("/api/timeline", params={"session": "s1"}).status_code == 401
        assert client.get("/api/task", params={"session": "s1", "task": 1}).status_code == 401


def test_page_and_app_js_embed_token(tmp_path: Path) -> None:
    """审核缺陷回归：app.js 的占位符必须被替换，否则页面拿字面量连 WS 必被拒。"""
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        assert client.get("/").status_code == 200
        script = client.get("/app.js").text
        assert "secret123" in script
        assert "__AGENT_TOKEN__" not in script


def test_ws_rejects_origin_prefix_bypass(tmp_path: Path) -> None:
    """审核缺陷回归：Origin 前缀域（localhost.evil.com）不得被当成本机来源。"""
    with TestClient(create_app(make_orch(tmp_path), token="secret123")) as client:
        for evil in ("http://localhost.evil.com", "http://127.0.0.1.evil.com"):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/ws?token=secret123", headers={"Origin": evil}
                ):
                    pass


def test_write_token_file_permissions(tmp_path: Path, monkeypatch) -> None:
    from agent.server import __main__ as server_main

    target = tmp_path / "agent_token"
    monkeypatch.setattr(server_main, "TOKEN_PATH", str(target))
    token = server_main._write_token()
    assert len(token) == 32
    assert target.read_text(encoding="utf-8") == token
    assert target.stat().st_mode & 0o777 == 0o600


# ---------- shell 环境变量隔离 ----------

async def test_shell_env_hides_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MAIN_API_KEY", "SECRET_VALUE_XYZ")
    output = await make_shell_tool(tmp_path).handler({"command": "env"})
    assert "SECRET_VALUE_XYZ" not in output
    assert "AGENT_MAIN_API_KEY" not in output
    assert "PATH" in output  # allowlist 保留


# ---------- bwrap 沙箱 ----------

@pytest.mark.skipif(sandbox_mode() != "bwrap", reason="bwrap 不可用则仅 env 隔离")
async def test_bwrap_blocks_write_outside_project(tmp_path: Path) -> None:
    output = await make_shell_tool(tmp_path).handler({"command": "echo x > /etc/bwrap_forbidden"})
    assert "exit code: 0" not in output  # 系统区只读，写入必失败


@pytest.mark.skipif(sandbox_mode() != "bwrap", reason="bwrap 不可用则仅 env 隔离")
async def test_bwrap_allows_write_inside_project(tmp_path: Path) -> None:
    output = await make_shell_tool(tmp_path).handler(
        {"command": "echo hi > inside.txt && cat inside.txt"}
    )
    assert "hi" in output
    assert (tmp_path / "inside.txt").exists()


@pytest.mark.skipif(sandbox_mode() != "bwrap", reason="bwrap 不可用则仅 env 隔离")
async def test_bwrap_hides_home_secrets(tmp_path: Path, monkeypatch) -> None:
    # home 与项目 root 必须分离（否则 root bind 会覆盖 mask——这正是本测试最初踩的坑）
    fake_home = tmp_path / "fake_home"
    project = tmp_path / "project"
    (fake_home / ".ssh").mkdir(parents=True)
    (fake_home / ".ssh" / "id_rsa").write_text("私钥内容XYZ", encoding="utf-8")
    (fake_home / ".config").mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    output = await make_shell_tool(project).handler(
        {"command": "cat ~/.ssh/id_rsa 2>&1 || true; ls ~/.config 2>&1 || true; echo done"}
    )
    assert "私钥内容XYZ" not in output  # .ssh 被遮蔽，读不到真实内容
    assert "done" in output


# ---------- symlink 防护 ----------

async def test_read_file_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("机密内容", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    from agent.tools import ToolError, build_registry

    registry = build_registry(tmp_path, write=False, shell=False)
    with pytest.raises(ToolError, match="越界"):
        await registry.call("read_file", {"path": "link.txt"})


async def test_grep_and_glob_skip_symlink_files(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("机密标记XYZ", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    (tmp_path / "normal.txt").write_text("普通内容", encoding="utf-8")
    from agent.tools import build_registry

    registry = build_registry(tmp_path, write=False, shell=False)
    grep_out = await registry.call("grep", {"pattern": "机密标记XYZ"})
    assert grep_out == "(无匹配)"  # symlink 文件被跳过，外部内容不泄露
    glob_out = await registry.call("glob", {"pattern": "*.txt"})
    assert "link.txt" not in glob_out
    assert "normal.txt" in glob_out


def test_bwrap_args_order_keeps_dev_after_root(tmp_path: Path) -> None:
    """回归：--ro-bind / / 必须在 --dev-bind /dev 之前，否则 /dev 被盖住（Python 无法启动）。"""
    from agent.tools.shell_tool import _wrap_command

    cmd = _wrap_command("echo hi", tmp_path)
    if "--dev-bind" not in cmd:
        import pytest as _pytest

        _pytest.skip("bwrap 不可用")
    assert cmd.index("--ro-bind / /") < cmd.index("--dev-bind /dev /dev")
    assert cmd.index("--dev-bind /dev /dev") < cmd.index("--tmpfs /tmp")


@pytest.mark.skipif(sandbox_mode() != "bwrap", reason="bwrap 不可用则仅 env 隔离")
async def test_bwrap_python_can_start(tmp_path: Path) -> None:
    """回归：沙箱内 Python 能初始化（需要可用的 /dev/urandom）。"""
    output = await make_shell_tool(tmp_path).handler({"command": "python3 -c 'print(42)'"})
    assert "exit code: 0" in output and "42" in output
