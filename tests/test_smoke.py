"""真实 API 冒烟逻辑测试（FakeProvider 驱动，零网络；真实冒烟由用户手动触发）。"""

from agent.providers import AuthError, ChatResult, FakeProvider, ToolCall
from agent.smoke import check_role

SETTINGS = {"protocol": "openai", "model": "m", "api_key": "x"}


async def test_check_role_success(capsys) -> None:
    fake = FakeProvider(
        [
            ChatResult(text="ok"),
            ChatResult(tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})]),
        ]
    )
    ok = await check_role("main", SETTINGS, factory=lambda **kw: fake)
    assert ok is True
    out = capsys.readouterr().out
    assert "连接正常" in out
    assert "工具调用正常" in out


async def test_check_role_connect_failure(capsys) -> None:
    fake = FakeProvider([AuthError("no")])
    ok = await check_role("main", SETTINGS, factory=lambda **kw: fake)
    assert ok is False
    assert "连接失败" in capsys.readouterr().out


async def test_check_role_invalid_config(capsys) -> None:
    def factory(**kw):
        raise ValueError("未知协议")

    from agent.core.config import ConfigError

    def raising_factory(**kw):
        raise ConfigError("未知协议")

    ok = await check_role("main", SETTINGS, factory=raising_factory)
    assert ok is False
    assert "配置无效" in capsys.readouterr().out


async def test_check_role_no_tool_call_is_warning_not_failure(capsys) -> None:
    fake = FakeProvider([ChatResult(text="ok"), ChatResult(text="我不会用工具")])
    ok = await check_role("executor", SETTINGS, factory=lambda **kw: fake)
    assert ok is True
    assert "未发起工具调用" in capsys.readouterr().out
