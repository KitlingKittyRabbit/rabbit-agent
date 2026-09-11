"""真实 API 冒烟：uv run python -m agent.smoke

读取 .providers.toml，逐角色 ping + 工具调用往返检查。
手动触发（规则四：真实 API 冒烟单独标记），不进默认测试套件。
"""

import asyncio

from .core.config import ConfigError, make_provider
from .core.provider_store import load_providers
from .providers import Message, ToolSpec

STORE_PATH = ".providers.toml"

_ECHO_TOOL = ToolSpec(
    name="echo",
    description="原样返回输入",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


async def check_role(role: str, settings: dict, factory=make_provider) -> bool:
    try:
        provider = factory(**settings)
    except ConfigError as e:
        print(f"✗ {role}: 配置无效（{e}）")
        return False
    try:
        result = await provider.chat([Message(role="user", content="回复 ok 两个字即可")])
        print(f"✓ {role}: 连接正常（{settings.get('model')}），回复: {result.text[:50]!r}")
    except Exception as e:
        print(f"✗ {role}: 连接失败（{type(e).__name__}: {e}）")
        return False
    try:
        probe = await provider.chat(
            [Message(role="user", content="请调用 echo 工具，参数 text 为 hello")],
            tools=[_ECHO_TOOL],
        )
        if probe.tool_calls:
            print(f"✓ {role}: 工具调用正常（{probe.tool_calls[0].name}）")
        else:
            print(f"? {role}: 模型未发起工具调用（回复 {probe.text[:40]!r}），请人工留意")
    except Exception as e:
        print(f"? {role}: 工具调用检查异常（{e}）")
    return True


async def _main() -> int:
    stored = load_providers(STORE_PATH)
    if not stored:
        print(f"{STORE_PATH} 为空——先用 /connect_provider 连接服务商")
        return 1
    ok = True
    for role, settings in stored.items():
        ok = await check_role(role, settings) and ok
    return 0 if ok else 1


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
