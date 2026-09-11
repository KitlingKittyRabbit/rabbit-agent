"""环境代理鲁棒性测试：用户环境根因（socks:// 代理变量导致 SDK 初始化 ValueError）的锁定。"""

from agent.providers import AnthropicCompatProvider, OpenAICompatProvider

_SOCKS_ENV = {
    "ALL_PROXY": "socks://127.0.0.1:7890/",
    "all_proxy": "socks://127.0.0.1:7890/",
}


def test_openai_provider_survives_socks_env_proxy(monkeypatch) -> None:
    """用户环境复现：socks:// scheme 的环境代理不得让 provider 构造崩溃。"""
    for key, value in _SOCKS_ENV.items():
        monkeypatch.setenv(key, value)
    # 修复前：httpx2 ValueError: Unknown scheme for proxy URL 'socks://127.0.0.1:7890/'
    provider = OpenAICompatProvider(
        base_url="https://api.deepseek.com/v1", api_key="sk-x", model="m"
    )
    assert provider is not None


def test_anthropic_provider_survives_socks_env_proxy(monkeypatch) -> None:
    for key, value in _SOCKS_ENV.items():
        monkeypatch.setenv(key, value)
    provider = AnthropicCompatProvider(api_key="sk-x", model="m")
    assert provider is not None
