"""模型列表获取测试：分协议端点、结构化能力解析、凭据只进请求头、失败不伪造。"""

import asyncio

import pytest

from agent.providers.listing import (
    ModelListingError,
    ModelListingUnsupported,
    fetch_models,
    model_list_url,
    parse_model_entry,
)


def test_model_list_url_per_protocol() -> None:
    assert model_list_url("openai", "http://x:1/v1") == "http://x:1/v1/models"
    assert model_list_url("openai", "http://x:1/v1/") == "http://x:1/v1/models"
    assert model_list_url("openai", None) == "https://api.openai.com/v1/models"
    assert model_list_url("anthropic", "https://api.anthropic.com") == (
        "https://api.anthropic.com/v1/models"
    )
    assert model_list_url("anthropic", "https://api.anthropic.com/v1") == (
        "https://api.anthropic.com/v1/models"
    )
    assert model_list_url("anthropic", "https://api.kimi.com/coding") == (
        "https://api.kimi.com/coding/v1/models"
    )
    with pytest.raises(ModelListingUnsupported):
        model_list_url("bogus", "http://x")


def test_parse_model_entry_structured_only() -> None:
    entry = parse_model_entry({
        "id": "m1", "display_name": "模型一",
        "context_window": 128_000, "max_output_tokens": 8_192,
        "supports_tools": True,
        "capabilities": {"reasoning": True, "reasoning_efforts": ["low", "medium", "high"]},
    }, "openai")
    assert entry["id"] == "m1" and entry["display_name"] == "模型一"
    cap = entry["capability"]
    assert cap["window"] == 128_000
    assert cap["max_output"] == 8_192
    assert cap["tools"] is True
    assert cap["reasoning_returned"] is True
    assert cap["reasoning_mode"] == "adjustable"
    assert cap["levels"] == ["low", "medium", "high"]
    assert cap["source"] == "provider"


def test_parse_model_entry_no_name_guessing() -> None:
    """id 含已知家族子串但无字段 → 全部未知（禁止按名称猜）。"""
    entry = parse_model_entry({"id": "gpt-5-fake-model"}, "openai")
    cap = entry["capability"]
    assert cap["window"] is None
    assert cap["reasoning_mode"] == "unknown"
    assert cap["levels"] is None  # 无字段=None（未知）
    assert cap["max_output"] is None
    assert cap["tools"] is None


def test_parse_model_entry_fixed_and_none_reasoning() -> None:
    fixed = parse_model_entry({"id": "r1", "supports_reasoning": True}, "openai")
    assert fixed["capability"]["reasoning_mode"] == "fixed"
    none = parse_model_entry({"id": "r2", "supports_reasoning": False}, "openai")
    assert none["capability"]["reasoning_mode"] == "none"


class _Resp:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class _Client:
    def __init__(self, response):
        self._response = response
        self.last_url = None
        self.last_headers = None

    async def get(self, url, headers):
        self.last_url, self.last_headers = url, headers
        return self._response

    async def aclose(self):
        pass


def test_fetch_models_uses_header_credentials_not_url() -> None:
    client = _Client(_Resp(payload={"data": [{"id": "a"}, {"id": "b"}]}))
    models = asyncio.run(
        fetch_models("openai", "http://x:1/v1", "sk-secret", client=client)
    )  # noqa: E501 保留可读性
    assert [m["id"] for m in models] == ["a", "b"]
    assert client.last_url == "http://x:1/v1/models"
    assert "sk-secret" not in client.last_url
    assert client.last_headers["Authorization"] == "Bearer sk-secret"


def test_fetch_models_anthropic_headers() -> None:
    client = _Client(_Resp(payload={"data": [{"id": "claude-x"}]}))
    asyncio.run(fetch_models("anthropic", "https://api.anthropic.com", "sk-a", client=client))
    assert client.last_headers["x-api-key"] == "sk-a"
    assert client.last_headers["anthropic-version"]
    assert client.last_url == "https://api.anthropic.com/v1/models"


def test_fetch_models_failures_raise_not_fake() -> None:
    with pytest.raises(ModelListingError):
        asyncio.run(fetch_models("openai", "http://x", "k", client=_Client(_Resp(status=401))))
    with pytest.raises(ModelListingError):
        asyncio.run(fetch_models("openai", "http://x", "k", client=_Client(_Resp(bad_json=True))))
    with pytest.raises(ModelListingError):
        empty = _Client(_Resp(payload={"data": []}))
        asyncio.run(fetch_models("openai", "http://x", "k", client=empty))
    with pytest.raises(ModelListingUnsupported):
        asyncio.run(fetch_models("bogus", "http://x", "k", client=_Client(_Resp(payload={}))))


async def test_listing_sends_opencode_session_headers() -> None:
    """OpenCode 主机的 /models 也要带 UA + x-opencode-session；其它主机不带。"""
    import httpx2

    from agent.providers.listing import fetch_models

    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["ua"] = request.headers.get("user-agent")
        seen["sid"] = request.headers.get("x-opencode-session")
        return httpx2.Response(200, json={"data": [{"id": "m1"}]})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    models = await fetch_models("openai", "https://opencode.ai/zen/go/v1", "sk",
                                client=client, session_id="sess-list")
    assert models and models[0]["id"] == "m1"
    assert seen["sid"] == "sess-list"
    assert seen["ua"] and "rabbit-agent" in seen["ua"]

    seen.clear()
    client2 = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    await fetch_models("openai", "https://api.deepseek.com/v1", "sk", client=client2)
    assert seen["sid"] is None
    await client.aclose()
    await client2.aclose()
