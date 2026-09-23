"""ChatGPT(Codex) 登录：PKCE/URL、JWT account_id、刷新、登录回调（全程离线）。"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest

import agent.providers.codex_auth as codex_auth
from agent.providers.codex_auth import (
    AUTHORIZE_URL,
    CLIENT_ID,
    CodexAuthError,
    account_id_from_tokens,
    build_authorize_url,
    load_tokens,
    login,
    make_pkce,
    needs_refresh,
    refresh_tokens,
    save_tokens,
    tokens_from_response,
    valid_access_token,
)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _jwt(claims: dict) -> str:
    header = _b64url(json.dumps({"alg": "none"}).encode())
    payload = _b64url(json.dumps(claims).encode())
    return f"{header}.{payload}.sig"


def test_pkce_and_authorize_url() -> None:
    verifier, challenge = make_pkce()
    assert len(verifier) >= 43
    assert _b64url(hashlib.sha256(verifier.encode()).digest()) == challenge

    url = build_authorize_url(state="st", code_challenge=challenge)
    query = parse_qs(urlparse(url).query)
    assert url.startswith(AUTHORIZE_URL + "?")
    assert query["client_id"] == [CLIENT_ID]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [challenge]
    assert query["id_token_add_organizations"] == ["true"]
    assert query["codex_cli_simplified_flow"] == ["true"]
    assert query["state"] == ["st"]
    assert "offline_access" in query["scope"][0]


def test_account_id_from_nested_claim() -> None:
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-nested"}})
    assert account_id_from_tokens({"id_token": token}) == "acct-nested"
    assert account_id_from_tokens({"access_token": token}) == "acct-nested"
    assert account_id_from_tokens(
        {"id_token": _jwt({"chatgpt_account_id": "acct-top"})}) == "acct-top"
    assert account_id_from_tokens(
        {"access_token": _jwt({"organizations": [{"id": "org-1"}]})}) == "org-1"
    assert account_id_from_tokens({"access_token": "not-a-jwt"}) is None


def test_tokens_from_response_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    id_token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-9"}})
    tokens = tokens_from_response(
        {"access_token": "at", "refresh_token": "rt", "expires_in": 60, "id_token": id_token})
    assert tokens["accountId"] == "acct-9"

    save_tokens(tokens, path)
    assert load_tokens(path) == tokens                      # 落盘（本测试为文件模式）
    assert needs_refresh(tokens, now_ms=tokens["expires"] - 60_000) is True
    assert needs_refresh(tokens, now_ms=tokens["expires"] - 10 * 60_000) is False
    assert needs_refresh({"access": "x"}) is True           # 无过期时间：按需要刷新处理


async def test_refresh_rotates_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    save_tokens({"access": "old", "refresh": "rt-old", "expires": 1, "accountId": "a"}, path)
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["body"] = dict(parse_qs(request.content.decode()))
        return httpx2.Response(200, json={
            "access_token": "new-at", "refresh_token": "rt-new",
            "expires_in": 3600,
            "id_token": _jwt({"chatgpt_account_id": "a"}),
        })

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    access, account = await valid_access_token(path, client=client)
    await client.aclose()

    assert seen["url"] == codex_auth.TOKEN_URL
    assert seen["body"]["grant_type"] == ["refresh_token"]
    assert seen["body"]["refresh_token"] == ["rt-old"]
    assert seen["body"]["client_id"] == [CLIENT_ID]
    assert (access, account) == ("new-at", "a")
    assert load_tokens(path)["refresh"] == "rt-new"          # 轮换后的 refresh 已回写


async def test_valid_access_token_without_login(tmp_path: Path) -> None:
    with pytest.raises(CodexAuthError, match="尚未登录"):
        await valid_access_token(tmp_path / "keys.json")


async def test_refresh_failure_is_explicit(tmp_path: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": "invalid_grant"})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    with pytest.raises(CodexAuthError, match="刷新令牌失败"):
        await refresh_tokens({"access": "x", "refresh": "y", "expires": 1}, client=client)
    await client.aclose()


def test_login_flow_with_local_callback(tmp_path: Path, monkeypatch) -> None:
    """模拟浏览器：拿到授权 URL 后访问本地回调，验证 PKCE 配对与令牌落盘。"""
    path = tmp_path / "keys.json"
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = dict(parse_qs(request.content.decode()))
        return httpx2.Response(200, json={
            "access_token": "at-login", "refresh_token": "rt-login",
            "expires_in": 3600,
            "id_token": _jwt({"chatgpt_account_id": "acct-login"}),
        })

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    def fake_fetch(url: str) -> None:
        query = parse_qs(urlparse(url).query)
        state = query["state"][0]
        challenge = query["code_challenge"][0]

        def hit() -> None:
            import time
            time.sleep(0.3)
            with urllib.request.urlopen(
                f"http://127.0.0.1:{codex_auth.CALLBACK_PORT}/auth/callback"
                f"?code=abc&state={state}", timeout=5) as resp:
                resp.read()
        threading.Thread(target=hit, daemon=True).start()
        seen["challenge"] = challenge

    tokens = login(path=path, open_browser=False, timeout=10, client=client,
                   on_url=fake_fetch)
    client.close()

    assert seen["body"]["grant_type"] == ["authorization_code"]
    assert seen["body"]["code"] == ["abc"]
    assert seen["body"]["redirect_uri"] == [codex_auth.REDIRECT_URI]
    verifier = seen["body"]["code_verifier"][0]
    assert _b64url(hashlib.sha256(verifier.encode()).digest()) == seen["challenge"]
    assert tokens["accountId"] == "acct-login"
    assert load_tokens(path)["access"] == "at-login"


async def test_concurrent_refresh_rotates_only_once(tmp_path: Path, monkeypatch) -> None:
    """并发取令牌只刷新一次（refresh_token 轮换不能重复使用）。

    假刷新故意 await 挂起：没有锁时三个协程会各刷一次（calls=3），测试必须失败。
    """
    path = tmp_path / "keys.json"
    save_tokens({"access": "old", "refresh": "rt-old", "expires": 1, "accountId": "a"}, path)
    calls = {"n": 0}

    async def fake_refresh(tokens: dict, *, client: object | None = None) -> dict:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return {"access": "new", "refresh": "rt-new",
                "expires": 4102444800000, "accountId": "a"}

    monkeypatch.setattr(codex_auth, "refresh_tokens", fake_refresh)
    results = await asyncio.gather(*(valid_access_token(path) for _ in range(3)))

    assert calls["n"] == 1
    assert all(access == "new" for access, _ in results)
    assert load_tokens(path)["refresh"] == "rt-new"


def test_login_rejects_state_mismatch(tmp_path: Path, monkeypatch) -> None:
    def fake_fetch(url: str) -> None:
        def hit() -> None:
            import time
            time.sleep(0.3)
            with urllib.request.urlopen(
                f"http://127.0.0.1:{codex_auth.CALLBACK_PORT}/auth/callback"
                "?code=abc&state=WRONG", timeout=5) as resp:
                resp.read()
        threading.Thread(target=hit, daemon=True).start()

    client = httpx2.Client(transport=httpx2.MockTransport(
        lambda r: httpx2.Response(200, json={})))
    try:
        with pytest.raises(CodexAuthError, match="state"):
            login(path=tmp_path / "keys.json", open_browser=False, timeout=10,
                  client=client, on_url=fake_fetch)
    finally:
        client.close()
