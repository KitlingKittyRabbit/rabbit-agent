"""ChatGPT（Codex）OAuth 登录与令牌管理。

与 opencode/Codex 使用同一套公开客户端常量：
client_id、issuer、本地回调端口 1455、PKCE(S256)、附加参数。
令牌存进本项目自己的密钥存储（系统钥匙串优先，回退 keys.json），
不读写 ~/.codex/auth.json，避免影响用户已有的 Codex 登录。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx2

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/oauth/authorize"
TOKEN_URL = f"{ISSUER}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455
SCOPE = "openid profile email offline_access"
TOKEN_STORE_KEY = "openai-codex"       # 钥匙串/keys.json 里的条目名
ORIGINATOR = "rabbit-agent"
USER_AGENT = "rabbit-agent/0.1"
REFRESH_SKEW = 120.0                   # 过期前多少秒提前刷新


class CodexAuthError(Exception):
    """登录/刷新失败。"""


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def make_pkce() -> tuple[str, str]:
    """返回 (code_verifier, code_challenge)，S256。"""
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(*, state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": ORIGINATOR,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def jwt_claims(token: str | None) -> dict:
    """解码 JWT 的 payload（仅解码，不校验签名）。"""
    if not token or token.count(".") < 2:
        return {}
    try:
        payload = json.loads(_b64url_decode(token.split(".")[1]))
    except (ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def account_id_from_tokens(tokens: dict) -> str | None:
    """按 opencode 的取值链：id_token → access_token → organizations[0]。"""
    for key in ("id_token", "access_token"):
        claims = jwt_claims(tokens.get(key))
        nested = claims.get("https://api.openai.com/auth") or {}
        found = claims.get("chatgpt_account_id") or nested.get("chatgpt_account_id")
        if found:
            return str(found)
        orgs = claims.get("organizations") or []
        if orgs and isinstance(orgs[0], dict) and orgs[0].get("id"):
            return str(orgs[0]["id"])
    return tokens.get("accountId") or None


def tokens_from_response(payload: dict) -> dict:
    """token 端点响应 → 我们的存储结构。"""
    access = str(payload.get("access_token") or "")
    refresh = str(payload.get("refresh_token") or "")
    if not access or not refresh:
        raise CodexAuthError("登录响应缺少 access_token/refresh_token")
    expires_in = payload.get("expires_in") or 3600
    tokens = {
        "access": access,
        "refresh": refresh,
        "expires": int(time.time() * 1000) + int(expires_in) * 1000,
        "id_token": str(payload.get("id_token") or ""),
    }
    account_id = account_id_from_tokens(tokens)
    if account_id:
        tokens["accountId"] = account_id
    return tokens


def load_tokens(path: str | Path | None = None) -> dict | None:
    from ..core.keystore import load_keys  # 延迟导入：避免 providers↔core 循环

    raw = load_keys(path).get(TOKEN_STORE_KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) and data.get("access") else None


def save_tokens(tokens: dict, path: str | Path | None = None) -> None:
    from ..core.keystore import save_key

    save_key(TOKEN_STORE_KEY, json.dumps(tokens, ensure_ascii=False), path)


def delete_tokens(path: str | Path | None = None) -> None:
    from ..core.keystore import delete_key

    delete_key(TOKEN_STORE_KEY, path)


def needs_refresh(tokens: dict, *, now_ms: int | None = None) -> bool:
    expires = tokens.get("expires")
    if not isinstance(expires, (int, float)):
        return True
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    return expires - now <= REFRESH_SKEW * 1000


async def refresh_tokens(tokens: dict, *, client: object | None = None) -> dict:
    """用 refresh_token 换新令牌（会轮换 refresh_token，必须回写）。"""
    owns = client is None
    if client is None:
        client = httpx2.AsyncClient(trust_env=False, timeout=30.0)
    try:
        resp = await client.post(TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": tokens.get("refresh") or "",
            "client_id": CLIENT_ID,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})
    finally:
        if owns:
            await client.aclose()
    if getattr(resp, "status_code", 0) != 200:
        raise CodexAuthError(f"刷新令牌失败（HTTP {getattr(resp, 'status_code', '?')}）")
    try:
        payload = resp.json()
    except ValueError as e:
        raise CodexAuthError(f"刷新响应不是 JSON: {e}") from e
    new_tokens = tokens_from_response(payload)
    if not new_tokens.get("accountId"):
        new_tokens["accountId"] = tokens.get("accountId")
    return new_tokens


_REFRESH_LOCK = asyncio.Lock()


async def valid_access_token(path: str | Path | None = None, *,
                             client: object | None = None) -> tuple[str, str]:
    """返回 (access_token, account_id)；过期自动刷新并回写。

    刷新加锁：refresh_token 会轮换，main/executor 并发时不能同时用同一个去换。
    """
    tokens = load_tokens(path)
    if not tokens:
        raise CodexAuthError("尚未登录 ChatGPT：请在设置里点“登录 ChatGPT”")
    if needs_refresh(tokens):
        async with _REFRESH_LOCK:
            tokens = load_tokens(path)          # 可能已被另一个协程刷新
            if tokens and needs_refresh(tokens):
                tokens = await refresh_tokens(tokens, client=client)
                save_tokens(tokens, path)
            if not tokens:
                raise CodexAuthError("尚未登录 ChatGPT：请在设置里点“登录 ChatGPT”")
    return str(tokens["access"]), str(tokens.get("accountId") or "")


def _exchange_code(code: str, verifier: str, *, client: object | None = None) -> dict:
    owns = client is None
    if client is None:
        client = httpx2.Client(trust_env=False, timeout=30.0)
    try:
        resp = client.post(TOKEN_URL, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})
    finally:
        if owns:
            client.close()
    if getattr(resp, "status_code", 0) != 200:
        raise CodexAuthError(f"换取令牌失败（HTTP {getattr(resp, 'status_code', '?')}）")
    return tokens_from_response(resp.json())


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:   # noqa: N802 (http.server 接口)
        query = parse_qs(urlparse(self.path).query)
        code = (query.get("code") or [""])[0]
        state = (query.get("state") or [""])[0]
        self.server.oauth_result = {"code": code, "state": state}   # type: ignore[attr-defined]
        body = (b"<html><body style='font-family:sans-serif'>"
                b"<h3>Login complete</h3><p>You can close this page.</p>"
                b"</body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return


def login(*, path: str | Path | None = None, open_browser: bool = True,
          timeout: float = 300.0, client: object | None = None,
          on_url=None) -> dict:
    """打开浏览器完成 OAuth（本地 1455 回调），成功后写入令牌存储。"""
    verifier, challenge = make_pkce()
    state = secrets.token_urlsafe(24)
    url = build_authorize_url(state=state, code_challenge=challenge)
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    server.timeout = 1.0
    server.oauth_result = None            # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if on_url is not None:
            on_url(url)
        if open_browser:
            webbrowser.open(url)
        deadline = time.monotonic() + timeout
        result = None
        while time.monotonic() < deadline:
            result = getattr(server, "oauth_result", None)
            if result:
                break
            time.sleep(0.2)
        if not result or not result.get("code"):
            raise CodexAuthError("登录超时或未拿到授权码")
        if result.get("state") != state:
            raise CodexAuthError("登录 state 校验失败（可能是伪造回调）")
        tokens = _exchange_code(result["code"], verifier, client=client)
        save_tokens(tokens, path)
        return tokens
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        logger.info("ChatGPT 登录流程结束")
