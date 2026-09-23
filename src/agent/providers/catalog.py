"""models.dev 公共模型目录：按需拉取 + 本地缓存 + 能力映射（零维护成本的数据源）。

- 数据源：https://models.dev/api.json（社区维护，opencode 也用同一份）
- 缓存：<keys 目录>/models-dev.json，TTL 7 天；重拉失败继续使用过期缓存，仍不可用则静默按未知处理
- 只做补齐，不覆盖：用户覆盖 > provider /models 返回 > 本目录 > 未知
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx2

from .base import ModelCapability

CATALOG_URL = "https://models.dev/api.json"
CATALOG_TTL_SECONDS = 7 * 24 * 3600

# 我们的档位词汇（off 表示关闭思考）
_FIXED_LEVELS = ("off", "low", "medium", "high")  # budget_tokens（Anthropic 风格预算）


def load_catalog(path: str | Path) -> dict | None:
    """读缓存；不存在/损坏返回 None。"""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("providers"), dict):
        return None
    return data


def save_catalog(path: str | Path, providers: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "fetched_at": time.time(), "providers": providers,
    }, ensure_ascii=False), encoding="utf-8")
    os.chmod(p, 0o600)


def catalog_stale(path: str | Path, now: float | None = None) -> bool:
    data = load_catalog(path)
    if data is None:
        return True
    now = time.time() if now is None else now
    return now - float(data.get("fetched_at") or 0) > CATALOG_TTL_SECONDS


async def fetch_catalog(client: object | None = None) -> dict:
    """拉取 models.dev api.json，返回 {provider_id: {model_id: entry}}。失败抛异常。"""
    owns = client is None
    if client is None:
        client = httpx2.AsyncClient(trust_env=False, timeout=15)
    try:
        response = await client.get(CATALOG_URL)
    finally:
        if owns:
            await client.aclose()
    if getattr(response, "status_code", 0) != 200:
        raise ValueError(f"models.dev 返回 HTTP {getattr(response, 'status_code', '?')}")
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("models.dev 响应不是对象")
    return data


def catalog_models(catalog_data: dict | None, catalog_name: str) -> dict[str, dict]:
    """取某 provider 的 {model_id: 原始条目}；缺失返回空。"""
    if not catalog_data or not catalog_name:
        return {}
    provider = catalog_data.get("providers", {}).get(catalog_name)
    if not isinstance(provider, dict):
        return {}
    models = provider.get("models")
    if not isinstance(models, dict):
        return {}
    return {str(k): v for k, v in models.items() if isinstance(v, dict)}


def capability_from_catalog(entry: dict, protocol: str) -> ModelCapability:
    """把 models.dev 条目映射为能力（未知字段一律 None，不猜测）。

    - limit.context/output → 窗口/最大输出；tool_call → tools
    - reasoning=False → none；reasoning=True:
      - effort 选项 → 可调（toggle 时补 off）
      - budget_tokens（Anthropic 风格预算）→ 可调（固定四档 off/low/medium/high）
      - 仅 toggle 或无 options → 固定（可开但不可调档）
    - interleaved.field 或 anthropic 协议 → 返回 reasoning
    """
    limit = entry.get("limit") if isinstance(entry.get("limit"), dict) else {}
    window = limit.get("context")
    if not (isinstance(window, int) and window > 0):
        window = None
    max_output = limit.get("output")
    if not (isinstance(max_output, int) and max_output > 0):
        max_output = None
    tools = entry.get("tool_call")
    tools = tools if isinstance(tools, bool) else None

    reasoning = entry.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, bool) else None
    options = entry.get("reasoning_options")
    options = options if isinstance(options, list) else []
    toggle = any(isinstance(o, dict) and o.get("type") == "toggle" for o in options)
    effort_values: list[str] = []
    for opt in options:
        if isinstance(opt, dict) and opt.get("type") == "effort" and isinstance(
                opt.get("values"), list):
            effort_values = [str(v) for v in opt["values"]]
            break
    has_budget = any(isinstance(o, dict) and o.get("type") == "budget_tokens"
                     for o in options)

    mode = "unknown"
    levels: tuple[str, ...] | None = None
    if reasoning is False:
        mode, levels = "none", ()
    elif reasoning is True:
        if effort_values:
            mode = "adjustable"
            if protocol == "anthropic":
                # anthropic 档位最终按思考预算发送：目录档位词汇不可用时改用固定四档，
                # 禁止出现"选了档位但请求里静默消失"（如 max）
                levels = _FIXED_LEVELS
            else:
                levels = tuple((["off"] if toggle else []) + effort_values)
        elif has_budget:
            mode = "adjustable"
            levels = _FIXED_LEVELS
        else:
            mode = "fixed"
            levels = ()
    interleaved = entry.get("interleaved")
    reasoning_returned = None
    if isinstance(interleaved, dict) and interleaved.get("field"):
        reasoning_returned = True
    elif protocol == "anthropic" and reasoning is True:
        reasoning_returned = True

    return ModelCapability(
        window=window,
        reasoning_returned=reasoning_returned,
        reasoning_mode=mode,
        levels=levels,
        max_output=max_output,
        tools=tools,
        source="catalog",
    )
