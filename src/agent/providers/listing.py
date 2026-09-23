"""模型列表获取与结构化能力解析（分协议端点 + 只读结构化字段）。

设计约束：
- 不使用模型名称子串猜测能力；未返回的字段一律 None（未知）
- 凭据只作为请求头在内存中使用，绝不进入 URL/日志/事件
- provider 没有模型列表端点时不猜测请求（抛 ModelListingUnsupported）
"""

from __future__ import annotations

import httpx2

from .base import ModelCapability, is_opencode_host

ANTHROPIC_VERSION = "2023-06-01"


class ModelListingError(Exception):
    """模型列表获取失败（网络/鉴权/协议错误）。"""


class ModelListingUnsupported(Exception):
    """该 provider/协议没有可用的模型列表端点。"""


def model_list_url(protocol: str, base_url: str | None) -> str:
    """按协议给出正确的模型列表端点；无端点抛 ModelListingUnsupported。"""
    base = (base_url or "").rstrip("/")
    if protocol == "openai":
        if not base:
            base = "https://api.openai.com/v1"
        return f"{base}/models"
    if protocol == "anthropic":
        if not base:
            base = "https://api.anthropic.com"
        if base.endswith("/v1"):
            return f"{base}/models"
        return f"{base}/v1/models"
    raise ModelListingUnsupported(f"未知协议，无法获取模型列表: {protocol!r}")


def _pick_int(item: dict, *fields: str) -> int | None:
    for field in fields:
        value = item.get(field)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _pick_bool(item: dict, *fields: str) -> bool | None:
    for field in fields:
        value = item.get(field)
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            for key in ("enabled", "supported", "value"):
                inner = value.get(key)
                if isinstance(inner, bool):
                    return inner
    return None


def _reasoning_levels(item: dict) -> tuple[str, ...] | None:
    """只接受结构化字段；不接受按名称推断。"""
    raw = None
    for field in ("reasoning_efforts", "reasoning_effort_levels", "reasoning_levels"):
        if isinstance(item.get(field), list):
            raw = item[field]
            break
    caps = item.get("capabilities")
    if raw is None and isinstance(caps, dict):
        for field in ("reasoning_efforts", "reasoning_effort_levels", "reasoning_levels"):
            if isinstance(caps.get(field), list):
                raw = caps[field]
                break
    if raw is None:
        return None  # 无字段=None（未知），不得用空元组冒充
    allowed = {"off", "low", "medium", "high"}
    return tuple(str(v).lower() for v in raw if str(v).lower() in allowed)


def parse_model_entry(item: dict, protocol: str) -> dict:
    """把 provider 返回的单个模型条目转成结构化字典（缺字段=None，不猜测）。"""
    model_id = str(item.get("id") or item.get("name") or item.get("model") or "")
    display = str(item.get("display_name") or item.get("displayName") or model_id)
    window = _pick_int(
        item,
        "context_window", "context_length", "max_context_length",
        "max_input_tokens", "input_token_limit",
    )
    max_output = _pick_int(
        item, "max_output_tokens", "max_completion_tokens", "output_token_limit",
    )
    tools = _pick_bool(item, "supports_tools", "tool_calling")
    caps = item.get("capabilities")
    if tools is None and isinstance(caps, dict):
        tools = _pick_bool(caps, "tool_calls", "tools", "function_calling")
    reasoning_returned = _pick_bool(item, "supports_reasoning", "reasoning")
    if reasoning_returned is None and isinstance(caps, dict):
        reasoning_returned = _pick_bool(caps, "reasoning", "thinking")
    levels = _reasoning_levels(item)
    if levels:
        mode = "adjustable"
    elif reasoning_returned is True:
        mode = "fixed"
    elif reasoning_returned is False:
        mode = "none"
    else:
        mode = "unknown"
    capability = ModelCapability(
        window=window, reasoning_returned=reasoning_returned, reasoning_mode=mode,
        levels=tuple(levels) if levels is not None else (),
        max_output=max_output, tools=tools, source="provider",
    )
    return {
        "id": model_id,
        "display_name": display,
        "raw": item,
        "capability": {
            **capability.as_dict(),
            "levels": list(levels) if levels is not None else None,
        },
    }


def _entries(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("data", "models", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)]
    if isinstance(payload, list):
        return [v for v in payload if isinstance(v, dict)]
    return []


async def fetch_models(
    protocol: str,
    base_url: str | None,
    api_key: str,
    *,
    timeout: float = 10.0,
    client: object | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """获取并结构化模型列表；失败抛 ModelListingError（不静默回退、不伪造）。"""
    url = model_list_url(protocol, base_url)  # 可能抛 Unsupported
    headers = {"Accept": "application/json"}
    if is_opencode_host(base_url):
        headers["User-Agent"] = "rabbit-agent/0.1"
        headers["x-opencode-session"] = session_id or "rabbit-list"
    if api_key:
        if protocol == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = ANTHROPIC_VERSION
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    owns_client = client is None
    if client is None:
        client = httpx2.AsyncClient(trust_env=False, timeout=timeout)
    try:
        response = await client.get(url, headers=headers)
    except Exception as e:
        raise ModelListingError(f"模型列表请求失败: {type(e).__name__}: {e}") from e
    finally:
        if owns_client and client is not None:
            await client.aclose()
    if getattr(response, "status_code", 0) != 200:
        raise ModelListingError(f"模型列表返回 HTTP {getattr(response, 'status_code', '?')}")
    try:
        payload = response.json()
    except Exception as e:
        raise ModelListingError(f"模型列表响应不是合法 JSON: {e}") from e
    entries = _entries(payload)
    if not entries:
        raise ModelListingError("模型列表为空或格式无法识别")
    return [parse_model_entry(item, protocol) for item in entries]


def _capability_gives(cap: ModelCapability | None) -> bool:
    """该层是否显式给出过任何值（False/空 levels 也算显式）。"""
    if cap is None:
        return False
    if cap.reasoning_mode != "unknown":
        return True
    return any(
        value is not None
        for value in (cap.window, cap.reasoning_returned, cap.tools, cap.max_output, cap.levels)
    )


def merge_capability(
    metadata: ModelCapability | None,
    user: ModelCapability | None = None,
) -> ModelCapability:
    """合并能力：用户显式覆盖 > provider 元数据 > 未知。

    只用 `is not None` 判定，False/0/空 levels 都是显式值：
    - tools=False、reasoning_returned=False 必须保留
    - window<=0 视为无效（未知）
    - levels=None 表示未知；levels=() 表示明确为空（如 fixed）
    """

    def first(*values):
        for value in values:
            if value is not None:
                return value
        return None

    def first_window(*values):
        for value in values:
            if isinstance(value, int) and value > 0:
                return value
        return None

    mode_user = user.reasoning_mode if user and user.reasoning_mode != "unknown" else None
    mode_meta = (
        metadata.reasoning_mode if metadata and metadata.reasoning_mode != "unknown" else None
    )
    mode = first(mode_user, mode_meta) or "unknown"
    levels = first(user.levels if user else None, metadata.levels if metadata else None)
    if mode == "unknown" and levels:
        mode = "adjustable"
    if mode == "adjustable" and levels is None:
        levels = None  # 可调但未知支持哪些档：控件应禁用/提示声明
    if mode in ("fixed", "none") and levels is None:
        levels = ()
    source_name = "unknown"
    if _capability_gives(user):
        source_name = "user"
    elif _capability_gives(metadata):
        source_name = "provider"
    return ModelCapability(
        window=first_window(user.window if user else None, metadata.window if metadata else None),
        reasoning_returned=first(
            user.reasoning_returned if user else None,
            metadata.reasoning_returned if metadata else None,
        ),
        reasoning_mode=mode,
        levels=levels,
        max_output=first_window(
            user.max_output if user else None, metadata.max_output if metadata else None
        ),
        tools=first(user.tools if user else None, metadata.tools if metadata else None),
        interleaved=first(
            user.interleaved if user else None, metadata.interleaved if metadata else None
        ),
        source=source_name,
    )
