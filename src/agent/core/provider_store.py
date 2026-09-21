"""多 provider 注册表：.providers.toml（只存非敏感配置）+ keys.json（按 provider_id 存密钥）。

新结构（旧 [main]/[executor] 段自动迁移，不丢配置）：

    [roles]
    main_provider = "p-xxxx"
    main_model = "m"
    main_reasoning_effort = "medium"

    [providers."p-xxxx"]
    name = "openai · api.example.com/v1"
    protocol = "openai"
    base_url = "https://api.example.com/v1"

    [providers."p-xxxx".model_overrides."m"]
    window = 128000
    reasoning_mode = "adjustable"
    levels = "off,low,medium,high"
    max_output = 8192
    tools = true
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

from .keystore import save_key

_PROVIDER_FIELDS = ("name", "protocol", "base_url", "catalog")
_OVERRIDE_FIELDS = ("window", "reasoning_returned", "reasoning_mode", "levels",
                    "max_output", "tools")


def derive_provider_id(protocol: str, base_url: str | None) -> str:
    """稳定 ID：同协议+同地址视为同一 provider（幂等连接不会产生重复条目）。"""
    raw = f"{protocol}|{(base_url or '').rstrip('/')}"
    return "p-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def provider_label(protocol: str, base_url: str | None) -> str:
    base = (base_url or "").replace("https://", "").replace("http://", "").rstrip("/")
    return f"{protocol} · {base}" if base else f"{protocol}（默认端点）"


def _default_registry() -> dict:
    return {"roles": {}, "providers": {}}


def load_registry(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        return _default_registry()
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _default_registry()
    if not isinstance(data, dict):
        return _default_registry()
    roles_raw = data.get("roles") if isinstance(data.get("roles"), dict) else {}
    providers_raw = data.get("providers") if isinstance(data.get("providers"), dict) else {}
    registry = _default_registry()
    for role in ("main", "executor"):
        pid = roles_raw.get(f"{role}_provider")
        model = roles_raw.get(f"{role}_model")
        if isinstance(pid, str) and pid and isinstance(model, str) and model:
            registry["roles"][role] = {
                "provider_id": pid,
                "model": model,
                "reasoning_effort": str(roles_raw.get(f"{role}_reasoning_effort") or "off"),
            }
    for pid, section in providers_raw.items():
        if not isinstance(section, dict):
            continue
        entry = {
            "name": str(section.get("name") or pid),
            "protocol": str(section.get("protocol") or ""),
            "base_url": section.get("base_url"),
            "catalog": str(section.get("catalog") or ""),
            "model_overrides": {},
        }
        overrides = section.get("model_overrides")
        if isinstance(overrides, dict):
            for model_id, values in overrides.items():
                if not isinstance(values, dict):
                    continue
                entry["model_overrides"][str(model_id)] = clean_override(values)
        entry["models"] = []
        models_section = section.get("models")
        if isinstance(models_section, dict):
            for model_id, values in models_section.items():
                if not isinstance(values, dict):
                    continue
                cap = clean_override(values)
                entry["models"].append({
                    "id": str(model_id),
                    "display_name": str(values.get("display_name") or model_id),
                    "provider_id": str(pid),
                    "provider": entry["name"],
                    "capability": {**cap, "source": "provider"},
                })
        fetched = section.get("models_fetched_at")
        entry["models_fetched_at"] = float(fetched) if isinstance(fetched, (int, float)) else 0.0
        registry["providers"][str(pid)] = entry
    return registry


def clean_override(values: dict) -> dict:
    out: dict = {}
    returned = values.get("reasoning_returned")
    if isinstance(returned, bool):
        out["reasoning_returned"] = returned
    window = values.get("window")
    if isinstance(window, int) and window > 0:
        out["window"] = window
    mode = values.get("reasoning_mode")
    if isinstance(mode, str) and mode in ("unknown", "fixed", "none", "adjustable"):
        out["reasoning_mode"] = mode
    levels = values.get("levels")
    if isinstance(levels, str):
        out["levels"] = [lv for lv in (s.strip() for s in levels.split(",")) if lv]
    elif isinstance(levels, list):
        out["levels"] = [str(lv) for lv in levels]
    max_output = values.get("max_output")
    if isinstance(max_output, int) and max_output > 0:
        out["max_output"] = max_output
    tools = values.get("tools")
    if isinstance(tools, bool):
        out["tools"] = tools
    return out


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return _quote(",".join(str(v) for v in value))
    return _quote(str(value))


def save_registry(path: str | Path, registry: dict) -> None:
    lines: list[str] = []
    roles = registry.get("roles") or {}
    if roles:
        lines.append("[roles]")
        for role in ("main", "executor"):
            binding = roles.get(role)
            if not binding:
                continue
            lines.append(f'{role}_provider = {_quote(str(binding.get("provider_id") or ""))}')
            lines.append(f'{role}_model = {_quote(str(binding.get("model") or ""))}')
            lines.append(
                f'{role}_reasoning_effort = '
                f'{_quote(str(binding.get("reasoning_effort") or "off"))}'
            )
        lines.append("")
    for pid, entry in (registry.get("providers") or {}).items():
        lines.append(f'[providers."{pid}"]')
        for field in _PROVIDER_FIELDS:
            value = entry.get(field)
            if value is not None and value != "":
                lines.append(f"{field} = {_format_value(value)}")
        lines.append("")
        fetched = entry.get("models_fetched_at") or 0.0
        if fetched:
            lines.append(f"models_fetched_at = {float(fetched)}")
        lines.append("")
        for model_id, override in (entry.get("model_overrides") or {}).items():
            if not override:
                continue
            lines.append(f'[providers."{pid}".model_overrides."{model_id}"]')
            for field in _OVERRIDE_FIELDS:
                if field in override and override[field] is not None:
                    lines.append(f"{field} = {_format_value(override[field])}")
            lines.append("")
        for model in entry.get("models") or []:
            model_id = str(model.get("id") or "")
            if not model_id:
                continue
            cap = model.get("capability") or {}
            lines.append(f'[providers."{pid}".models."{model_id}"]')
            lines.append(f'display_name = {_quote(str(model.get("display_name") or model_id))}')
            for field in _OVERRIDE_FIELDS:
                value = cap.get(field)
                if value is not None:
                    lines.append(f"{field} = {_format_value(value)}")
            lines.append("")
    p = Path(path)
    p.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(p, 0o600)


def migrate_legacy(path: str | Path, keys_path: str | Path | None = None) -> bool:
    """旧结构 [main]/[executor]（含 api_key）→ 新注册表 + keys.json(按 provider_id)。"""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or "providers" in data:
        return False
    legacy_roles = {
        role: data.get(role) for role in ("main", "executor")
        if isinstance(data.get(role), dict)
    }
    if not legacy_roles:
        return False
    registry = _default_registry()
    for role, section in legacy_roles.items():
        protocol = str(section.get("protocol") or "")
        if not protocol:
            continue
        base_url = section.get("base_url")
        model = str(section.get("model") or "")
        pid = derive_provider_id(protocol, base_url)
        registry["providers"].setdefault(pid, {
            "name": provider_label(protocol, base_url),
            "protocol": protocol,
            "base_url": base_url,
            "model_overrides": {},
        })
        registry["roles"][role] = {
            "provider_id": pid,
            "model": model,
            "reasoning_effort": str(section.get("reasoning_effort") or "off"),
        }
        window = section.get("context_window")
        if isinstance(window, str) and window.isdigit():
            window = int(window)
        if isinstance(window, int) and window > 0 and model:
            registry["providers"][pid]["model_overrides"].setdefault(model, {})["window"] = window
        key = section.get("api_key")
        if isinstance(key, str) and key:
            save_key(pid, key, keys_path)
    save_registry(p, registry)
    return True
