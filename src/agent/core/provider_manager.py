"""provider 运行时管理：注册表/实例/凭据/能力/目录/连接/状态（从 Orchestrator 拆出）。

职责单一：只管 provider 与角色绑定；会话/项目/事件/WS 仍在 Orchestrator。
Orchestrator 通过 __getattr__ 委托这些实现，保持旧 API 与测试兼容。
"""

from __future__ import annotations

import dataclasses
import ipaddress
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from ..providers import Message, ModelCapability, Provider
from ..providers.catalog import (
    capability_from_catalog,
    catalog_models,
    catalog_stale,
    fetch_catalog,
    load_catalog,
    save_catalog,
)
from ..providers.listing import ModelListingError, _capability_gives, merge_capability
from . import keystore
from .config import make_provider
from .context import context_budget
from .keystore import delete_key, load_keys, migrate_keys, save_key
from .presets import PRESETS
from .provider_store import (
    clean_override,
    derive_provider_id,
    load_registry,
    migrate_legacy,
    provider_label,
    save_registry,
)


def _key_required_hosts() -> frozenset[str]:
    """需要密钥的官方主机：PRESETS 中 needs_key 的地址 + 官方默认端点主机（不按名称猜）。"""
    hosts = {"api.openai.com", "api.anthropic.com"}
    for spec in PRESETS.values():
        if spec.get("needs_key") and spec.get("base_url"):
            host = urlparse(str(spec["base_url"])).hostname
            if host:
                hosts.add(host)
    return frozenset(hosts)


_KEY_REQUIRED_HOSTS = _key_required_hosts()


def _client_base_url(provider: Provider) -> str | None:
    """从 SDK client 反查 base_url（找不到就 None，绝不碰 api_key）。"""
    base = getattr(getattr(provider, "_client", None), "base_url", None)
    return str(base).rstrip("/") if base else None


def _infer_protocol(provider: Provider) -> str:
    name = type(provider).__name__.lower()
    if "codexresponses" in name:
        return "openai-responses"
    if "anthropic" in name:
        return "anthropic"
    if "openai" in name:
        return "openai"
    return ""


class ProviderManager:
    """provider 与角色绑定运行时：注册表/实例/凭据/能力/目录/连接/状态。"""

    _OFFICIAL_HOSTS = {"openai": "api.openai.com", "anthropic": "api.anthropic.com"}

    def __init__(
        self,
        *,
        store_path: str | Path | None,
        keys_path,
        provider_factory: Callable[..., Provider] = make_provider,
        catalog_fetcher: Callable[[], object] | None = None,
        main_provider: Provider | None = None,
        executor_provider: Provider | None = None,
    ) -> None:
        self.main_provider = main_provider
        self.executor_provider = executor_provider
        self._store_path = Path(store_path) if store_path is not None else None
        self._keys_path = keys_path
        self._provider_factory = provider_factory
        self._catalog_fetcher = catalog_fetcher
        self._catalog_data: dict | None = None
        self._providers: dict[str, dict] = {}   # provider_id → 目录/元数据/覆盖
        # (role, provider_id, model) → 实例：main/executor 共用 provider 时互不污染
        self._instances: dict[tuple[str, str, str], Provider] = {}
        # 静态配置（config.toml/env）带入的凭据：仅内存，用于状态与目录判定
        self._runtime_keys: dict[str, str] = {}
        self._roles: dict[str, dict] = {}       # role → {provider_id, model, reasoning_effort}
        self._windows: dict[str, int | None] = {"main": None, "executor": None}
        self._window_sources: dict[str, str] = {"main": "unknown", "executor": "unknown"}
        self._efforts: dict[str, str] = {"main": "off", "executor": "off"}
        self._capabilities: dict[str, dict] = {"main": {}, "executor": {}}
        self._catalog_cache_ttl = 300.0
        migrate_keys(self._keys_path)   # 明文密钥迁入系统钥匙串（不可用则保持文件）
        self._load_registry()
        # 公共目录：启动只读磁盘缓存，绝不外呼；过期/缺失留给按需拉取
        cached = load_catalog(self._catalog_path())
        if cached is not None:
            self._catalog_data = cached
        self._cleanup_legacy_unused_keys()  # 遗留 "unused" 占位不得继续冒充凭据
        self._migrate_legacy_role_keys()    # 旧角色名凭据迁移到 provider_id
        for role in list(self._roles):
            self._activate(role)  # 注册表中的角色绑定必须在启动时生效
        self._adopt_static(main_provider, executor_provider)

    def _load_registry(self) -> None:
        if self._store_path is None:
            return
        migrate_legacy(self._store_path, self._keys_path)
        registry = load_registry(self._store_path)
        self._roles.update(registry.get("roles") or {})
        for pid, entry in (registry.get("providers") or {}).items():
            self._providers[pid] = {
                "name": entry.get("name") or pid,
                "protocol": entry.get("protocol") or "",
                "base_url": entry.get("base_url"),
                "catalog": entry.get("catalog") or "",
                "model_overrides": dict(entry.get("model_overrides") or {}),
                "models": list(entry.get("models") or []),
                "error": None,
                "fetched_at": entry.get("models_fetched_at") or 0.0,
            }


    def _save_registry(self) -> None:
        if self._store_path is None:
            return
        save_registry(self._store_path, {
            "roles": self._roles,
            "providers": {
                pid: {
                    "name": e["name"], "protocol": e["protocol"], "base_url": e["base_url"],
                    "catalog": e.get("catalog") or "",
                    "model_overrides": e["model_overrides"],
                    "models": e["models"],
                    "models_fetched_at": e["fetched_at"],
                }
                for pid, e in self._providers.items()
            },
        })


    def _provider_key(self, pid: str) -> str:
        try:
            return load_keys(self._keys_path).get(pid, "")
        except Exception:
            return ""


    def _endpoint_needs_key(self, entry: dict) -> bool:
        """官方/默认端点必须有密钥；显式本地/自定义端点（含 anthropic）允许占位密钥。"""
        base_url = entry.get("base_url")
        if not base_url:
            return True  # SDK 默认官方端点
        raw = str(base_url).strip()
        if "://" not in raw:
            return True  # 无 scheme 的地址不是明确的本地端点：按官方处理
        try:
            parsed = urlparse(raw)
            host = (parsed.hostname or "").rstrip(".").lower()
            path = parsed.path or ""
        except ValueError:
            return True  # 畸形 URL 不是明确的本地端点：按官方处理
        if host == "chatgpt.com" and "/backend-api/codex" in path:
            return False     # ChatGPT 会员登录（Codex 端点）：凭据走 OAuth，不要 API key
        return host in _KEY_REQUIRED_HOSTS


    def _resolve_key(
        self, pid: str, protocol: str, base_url: str | None, user_key: str = ""
    ) -> str | None:
        """唯一密钥解析入口（连接/启动/状态/目录共用）。

        按来源优先级：用户本次输入 > keys.json > 内存静态 key；每个来源独立归一化，
        任意来源的 "unused" 对需要密钥的端点都视为无凭据。
        返回真实 key / 本地端点占位 "unused" / None（缺少凭据）。
        """
        needs_key = self._endpoint_needs_key({"protocol": protocol, "base_url": base_url})
        for candidate in (user_key, self._provider_key(pid), self._runtime_keys.get(pid)):
            if candidate and candidate != "unused":
                return candidate
        return None if needs_key else "unused"


    def _usable_key(self, pid: str) -> str | None:
        entry = self._providers.get(pid) or {}
        return self._resolve_key(pid, entry.get("protocol") or "", entry.get("base_url"))


    def _cleanup_legacy_unused_keys(self) -> None:
        """启动清理：需要密钥的 provider 的历史 "unused" 占位不再保留（不动真实密钥）。"""
        if not self._providers:
            return
        try:
            keys = load_keys(self._keys_path)
        except Exception:
            return
        for pid in [p for p, v in keys.items() if v == "unused"]:
            entry = self._providers.get(pid)
            if entry is None:
                continue
            if self._endpoint_needs_key(entry):
                delete_key(pid, self._keys_path)


    def _migrate_legacy_role_keys(self) -> None:
        """旧版 keys.json 以角色名（main/executor）存凭据 → 迁移到绑定 provider_id。

        迁移条件：角色键看起来是真实 key（>=20 字符），且 pid 键缺失、为占位
        "unused" 或明显不是 key（过短）。已存在的真实 pid 键（运行时新凭据）优先，
        不覆盖；迁移成功后删除旧角色键，避免下次再被误读。
        """
        for role in ("main", "executor"):
            try:
                keys = load_keys(self._keys_path)  # 每角色重读：上一步迁移已落盘
            except Exception:
                return
            legacy = keys.get(role) or ""
            if legacy == "unused" or len(legacy) < 20:
                continue
            pid = (self._roles.get(role) or {}).get("provider_id")
            if not pid:
                continue
            stored = keys.get(pid) or ""
            if stored and stored != "unused" and len(stored) >= 20:
                if stored == legacy:
                    delete_key(role, self._keys_path)  # 与 pid 同值的重复角色条目：清理
                continue  # 已有可信的运行时凭据
            save_key(pid, legacy, self._keys_path)
            delete_key(role, self._keys_path)


    def _provider_configured(self, pid: str) -> bool:
        return self._usable_key(pid) is not None

    def _provider_has_key(self, pid: str) -> bool:
        """是否存在可删除的显式凭据（本地端点占位密钥不算）。"""
        return self._provider_key(pid) not in ("", "unused") or pid in self._runtime_keys

    def disconnect_provider(self, provider_id: str) -> dict:
        """断开服务商：删除凭据并解绑引用它的角色；条目保留，便于换 key 重连。"""
        pid = str(provider_id or "")
        if not pid:
            return {"type": "provider_result", "ok": False, "message": "缺少 provider_id"}
        if not self._provider_has_key(pid):
            return {"type": "provider_result", "ok": False, "message": "该服务商没有可断开的凭据"}
        entry = self._providers.get(pid) or {}
        needs_key = self._endpoint_needs_key(entry)
        delete_key(pid, self._keys_path)
        self._runtime_keys.pop(pid, None)
        if needs_key:
            # 需要密钥的端点：解绑角色，换 key 后重新连接
            for role, binding in list(self._roles.items()):
                if (binding or {}).get("provider_id") == pid:
                    self._roles[role] = {}
                    self._capabilities[role] = {}
                    self._efforts[role] = "off"
        for key in [k for k in self._instances if k[1] == pid]:
            self._instances.pop(key, None)
        for role in ("main", "executor"):
            self._activate(role)
        self._save_registry()
        return {"type": "provider_result", "ok": True,
                "message": "已断开连接（凭据已删除，可重新连接）"}


    def _role_usable(self, role: str) -> bool:
        """角色当前绑定是否可用（无绑定或绑定 provider 缺凭据 → 可被新连接接管）。"""
        binding = self._roles.get(role)
        if not binding:
            return False
        return self._provider_configured(binding.get("provider_id") or "")


    def _adopt_static(
        self, main_provider: Provider | None, executor_provider: Provider | None
    ) -> None:
        """静态配置 provider：仅在优先链低优先级处兜底。

        凭据/实例优先级（贯穿状态、active provider 与请求体）：
            keys.json 运行时密钥 > 本轮内存注册的密钥 > config.toml/env 静态 provider
            > 本地端点占位密钥
        运行时绑定 + keys.json 凭据存在时，静态 provider 不得覆盖已恢复实例。
        """
        for role, provider in (("main", main_provider), ("executor", executor_provider)):
            if provider is None:
                continue
            protocol = _infer_protocol(provider)
            base_url = _client_base_url(provider)
            model = getattr(provider, "_model", "") or "static-model"
            pid = derive_provider_id(protocol, base_url)
            self._providers.setdefault(pid, {
                "name": provider_label(protocol, base_url) if protocol else "静态配置",
                "protocol": protocol,
                "base_url": base_url, "catalog": "",
                "model_overrides": {}, "models": [],
                "error": None, "fetched_at": 0.0,
            })
            static_key = str(getattr(provider, "_api_key", "") or "")
            if static_key and static_key != "unused":
                self._runtime_keys[pid] = static_key  # 仅内存，永不写入 keys.json
            binding = self._roles.get(role) or {}
            bound_to_static = (
                binding.get("provider_id") == pid and binding.get("model") == model
            )
            if bound_to_static and self._provider_key(pid) not in ("", "unused"):
                self._activate(role)  # 运行时密钥优先：保留已恢复实例
                continue
            if protocol in ("openai", "anthropic"):
                if self._resolve_key(pid, protocol, base_url) is None:
                    self._activate(role)  # 需要密钥却无有效凭据：不得安装不可用实例
                    continue
            if bound_to_static or not binding:
                self._instances[(role, pid, model)] = provider
                self._roles.setdefault(role, {
                    "provider_id": pid, "model": model,
                    "reasoning_effort": str(
                        getattr(provider, "_reasoning_effort", "off") or "off"),
                })
            self._activate(role)


    def _activate(self, role: str) -> None:
        """把角色绑定解析为运行中的 provider 实例并应用能力。"""
        binding = self._roles.get(role) or {}
        pid, model = binding.get("provider_id"), binding.get("model")
        provider = self._instances.get((role, pid, model)) if pid and model else None
        if provider is None and pid and model and pid in self._providers:
            provider = self._build_instance(pid, model, role)
            if provider is not None:
                self._instances[(role, pid, model)] = provider
        if role == "main":
            self.main_provider = provider
        else:
            self.executor_provider = provider
        self._apply_capability(role)


    def _build_instance(self, pid: str, model: str, role: str) -> Provider | None:
        """按角色绑定构建实例；缺凭据返回 None（不得构造空 key 的可用实例）。"""
        if role not in ("main", "executor"):
            return None
        entry = self._providers.get(pid)
        if entry is None or not entry["protocol"] or not model:
            return None
        api_key = self._usable_key(pid)
        if api_key is None:
            return None
        capability = self._effective_capability(pid, model)
        binding = self._roles.get(role) or {}
        return self._provider_factory(
            protocol=entry["protocol"], base_url=entry["base_url"], model=model,
            api_key=api_key, context_window=capability.window,
            reasoning_effort=binding.get("reasoning_effort") or "off",
            echo_reasoning_field=capability.interleaved,
        )


    def _instance_for_listing(self, pid: str) -> Provider:
        """模型目录专用实例：固定 effort=off，绝不继承任一角色的思考强度。"""
        entry = self._providers.get(pid)
        if entry is None:
            raise ModelListingError(f"provider 不存在: {pid}")
        api_key = self._usable_key(pid)
        if api_key is None:
            raise ModelListingError("缺少凭据，无法获取模型列表")
        return self._provider_factory(
            protocol=entry["protocol"], base_url=entry["base_url"], model="__listing__",
            api_key=api_key, context_window=None, reasoning_effort="off",
        )


    def context_window(self, role: str) -> tuple[int | None, str]:
        return self._windows.get(role), self._window_sources.get(role, "unknown")


    def reasoning_effort(self, role: str) -> str:
        return self._efforts.get(role, "off")


    def context_budget(self, role: str) -> dict | None:
        window, _ = self.context_window(role)
        provider = self.main_provider if role == "main" else self.executor_provider
        reserve = 0
        if provider is not None and hasattr(provider, "reasoning_reserve"):
            try:
                reserve = int(provider.reasoning_reserve(self.reasoning_effort(role)))
            except Exception:
                reserve = 0
        return context_budget(window, reserve)


    def _user_capability(self, pid: str, model: str) -> ModelCapability:
        entry = self._providers.get(pid) or {}
        override = (entry.get("model_overrides") or {}).get(model) or {}
        levels = override.get("levels")
        return ModelCapability(
            window=override.get("window"),
            reasoning_returned=override.get("reasoning_returned"),
            reasoning_mode=override.get("reasoning_mode") or "unknown",
            levels=tuple(levels) if isinstance(levels, list) else None,
            max_output=override.get("max_output"),
            tools=override.get("tools"),
            source="user",
        )


    def _entry_model(self, pid: str, model: str) -> dict | None:
        for item in (self._providers.get(pid) or {}).get("models", []):
            if item.get("id") == model:
                return item
        return None


    def _catalog_path(self) -> Path:
        if self._keys_path is not None:
            return Path(self._keys_path).parent / "models-dev.json"
        return keystore.KEYS_DIR / "models-dev.json"


    @staticmethod
    def _host_of(url) -> str:
        if not url:
            return ""
        try:
            return (urlparse(str(url)).hostname or "").rstrip(".").lower()
        except ValueError:
            return ""

    @staticmethod
    def _endpoint_key(url) -> tuple[str, str, int, str] | None:
        """端点指纹：scheme + host + port + 规范化路径（区分本地端口，避免同名主机混淆）。"""
        if not url:
            return None
        try:
            parsed = urlparse(str(url))
        except ValueError:
            return None
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = "/" + (parsed.path or "").strip("/")
        return (parsed.scheme or "https", host, port, path)

    @staticmethod
    def _is_private_host(host: str) -> bool:
        """本地/私网主机：地址不构成稳定身份，不参与目录地址匹配（避免同主机混淆）。"""
        if not host or host == "localhost" or host.endswith(
                (".localhost", ".local", ".internal", ".lan", ".home")):
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return not ip.is_global  # 环回/私网/链路本地/CGNAT(100.64/10)/ULA 全部覆盖

    @staticmethod
    def _paths_match(pa: str, pb: str) -> bool:
        """路径同一性：相等，或仅差一个版本段（/coding 对 /coding/v1；/v1 对 /v1beta 不算）。"""
        if pa == pb:
            return True
        longer, shorter = (pa, pb) if len(pa) > len(pb) else (pb, pa)
        base = shorter.rstrip("/")
        if not longer.startswith(base + "/"):
            return False
        suffix = longer[len(base):].strip("/")
        return suffix.startswith("v") and suffix[1:].isdigit()

    @classmethod
    def _same_endpoint(cls, a, b) -> bool:
        """同一端点：scheme/host/port 相同，且路径仅差至多一个版本段。"""
        ka, kb = cls._endpoint_key(a), cls._endpoint_key(b)
        if ka is None or kb is None:
            return False
        if ka[:3] != kb[:3]:
            return False
        return cls._paths_match(ka[3], kb[3])

    def _preset_for_entry(self, entry: dict) -> dict:
        """按 protocol+base_url 精确或官方主机找到匹配的预设。"""
        protocol = entry.get("protocol") or ""
        base_url = entry.get("base_url")
        host = self._host_of(base_url)
        for spec in PRESETS.values():
            if spec.get("protocol") != protocol:
                continue
            if spec.get("base_url") == base_url:
                return spec
            preset_host = self._host_of(spec.get("base_url")) or self._OFFICIAL_HOSTS.get(
                str(spec.get("protocol") or ""), ""
            )
            if host and preset_host and host == preset_host:
                return spec
        return {}

    def _catalog_candidates(self, entry: dict) -> list[str]:
        """公共目录候选名（按优先级）：显式记录 > 别名 > 预设 > 目录里 api 主机一致者。

        最后一条对改名免疫：models.dev 条目带 api 地址，名字怎么改都能按地址找到。
        """
        names: list[str] = []

        def add(value) -> None:
            for name in ([value] if isinstance(value, str) else list(value or ())):
                name = str(name or "")
                if name and name not in names:
                    names.append(name)

        add(entry.get("catalog"))
        add(entry.get("catalog_aliases"))
        spec = self._preset_for_entry(entry)
        add(spec.get("catalog"))
        add(spec.get("catalog_aliases"))
        base_url = entry.get("base_url")
        if base_url and self._catalog_data and not self._is_private_host(
                self._host_of(base_url)):
            # 公网端点才按目录 api 地址匹配（改名免疫）；本地/私网地址不构成稳定身份
            for pid, provider in (self._catalog_data.get("providers") or {}).items():
                if not isinstance(provider, dict):
                    continue
                if self._same_endpoint(provider.get("api"), base_url):
                    add(pid)
        return names

    def _catalog_name_for(self, entry: dict) -> str:
        """首个在（已加载的）公共目录里存在的候选名；无数据时返回首个候选（判映射有无）。"""
        names = self._catalog_candidates(entry)
        if not names:
            return ""
        if self._catalog_data:
            for name in names:
                if catalog_models(self._catalog_data, name):
                    return name
        return names[0]

    def _any_catalog_mapped(self) -> bool:
        return any(self._catalog_name_for(e) for e in self._providers.values())


    async def _ensure_catalog(self) -> None:
        """按需加载公共目录：只给映射到目录的 provider 用；缓存新鲜直接用，失败静默。"""
        if self._catalog_data is None:
            cached = load_catalog(self._catalog_path())
            if cached is not None:
                self._catalog_data = cached
        if not self._any_catalog_mapped():
            return
        if self._catalog_data is None or catalog_stale(self._catalog_path()):
            fetcher = self._catalog_fetcher or fetch_catalog
            try:
                providers = await fetcher()
            except Exception:
                return  # 拉取失败：不报错、不污染，目录保持未知
            if isinstance(providers, dict) and providers:
                save_catalog(self._catalog_path(), providers)
                self._catalog_data = {"providers": providers}


    def _catalog_capability(self, pid: str, model: str) -> ModelCapability | None:
        """公共目录能力：逐个候选找"真包含该模型"的条目（避免首个命中遮蔽正确项）。"""
        entry = self._providers.get(pid) or {}
        if self._catalog_data is None:
            return None
        protocol = entry.get("protocol") or ""
        for name in self._catalog_candidates(entry):
            item = catalog_models(self._catalog_data, name).get(model)
            if item is not None:
                return capability_from_catalog(item, protocol)
        return None


    def _effective_capability(self, pid: str, model: str) -> ModelCapability:
        item = self._entry_model(pid, model)
        metadata: ModelCapability | None = None
        if item is not None:
            raw = item.get("capability") or {}
            levels = raw.get("levels")
            metadata = ModelCapability(
                window=raw.get("window"),
                reasoning_returned=raw.get("reasoning_returned"),
                reasoning_mode=raw.get("reasoning_mode") or "unknown",
                levels=tuple(levels) if isinstance(levels, list) else None,
                max_output=raw.get("max_output"),
                tools=raw.get("tools"),
                source="provider",
            )
        user = self._user_capability(pid, model)
        catalog = self._catalog_capability(pid, model)
        merged = merge_capability(merge_capability(catalog, metadata), user)
        source = "unknown"
        if _capability_gives(user):
            source = "user"
        elif _capability_gives(metadata):
            source = "provider"
        elif _capability_gives(catalog):
            source = "catalog"
        return dataclasses.replace(merged, source=source)


    def _apply_capability(self, role: str) -> None:
        binding = self._roles.get(role) or {}
        pid, model = binding.get("provider_id"), binding.get("model")
        capability = self._effective_capability(pid, model) if pid and model else ModelCapability()
        self._windows[role] = capability.window
        self._window_sources[role] = capability.source
        self._capabilities[role] = capability.as_dict()
        levels = capability.levels or ()
        effort = str(binding.get("reasoning_effort") or "off")
        if capability.reasoning_mode == "adjustable" and effort in levels:
            self._efforts[role] = effort
        else:
            self._efforts[role] = "off"
        provider = self.main_provider if role == "main" else self.executor_provider
        if provider is not None:
            provider.context_window = capability.window
            provider._reasoning_effort = (
                self._efforts[role] if capability.reasoning_mode == "adjustable" else None
            )
            # 目录声明 interleaved 时补齐实例的回传字段；已有值（显式配置/自愈所得）不覆盖
            if capability.interleaved and not getattr(provider, "_echo_reasoning_field", None):
                provider._echo_reasoning_field = capability.interleaved
            if hasattr(provider, "reasoning_reserve"):
                provider._thinking_budget = provider.reasoning_reserve(self._efforts[role])


    async def list_models(self, provider_id: str | None = None, refresh: bool = False) -> dict:
        """聚合所有已配置 provider 的模型目录；单点失败不影响其它 provider。

        成功刷新后集中持久化一次（原始 provider 元数据 + fetched_at）；
        失败只记录错误，保留最后一次成功的目录缓存。
        """
        now = time.time()
        targets = [provider_id] if provider_id else list(self._providers)
        changed = False
        for pid in targets:
            entry = self._providers.get(pid)
            if entry is None:
                continue
            if (not refresh and entry["models"]
                    and now - entry["fetched_at"] < self._catalog_cache_ttl):
                continue  # 命中有效缓存：不请求、不写盘
            try:
                provider = self._instance_for_listing(pid)
                raw_models = await provider.list_models()
            except Exception as e:
                entry["error"] = f"无法获取模型列表: {type(e).__name__}: {e}"
                continue  # 保留旧 models/fetched_at（内存与磁盘都不清）
            if not raw_models:
                entry["error"] = "无法获取模型列表: provider 返回空列表"
                continue  # 同上：不覆盖最后一次成功目录
            entry["models"] = [
                {
                    "id": str(raw.get("id") or ""),
                    "display_name": raw.get("display_name") or str(raw.get("id") or ""),
                    "provider_id": pid,
                    "provider": entry["name"],
                    "capability": raw.get("capability") or {},  # 只存 provider 原始元数据
                }
                for raw in raw_models if raw.get("id")
            ]
            entry["error"] = None
            entry["fetched_at"] = now
            changed = True
        if changed:
            await self._ensure_catalog()  # 目录有更新才考虑公共目录补齐
            self._save_registry()  # 一次刷新流程只写盘一次
        for role in ("main", "executor"):
            if self._roles.get(role):
                self._apply_capability(role)
        return self.model_catalog()


    def model_catalog(self) -> dict:
        providers = []
        for pid, entry in self._providers.items():
            status = "error" if entry["error"] else ("ok" if entry["models"] else "empty")
            providers.append({
                "id": pid,
                "name": entry["name"],
                "protocol": entry["protocol"],
                "base_url": entry["base_url"],
                "status": status,
                "error": entry["error"],
                "models": [
                    {**m, "capability": self._effective_capability(pid, m["id"]).as_dict()}
                    for m in entry["models"]
                ],
                "fetched_at": entry["fetched_at"] or None,
                "configured": self._provider_configured(pid),
                "overrides": entry["model_overrides"],
            })
        return {"type": "model_catalog", "providers": providers}


    def set_model_capability(self, provider_id: str, model: str, override: dict) -> dict:
        """用户能力覆盖（绑定 provider_id+model_id）；None 值清除该字段。"""
        entry = self._providers.get(provider_id)
        if entry is None or not model:
            return {"type": "provider_result", "role": "", "ok": False,
                    "message": "provider 或模型不存在"}
        cleaned = clean_override(override)
        stored = dict(entry["model_overrides"].get(model) or {})
        clearing = {k for k, v in (override or {}).items() if v is None}
        for key in clearing:
            stored.pop(key, None)
        stored.update(cleaned)
        if stored:
            entry["model_overrides"][model] = stored
        else:
            entry["model_overrides"].pop(model, None)
        self._save_registry()
        for role in ("main", "executor"):
            binding = self._roles.get(role) or {}
            if binding.get("provider_id") == provider_id and binding.get("model") == model:
                self._apply_capability(role)
        return {"type": "provider_result", "role": "", "ok": True,
                "message": f"已更新 {model} 的能力设置"}


    def set_context_window(self, role: str, window: int | None) -> dict:
        """用户显式覆盖窗口（None 清除）；写入 provider_id+model 覆盖。"""
        binding = self._roles.get(role)
        if role not in ("main", "executor") or not binding:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "该角色尚未绑定 provider"}
        if window is not None and (not isinstance(window, int) or window <= 0):
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "窗口必须是正整数"}
        result = self.set_model_capability(
            binding["provider_id"], binding["model"], {"window": window}
        )
        result["role"] = role
        self._apply_capability(role)
        return result


    async def set_model(self, role: str, provider_id: str, model: str) -> dict:
        """切换角色模型：provider 已配置则复用实例与凭据，不重输 key。"""
        if role not in ("main", "executor"):
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"未知角色: {role}"}
        if not model or provider_id not in self._providers:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "provider 或模型不存在"}
        instance = self._instances.get((role, provider_id, model))
        if instance is None:
            instance = self._build_instance(provider_id, model, role)
            if instance is None:
                return {"type": "provider_result", "role": role, "ok": False,
                        "message": "该 provider 缺少凭据，请先在设置中连接"}
            try:
                await instance.chat([Message(role="user", content="ping")])
            except Exception as e:
                return {"type": "provider_result", "role": role, "ok": False,
                        "message": f"模型切换失败（未保存）: {type(e).__name__}: {e}"}
            self._instances[(role, provider_id, model)] = instance
        old_effort = self.reasoning_effort(role)
        self._roles[role] = {
            "provider_id": provider_id, "model": model, "reasoning_effort": old_effort,
        }
        self._save_registry()
        self._activate(role)
        note = ""
        if old_effort != "off" and self.reasoning_effort(role) == "off":
            note = f"（原思考强度 {old_effort} 不被该模型支持，已关闭）"
        return {"type": "provider_result", "role": role, "ok": True,
                "message": f"模型已切换为 {model}{note}"}


    def set_reasoning_effort(self, role: str, effort: str) -> dict:
        if role not in ("main", "executor"):
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"未知角色: {role}"}
        provider = self.main_provider if role == "main" else self.executor_provider
        if provider is None or not self._roles.get(role):
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "该角色尚未配置 provider，请先在设置中连接"}
        capability = self._capabilities.get(role) or {}
        mode = capability.get("reasoning_mode", "unknown")
        levels_raw = capability.get("levels")
        if mode == "fixed":
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "该模型思考固定开启，不支持调节强度"}
        if mode == "none":
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "该模型不支持推理"}
        if mode != "adjustable":
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "该模型能力未知，可在模型能力设置中声明后使用"}
        if levels_raw is None or len(levels_raw) == 0:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "该模型可调但档位未知或为空，请在模型能力设置中声明支持的档位"}
        levels = tuple(levels_raw)
        if effort not in levels:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"该模型不支持思考强度 {effort}（支持: {', '.join(levels)}）"}
        if effort != "off" and hasattr(provider, "reasoning_reserve"):
            reserve = provider.reasoning_reserve(effort)
            if not isinstance(reserve, int) or reserve <= 0:
                return {"type": "provider_result", "role": role, "ok": False,
                        "message": (f"该 provider 的协议不识别思考强度 {effort}，"
                                    "请在模型能力设置中只声明其支持的档位")}
        self._roles[role]["reasoning_effort"] = effort
        self._save_registry()
        try:
            provider._reasoning_effort = effort
            if hasattr(provider, "reasoning_reserve"):
                provider._thinking_budget = provider.reasoning_reserve(effort)
        except Exception as e:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"思考强度设置失败: {type(e).__name__}: {e}"}
        self._efforts[role] = effort
        return {"type": "provider_result", "role": role, "ok": True,
                "message": f"思考强度已设为 {effort}"}


    def _presets_status(self) -> dict:
        """预设服务商 → UI 行（label/是否已连接/对应 provider_id），凭据全局共享。"""
        out: dict[str, dict] = {}
        for name, spec in PRESETS.items():
            spec_pid = derive_provider_id(str(spec["protocol"]), spec.get("base_url"))
            out[name] = {**spec, "provider_id": spec_pid,
                         "configured": self._provider_configured(spec_pid)}
        return out


    def _role_status(self, role: str) -> dict:
        window, window_source = self.context_window(role)
        capability = self._capabilities.get(role) or {}
        binding = self._roles.get(role) or {}
        provider = self.main_provider if role == "main" else self.executor_provider
        pid = binding.get("provider_id") or ""
        entry = self._providers.get(pid) or {}
        model = binding.get("model") or ""
        levels = capability.get("levels")
        return {
            "configured": provider is not None and bool(model),
            "provider_id": pid,
            "provider": entry.get("name") or "",
            "protocol": entry.get("protocol") or "",
            "base_url": entry.get("base_url"),
            "model": model,
            "window": window,
            "window_source": window_source,
            "effort": self.reasoning_effort(role),
            "efforts": list(levels) if levels is not None else None,
            "reasoning_mode": capability.get("reasoning_mode", "unknown"),
            "reasoning_returned": capability.get("reasoning_returned"),
            "max_output": capability.get("max_output"),
            "tools": capability.get("tools"),
        }


    async def connect_provider(
        self,
        role: str,
        *,
        protocol: str,
        base_url: str | None,
        model: str,
        api_key: str,
        preset: str = "",
        reasoning_effort: str = "off",
    ) -> dict:
        """连接流程：校验凭据 → 拉真实目录（定默认模型）→ 能力校验 → ping → 保存并绑定角色。

        role="" 表示服务商级连接：凭据全局共享，为尚无绑定的角色自动选默认模型；
        role=main/executor 表示只为该角色连接。不支持的思考强度必须明确报错，禁止静默降为 off。
        """
        if role not in ("", "main", "executor"):
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"未知角色: {role}（可选: 空 / main / executor）"}
        if protocol not in ("openai", "anthropic", "openai-responses"):
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "协议无效"}
        pid = derive_provider_id(protocol, base_url)
        effective_key = self._resolve_key(pid, protocol, base_url, api_key)
        if effective_key is None:
            message = "缺少 API key（官方端点需要凭据；本地/自定义端点请显式填写 base_url）"
            return {"type": "provider_result", "role": role, "ok": False, "message": message}

        preset_model = str((PRESETS.get(preset) or {}).get("model") or "")
        requested = model.strip()
        entry = self._providers.setdefault(pid, {
            "name": provider_label(protocol, base_url), "protocol": protocol,
            "base_url": base_url, "model_overrides": {}, "models": [],
            "error": None, "fetched_at": 0.0,
        })
        preset_spec = PRESETS.get(preset)
        if preset_spec and preset_spec.get("catalog") and not entry.get("catalog"):
            entry["catalog"] = preset_spec["catalog"]
        try:
            candidate = self._provider_factory(
                protocol=protocol, base_url=base_url,
                model=requested or preset_model or "__listing__",
                api_key=effective_key, context_window=None, reasoning_effort="off",
            )
        except Exception as e:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"配置无效（未保存）: {e}"}
        # 先拉真实模型目录（不依赖 model）：失败不阻塞已有模型的连接，但必须可见
        try:
            raw_models = await candidate.list_models()
            listing_error = None
        except Exception as e:
            raw_models = []
            listing_error = f"无法获取模型列表: {type(e).__name__}: {e}"
        if raw_models:
            await self._ensure_catalog()
            entry["models"] = [
                {
                    "id": str(raw.get("id") or ""),
                    "display_name": raw.get("display_name") or str(raw.get("id") or ""),
                    "provider_id": pid,
                    "provider": entry["name"],
                    "capability": raw.get("capability") or {},  # 只存 provider 原始元数据
                }
                for raw in raw_models if raw.get("id")
            ]
            entry["error"] = None
            entry["fetched_at"] = time.time()
        else:
            # 重连时目录拉取失败不得清空并持久化最后一次成功目录
            entry["error"] = listing_error or "无法获取模型列表: provider 返回空列表"

        listed = [item["id"] for item in entry["models"]]
        effective_model = requested
        if not effective_model:
            effective_model = preset_model if preset_model in listed else (
                listed[0] if listed else preset_model)
        if not effective_model:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": "未能获取模型列表，无法确定使用的模型；可在高级设置中手动指定模型"}

        capability = self._effective_capability(pid, effective_model)
        if reasoning_effort != "off":
            if capability.reasoning_mode != "adjustable":
                return {"type": "provider_result", "role": role, "ok": False,
                        "message": ("该模型的思考能力未知或固定，无法设置思考强度；"
                                    "可在模型能力设置中声明后重试")}
            if reasoning_effort not in (capability.levels or ()):
                levels = ", ".join(capability.levels or ()) or "无"
                return {"type": "provider_result", "role": role, "ok": False,
                        "message": f"该模型不支持思考强度 {reasoning_effort}（支持: {levels}）"}

        # 能力校验通过：用最终参数重建候选（factory 收到正确 effort），ping 验证后保存
        try:
            candidate = self._provider_factory(
                protocol=protocol, base_url=base_url, model=effective_model,
                api_key=effective_key, context_window=capability.window,
                reasoning_effort=(
                    reasoning_effort if capability.reasoning_mode == "adjustable" else "off"
                ),
                echo_reasoning_field=capability.interleaved,
            )
        except Exception as e:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"配置无效（未保存）: {e}"}
        candidate._reasoning_effort = (
            reasoning_effort if capability.reasoning_mode == "adjustable" else None
        )
        if reasoning_effort != "off" and hasattr(candidate, "reasoning_reserve"):
            reserve = candidate.reasoning_reserve(reasoning_effort)
            if not isinstance(reserve, int) or reserve <= 0:
                return {"type": "provider_result", "role": role, "ok": False,
                        "message": (f"该 provider 的协议不识别思考强度 {reasoning_effort}，"
                                    "请在模型能力设置中只声明其支持的档位")}
        if hasattr(candidate, "reasoning_reserve"):
            candidate._thinking_budget = candidate.reasoning_reserve(reasoning_effort)
        try:
            await candidate.chat([Message(role="user", content="ping")])
        except Exception as e:
            return {"type": "provider_result", "role": role, "ok": False,
                    "message": f"连接失败（未保存）: {type(e).__name__}: {e}"}

        unbound = [r for r in ("main", "executor") if not self._role_usable(r)]
        targets = [role] if role else unbound
        for target in targets:
            self._roles[target] = {
                "provider_id": pid, "model": effective_model,
                "reasoning_effort": reasoning_effort,
            }
            self._instances[(target, pid, effective_model)] = candidate
        if not role:
            # 已绑定到该 provider 的角色：丢弃旧凭据实例，稍后用新凭据重建
            for r, binding in self._roles.items():
                if binding.get("provider_id") == pid and r not in targets:
                    self._instances.pop((r, pid, binding.get("model")), None)
        self._save_registry()
        if effective_key and effective_key != "unused":
            save_key(pid, effective_key, self._keys_path)  # 密钥只落 keys.json；占位不落盘
        for r, binding in self._roles.items():
            if binding.get("provider_id") == pid:
                self._activate(r)
        if role:
            message = "连接成功，已切换并保存"
        elif targets:
            message = f"连接成功，已为 {', '.join(targets)} 选择默认模型 {effective_model}"
        else:
            message = "连接成功，已保存（角色模型未变）"
        return {"type": "provider_result", "role": targets[0] if targets else "",
                "ok": True, "message": message, "model": effective_model}

