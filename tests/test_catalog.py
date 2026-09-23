"""models.dev 公共目录：解析/映射/缓存/TTL（零网络，夹具数据）。"""

import json
import time

import pytest

from agent.providers.catalog import (
    capability_from_catalog,
    catalog_models,
    catalog_stale,
    fetch_catalog,
    load_catalog,
    save_catalog,
)

FIXTURE = {
    "deepseek": {"models": {
        "deepseek-flash": {
            "name": "DeepSeek V4.1 Flash", "reasoning": True, "tool_call": True,
            "reasoning_options": [{"type": "toggle"},
                                  {"type": "effort", "values": ["low", "high", "max"]}],
            "interleaved": {"field": "reasoning_content"},
            "limit": {"context": 1_000_000, "output": 384_000},
        },
        "deepseek-plain": {"reasoning": False, "tool_call": True,
                           "limit": {"context": 64_000, "output": 8_000}},
    }},
    "anthropic": {"models": {
        "claude-x": {"reasoning": True, "tool_call": True,
                     "reasoning_options": [{"type": "budget_tokens", "min": 1024}],
                     "limit": {"context": 400_000, "output": 64_000}},
    }},
    "zai": {"models": {
        "glm-x": {"reasoning": True, "tool_call": True,
                  "reasoning_options": [{"type": "toggle"}],
                  "limit": {"context": 200_000}},
    }},
    "moonshotai": {"models": {
        "kimi-x": {"reasoning": True, "tool_call": True, "reasoning_options": []},
    }},
}


def test_capability_mapping_toggle_and_effort():
    cap = capability_from_catalog(FIXTURE["deepseek"]["models"]["deepseek-flash"], "openai")
    assert cap.window == 1_000_000 and cap.max_output == 384_000
    assert cap.reasoning_mode == "adjustable"
    assert cap.levels == ("off", "low", "high", "max")  # toggle 补 off
    assert cap.tools is True and cap.reasoning_returned is True
    assert cap.interleaved == "reasoning_content"  # 回传字段名必须保留
    assert cap.source == "catalog"


def test_capability_interleaved_absent_stays_none():
    cap = capability_from_catalog(FIXTURE["zai"]["models"]["glm-x"], "openai")
    assert cap.interleaved is None


def test_capability_as_dict_includes_interleaved():
    from agent.providers import ModelCapability

    assert ModelCapability(interleaved="reasoning_content").as_dict()["interleaved"] == (
        "reasoning_content"
    )
    assert ModelCapability().as_dict()["interleaved"] is None


def test_capability_mapping_reasoning_none_and_toggle_only_and_budget():
    none = capability_from_catalog(FIXTURE["deepseek"]["models"]["deepseek-plain"], "openai")
    assert none.reasoning_mode == "none" and none.levels == ()
    fixed = capability_from_catalog(FIXTURE["zai"]["models"]["glm-x"], "openai")
    assert fixed.reasoning_mode == "fixed" and fixed.levels == ()
    budget = capability_from_catalog(FIXTURE["anthropic"]["models"]["claude-x"], "anthropic")
    assert budget.reasoning_mode == "adjustable"
    assert budget.levels == ("off", "low", "medium", "high")
    assert budget.reasoning_returned is True  # anthropic 协议返回 reasoning blocks


def test_capability_mapping_no_options_means_fixed():
    fixed = capability_from_catalog(FIXTURE["moonshotai"]["models"]["kimi-x"], "openai")
    assert fixed.reasoning_mode == "fixed" and fixed.levels == ()
    assert fixed.window is None  # 缺 limit 保持未知，不猜测


def test_capability_mapping_missing_fields_stay_none():
    cap = capability_from_catalog({"reasoning": False}, "openai")
    assert cap.window is None and cap.max_output is None and cap.tools is None


def test_catalog_cache_roundtrip_and_stale(tmp_path):
    path = tmp_path / "models-dev.json"
    assert load_catalog(path) is None
    save_catalog(path, FIXTURE)
    loaded = load_catalog(path)
    assert loaded is not None and loaded["fetched_at"] > 0
    assert catalog_models(loaded, "deepseek")["deepseek-flash"]["limit"]["context"] == 1_000_000
    assert catalog_models(loaded, "不存在") == {}
    assert not catalog_stale(path)
    # 过期缓存触发重拉
    loaded["fetched_at"] = time.time() - 8 * 24 * 3600
    path.write_text(json.dumps(loaded), encoding="utf-8")
    assert catalog_stale(path)
    # 损坏文件当缺失处理
    path.write_text("not json", encoding="utf-8")
    assert load_catalog(path) is None and catalog_stale(path)


async def test_fetch_catalog_with_fake_client():
    class Resp:
        status_code = 200

        def json(self):
            return FIXTURE

    class Client:
        async def get(self, url):
            assert url.endswith("models.dev/api.json")
            return Resp()

        async def aclose(self):
            pass

    data = await fetch_catalog(client=Client())
    assert "deepseek" in data

    class Bad:
        async def get(self, url):
            return type("R", (), {"status_code": 500})()

        async def aclose(self):
            pass

    with pytest.raises(ValueError):
        await fetch_catalog(client=Bad())


def test_anthropic_effort_values_never_leak_unknown_levels():
    """漏洞 A 回归：anthropic 协议的目录 effort 词汇（如 max）必须映射到预算已知档。"""
    entry = {"reasoning": True, "tool_call": True,
             "reasoning_options": [{"type": "toggle"},
                                   {"type": "effort", "values": ["low", "high", "max"]}],
             "limit": {"context": 200_000}}
    cap = capability_from_catalog(entry, "anthropic")
    assert cap.reasoning_mode == "adjustable"
    assert cap.levels == ("off", "low", "medium", "high")  # max 不得出现
    # 对照：openai 协议原样透传 max
    cap_o = capability_from_catalog(entry, "openai")
    assert cap_o.levels == ("off", "low", "high", "max")
