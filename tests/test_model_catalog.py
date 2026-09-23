"""预算数学与能力合并测试：无名称猜测、未知即未知、用户覆盖优先。"""

import agent.core.model_catalog as catalog
from agent.core.model_catalog import context_budget, next_output_reserve
from agent.providers.base import ModelCapability
from agent.providers.listing import merge_capability


def test_no_name_based_lookup_api_remains() -> None:
    """禁止再按模型名称子串猜能力：旧 API 必须已删除。"""
    assert not hasattr(catalog, "lookup_model")
    assert not hasattr(catalog, "resolve_window")
    assert not hasattr(catalog, "reasoning_efforts")


def test_context_budget_unknown_window_is_none() -> None:
    """窗口未知 → 不计算预算（不编造阈值、不主动压缩）。"""
    assert context_budget(None) is None
    assert context_budget(0) is None


def test_context_budget_math_with_reasoning_reserve() -> None:
    budget = context_budget(100_000, 16_384)
    assert budget["reasoning"] == 16_384
    assert budget["next_output"] == 4_096
    assert budget["safety"] == 5_000
    assert budget["compact_at"] == min(85_000, 100_000 - (4_096 + 16_384 + 5_000))
    assert context_budget(100_000, 0)["compact_at"] > budget["compact_at"]


def test_next_output_reserve() -> None:
    assert next_output_reserve(100_000) == 4_096
    assert next_output_reserve(8_000) == 1_024


def test_protocol_never_grants_reasoning_capability() -> None:
    """协议名不得赋予任何思考能力（含 Anthropic 兼容）。"""
    merged = merge_capability(None, None)
    assert merged.reasoning_mode == "unknown"
    assert merged.levels is None
    assert merged.window is None
    assert merged.source == "unknown"


def test_merge_boolean_and_window_edge_cases() -> None:
    """False 必须保留；window<=0 无效；levels None（未知）与 ()（明确为空）不混淆。"""
    meta = ModelCapability(
        window=0, tools=False, reasoning_returned=False,
        reasoning_mode="none", levels=(), source="provider",
    )
    merged = merge_capability(meta, None)
    assert merged.window is None  # 0 不是合法窗口
    assert merged.tools is False
    assert merged.reasoning_returned is False
    assert merged.levels == ()  # 明确为空 ≠ 未知

    unknown_levels = merge_capability(
        ModelCapability(reasoning_mode="adjustable", levels=None, source="provider"), None
    )
    assert unknown_levels.reasoning_mode == "adjustable"
    assert unknown_levels.levels is None  # 可调但档位未知

    fixed = merge_capability(
        ModelCapability(reasoning_returned=True, reasoning_mode="fixed", levels=None,
                        source="provider"), None
    )
    assert fixed.reasoning_mode == "fixed" and fixed.levels == ()

    # 用户 False 覆盖 provider True
    user = ModelCapability(tools=False, source="user")
    both = merge_capability(ModelCapability(tools=True, source="provider"), user)
    assert both.tools is False and both.source == "user"


def test_merge_precedence_metadata_then_user() -> None:
    meta = ModelCapability(
        window=123_000, reasoning_returned=True, reasoning_mode="fixed",
        levels=(), source="provider",
    )
    merged = merge_capability(meta, None)
    assert merged.window == 123_000
    assert merged.reasoning_mode == "fixed"
    assert merged.source == "provider"

    user = ModelCapability(window=777_000, source="user")
    merged_user = merge_capability(meta, user)
    assert merged_user.window == 777_000  # 用户显式覆盖
    assert merged_user.source == "user"
