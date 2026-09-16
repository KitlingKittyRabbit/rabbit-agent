"""provider 层：统一协议接口、两种协议适配与 fake。"""

from .anthropic_compat import AnthropicCompatProvider
from .base import (
    AuthError,
    ChatResult,
    ContextOverflowError,
    Message,
    ModelCapability,
    OnReasoning,
    OnText,
    Provider,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolSpec,
    Usage,
)
from .fake import FakeProvider
from .openai_compat import OpenAICompatProvider

__all__ = [
    "AnthropicCompatProvider",
    "AuthError",
    "ChatResult",
    "ContextOverflowError",
    "FakeProvider",
    "Message",
    "ModelCapability",
    "OnReasoning",
    "OnText",
    "OpenAICompatProvider",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
