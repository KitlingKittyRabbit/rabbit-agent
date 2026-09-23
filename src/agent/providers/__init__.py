"""provider 层：统一协议接口、协议适配（openai/anthropic/codex responses）与 fake。"""

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
from .codex_auth import (
    CodexAuthError,
    load_tokens,
)
from .codex_auth import (
    delete_tokens as codex_logout,
)
from .codex_auth import (
    login as codex_login,
)
from .fake import FakeProvider
from .openai_compat import OpenAICompatProvider
from .openai_responses import CodexResponsesProvider

__all__ = [
    "AnthropicCompatProvider",
    "CodexAuthError",
    "CodexResponsesProvider",
    "codex_login",
    "codex_logout",
    "load_tokens",
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
