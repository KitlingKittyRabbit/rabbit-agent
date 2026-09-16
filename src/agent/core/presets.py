"""服务商预设表：纯数据，随时可增删。model 为默认值，UI 只展示 label + 连接。"""

PRESETS: dict[str, dict] = {
    "kimi-coding": {
        "label": "Kimi Coding",
        "protocol": "anthropic",
        "base_url": "https://api.kimi.com/coding/",
        "model": "kimi-for-coding",
        "needs_key": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "protocol": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "needs_key": True,
    },
    "glm": {
        "label": "智谱 GLM",
        "protocol": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.6",
        "needs_key": True,
    },
    "qwen": {
        "label": "通义千问",
        "protocol": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "needs_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "protocol": "openai",
        "base_url": None,
        "model": "gpt-5",
        "needs_key": True,
    },
    "anthropic": {
        "label": "Anthropic",
        "protocol": "anthropic",
        "base_url": None,
        "model": "claude-sonnet-4-5",
        "needs_key": True,
    },
    "ollama": {
        "label": "Ollama（本地）",
        "protocol": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3",
        "needs_key": False,
    },
}
