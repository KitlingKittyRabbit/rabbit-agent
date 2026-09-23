"""服务商预设表：纯数据，随时可增删。model 为默认值，UI 只展示 label + 连接。"""

PRESETS: dict[str, dict] = {
    "kimi-coding": {
        "label": "Kimi Coding",
        "catalog": "kimi-code-plan-cn",
        "catalog_aliases": ("kimi-for-coding",),
        "protocol": "anthropic",
        "base_url": "https://api.kimi.com/coding/",
        "model": "kimi-for-coding",
        "needs_key": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "catalog": "deepseek",
        "protocol": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "needs_key": True,
    },
    "glm": {
        "label": "智谱 GLM",
        "catalog": "zai",
        "protocol": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.6",
        "needs_key": True,
    },
    "qwen": {
        "label": "通义千问",
        "catalog": "alibaba-cn",
        "protocol": "openai",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "needs_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "catalog": "openai",
        "protocol": "openai",
        "base_url": None,
        "model": "gpt-5",
        "needs_key": True,
    },
    "anthropic": {
        "label": "Anthropic",
        "catalog": "anthropic",
        "protocol": "anthropic",
        "base_url": None,
        "model": "claude-sonnet-4-5",
        "needs_key": True,
    },
    "opencode-go": {
        "label": "OpenCode Go",
        "protocol": "openai",
        "base_url": "https://opencode.ai/zen/go/v1",
        "model": "deepseek-v4.1-flash",
        "needs_key": True,
    },
    "chatgpt": {
        "label": "ChatGPT（会员登录）",
        "protocol": "openai-responses",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "model": "",           # 留空：连接时自动选本机 Codex 缓存里的第一个可用模型
        "needs_key": False,
        "login": "codex",
        "hint": "非官方接口（与 opencode 同路），可能随 OpenAI 变更失效，请自行评估账号风险；"
                "未装系统钥匙串时令牌会以明文存本机 keys.json（600）",
    },
    "ollama": {
        "label": "Ollama（本地）",
        "protocol": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3",
        "needs_key": False,
    },
}
