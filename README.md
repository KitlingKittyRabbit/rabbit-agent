# agent

## 需求定稿

**定位**：自用 agent 软件，后端先行，核心是"判断—执行分离"的 orchestrator

**场景**：
1. 主力编程助手
2. 后续姥姥聊天机器人（复用基础设施，单 agent 无写路径）

**角色**：
- 主 agent（顶尖模型）：读全局、规划、评价输出；只读 + 派发，物理无写工具
- 执行 subagent（模型可配置，默认本地）：按提示词写，自己跑命令验证，有最大步数上限
- 探索 subagent：开放探索返回摘要

**交互模型**：主 agent 用提示词派活，subagent 输出文本；主 agent 可用读工具自行核实

**派发不阻塞**：call_subagent 立即返回任务号，subagent 后台运行，完成后输出作为事件回注主 agent 输入队列；主 agent 等待期间可继续交互、继续派发（并行是其自然副产品）

**纪律机制**：项目纪律（AGENTS.md）进提示词；plan 模式开关使全局只读（写与 shell 同时摘除）；介入方式由项目定

**上下文管理**：默认摘要压缩（B）——主上下文接近阈值时把旧历史压成 [前情摘要] 替换，只在 user 边界切割；兜底截断重试（A）——撞 ContextOverflowError 砍半重试 ≤2 次，subagent 仅享兜底

**多会话**：会话注册表（id → 独立消息历史/驱动循环/任务表）；事件按 session_id 广播路由，多客户端各自订阅互不偷；SQLite 分会话持久化；CLI `/new`、`/sessions`、`/switch`

**web 前端**：三栏 IDE 形态——左栏项目→会话树（多项目 `.projects.toml`），中栏对话流（markdown + 内联任务/提问/确认卡），右栏文件|任务页签（只读文件查看、subagent 实时步骤流）；浏览器开 `http://127.0.0.1:8000`

**澄清通道**：subagent 遇规格歧义用 ask 工具回问（限时 300 秒），主 agent 用 answer_task 回答

**运行与安全**：CLI 未检测到 server 时自动拉起；/stop 中断当前会话的 turn 与全部 subagent；危险 shell 命令需用户确认（超时自动拒）；写操作与 shell 落审计日志（JSONL）；token 用量与上下文水位随 turn 上报

**provider 层**：角色→provider 可配置；base_url + key + 协议三要素；覆盖 coding plan / 按量 API / Ollama；`/connect_provider` 向导热切换不重启，状态存 .providers.toml；`uv run python -m agent.smoke` 真实连接体检（手动触发）

**技术栈**：Python + FastAPI + SQLite，本地优先部署，第一个客户端为 CLI

**MVP 切口**：一个主 agent + 一个执行 subagent + CLI，跑通"提示→主 agent 读/派→subagent 写→输出"闭环，含 plan 开关；探索 subagent 第二期

**暂缓**：长期记忆、MCP、姥姥场景细化、探索 subagent、并行上限与任务面板、repo map、TUI/桌面前端
