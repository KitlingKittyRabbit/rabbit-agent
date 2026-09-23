/* 时间线渲染的纯判定逻辑（零 DOM），浏览器与 node --test 共用。 */

export const WORK_EVENT_TYPES = [
  "activity_text_delta",
  "tool_started",
  "subagent_queued",
  "subagent_started",
  "subagent_text_delta",
  "subagent_tool_started",
  "subagent_tool_finished",
  "subagent_completed",
  "subagent_failed",
];

export function hasWork(events) {
  return (events || []).some((e) => WORK_EVENT_TYPES.includes(e.type));
}

export function isActionEvent(type) {
  return type === "tool_started" || type === "subagent_tool_started";
}

export function countActions(events) {
  return (events || []).filter((e) => isActionEvent(e.type)).length;
}

export function routeActionEvent(turnsById, ev) {
  // 按事件自带的 turn_id 路由：只更新所属 turn；finished/文本等非动作事件不计
  if (!isActionEvent(ev.type)) return null;
  const turn = (turnsById || {})[ev.turn_id];
  if (!turn) return null;
  turn.actions += 1;
  return turn;
}

export function isSystemTurn(turn) {
  return !turn.user_message && !hasWork(turn.events);
}

export function workingCounts(events) {
  const list = events || [];
  // 只数真实动作：主工具 + 子代理工具；finished/completed/文本等不重复计数
  const actions = countActions(list);
  return { actions, tasks: subagentTaskCount(list), total: actions };
}

export function subagentTaskCount(events) {
  const ids = new Set(
    (events || [])
      .filter((e) => e.type.startsWith("subagent_"))
      .map((e) => e.task_id)
      .filter((x) => x !== undefined && x !== null),
  );
  return ids.size;
}

export function historySummary(durationLabel, events) {
  const { actions, tasks } = workingCounts(events);
  return `Worked ${durationLabel} · ${actions} actions · ${tasks} subagent`;
}

export function taskProgressText(info) {
  const {
    status = "running", steps = 0, maxSteps = 0, actions = null,
    lastAction = "", stopReason = "",
  } = info || {};
  let stateText = taskStatusText(status);
  if (stopReason === "max_steps") stateText = "已达最大步数上限";
  else if (status === "unconfigured") stateText = "executor 未配置 · 任务未执行";
  const parts = [stateText, maxSteps ? `步数 ${steps}/${maxSteps}` : `步数 ${steps}`];
  if (actions !== null && actions !== undefined) parts.push(`动作 ${actions}`);
  if (lastAction) parts.push(`最近：${lastAction}`);
  return parts.join(" · ");
}

export function liveSummary(actions, subagents) {
  return `Worked · ${actions} actions · ${subagents} subagent`;
}

export function turnTaskSummary(tasks) {
  const list = (tasks || []).filter((t) => t && t.id !== undefined && t.id !== null);
  if (!list.length) return "";
  return list.map((t) => `任务 #${t.id} ${taskStatusText(t.status)}`).join(" · ");
}

export function turnSummaryText(actions, subagents, tasks, durationLabel = "") {
  const parts = [];
  if (durationLabel) parts.push(`用时 ${durationLabel}`);
  const taskText = turnTaskSummary(tasks);
  if (taskText) parts.push(taskText);
  else parts.push("处理中");
  if (actions) parts.push(`本轮动作 ${actions} 次`);
  return parts.join(" · ");
}

export function taskCardLines(task) {
  const id = task && task.id !== undefined && task.id !== null ? task.id : "?";
  const title = `Task #${id} · ${(task && task.title) || ""}`;
  const line = taskProgressText({
    status: (task && task.status) || "running",
    steps: (task && task.steps_used) || 0,
    maxSteps: (task && task.max_steps) || 0,
    actions: (task && task.actions !== undefined) ? task.actions : null,
    lastAction: (task && task.last_action) || "",
    stopReason: (task && task.stop_reason) || "",
  });
  return { title, line };
}

export function finalPreview(text, limit = 160) {
  const value = String(text || "");
  if (value.length <= limit) return { text: value, truncated: false };
  return { text: `${value.slice(0, limit)}…`, truncated: true };
}

export function ringLabel(payload) {
  const k = (n) => (n >= 1_000 ? `${Math.round(n / 1_000)}k` : `${n}`);
  if (!payload) {
    return { text: "?", sub: "未知", percent: 0, unknown: true };
  }
  const prefix = payload.exact ? "" : "约";
  if (!payload.window) {
    return {
      text: "?", sub: `${prefix}${k(payload.used_tokens || 0)} / ?`,
      percent: 0, unknown: true,
    };
  }
  const percent = Math.max(0, Math.min(100,
    payload.percent ?? Math.round(payload.used_tokens / payload.window * 100)));
  return {
    text: `${percent}%`,
    sub: `${prefix}${k(payload.used_tokens)} / ${k(payload.window)}（${prefix ? "估算" : "精确"}）`,
    percent, unknown: false,
  };
}

export function effortLabel(effort) {
  return { off: "关闭", low: "低", medium: "中", high: "高", max: "最高" }[effort] || effort || "默认";
}

export function composerLayout(width) {
  return Number(width) < 760 ? "narrow" : "wide";
}

export function shouldStickToBottom(scrollHeight, scrollTop, clientHeight, slack = 40) {
  // 时间线刷新后是否该跟随到底部：用户上翻查看历史时不得被拽回
  return Number(scrollHeight) - Number(scrollTop) - Number(clientHeight) < slack;
}

export function liveBufferStale(round, inflightAssistantCount) {
  // 实时缓冲是否已被 inflight 快照收录：按轮次身份判定，不靠文本包含。
  // round = 缓冲创建时已完成的模型回合数；inflight 中 assistant 条数更多 → 本轮已入快照。
  return Number(inflightAssistantCount) > Number(round ?? 0);
}

export function shouldDropShell(actions, subagents, bodyChildren) {
  return actions + subagents === 0 && bodyChildren === 0;
}

export function contextLabel(ctx) {
  if (!ctx) return "上下文 约 0 tokens";
  return `历史 ${ctx.messages} 条 · 上下文 约 ${ctx.tokens} tokens (${ctx.percent}%)`;
}

export function sessionLabel(session, allSessions) {
  const base = session.title || "新会话";
  const sameTitle = (allSessions || []).filter((s) => (s.title || "新会话") === base);
  if (!session.title || sameTitle.length > 1) return `${base} · #${session.id.slice(0, 4)}`;
  return base;
}

export function mergeTask(existing, id, patch) {
  return {...(existing || {}), ...patch, id};
}

export const TASK_STATUS_TEXT = {
  queued: "排队中",
  running: "执行中",
  done: "已完成",
  error: "执行失败",
  cancelled: "已取消",
  unconfigured: "未执行",
  incomplete: "未完成",
};

export function taskStatusText(status) {
  return TASK_STATUS_TEXT[status] || status || "";
}

export function inspectorLines(info) {
  const {
    status = "running", steps = 0, maxSteps = 0, actions = null,
    lastAction = "", stopReason = "", durationLabel = "", model = "",
  } = info || {};
  const first = taskProgressText({ status, steps, maxSteps, actions, lastAction, stopReason });
  const second = [durationLabel, model].filter(Boolean).join(" · ");
  return [first, second];
}

export function roleStatusText(roleStatus) {
  if (!roleStatus || !roleStatus.configured) return "当前：未配置";
  const name = roleStatus.preset || roleStatus.protocol || "custom";
  const model = roleStatus.model ? ` · ${roleStatus.model}` : "";
  return `当前：${name}${model} · 已连接`;
}

export function planLabel(on) {
  return on ? "plan 模式：已开启（不会修改文件）" : "plan 模式：已关闭";
}

export function planShort(on) {
  return on ? "plan：开" : "plan：关";
}

export function groupActivityEvents(events) {
  const groups = [];
  let text = null;
  let reasoning = null;
  for (const e of events || []) {
    if (e.type === "subagent_text_delta") {
      reasoning = null;
      const piece = e.text || "";
      if (!piece) continue;
      if (text) {
        text.text += piece;
      } else {
        text = { kind: "text", ts: e.ts, text: piece };
        groups.push(text);
      }
      continue;
    }
    if (e.type === "reasoning_delta") {
      text = null;
      const piece = e.text || "";
      if (!piece) continue;
      if (reasoning) {
        reasoning.text += piece;
      } else {
        reasoning = { kind: "reasoning", ts: e.ts, text: piece };
        groups.push(reasoning);
      }
      continue;
    }
    text = null;
    reasoning = null;
    if (e.type === "subagent_tool_started") {
      groups.push({ kind: "tool_started", ts: e.ts, name: e.name, arguments: e.arguments });
    } else if (e.type === "subagent_tool_finished") {
      groups.push({ kind: "tool_finished", ts: e.ts, name: e.name, status: e.status, result: e.result });
    } else if (e.type === "subagent_started") {
      groups.push({ kind: "status", ts: e.ts, label: "开始" });
    } else if (e.type === "subagent_completed") {
      groups.push({ kind: "status", ts: e.ts, label: "完成" });
    } else if (e.type === "subagent_failed") {
      groups.push({ kind: "status", ts: e.ts, label: `失败（${e.status || ""}）` });
    }
  }
  return groups;
}

export function pairToolEvents(events) {
  const pending = new Map();  // name -> FIFO 未配对 started
  const pairs = [];
  for (const e of events || []) {
    if (e.type === "subagent_tool_started" || e.type === "tool_started") {
      const queue = pending.get(e.name) || [];
      queue.push({ name: e.name, arguments: e.arguments, ts: e.ts });
      pending.set(e.name, queue);
    } else if (e.type === "subagent_tool_finished" || e.type === "tool_finished") {
      const queue = pending.get(e.name);
      if (queue && queue.length) {
        const start = queue.shift();
        pairs.push({
          name: start.name, arguments: start.arguments, ts: start.ts,
          status: e.status, result: e.result, finishedTs: e.ts,
        });
      } else {
        pairs.push({ name: e.name, arguments: null, ts: e.ts, status: e.status, result: e.result, finishedTs: e.ts });
      }
    }
  }
  return pairs;
}

const SHELL_WRITE_HINTS = [">", ">>", "tee ", "touch ", "mkdir ", "cp ", "mv ", "rm ", "sed -i"];

export function shellMayChangeFiles(command) {
  const text = String(command || "");
  return SHELL_WRITE_HINTS.some((hint) => text.includes(hint));
}

export function extractChanges(events) {
  const changes = [];
  for (const pair of pairToolEvents(events)) {
    if (pair.name === "write_file" || pair.name === "edit_file") {
      changes.push({ kind: "file", label: shortArg(pair.arguments) || pair.arguments || "", tool: pair.name });
    } else if (pair.name === "run_shell" && shellMayChangeFiles(shortArg(pair.arguments))) {
      changes.push({ kind: "shell", label: shortArg(pair.arguments), tool: "run_shell" });
    }
  }
  return changes;
}

export function shortArg(args) {
  try {
    const obj = JSON.parse(args || "{}");
    return obj.path || obj.pattern || obj.command || obj.title || obj.question || "";
  } catch {
    return String(args || "").slice(0, 60);
  }
}

export function breadcrumb(path) {
  const clean = String(path || ".").replace(/^\.?\/?/, "").replace(/\/+$/, "");
  return ["项目", ...clean.split("/").filter(Boolean)];
}

export function parentPath(path) {
  const clean = String(path || ".").replace(/^\.?\/?/, "").replace(/\/+$/, "");
  if (!clean) return ".";
  const parts = clean.split("/").filter(Boolean);
  parts.pop();
  return parts.length ? parts.join("/") : ".";
}

export function taskCardAction(ev) {
  const t = ev.type;
  if (t === "subagent_queued") {
    return { kind: "create", status: "queued", title: ev.text || "" };
  }
  if (t === "subagent_started") return { kind: "update", status: "running" };
  if (t === "subagent_completed") {
    return { kind: "update", status: "done", output: ev.text || "" };
  }
  if (t === "subagent_failed") {
    return { kind: "update", status: ev.status || "error", output: ev.text || "" };
  }
  if (t === "subagent_step") return { kind: "update", steps: ev.steps_used, maxSteps: ev.max_steps };
  if (t === "subagent_tool_started") return { kind: "update", countAction: true };
  return { kind: "none" };
}

export function planTaskCard(registry, taskId, hasContainer) {
  // 同一任务只允许一张卡：首次且能落位才 create，其余 update，无处落位 skip
  if (!registry[taskId]) {
    if (!hasContainer) return "skip";
    registry[taskId] = true;
    return "create";
  }
  return "update";
}

/* ---------- 模型选择器（数据来自 provider 列表，不来自 PRESETS） ---------- */

export function modelMenuState(catalog) {
  if (catalog == null) return { loading: true, groups: [], models: [], errors: [] };
  const providers = (catalog && catalog.providers) || [];
  const groups = providers.map((p) => ({
    id: p.id,
    provider: p.name || p.id,
    status: p.status || (p.error ? "error" : "empty"),
    error: p.error || null,
    configured: p.configured !== false,
    models: p.models || [],
  }));
  const models = groups.flatMap((g) => g.models);
  const errors = groups.filter((g) => g.error).map((g) => `${g.provider}: ${g.error}`);
  return { loading: false, groups, models, errors };
}

export function modelSelectAction(providerId, currentProviderId, targetConfigured) {
  if (!targetConfigured) return "connect";  // 目标 provider 尚未配置 → 进设置
  return providerId === currentProviderId ? "set_model" : "switch_provider";
}

export function filterModels(models, query) {
  const q = String(query || "").trim().toLowerCase();
  const list = models || [];
  if (!q) return list;
  return list.filter((m) => (
    `${m.id || ""} ${m.display_name || ""}`.toLowerCase().includes(q)
  ));
}



/* 能力数据唯一结构 = provider_status.roles.<role>：
   window / window_source / reasoning_mode / efforts（null=未知，[]=明确为空）/
   reasoning_returned / max_output / tools / provider_id / model */

const SOURCE_LABELS = { user: "用户设置", provider: "provider 返回", catalog: "公共目录", unknown: "未知" };

export function capabilityFormState(role) {
  const cap = role || {};
  const levels = cap.efforts;
  return {
    window: cap.window ?? "",
    maxOutput: cap.max_output ?? "",
    mode: cap.reasoning_mode || "unknown",
    levelsText: levels == null ? "" : levels.join(","),
    levelsUnknown: levels == null,
    tools: cap.tools == null ? "" : (cap.tools ? "yes" : "no"),
    source: SOURCE_LABELS[cap.window_source] || "未知",
    sourceKey: cap.window_source || "unknown",
    providerId: cap.provider_id || "",
    model: cap.model || "",
  };
}

export function capabilityEditorState(roleStatuses, roleName) {
  const role = (roleStatuses || {})[roleName] || {};
  const form = capabilityFormState(role);
  return {
    roleName,
    empty: !role.provider_id || !role.model,
    title: role.model
      ? `模型能力：${role.provider || role.provider_id} · ${role.model}（${form.source}）`
      : "模型能力（尚未绑定 provider）",
    target: { provider: role.provider_id || "", model: role.model || "" },
    form,
  };
}

function intOrNull(value) {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

export function capabilityOverrideFromForm(values) {
  const form = values || {};
  const mode = ["unknown", "fixed", "none", "adjustable"].includes(form.mode)
    ? form.mode : "unknown";
  const typed = String(form.levelsText || "").split(",").map((x) => x.trim()).filter(Boolean);
  let levels;
  if (mode === "unknown") {
    levels = null;  // 未知：不得写成空数组
  } else if (mode === "fixed" || mode === "none") {
    levels = [];  // 明确不支持调节
  } else {
    // 可调：空文本且原本就是未知 → 保持未知；否则按输入（空=明确为空）
    levels = form.levelsUnknown && typed.length === 0 ? null : typed;
  }
  return {
    window: intOrNull(form.window),
    max_output: intOrNull(form.maxOutput),
    reasoning_mode: mode,
    levels,
    tools: form.tools === "" || form.tools == null ? null : form.tools === "yes",
  };
}

export function effortState(capability) {
  const mode = (capability && capability.reasoning_mode) || "unknown";
  const raw = capability && capability.levels;
  if (mode === "adjustable") {
    if (!Array.isArray(raw) || raw.length === 0) {
      // 可调但档位未知/为空：禁止空菜单，提示先声明
      return { mode: "unknown-levels", selectable: false, levels: [], label: "档位未知" };
    }
    return { mode, selectable: true, levels: raw, label: "可调" };
  }
  if (mode === "fixed") return { mode, selectable: false, levels: [], label: "固定" };
  if (mode === "none") return { mode, selectable: false, levels: [], label: "关闭" };
  return { mode: "unknown", selectable: false, levels: [], label: "未知" };
}

export function reasoningTarget(ev) {
  return ev && ev.task_id ? `task:${ev.task_id}` : "main";
}

/* ---------- 服务商列表（凭据全局共享）与角色模型选项 ---------- */

export function providerListState(providerStatus) {
  const presets = (providerStatus && providerStatus.presets) || {};
  const providers = (providerStatus && providerStatus.providers) || [];
  const byId = new Map(providers.map((p) => [p.id, p]));
  const rows = Object.entries(presets).map(([name, spec]) => {
    const live = byId.get(spec.provider_id);
    return {
      kind: "preset",
      name,
      label: spec.label || name,
      providerId: spec.provider_id || "",
      configured: Boolean(spec.configured),
      needsKey: spec.needs_key !== false,
      hasKey: live ? Boolean(live.has_key) : false,
      login: spec.login || "",
      hint: spec.hint || "",
      loggedIn: Boolean(((providerStatus || {}).codex_login || {}).logged_in),
      modelCount: live ? live.model_count || 0 : 0,
    };
  });
  const presetIds = new Set(rows.map((r) => r.providerId));
  for (const p of providers) {
    if (presetIds.has(p.id)) continue;
    rows.push({
      kind: "custom",
      name: p.id,
      label: p.name || p.id,
      providerId: p.id,
      configured: p.configured !== false,
      needsKey: false,
      hasKey: Boolean(p.has_key),
      modelCount: p.model_count || 0,
      protocol: p.protocol || "",
      baseUrl: p.base_url || "",
    });
  }
  return rows;
}

export function roleModelOptions(catalog, providerStatus) {
  const configured = new Set(
    ((providerStatus && providerStatus.providers) || [])
      .filter((p) => p.configured !== false)
      .map((p) => p.id)
  );
  const groups = [];
  for (const p of (catalog && catalog.providers) || []) {
    if (!configured.has(p.id)) continue;
    groups.push({
      providerId: p.id,
      provider: p.name || p.id,
      models: (p.models || []).map((m) => ({ id: m.id, name: m.display_name || m.id })),
    });
  }
  return groups;
}

export function connectRequestForRow(row, key) {
  if (!row) return null;
  if (row.kind === "custom") {
    return {
      type: "connect_provider",
      protocol: row.protocol || "",
      base_url: row.baseUrl || null,
      api_key: key || "",
    };
  }
  if (!row.needsKey) {
    return {type: "connect_provider", preset: row.name, api_key: ""};
  }
  return {type: "connect_provider", preset: row.name, api_key: key || ""};
}


export function connectRowPlan(row) {
  if (!row) return {kind: "none"};
  if (row.kind === "custom") {
    return row.protocol ? {kind: "key"} : {kind: "guide-advanced"};
  }
  return row.needsKey ? {kind: "key"} : {kind: "direct"};
}


export function effortChipVisible(roleStatus) {
  return effortState({
    reasoning_mode: roleStatus && roleStatus.reasoning_mode,
    levels: roleStatus && roleStatus.efforts,
  }).selectable;
}

export function ringVisible(window) {
  return typeof window === "number" && window > 0;
}


export function roleEffortVisible(roleStatus) {
  return Boolean(roleStatus && roleStatus.configured) && effortChipVisible(roleStatus);
}


export function executorMessageView(m) {
  return {
    role: m && m.role,
    text: (m && m.content) || "",
    reasoning: (m && m.reasoning) || "",
    calls: ((m && m.tool_calls) || []).map((tc) => ({ name: tc.name, arguments: tc.arguments })),
    isToolResult: Boolean(m && m.role === "tool"),
  };
}

const CODE_SPAN = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;
const INDENTED = /^(?: {4}|\t)/;
const MATH_SPAN = /(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<!\\)\$(?!\s)([^\s$](?:[^$\n]*?[^\s$])?)(?<!\\)\$(?!\d)/g;

export function extractMath(text) {
  // markdown 会吃掉 \( \[ 的反斜杠：先摘出公式留占位，再由 KaTeX 回填。
  // 围栏/行内代码整段不动；缩进代码块按行判定（段首或空行后的缩进才是代码，
  // 列表/段落的缩进续行仍是正文，公式照常提取）。
  const items = [];
  const replace = (seg, ranges) => seg.replace(MATH_SPAN, (match, d1, d2, i1, i2, offset) => {
    if (ranges && ranges.some((r) => offset >= r[0] && offset <= r[1])) return match;
    const tex = (d1 ?? d2 ?? i1 ?? i2 ?? "").trim();
    if (!tex) return match;
    const display = d1 !== undefined || d2 !== undefined;
    items.push({tex: tex, display: display});
    return `@@KATEX${items.length - 1}@@`;
  });
  const parts = String(text || "").split(CODE_SPAN);
  const out = parts.map((seg, idx) => {
    if (idx % 2 === 1) return seg;   // 代码段不动
    return replace(seg, indentedCodeRanges(seg));
  });
  return {text: out.join(""), items: items};
}

function indentedCodeRanges(seg) {
  // 缩进代码块的字符区间：段首或空行后的缩进行；列表/段落续行不算代码
  const ranges = [];
  const lines = seg.split("\n");
  let pos = 0;
  let start = -1;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    let isCode = false;
    if (INDENTED.test(line)) {
      const prev = i > 0 ? lines[i - 1] : "";
      isCode = !(prev.trim() !== "" && !INDENTED.test(prev));
    }
    if (isCode && start < 0) start = pos;
    if (!isCode && start >= 0) { ranges.push([start, pos - 1]); start = -1; }
    pos += line.length + 1;
  }
  if (start >= 0) ranges.push([start, seg.length - 1]);
  return ranges;
}

export function diffSummary(events) {
  // 汇总回合内所有 task_diff：{files:[{path,added,removed}], added, removed, label}
  const files = new Map();
  for (const e of events || []) {
    if (!e || e.type !== "task_diff" || !e.text) continue;
    let payload = null;
    try { payload = JSON.parse(e.text); } catch (err) { continue; }
    for (const f of (payload && payload.files) || []) {
      if (!f || !f.path) continue;
      const cur = files.get(f.path) || {path: f.path, added: 0, removed: 0};
      cur.added += Number(f.added) || 0;
      cur.removed += Number(f.removed) || 0;
      files.set(f.path, cur);
    }
  }
  const list = [...files.values()].sort((a, b) => a.path.localeCompare(b.path));
  return {
    files: list,
    added: list.reduce((sum, f) => sum + f.added, 0),
    removed: list.reduce((sum, f) => sum + f.removed, 0),
    label: list.length ? `${list.length} 个文件已更改` : "",
  };
}

export function mentionQuery(text, caret) {
  // 光标前最近的 @片段（行首/空白后触发；词中 @ 不触发）
  const upto = String(text || "").slice(0, Math.max(0, Number(caret) || 0));
  const match = /(^|[\s(\[])(@[^\s@\[\]()]*)$/.exec(upto);
  if (!match) return null;
  const fragment = match[2];
  return {start: upto.length - fragment.length, query: fragment.slice(1)};
}

export function insertMention(text, start, end, path) {
  // 用 @路径 替换 @片段；后面紧贴文字时补一个空格
  const src = String(text || "");
  const before = src.slice(0, start);
  const after = src.slice(end);
  const inserted = `@${path}`;
  const pad = after && !/^[\s,，。;；)]/.test(after) ? " " : "";
  return {text: before + inserted + pad + after, caret: (before + inserted + pad).length};
}

export function commandQuery(text, caret) {
  // 光标前的 /命令片段（行首或空白后触发）
  const upto = String(text || "").slice(0, Math.max(0, Number(caret) || 0));
  const match = /(^|[\s])(\/[A-Za-z0-9_-]*)$/.exec(upto);
  if (!match) return null;
  const fragment = match[2];
  return {start: upto.length - fragment.length, query: fragment};
}

export function pendingBindPlan(batchText, pendingTexts) {
  // 本回合文本从开头起能匹配哪些待绑定气泡：
  // - skip：被并入上一回合、无法单独撤销的前导气泡
  // - bind：本回合可绑定的气泡数（合并消息时 >1）
  const text = String(batchText || "");
  const list = Array.isArray(pendingTexts) ? pendingTexts : [];
  if (!text || !list.length) return {skip: list.length, bind: 0};
  for (let first = 0; first < list.length; first += 1) {
    const head = String(list[first] || "");
    // 必须从批次开头整行匹配（行边界），避免“继续”错配“继续做这件事”
    if (!head || !text.startsWith(head)) continue;
    const after = text.slice(head.length, head.length + 1);
    if (after && after !== "\n") continue;
    let joined = head;
    let bind = 1;
    for (let i = first + 1; i < list.length; i += 1) {
      const next = `${joined}\n${String(list[i] || "")}`;
      if (!text.startsWith(next)) break;
      const tail = text.slice(next.length, next.length + 1);
      if (tail && tail !== "\n") break;
      joined = next;
      bind += 1;
    }
    return {skip: first, bind};
  }
  return {skip: list.length, bind: 0};
}
