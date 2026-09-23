"use strict";
/* rabbit-agent web client：Turn/Working/Final 结构的时间线 UI。
   数据源：/api/timeline（持久化历史）+ WS execution events（实时）。 */
import {
  breadcrumb,
  capabilityEditorState,
  capabilityOverrideFromForm,
  connectRequestForRow,
  connectRowPlan,
  composerLayout,
  effortLabel,
  executorMessageView,
  effortState,
  extractChanges,
  finalPreview,
  filterModels,
  groupActivityEvents,
  hasWork,
  inspectorLines,
  extractMath,
  isSystemTurn,
  liveBufferStale,
  mergeTask,
  modelMenuState,
  modelSelectAction,
  pairToolEvents,
  parentPath,
  planLabel,
  planTaskCard,
  providerListState,
  roleEffortVisible,
  roleModelOptions,
  reasoningTarget,
  ringLabel,
  ringVisible,
  routeActionEvent,
  sessionLabel,
  shouldDropShell,
  shouldStickToBottom,
  shortArg,
  taskCardAction,
  taskCardLines,
  taskStatusText,
  turnSummaryText,
} from "./timeline_logic.mjs?v=2";

const $ = (id) => document.getElementById(id);
const TOKEN = "__AGENT_TOKEN__";
const state = {
  projects: {}, sessions: {}, current: null, activity: new Set(),
  tasks: {}, liveTurn: null, planMode: false, wantNewSession: false,
  providerStatus: null, liveTaskProgress: {}, turnsById: {},
  taskCards: {}, taskCardEls: {}, fsPath: ".",
  modelCatalog: null, execLive: {},
};

/* ---------- 基础设施 ---------- */
const ws = new WebSocket(`ws://${location.host}/ws?token=${TOKEN}`);
ws.onopen = () => {
  send({type: "list_projects"});
  send({type: "list_sessions"});
  send({type: "get_provider_status"});
};
ws.onmessage = (e) => handle(JSON.parse(e.data));
ws.onclose = () => setStatus("连接已断开，请刷新");
function send(obj) { ws.send(JSON.stringify(obj)); }
async function api(path) {
  const sep = path.includes("?") ? "&" : "?";
  const resp = await fetch(`${path}${sep}token=${TOKEN}`);
  return resp.json();
}
function setStatus(text) { $("turn-status").textContent = text; }
function renderPlanLabel() {
  $("plan-label").textContent = state.planMode ? "Plan" : "Build";
  $("btn-plan-mode").title = planLabel(state.planMode);
  document.querySelectorAll("#plan-menu .menu-item").forEach((el) => {
    el.classList.toggle("active", (el.dataset.plan === "on") === state.planMode);
  });
}
function stream() { return $("stream"); }
function scrollDown() { const s = stream(); s.scrollTop = s.scrollHeight; }
function escapeHtml(text) { const d = document.createElement("div"); d.textContent = text; return d.innerHTML; }
function sanitize(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  doc.querySelectorAll("script, iframe, object, embed, link, meta").forEach(el => el.remove());
  doc.querySelectorAll("*").forEach(el => {
    for (const attr of [...el.attributes]) {
      if (attr.name.toLowerCase().startsWith("on")) el.removeAttribute(attr.name);
      if ((attr.name === "href" || attr.name === "src") && /^\s*javascript:/i.test(attr.value)) el.removeAttribute(attr.name);
    }
  });
  return doc.body.innerHTML;
}
function fmtTime(ts) { const d = new Date(ts * 1000); return d.toTimeString().slice(0, 8); }
function fmtDur(a, b) { if (!b) return "…"; return Math.max(0, Math.round(b - a)) + "s"; }

/* ---------- 事件路由 ---------- */
function handle(ev) {
  const t = ev.type;
  if (t === "project_list") { state.projects = Object.fromEntries(ev.projects.map(p => [p.id, p])); renderTree(); return; }
  if (t === "session_list") {
    state.sessions = Object.fromEntries(ev.sessions.map(s => [s.id, s]));
    if (!state.current && ev.sessions.length) selectSession(ev.sessions[0].id);
    renderTree(); return;
  }
  if (t === "session_created") {
    state.sessions[ev.session] = {id: ev.session, title: ev.title, project: ev.project};
    if (state.wantNewSession) { state.wantNewSession = false; selectSession(ev.session); }
    renderTree(); return;
  }
  if (t === "session_updated") {
    if (state.sessions[ev.session]) state.sessions[ev.session].title = ev.title;
    renderTree();
    applyCurrentTitle();
    return;
  }
  if (t === "session_deleted") {
    delete state.sessions[ev.session]; state.activity.delete(ev.session);
    if (ev.session === state.current) {
      const rest = Object.keys(state.sessions); state.current = null;
      if (rest.length) selectSession(rest[0]);
      else { state.wantNewSession = true; send({type: "new_session", title: ""}); }
    }
    renderTree(); return;
  }
  if (t === "project_created") { send({type: "list_projects"}); return; }
  if (t === "project_result") { $("p-result").textContent = ev.message; if (ev.ok) setTimeout(closeModal2, 600); return; }
  if (t === "provider_result") {
    const keyOpen = !$("key-modal-mask").classList.contains("hidden");
    const target = keyOpen ? $("key-result")
      : (lastProviderAction === "advanced" ? $("adv-result") : $("m-result"));
    setResult(target, ev.message, ev.ok);
    if (ev.ok && keyOpen) setTimeout(closeKeyModal, 700);
    send({type: "get_provider_status"});  // 成功/失败都立即刷新状态
    send({type: "list_models"});
    return;
  }
  if (t === "provider_status") {
    state.providerStatus = ev;
    if (typeof ev.plan_mode === "boolean") {
      state.planMode = ev.plan_mode;
      renderPlanLabel();
    }
    renderProviderStatus();
    return;
  }
  if (t === "model_catalog") {
    state.modelCatalog = ev;
    renderModelModal();
    if (!$("modal-mask").classList.contains("hidden")) renderRoleSettings();
    if (!$("tab-executor").classList.contains("hidden")) renderExecutorHeader();
    if (state.providerStatus) renderModelButton(state.providerStatus.roles || {});
    return;
  }
  if (t === "plan_mode") {
    state.planMode = ev.on;
    renderPlanLabel();
    return;
  }
  if (t === "session_list") return;

  const sid = ev.session;
  if (sid && sid !== state.current) {
    if (["turn_started","final_completed","subagent_completed"].includes(t)) { state.activity.add(sid); renderTree(); }
    return;
  }

  // 状态与控制类
  if (t === "usage") {
    state.usage = ev;  // 本轮累计消耗：只在设置面板展示
    renderUsageLine();
    return;
  }
  if (t === "context") { state.context = ev; renderRing(ev); return; }
  if (t === "compacted") {
    $("ctx-badge").classList.remove("hidden");
    flash(`上下文已压缩：约 ${ev.before_tokens} → ${ev.after_tokens} tokens`);
    return;
  }
  if (t === "confirm_request") { confirmCard(ev); return; }
  if (t === "task_question") { note(`任务 #${ev.id} 提问: ${ev.question}`); return; }
  if (t === "user_to_executor") {
    // 用户直接对执行者说的话：主时间线灰色提示，执行者窗口同步刷新
    note(`你对执行者说：${ev.text}`);
    maybeRefreshExecutor(ev.session);
    return;
  }
  if (t === "error") { note(`[错误] ${ev.message}`); return; }
  if (t === "executor_report") { renderExecReport(ev.on); return; }
  if (t === "stopped") { setStatus(""); setBusy(false); note("[已中断]"); return; }
  if (t === "turn_end") { setStatus(""); setBusy(false); return; }

  // 执行事件 → 时间线
  const execDelta = t === "subagent_text_delta"
    || (t === "reasoning_delta" && ev.actor === "subagent");
  if (EXEC_EVENT_TYPES.has(t) && (t !== "reasoning_delta" || ev.actor === "subagent")
      && !execDelta) {
    clearExecLive(ev.task_id);      // 该轮已入 messages/inflight，实时缓冲退役
    maybeRefreshExecutor(ev.session);
  }
  routeExecution(ev);
}

function flash(text) { setStatus(text); setTimeout(() => setStatus(""), 4000); }
function note(text) {
  const el = document.createElement("div"); el.className = "msg-note"; el.textContent = text;
  stream().appendChild(el); scrollDown();
}

function renderRing(payload) {
  const info = ringLabel(payload);
  $("ctx-ring-text").textContent = info.text;
  const ring = $("ring-fg");
  const circumference = 2 * Math.PI * 15.9;
  ring.style.strokeDasharray = `${circumference}`;
  ring.style.strokeDashoffset = `${circumference * (1 - info.percent / 100)}`;
  ring.classList.toggle("unknown", info.unknown);
  if (!$("ctx-ring-wrap").classList.contains("hidden")) {
    $("ctx-ring-wrap").title = `上下文：${info.sub}`;
  }
}

function renderUsageLine() {
  const el = $("m-usage");
  if (!el) return;
  const u = state.usage;
  if (!u) { el.textContent = ""; return; }
  el.textContent = `本轮累计消耗：输入 ${u.input} / 输出 ${u.output} tokens` +
    (u.reasoning ? `（含思考 ${u.reasoning}）` : "");
}

/* ---------- Turn 结构构建 ---------- */
function userBubble(text) {
  const el = document.createElement("div"); el.className = "msg-user"; el.textContent = text;
  stream().appendChild(el); scrollDown();
}

function makeTurn(container) {
  const turn = document.createElement("div"); turn.className = "turn";
  (container || stream()).appendChild(turn); scrollDown();
  return turn;
}

function systemDivider(text) {
  const el = document.createElement("div");
  el.className = "sys-divider"; el.textContent = text || "任务事件";
  return el;
}

function makeTurnShell(turn) {
  const working = document.createElement("div"); working.className = "working open";
  const head = document.createElement("div"); head.className = "working-head";
  head.innerHTML = `<span class="chevron">▶</span><span class="spinner">◉</span><span class="summary">Working…</span>`;
  const body = document.createElement("div"); body.className = "working-body";
  working.append(head, body); turn.appendChild(working);
  head.onclick = () => working.classList.toggle("open");
  return {working, head, body, summary: head.querySelector(".summary"), spinner: head.querySelector(".spinner")};
}

function ensureShell(lt) {
  if (!lt.shell) lt.shell = makeTurnShell(lt.turn);
  return lt.shell;
}

function renderFinalText(el, text) {
  const preview = finalPreview(text, 160);
  renderMarkdown(el, preview.text);
  if (!preview.truncated) return;
  const wrap = el.closest(".final") || el;
  const btn = document.createElement("button");
  btn.className = "final-toggle"; btn.textContent = "展开全文";
  btn.onclick = () => {
    renderMarkdown(el, text);
    btn.remove();
  };
  wrap.appendChild(btn);
}

function renderMarkdown(el, text) {
  const picked = extractMath(text);
  const html = sanitize(marked.parse(picked.text));
  el.innerHTML = html.replace(/@@KATEX(\d+)@@/g, (match, i) => {
    const item = picked.items[Number(i)];
    if (!item || typeof katex === "undefined") return match;
    try {
      return katex.renderToString(item.tex, {displayMode: item.display, throwOnError: false});
    } catch (e) { return match; }
  });
}

function renderMath(el) {
  // LLM 习惯的 LaTeX 分隔符；代码块/行内代码不处理；失败保持原文
  if (typeof renderMathInElement !== "function") return;
  try {
    renderMathInElement(el, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\[", right: "\\]", display: true},
        {left: "\\(", right: "\\)", display: false},
        {left: "$", right: "$", display: false},
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      throwOnError: false,
    });
  } catch (e) { /* 渲染失败时保持原文展示 */ }
}

function reasoningDetails(container, label, beforeEl, extraClass = "") {
  const details = document.createElement("details");
  details.className = extraClass ? `thinking ${extraClass}` : "thinking";
  const summary = document.createElement("summary");
  summary.textContent = label;
  const body = document.createElement("div");
  body.className = "thinking-text";
  details.append(summary, body);
  if (beforeEl && beforeEl.parentElement === container) container.insertBefore(details, beforeEl);
  else container.appendChild(details);
  return body;
}

function appendReasoning(lt, text) {
  if (!text) return;
  if (!lt.reasoningEl || !lt.reasoningEl.isConnected) {
    lt.reasoningEl = reasoningDetails(lt.turn, "思考（主 agent）", lt.finalEl);
  }
  lt.reasoningEl.textContent += text;
  scrollDown();
}

function appendTaskReasoning(taskId, text) {
  if (!text) return;
  const card = state.taskCardEls[taskId];
  if (!card || !card.isConnected) return;
  let body = card.querySelector(".task-thinking .thinking-text");
  if (!body) body = reasoningDetails(card, `思考（子任务 #${taskId}）`, null, "task-thinking");
  body.textContent += text;
  scrollDown();
}

function paintTurnSummary(lt, durationLabel = "") {
  if (!lt || !lt.shell) return;
  const tasks = Object.entries(lt.tasks || {}).map(([id, status]) => ({ id: Number(id), status }));
  lt.shell.summary.textContent = turnSummaryText(lt.actions, lt.subagents, tasks, durationLabel);
}

function markTurnTask(lt, taskId, status) {
  if (!lt) return;
  lt.tasks[taskId] = status;
  paintTurnSummary(lt);
}

function addWorkingText(body, text) {
  const el = document.createElement("div"); el.className = "working-text"; el.textContent = text;
  body.appendChild(el); scrollDown();
}

const TOOL_ICONS = {ls: "📁", read_file: "📖", grep: "🔍", glob: "🗂", write_file: "✏️", edit_file: "✏️", run_shell: "$", call_subagent: "➤", answer_task: "↩", ask: "?"};
function addToolRow(body, name, args) {
  const row = document.createElement("div"); row.className = "tool-row"; row.dataset.tool = `${name}:${args || ""}`;
  const argsText = compactArgs(name, args);
  row.innerHTML = `<span class="t-icon">…</span><span class="t-name">${escapeHtml(TOOL_ICONS[name] || "🔧")} ${escapeHtml(displayToolName(name))}</span><span class="t-detail">${escapeHtml(argsText)}</span>`;
  body.appendChild(row); scrollDown();
  return row;
}
function displayToolName(name) {
  return {ls: "List", read_file: "Read", grep: "Search", glob: "Find", write_file: "Write", edit_file: "Edit", run_shell: "Shell", call_subagent: "Dispatch", answer_task: "Answer", ask: "Ask"}[name] || name;
}
function compactArgs(name, args) {
  if (!args) return "";
  try {
    const obj = JSON.parse(args);
    return obj.path || obj.pattern || obj.command || obj.title || obj.question || "";
  } catch { return String(args).slice(0, 60); }
}
function finishToolRow(body, name, args, status, result) {
  const key = `${name}:${args || ""}`;
  const row = [...body.children].reverse().find(el => el.dataset && el.dataset.tool === key && !el.classList.contains("done"));
  if (!row) return;
  const icon = status === "success" ? "✓" : "✗";
  row.querySelector(".t-icon").textContent = icon;
  row.classList.add("done");
  if (status !== "success") { row.classList.add(status || "error"); row.title = (result || "").slice(0, 500); }
  else if (result) row.title = result.slice(0, 500);
  scrollDown();
}

function ensureTaskCard(taskId, title, status, container) {
  // 去重：同一任务只允许一张卡；已存在则原地更新状态（决策由纯函数给出）
  const plan = planTaskCard(state.taskCards, taskId, Boolean(container));
  if (plan === "skip") return null;
  let card = state.taskCardEls[taskId];
  if (plan === "update") {
    if (card && card.isConnected) updateTaskCard(taskId, status);
    return card || null;
  }
  if (!state.liveTaskProgress[taskId]) {
    const task = state.tasks[taskId] || {};
    state.liveTaskProgress[taskId] = {
      steps: task.steps_used || 0, actions: 0, last: task.last_action || "",
    };
  }
  const task = state.tasks[taskId] || {};
  const prog = state.liveTaskProgress[taskId];
  const lines = taskCardLines({
    id: taskId, title: title || task.title || "", status,
    steps_used: prog.steps || task.steps_used || 0,
    max_steps: task.max_steps || state.providerStatus?.max_steps?.executor || 0,
    actions: prog.actions,
    last_action: prog.last || task.last_action || "",
    stop_reason: task.stop_reason || "",
  });
  card = document.createElement("div");
  card.className = "task-card"; card.dataset.task = taskId; card.dataset.status = status;
  card.innerHTML = `<div class="tc-title"></div><div class="tc-progress"></div>`;
  card.querySelector(".tc-title").textContent = lines.title;
  card.querySelector(".tc-progress").textContent = lines.line;
  card.onclick = () => openExecutorTab();
  container.appendChild(card); state.taskCardEls[taskId] = card; scrollDown();
  return card;
}

function paintTaskCard(taskId) {
  const prog = state.liveTaskProgress[taskId] || {steps: 0, actions: null, last: ""};
  const task = state.tasks[taskId] || {};
  const status = task.status || "running";
  const lines = taskCardLines({
    id: taskId,
    title: task.title || "",
    status,
    steps_used: prog.steps || task.steps_used || 0,
    max_steps: task.max_steps || state.providerStatus?.max_steps?.executor || 0,
    actions: prog.actions,
    last_action: prog.last || task.last_action || "",
    stop_reason: task.stop_reason || "",
  });
  document.querySelectorAll(`.task-card[data-task="${taskId}"]`).forEach((el) => {
    el.dataset.status = status;
    const progressEl = el.querySelector(".tc-progress");
    if (progressEl) progressEl.textContent = lines.line;
  });
}

function updateTaskCard(taskId, status, output) {
  const card = state.taskCardEls[taskId];
  if (!card || !card.isConnected) return;
  card.dataset.status = status;
  if (output) card.title = output.slice(0, 500);
  paintTaskCard(taskId);
}

/* ---------- 实时执行事件 ---------- */
function routeExecution(ev) {
  const t = ev.type;
  if (t === "subagent_text_delta") {
    execAppendLive(ev.task_id, "text", ev.text || "");
    return;
  }
  if (t === "turn_started") {
    const turnEl = makeTurn();
    let shell = null;
    if (ev.text) shell = makeTurnShell(turnEl);
    else turnEl.appendChild(systemDivider());  // 事件驱动 turn：无用户消息
    const lt = {
      turnId: ev.turn_id, turn: turnEl, shell, startedAt: Date.now(),
      actions: 0, subagents: 0, finalEl: null, finalText: "", tasks: {},
    };
    state.turnsById[ev.turn_id] = lt;
    state.liveTurn = lt;
    setStatus("Working…");
    setBusy(true);
    return;
  }
  // 按事件自带 turn_id 路由：旧 subagent 的事件不会串进新 turn
  const lt = state.turnsById[ev.turn_id];
  if (t === "activity_text_delta") { if (lt) addWorkingText(ensureShell(lt).body, ev.text); return; }
  if (t === "reasoning_delta") {
    if (reasoningTarget(ev) === "main") {
      if (lt) appendReasoning(lt, ev.text || "");
    } else if (ev.task_id !== undefined && ev.task_id !== null) {
      appendTaskReasoning(ev.task_id, ev.text || "");
      execAppendLive(ev.task_id, "reasoning", ev.text || "");
    }
    return;
  }
  if (t === "tool_started") {
    if (lt) {
      addToolRow(ensureShell(lt).body, ev.name, ev.arguments);
      routeActionEvent(state.turnsById, ev);
      paintTurnSummary(lt);
    }
    return;
  }
  if (t === "tool_finished") { if (lt && lt.shell) { finishToolRow(lt.shell.body, ev.name, ev.arguments, ev.status, ev.result); } return; }
  if (t === "subagent_queued") {
    const action = taskCardAction(ev);
    if (lt) {
      ensureTaskCard(ev.task_id, action.title, action.status, ensureShell(lt).body);
      lt.subagents++;
      markTurnTask(lt, ev.task_id, action.status);
    } else {
      paintTaskCard(ev.task_id);
    }
    upsertTask(ev.task_id, {title: action.title, status: action.status}); return;
  }
  if (t === "subagent_tool_started") {
    const prog = state.liveTaskProgress[ev.task_id] ||
      (state.liveTaskProgress[ev.task_id] = {steps: 0, actions: 0, last: ""});
    prog.actions++;
    if (lt) {
      ensureTaskCard(ev.task_id, "", "running", ensureShell(lt).body);
      routeActionEvent(state.turnsById, ev);  // 旧任务事件只加旧 turn
      markTurnTask(lt, ev.task_id, "running");
    } else {
      paintTaskCard(ev.task_id);
    }
    prog.last = `${displayToolName(ev.name)} ${compactArgs(ev.name, ev.arguments)}`.trim();
    upsertTask(ev.task_id, {last_action: prog.last});
    paintTaskCard(ev.task_id);
    return;
  }
  if (t === "subagent_step") {
    const action = taskCardAction(ev);
    const prog = state.liveTaskProgress[ev.task_id] ||
      (state.liveTaskProgress[ev.task_id] = {steps: 0, actions: 0, last: ""});
    prog.steps = action.steps;
    if (lt) {
      ensureTaskCard(ev.task_id, "", "running", ensureShell(lt).body);
      markTurnTask(lt, ev.task_id, "running");
    } else {
      paintTaskCard(ev.task_id);
    }
    upsertTask(ev.task_id, {steps_used: action.steps, max_steps: action.maxSteps});
    paintTaskCard(ev.task_id);
    return;
  }
  if (t === "subagent_tool_finished") return;  // 不计二次动作，避免重复累计
  if (t === "subagent_started") {
    const action = taskCardAction(ev);
    upsertTask(ev.task_id, {status: action.status});
    if (lt) {
      ensureTaskCard(ev.task_id, state.tasks[ev.task_id]?.title || "", action.status, ensureShell(lt).body);
      markTurnTask(lt, ev.task_id, action.status);
    } else {
      paintTaskCard(ev.task_id);
    } return;
  }
  if (t === "subagent_completed") {
    const action = taskCardAction(ev);
    const hitLimit = action.output.startsWith("[已达最大步数上限");
    upsertTask(ev.task_id, {status: action.status, stop_reason: hitLimit ? "max_steps" : ""});
    if (lt) {
      ensureTaskCard(ev.task_id, state.tasks[ev.task_id]?.title || "", action.status, ensureShell(lt).body);
      updateTaskCard(ev.task_id, action.status, action.output);
      markTurnTask(lt, ev.task_id, action.status);
    } else {
      paintTaskCard(ev.task_id);
    } return;
  }
  if (t === "subagent_failed") {
    const action = taskCardAction(ev);
    upsertTask(ev.task_id, {status: action.status});
    if (lt) {
      ensureTaskCard(ev.task_id, state.tasks[ev.task_id]?.title || "", action.status, ensureShell(lt).body);
      updateTaskCard(ev.task_id, action.status, action.output);
      markTurnTask(lt, ev.task_id, action.status);
    } else {
      paintTaskCard(ev.task_id);
    } return;
  }
  if (t === "final_started") {
    if (lt) {
      const finalEl = document.createElement("div"); finalEl.className = "final";
      finalEl.innerHTML = `<div class="final-label">Agent</div><div class="final-text"></div>`;
      lt.turn.appendChild(finalEl); lt.finalEl = finalEl; scrollDown();
    }
    return;
  }
  if (t === "final_text_delta") { if (lt && lt.finalEl) { lt.finalText += ev.text; lt.finalEl.querySelector(".final-text").textContent = lt.finalText; scrollDown(); } return; }
  if (t === "final_completed") {
    if (lt && lt.finalEl) renderFinalText(lt.finalEl.querySelector(".final-text"), lt.finalText);
    return;
  }
  if (t === "turn_completed") { finishLiveTurn(ev.turn_id, "completed"); return; }
  if (t === "turn_cancelled") { finishLiveTurn(ev.turn_id, "cancelled"); return; }
  if (t === "turn_failed") { finishLiveTurn(ev.turn_id, "failed"); return; }
}

function finishLiveTurn(turnId, outcome) {
  const lt = state.turnsById[turnId];
  if (!lt) return;
  if (lt.shell) {
    if (shouldDropShell(lt.actions, lt.subagents, lt.shell.body.childElementCount)) {
      lt.shell.working.remove();  // 无动作的纯回答：不留空壳（与历史渲染一致）
    } else {
      const icon = {completed: "✓", cancelled: "■", failed: "✗"}[outcome] || "✓";
      lt.shell.spinner.textContent = icon;
      paintTurnSummary(lt, `${Math.max(0, Math.round((Date.now() - lt.startedAt) / 1000))}s`);
      lt.shell.working.classList.remove("open");
    }
  }
  if (state.liveTurn === lt) {
    state.liveTurn = null;
    setStatus("");
    setBusy(false);
  }
  scrollDown();
}

/* ---------- 历史时间线加载与渲染 ---------- */
async function loadTimeline() {
  stream().innerHTML = "";
  if (!state.current) return;
  const sid = state.current;
  const data = await api(`/api/timeline?session=${sid}`);
  if (state.current !== sid) return;  // 快速切换时丢弃过期响应
  state.tasks = Object.fromEntries((data.tasks || []).map(t => [t.id, t]));
  state.context = data.context;
  renderRing(data.context);
  setBusy((data.turns || []).some((t) => t.status === "running"));
  for (const turn of data.turns || []) renderHistoryTurn(turn);
  for (const e of data.loose_events || []) {
    if (e.type === "user_to_executor") note(`你对执行者说：${e.text || ""}`);
  }
  scrollDown();
  updateExecButton();
}

function renderHistoryTurn(turn) {
  if (turn.user_message) userBubble(turn.user_message);
  const events = turn.events || [];
  const turnEl = makeTurn();
  if (hasWork(events)) {
    const shell = makeTurnShell(turnEl);
    shell.working.classList.remove("open");
    shell.spinner.textContent = {completed: "✓", cancelled: "■", error: "✗"}[turn.status] || "✓";
    const actionsByTask = {};
    const turnTasks = {};
    for (const e of events) {
      if (e.type === "subagent_tool_started") {
        actionsByTask[e.task_id] = (actionsByTask[e.task_id] || 0) + 1;
      } else if (e.type === "subagent_queued" || e.type === "subagent_started"
                 || e.type === "subagent_completed" || e.type === "subagent_failed") {
        turnTasks[e.task_id] = e.type === "subagent_queued" ? "queued"
          : e.type === "subagent_started" ? "running"
          : e.type === "subagent_completed" ? "done" : (e.status || "error");
      }
    }
    const counts = { actions: 0, subagents: Object.keys(turnTasks).length };
    for (const e of events) {
      if (e.type === "tool_started" || e.type === "subagent_tool_started") counts.actions++;
    }
    const taskList = Object.entries(turnTasks).map(([id, status]) => ({ id: Number(id), status }));
    shell.summary.textContent = turnSummaryText(
      counts.actions, counts.subagents, taskList,
      turn.created_at && turn.completed_at ? fmtDur(turn.created_at, turn.completed_at) : "",
    );
    for (const e of events) {
      if (e.type === "activity_text_delta") addWorkingText(shell.body, e.text);
      else if (e.type === "tool_started") {
        const row = addToolRow(shell.body, e.name, e.arguments);
        row.querySelector(".t-icon").textContent = "✓";
        row.classList.add("done");
      }
      else if (e.type === "subagent_queued" || e.type === "subagent_started"
               || e.type === "subagent_completed" || e.type === "subagent_failed") {
        const task = state.tasks[e.task_id] || {};
        const status = e.type === "subagent_queued" ? "queued"
          : e.type === "subagent_started" ? "running"
          : e.type === "subagent_completed" ? "done" : (e.status || "error");
        ensureTaskCard(e.task_id, e.text || task.title, status, shell.body);
        if (state.liveTaskProgress[e.task_id]) {
          state.liveTaskProgress[e.task_id].actions = actionsByTask[e.task_id] || 0;
        }
        paintTaskCard(e.task_id);
      }
    }
  } else if (isSystemTurn(turn)) {
    turnEl.appendChild(systemDivider());  // 事件驱动 turn：无用户消息、无动作
  }
  for (const e of events) {
    if (e.type === "user_to_executor") {
      const hint = document.createElement("div");
      hint.className = "msg-note";
      hint.textContent = `你对执行者说：${e.text || ""}`;
      turnEl.appendChild(hint);
    }
  }
  const mainReasoning = events
    .filter(e => e.type === "reasoning_delta" && e.task_id == null)
    .map(e => e.text || "")
    .join("");
  if (mainReasoning) reasoningDetails(turnEl, "思考（主 agent）").textContent = mainReasoning;
  const reasoningByTask = {};
  for (const e of events) {
    if (e.type === "reasoning_delta" && e.task_id != null) {
      (reasoningByTask[e.task_id] = reasoningByTask[e.task_id] || []).push(e.text || "");
    }
  }
  for (const [taskId, chunks] of Object.entries(reasoningByTask)) {
    appendTaskReasoning(Number(taskId), chunks.join(""));
  }
  if (turn.final_text) {
    const finalEl = document.createElement("div"); finalEl.className = "final";
    finalEl.innerHTML = `<div class="final-label">Agent</div><div class="final-text"></div>`;
    renderFinalText(finalEl.querySelector(".final-text"), turn.final_text);
    turnEl.appendChild(finalEl);
  }
}

/* ---------- Task inspector ---------- */
function upsertTask(id, patch) {
  state.tasks[id] = mergeTask(state.tasks[id], id, patch);
}function renderActivity(body, events) {
  body.innerHTML = "";
  for (const g of groupActivityEvents(events)) {
    const row = document.createElement("div"); row.className = "act-row";
    const time = `<span class="act-time">${fmtTime(g.ts)}</span>`;
    if (g.kind === "text") row.innerHTML = `${time}<span class="act-text">${escapeHtml(g.text)}</span>`;
    else if (g.kind === "reasoning") row.innerHTML = `${time}<span class="act-reasoning">${escapeHtml(g.text)}</span>`;
    else if (g.kind === "tool_started") row.innerHTML = `${time}▶ ${escapeHtml(displayToolName(g.name))} <span class="t-detail">${escapeHtml(compactArgs(g.name, g.arguments))}</span>`;
    else if (g.kind === "tool_finished") row.innerHTML = `${time}${g.status === "success" ? "✓" : "✗"} ${escapeHtml(displayToolName(g.name))}`;
    else if (g.kind === "status") row.innerHTML = `${time}${escapeHtml(g.label)}`;
    else continue;
    body.appendChild(row);
  }
}

function renderChanges(body, events) {
  body.innerHTML = "";
  const changes = extractChanges(events);
  if (!changes.length) { body.innerHTML = `<div class="act-row">（无文件修改）</div>`; return; }
  const head = document.createElement("div"); head.className = "act-row"; head.textContent = "变更";
  body.appendChild(head);
  for (const c of changes) {
    const el = document.createElement("div"); el.className = "change-file";
    el.textContent = c.kind === "shell"
      ? `Shell 可能变更：${c.label}`
      : (c.label || c.tool);
    body.appendChild(el);
  }
}

function renderValidation(body, events) {
  body.innerHTML = "";
  const shells = pairToolEvents(events).filter(p => p.name === "run_shell");
  if (!shells.length) { body.innerHTML = `<div class="act-row">（未执行验证命令）</div>`; return; }
  for (const pair of shells) {
    const result = pair.result || "";
    const exitOk = result.includes("exit code: 0");
    const el = document.createElement("div"); el.className = "valid-cmd";
    el.innerHTML = `<div class="mono">${escapeHtml(shortArg(pair.arguments) || "(命令)")}</div>
      <div class="${exitOk ? "exit-ok" : "exit-bad"}">${exitOk ? "exit 0" : "exit ✗"}</div>
      <pre>${escapeHtml(result.split("\n").slice(0, 5).join("\n"))}</pre>`;
    body.appendChild(el);
  }
}

/* ---------- 右栏任务列表 ---------- *//* ---------- 左栏（项目/会话树） ---------- */
function renderTree() {
  const tree = $("project-tree"); tree.innerHTML = "";
  for (const [pid, p] of Object.entries(state.projects)) {
    const pEl = document.createElement("div");
    const head = document.createElement("div"); head.className = "project-name";
    head.innerHTML = `<span>📁 ${escapeHtml(p.name)}</span>`;
    const add = document.createElement("button"); add.className = "add-session"; add.textContent = "+";
    add.title = "新会话"; add.onclick = () => { state.wantNewSession = true; send({type: "new_session", project: pid, title: ""}); };
    head.appendChild(add); pEl.appendChild(head);
    const sessions = Object.values(state.sessions).filter(s => s.project === pid);
    for (const s of sessions) {
      const el = document.createElement("div");
      el.className = "session-item" + (s.id === state.current ? " active" : "") + (state.activity.has(s.id) ? " has-activity" : "");
      const dot = document.createElement("span"); dot.className = "dot";
      const label = document.createElement("span"); label.className = "label";
      label.textContent = sessionLabel(s, sessions);
      const actions = document.createElement("span"); actions.className = "actions";
      const rename = document.createElement("button"); rename.textContent = "✎"; rename.title = "重命名";
      rename.onclick = (e) => {
        e.stopPropagation();
        const title = prompt("重命名会话", s.title || "");
        if (title !== null && title.trim()) send({type: "rename_session", session: s.id, title: title.trim()});
      };
      const del = document.createElement("button"); del.textContent = "✕"; del.title = "删除";
      del.onclick = (e) => {
        e.stopPropagation();
        if (confirm(`删除会话「${s.title || s.id.slice(0,6)}」？对话历史将一并删除。`))
          send({type: "delete_session", session: s.id});
      };
      actions.append(rename, del);
      el.append(dot, label, actions);
      el.onclick = () => selectSession(s.id);
      pEl.appendChild(el);
    }
    tree.appendChild(pEl);
  }
  applyCurrentTitle();
}

function applyCurrentTitle() {
  const s = state.current ? state.sessions[state.current] : null;
  if (!s) { $("current-title").textContent = "agent"; return; }
  const siblings = Object.values(state.sessions).filter(x => x.project === s.project);
  $("current-title").textContent = sessionLabel(s, siblings);  // 与左侧同一标签
}

function selectSession(sid) {
  state.current = sid; state.activity.delete(sid); state.liveTurn = null;
  state.tasks = {}; state.liveTaskProgress = {}; state.turnsById = {};
  state.taskCards = {}; state.taskCardEls = {}; state.execLive = {};
  setBusy(false);
  stream().innerHTML = ""; state.usage = null; state.context = null;
  $("ctx-badge").classList.add("hidden");
  renderUsageLine();
  renderTree(); renderFiles(); loadTimeline();
  if (!$("tab-executor").classList.contains("hidden")) {   // 执行者页可见时跟随会话切换
    refreshExecutor();
    renderExecutorHeader();
  }
}

/* ---------- 文件栏（单层显示 + 面包屑 + 返回上级） ---------- */
async function renderFiles(path = ".") {
  if (!state.current || !state.sessions[state.current]) return;
  const page = $("tab-files");
  const pid = state.sessions[state.current].project;
  page.innerHTML = "";
  state.fsPath = path;

  const crumbs = document.createElement("div"); crumbs.className = "fs-crumbs";
  const parts = breadcrumb(path);
  parts.forEach((part, idx) => {
    const seg = document.createElement("span");
    seg.className = "fs-crumb";
    seg.textContent = part;
    seg.onclick = () => {
      if (idx === 0) { renderFiles("."); return; }
      renderFiles(parts.slice(1, idx + 1).join("/"));
    };
    crumbs.appendChild(seg);
    if (idx < parts.length - 1) {
      const sep = document.createElement("span"); sep.className = "fs-sep"; sep.textContent = "/";
      crumbs.appendChild(sep);
    }
  });
  page.appendChild(crumbs);

  if (path !== ".") {
    const up = document.createElement("div");
    up.className = "fs-up"; up.textContent = "↑ 返回上级";
    up.onclick = () => renderFiles(parentPath(path));
    page.appendChild(up);
  }

  const data = await api(`/api/ls?project=${pid}&path=${encodeURIComponent(path)}`);
  if (data.error) {
    const err = document.createElement("div");
    err.className = "fs-error"; err.textContent = `读取目录失败：${data.error}`;
    page.appendChild(err);
    return;
  }
  const list = document.createElement("div"); list.className = "fs-list";
  for (const line of (data.result || "").split("\n")) {
    if (!line) continue;
    const isDir = line.endsWith("/");
    const name = isDir ? line.slice(0, -1) : line;
    const el = document.createElement("div");
    el.className = "fs-entry" + (isDir ? " dir" : " file");
    el.textContent = (isDir ? "📁 " : "📄 ") + name;
    el.onclick = () => isDir
      ? renderFiles(path === "." ? name : `${path}/${name}`)
      : openFile(path === "." ? name : `${path}/${name}`);
    list.appendChild(el);
  }
  page.appendChild(list);
}

async function openFile(path) {
  if (!state.current || !state.sessions[state.current]) return;
  const pid = state.sessions[state.current].project;
  const data = await api(`/api/read?project=${pid}&path=${encodeURIComponent(path)}&limit=500`);
  const page = $("tab-files");
  const view = document.createElement("div"); view.className = "file-view";
  const head = document.createElement("div"); head.className = "file-path";
  const back = document.createElement("button"); back.textContent = "← 返回";
  back.onclick = () => renderFiles(state.fsPath || ".");
  head.append(back, document.createTextNode(path));
  const pre = document.createElement("pre");
  pre.textContent = data.error ? `读取文件失败：${data.error}` : (data.result || "");
  if (data.error) pre.className = "fs-error";
  view.append(head, pre);
  page.innerHTML = ""; page.appendChild(view);
}

/* ---------- 确认卡 ---------- */
function confirmCard(ev) {
  const el = document.createElement("div"); el.className = "card";
  const h = document.createElement("div"); h.className = "card-title"; h.textContent = "⚠ 危险命令";
  const cmd = document.createElement("pre"); cmd.textContent = ev.command;
  const buttons = document.createElement("div"); buttons.className = "card-buttons";
  const yes = document.createElement("button"); yes.textContent = "允许"; yes.className = "primary";
  const no = document.createElement("button"); no.textContent = "拒绝";
  const answer = (allow) => {
    send({type: "confirm_response", id: ev.id, allow});
    yes.disabled = no.disabled = true;
    h.textContent = allow ? "⚠ 危险命令（已允许）" : "⚠ 危险命令（已拒绝）";
  };
  yes.onclick = () => answer(true); no.onclick = () => answer(false);
  buttons.append(yes, no); el.append(h, cmd, buttons);
  stream().appendChild(el); scrollDown();
}

/* ---------- 输入与命令 ---------- */
let mainBusy = false;
function setBusy(on) {
  mainBusy = on;
  const btn = $("btn-send");
  btn.classList.toggle("stop-mode", on);
  btn.title = on ? "停止" : "发送";
  btn.setAttribute("aria-label", on ? "停止" : "发送");
  btn.querySelector(".icon-send").classList.toggle("hidden", on);
  btn.querySelector(".icon-stop").classList.toggle("hidden", !on);
}
$("btn-send").onclick = () => {
  if (mainBusy) { send({type: "stop", session: state.current}); return; }
  sendInput();
};
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendInput(); }
});
function sendInput() {
  const text = $("input").value.trim();
  if (!text || !state.current) return;
  send({type: "user", session: state.current, text});
  userBubble(text); $("input").value = "";
}
$("btn-plan-mode").onclick = (e) => {
  e.stopPropagation();
  closeMenus();
  $("plan-menu").classList.toggle("hidden");
};
document.querySelectorAll("#plan-menu .menu-item").forEach((el) => {
  el.onclick = () => {
    closeMenus();
    send({type: "set_plan_mode", on: el.dataset.plan === "on"});
  };
});

/* ---------- 设置：服务商（凭据全局共享）+ 角色模型 + 高级 ---------- */

let pendingKeyAction = null;
let lastProviderAction = "settings";

function setResult(el, text, ok) {
  el.textContent = text || "";
  el.style.color = ok ? "#7BAE7F" : "#DC6B5E";
}

function openSettings() {
  send({type: "get_provider_status"});
  send({type: "list_models"});
  $("modal-mask").classList.remove("hidden");
  renderProviders();
  renderRoleSettings();
  renderCapabilityEditor();
  renderUsageLine();
}

function renderProviders() {
  const box = $("provider-list");
  box.innerHTML = "";
  const rows = providerListState(state.providerStatus);
  if (!rows.length) {
    const hint = document.createElement("div");
    hint.className = "hint"; hint.textContent = "正在读取服务商…";
    box.appendChild(hint);
    return;
  }
  for (const row of rows) {
    const el = document.createElement("div");
    el.className = "provider-row";
    const name = document.createElement("span");
    name.className = "provider-name"; name.textContent = row.label;
    const stateEl = document.createElement("span");
    stateEl.className = "provider-state" + (row.configured ? " ok" : "");
    stateEl.textContent = row.configured ? `已连接 · ${row.modelCount} 个模型` : "未连接";
    const actions = document.createElement("span");
    actions.className = "provider-actions";
    const btn = document.createElement("button");
    if (row.configured) {
      btn.textContent = "刷新";
      btn.onclick = () => send({type: "list_models", provider: row.providerId, refresh: true});
      actions.appendChild(btn);
      if (row.hasKey) {
        const off = document.createElement("button");
        off.textContent = "断开";
        off.title = "删除该服务商凭据（换 key 后可重新连接）";
        off.onclick = () => send({type: "disconnect_provider", provider_id: row.providerId});
        actions.appendChild(off);
      }
    } else {
      btn.textContent = "连接";
      btn.onclick = () => startConnectPreset(row);
      actions.appendChild(btn);
    }
    el.append(name, stateEl, actions);
    box.appendChild(el);
  }
}

function startConnectPreset(row) {
  lastProviderAction = "settings";
  const plan = connectRowPlan(row);
  if (plan.kind === "guide-advanced") {
    // 注册表缺少协议信息：引导到高级补全后重连（避免“协议无效”死路）
    $("advanced").open = true;
    $("adv-base-url").value = row.baseUrl || "";
    setResult($("adv-result"), "该 provider 缺少协议信息：请补全协议后重连", false);
    return;
  }
  if (plan.kind === "direct") {
    setResult($("m-result"), `正在连接 ${row.label}…`, true);
    send(connectRequestForRow(row, ""));
    return;
  }
  pendingKeyAction = (key) => send(connectRequestForRow(row, key));
  const desc = row.kind === "custom"
    ? "自定义端点：有密钥就填，本地/无鉴权端点可留空。"
    : "输入 API 密钥以连接账户；密钥只保存在本机。";
  openKeyModal(row.label, desc);
}

function openKeyModal(label, desc) {
  $("key-title").textContent = `连接 ${label}`;
  $("key-desc").textContent = desc;
  $("key-input").value = "";
  setResult($("key-result"), "", true);
  $("key-modal-mask").classList.remove("hidden");
  $("key-input").focus();
}

function closeKeyModal() {
  $("key-modal-mask").classList.add("hidden");
  pendingKeyAction = null;
}

$("key-submit").onclick = () => {
  if (!pendingKeyAction) return;
  pendingKeyAction($("key-input").value.trim());
};
$("key-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); $("key-submit").click(); }
});
$("key-close").onclick = () => closeKeyModal();
$("key-back").onclick = () => closeKeyModal();

function fillEffortSelect(sel, role, st) {
  sel.innerHTML = "";
  sel.onchange = null;
  const stateInfo = effortState({reasoning_mode: st.reasoning_mode, levels: st.efforts});
  if (!stateInfo.selectable) return;  // 调用方已保证可调
  for (const level of stateInfo.levels) {
    const o = document.createElement("option");
    o.value = level; o.textContent = effortLabel(level);
    if (level === (st.effort || "off")) o.selected = true;
    sel.appendChild(o);
  }
  sel.onchange = () => {
    if (sel.value) send({type: "set_reasoning_effort", role, effort: sel.value});
  };
}

function renderRoleSettings() {
  const box = $("role-list");
  box.innerHTML = "";
  const roles = state.providerStatus?.roles || {};
  const groups = roleModelOptions(state.modelCatalog, state.providerStatus);
  for (const role of ["main", "executor"]) {
    const st = roles[role] || {};
    const row = document.createElement("div");
    row.className = "role-row";
    const name = document.createElement("span");
    name.className = "role-name"; name.textContent = role;
    const modelSel = document.createElement("select");
    modelSel.className = "role-model";
    modelSel.title = `${role} 使用的模型`;
    let found = false;
    for (const g of groups) {
      const og = document.createElement("optgroup");
      og.label = g.provider;
      for (const m of g.models) {
        const o = document.createElement("option");
        o.value = `${g.providerId}|${m.id}`;
        o.textContent = m.name;
        if (g.providerId === st.provider_id && m.id === st.model) {
          o.selected = true; found = true;
        }
        og.appendChild(o);
      }
      modelSel.appendChild(og);
    }
    if (!found && st.model) {
      const o = document.createElement("option");
      o.value = `${st.provider_id}|${st.model}`;
      o.textContent = st.model; o.selected = true;
      modelSel.appendChild(o);
    }
    if (!groups.length) {
      const o = document.createElement("option");
      o.value = ""; o.textContent = "先连接服务商"; o.disabled = true;
      modelSel.appendChild(o);
    }
    modelSel.disabled = !groups.length;
    modelSel.onchange = () => {
      const [pid, model] = modelSel.value.split("|");
      if (pid && model) send({type: "set_model", role, provider_id: pid, model});
    };
    if (roleEffortVisible(st)) {
      const effortSel = document.createElement("select");
      effortSel.className = "role-effort";
      effortSel.title = `${role} 的思考强度`;
      fillEffortSelect(effortSel, role, st);
      row.append(name, modelSel, effortSel);
    } else {
      row.append(name, modelSel);  // 能力未知/不可调：不给用户看“未知”控件
    }
    box.appendChild(row);
  }
}

/* ---------- 高级：自定义端点 + 能力覆盖 ---------- */

$("adv-connect").onclick = () => {
  lastProviderAction = "advanced";
  const protocol = $("adv-protocol").value;
  const base_url = $("adv-base-url").value.trim() || null;
  const model = $("adv-model").value.trim();
  const api_key = $("adv-key").value.trim();
  setResult($("adv-result"), "正在连接…", true);
  send({type: "connect_provider", protocol, base_url, model, api_key});
};

function currentRoleName() {
  return $("cap-role").value === "executor" ? "executor" : "main";
}

function renderCapabilityEditor() {
  const editor = capabilityEditorState(state.providerStatus?.roles, currentRoleName());
  const form = editor.form;
  $("cap-title").textContent = editor.title;
  $("cap-window").value = form.window;
  $("cap-max-output").value = form.maxOutput;
  $("cap-mode").value = form.mode;
  $("cap-levels").value = form.levelsText;
  $("cap-tools").value = form.tools;
  const sourceEl = $("cap-source");
  sourceEl.textContent = `当前来源：${form.source}`;
  sourceEl.dataset.source = form.sourceKey;
  sourceEl.dataset.levelsUnknown = form.levelsUnknown ? "1" : "0";
}

function _capabilityFormValues() {
  return {
    window: $("cap-window").value,
    maxOutput: $("cap-max-output").value,
    mode: $("cap-mode").value,
    levelsText: $("cap-levels").value,
    levelsUnknown: $("cap-source").dataset.levelsUnknown === "1",
    tools: $("cap-tools").value,
  };
}

$("cap-save").onclick = () => {
  const editor = capabilityEditorState(state.providerStatus?.roles, currentRoleName());
  if (editor.empty) {
    $("cap-source").textContent = "尚未绑定 provider，无法保存能力";
    return;
  }
  send({type: "set_model_capability", provider: editor.target.provider,
        model: editor.target.model, override: capabilityOverrideFromForm(_capabilityFormValues())});
};
$("cap-clear").onclick = () => {
  const editor = capabilityEditorState(state.providerStatus?.roles, currentRoleName());
  if (editor.empty) return;
  send({type: "set_model_capability", provider: editor.target.provider,
        model: editor.target.model,
        override: {window: null, max_output: null, reasoning_mode: null, levels: null, tools: null,
                   reasoning_returned: null}});
};
$("cap-role").onchange = () => renderCapabilityEditor();

$("btn-settings").onclick = () => openSettings();
$("m-close").onclick = () => {
  $("modal-mask").classList.add("hidden");
  $("m-result").textContent = "";
};

function renderRingFromStatus(roles) {
  const used = (state.context && state.context.used_tokens) || 0;
  const exact = Boolean(state.context && state.context.exact);
  const window = (roles.main && roles.main.window) || null;
  const percent = window ? Math.round((used / window) * 100) : null;
  $("ctx-ring-wrap").classList.toggle("hidden", !ringVisible(window));
  renderRing({ used_tokens: used, exact, window, percent });
}

function renderProviderStatus() {
  const roles = state.providerStatus?.roles || {};
  renderModelButton(roles);
  renderEffortButton(roles.main);
  renderRingFromStatus(roles);  // 模型切换后窗口能力立即反映到环
  renderExecutorHeader();
  if (!$("modal-mask").classList.contains("hidden")) {
    renderProviders();
    renderRoleSettings();
    renderCapabilityEditor();
  }
}

function closeMenus() {
  for (const id of ["effort-menu", "plan-menu"]) $(id).classList.add("hidden");
}

/* ---------- 模型选择器（真实 provider 列表） ---------- */

function openModelModal() {
  $("model-modal-mask").classList.remove("hidden");
  renderModelModal();
  send({type: "list_models"});
}

function renderModelModal() {
  const groupBox = $("mm-groups");
  const errorBox = $("mm-error");
  groupBox.innerHTML = "";
  const menu = modelMenuState(state.modelCatalog);
  const query = $("mm-search").value;
  const manualSelect = $("mm-manual-provider");
  const configured = (menu.groups || []).filter((g) => g.configured !== false);
  manualSelect.innerHTML = "";
  for (const g of configured) {
    const opt = document.createElement("option");
    opt.value = g.id; opt.textContent = g.provider; manualSelect.appendChild(opt);
  }
  if (menu.errors && menu.errors.length) {
    errorBox.textContent = menu.errors.join("；");
    errorBox.classList.remove("hidden");
  } else if (menu.loading) {
    errorBox.classList.add("hidden");
  } else {
    errorBox.classList.add("hidden");
  }
  if (menu.loading) {
    const row = document.createElement("div");
    row.className = "act-row"; row.textContent = "正在获取模型列表…";
    groupBox.appendChild(row);
    return;
  }
  if (!menu.models.length) {
    const row = document.createElement("div");
    row.className = "act-row";
    row.textContent = "（无模型可显示，可在下方手动输入模型 ID）";
    groupBox.appendChild(row);
    return;
  }
  const current = state.providerStatus?.roles?.main || {};
  for (const group of menu.groups) {
    const models = filterModels(group.models, query);
    const head = document.createElement("div");
    head.className = "mm-group-title";
    const label = document.createElement("span");
    label.textContent = group.provider;
    head.appendChild(label);
    const refresh = document.createElement("button");
    refresh.className = "mm-refresh-one";
    refresh.textContent = "刷新";
    refresh.onclick = () => {
      state.modelCatalog = null;
      renderModelModal();
      send({type: "list_models", provider: group.id, refresh: true});
    };
    head.appendChild(refresh);
    groupBox.appendChild(head);
    if (group.error) {
      const row = document.createElement("div");
      row.className = "act-row";
      row.textContent = `${group.error}（可在下方手动输入模型 ID）`;
      groupBox.appendChild(row);
      // 不 continue：保留最后一次成功的模型缓存，仍可查看/选择
    }
    if (!models.length) {
      if (query) continue;
      const row = document.createElement("div");
      row.className = "act-row"; row.textContent = "（该 provider 暂无模型）";
      groupBox.appendChild(row);
      continue;
    }
    for (const model of models) {
      const item = document.createElement("button");
      const isCurrent = model.provider_id === current.provider_id && model.id === current.model;
      item.className = "menu-item mm-model" + (isCurrent ? " active" : "");
      const cap = model.capability || {};
      const badges = [];
      if (cap.window) badges.push(`${Math.round(cap.window / 1000)}k 窗口`);
      if (cap.reasoning_mode && cap.reasoning_mode !== "unknown") {
        badges.push(cap.reasoning_mode === "adjustable" ? "可调推理"
          : cap.reasoning_mode === "fixed" ? "固定推理" : "无推理");
      }
      item.textContent = `${isCurrent ? "● " : ""}${model.display_name || model.id}`
        + (badges.length ? `  · ${badges.join(" · ")}` : "");
      item.title = model.id;
      item.onclick = () => selectCatalogModel(group.id, model.id, group.configured !== false);
      groupBox.appendChild(item);
    }
  }
}

function selectCatalogModel(providerId, modelId, configured) {
  const action = modelSelectAction(
    providerId, state.providerStatus?.roles?.main?.provider_id || "", configured
  );
  closeModelModal();
  if (action === "connect") {
    openSettings();
    setResult($("m-result"), "该服务商尚未连接：请在服务商列表点击「连接」并填入 API 密钥", false);
    return;
  }
  send({type: "set_model", role: "main", provider_id: providerId, model: modelId});
}

function closeModelModal() {
  $("model-modal-mask").classList.add("hidden");
}

function renderModelButton(roles) {
  const main = roles.main || {};
  $("btn-model-label").textContent = main.configured
    ? (modelDisplayName(main.provider_id, main.model) || main.model)
    : "未配置";
  $("btn-model").title = main.configured
    ? `指挥者（主 agent）的模型：${main.model}（点击从 provider 列表选择）`
    : "指挥者尚未配置 provider，点击进入设置";
}

function modelDisplayName(providerId, modelId) {
  if (!modelId) return "";
  for (const g of roleModelOptions(state.modelCatalog, state.providerStatus)) {
    if (g.providerId !== providerId) continue;
    for (const m of g.models) if (m.id === modelId) return m.name || m.id;
  }
  return modelId;
}

function renderEffortButton(mainStatus) {
  const wrap = $("btn-effort").closest(".menu-wrap");
  if (!roleEffortVisible(mainStatus)) {
    // 能力未知/不可调：不显示思考控件，也不暴露内部原因
    wrap.classList.add("hidden");
    $("effort-menu").classList.add("hidden");
    $("effort-menu").innerHTML = "";
    return;
  }
  wrap.classList.remove("hidden");
  const stateInfo = effortState({
    reasoning_mode: mainStatus.reasoning_mode,
    levels: mainStatus.efforts,
  });
  const effort = (mainStatus && mainStatus.effort) || "off";
  $("btn-effort-label").textContent = effortLabel(effort);
  $("btn-effort").title = "指挥者（主 agent）的思考强度";
  const menu = $("effort-menu");
  menu.innerHTML = "";
  for (const level of stateInfo.levels) {
    const item = document.createElement("button");
    item.className = "menu-item" + (level === effort ? " active" : "");
    item.textContent = effortLabel(level);
    item.onclick = () => {
      closeMenus();
      send({type: "set_reasoning_effort", role: "main", effort: level});
    };
    menu.appendChild(item);
  }
}

$("btn-model").onclick = (e) => {
  e.stopPropagation();
  closeMenus();
  openModelModal();
};
$("btn-effort").onclick = (e) => {
  e.stopPropagation();
  $("effort-menu").classList.toggle("hidden");
};
document.addEventListener("click", () => closeMenus());

$("mm-close").onclick = () => closeModelModal();
$("mm-refresh").onclick = () => {
  state.modelCatalog = null;
  renderModelModal();
  send({type: "list_models", refresh: true});
};
$("mm-search").addEventListener("input", () => renderModelModal());
$("mm-use-manual").onclick = () => {
  const modelId = $("mm-manual").value.trim();
  const providerId = $("mm-manual-provider").value;
  if (!modelId) return;
  if (!providerId) { openSettings(); return; }
  selectCatalogModel(providerId, modelId, true);
};
$("btn-new-project").onclick = () => { $("modal2-mask").classList.remove("hidden"); browseDir(); };
$("p-cancel").onclick = () => { $("modal2-mask").classList.add("hidden"); $("p-result").textContent = ""; };
$("p-ok").onclick = () => {
  send({type: "create_project", name: $("p-name").value.trim(), path: $("p-path").value.trim()});
};
let dirCurrentPath = "";
async function browseDir(path) {
  const data = await api(path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse");
  if (data.error) { $("dir-current").textContent = data.error; return; }
  dirCurrentPath = data.path;
  $("dir-current").textContent = data.path;
  const list = $("dir-list"); list.innerHTML = "";
  const parent = document.createElement("div");
  parent.className = "dir-item parent"; parent.textContent = "..（上一级）";
  parent.onclick = () => browseDir(data.parent);
  list.appendChild(parent);
  for (const d of data.dirs) {
    const el = document.createElement("div");
    el.className = "dir-item"; el.textContent = "📁 " + d;
    el.onclick = () => browseDir(data.path + "/" + d);
    list.appendChild(el);
  }
}
$("p-pick").onclick = () => { $("p-path").value = dirCurrentPath; };
function closeModal2() { $("modal2-mask").classList.add("hidden"); $("p-result").textContent = ""; $("p-name").value = ""; $("p-path").value = ""; }

/* ---------- 侧栏拖动与收起 ---------- */
function makeResizable(handleId, side) {
  const handle = $(handleId);
  const pane = $(side);
  const varName = side === "left" ? "--left-w" : "--right-w";
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = pane.getBoundingClientRect().width;
    const move = (ev) => {
      const dx = ev.clientX - startX;
      const w = Math.max(120, Math.min(700, side === "left" ? startW + dx : startW - dx));
      $("app").style.setProperty(varName, w + "px");
    };
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });
}
makeResizable("div-left", "left");
makeResizable("div-right", "right");
function togglePane(side) {
  const pane = $(side);
  const varName = side === "left" ? "--left-w" : "--right-w";
  if (pane.style.display === "none") {
    pane.style.display = "";
    $("app").style.setProperty(varName, (side === "left" ? 240 : 320) + "px");
  } else {
    pane.style.display = "none";
    $("app").style.setProperty(varName, "0px");
  }
}
$("collapse-left").onclick = () => togglePane("left");
$("collapse-right").onclick = () => togglePane("right");

/* ---------- 响应式布局与初始渲染 ---------- */
function applyComposerLayout() {
  const narrow = composerLayout(window.innerWidth) === "narrow";
  $("composer").classList.toggle("narrow", narrow);
}
window.addEventListener("resize", applyComposerLayout);
applyComposerLayout();
renderPlanLabel();

/* tabs */
for (const tab of document.querySelectorAll(".tab")) {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    $("tab-files").classList.toggle("hidden", tab.dataset.tab !== "files");
    $("tab-executor").classList.toggle("hidden", tab.dataset.tab !== "executor");
    if (tab.dataset.tab === "executor") { refreshExecutor(); renderExecutorHeader(); }
  };
}

/* ---------- 执行者会话面板（右栏）：完整历史 + 直连输入 ---------- */

let execRefreshing = false;
let execPending = false;

function openExecutorTab() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === "executor");
  }
  $("tab-files").classList.add("hidden");
  $("tab-executor").classList.remove("hidden");
  refreshExecutor();
  renderExecutorHeader();
  updateExecButton();
}

function updateExecButton() {
  const busy = Object.values(state.tasks)
    .some((t) => t && (t.status === "running" || t.status === "queued"));
  state.execBusy = busy;
  const btn = $("exec-send");
  btn.classList.toggle("stop-mode", busy);
  btn.title = busy ? "中断执行者当前任务" : "发送";
  btn.setAttribute("aria-label", busy ? "中断执行者当前任务" : "发送");
  btn.querySelector(".icon-send").classList.toggle("hidden", busy);
  btn.querySelector(".icon-stop").classList.toggle("hidden", !busy);
}

function renderExecReport(on) {
  state.execReport = !!on;
  const btn = $("exec-report");
  btn.textContent = `回报：${on ? "开" : "关"}`;
  btn.classList.toggle("on", !!on);
}

function renderExecutorHeader() {
  const st = state.providerStatus?.roles?.executor || {};
  const modelSel = $("exec-model");
  const groups = roleModelOptions(state.modelCatalog, state.providerStatus);
  modelSel.innerHTML = "";
  let found = false;
  for (const g of groups) {
    const og = document.createElement("optgroup");
    og.label = g.provider;
    for (const m of g.models) {
      const o = document.createElement("option");
      o.value = `${g.providerId}|${m.id}`;
      o.textContent = m.name;
      if (g.providerId === st.provider_id && m.id === st.model) { o.selected = true; found = true; }
      og.appendChild(o);
    }
    modelSel.appendChild(og);
  }
  if (!found && st.model) {
    const o = document.createElement("option");
    o.value = `${st.provider_id}|${st.model}`;
    o.textContent = st.model; o.selected = true;
    modelSel.appendChild(o);
  }
  if (!groups.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "先连接服务商"; o.disabled = true;
    modelSel.appendChild(o);
  }
  modelSel.disabled = !groups.length;
  modelSel.onchange = () => {
    const [pid, model] = modelSel.value.split("|");
    if (pid && model) send({type: "set_model", role: "executor", provider_id: pid, model});
  };
  const effortSel = $("exec-effort");
  if (roleEffortVisible(st)) {
    effortSel.classList.remove("hidden");
    fillEffortSelect(effortSel, "executor", st);
  } else {
    effortSel.classList.add("hidden");   // 能力未知/不可调：不显示，避免"未知"噪音
    effortSel.innerHTML = "";
  }
}

async function refreshExecutor() {
  if (!state.current) return;
  if (execRefreshing) { execPending = true; return; }
  execRefreshing = true;
  const sid = state.current;
  try {
    const data = await api(`/api/executor_messages?session=${encodeURIComponent(sid)}`);
    if (state.current !== sid) return;   // 快速切换：丢弃过期响应
    renderExecReport(data.report !== false);
    renderExecutorMessages(data.messages || [], data.inflight || []);
  } finally {
    execRefreshing = false;
    if (execPending) { execPending = false; refreshExecutor(); }  // 尾调用补偿
  }
}

function maybeRefreshExecutor(session) {
  if (session === state.current && !$("tab-executor").classList.contains("hidden")) {
    refreshExecutor();
  }
}

function renderExecutorMessages(messages, inflight) {
  const box = $("exec-timeline");
  const stick = shouldStickToBottom(box.scrollHeight, box.scrollTop, box.clientHeight);
  const prevTop = box.scrollTop;
  box.innerHTML = "";
  if (!messages.length && !(inflight && inflight.length)) {
    const empty = document.createElement("div");
    empty.className = "exec-empty";
    empty.textContent = "（执行者还没有任何任务）";
    box.appendChild(empty);
    updateExecButton();
    return;
  }
  for (const raw of messages) {
    const m = executorMessageView(raw);
    if (m.isToolResult) {
      const el = document.createElement("div");
      el.className = "exec-msg-result";
      el.textContent = `↳ 工具结果：${m.text.slice(0, 300)}`;
      box.appendChild(el);
      continue;
    }
    if (m.role === "user") {
      const el = document.createElement("div");
      el.className = "exec-msg-user";
      el.textContent = m.text;
      renderMath(el);
      box.appendChild(el);
      continue;
    }
    if (m.role !== "assistant") continue;
    const el = document.createElement("div");
    el.className = "exec-msg-assistant";
    if (m.reasoning) {
      const details = document.createElement("details");
      details.className = "exec-msg-reasoning";
      const sum = document.createElement("summary");
      sum.textContent = "思考";
      const pre = document.createElement("pre");
      pre.textContent = m.reasoning;
      details.append(sum, pre);
      el.appendChild(details);
    }
    if (m.text) {
      const text = document.createElement("div");
      text.textContent = m.text;
      el.appendChild(text);
    }
    for (const call of m.calls) {
      const row = document.createElement("div");
      row.className = "exec-msg-tool";
      row.textContent = `🔧 ${call.name} ${JSON.stringify(call.arguments).slice(0, 120)}`;
      el.appendChild(row);
    }
    renderMath(el);
    box.appendChild(el);
  }
  renderExecutorInflight(inflight);
  renderExecLiveBuffers(inflight);
  if (stick) box.scrollTop = box.scrollHeight;
  else box.scrollTop = prevTop;
  updateExecButton();
}

function renderExecLiveBuffers(inflight) {
  const box = $("exec-timeline");
  const assistants = (inflight || [])
    .map((raw) => executorMessageView(raw))
    .filter((m) => m.role === "assistant").length;
  for (const [tid, buf] of Object.entries(state.execLive)) {
    if (!buf.text && !buf.reasoning) continue;
    const steps = state.liveTaskProgress[tid]?.steps;
    // 步数基线未知（如切会话后回包未到）时不退役：等事件驱动的清理，避免误清前缀
    if (steps !== undefined && liveBufferStale(steps, assistants)) {
      delete state.execLive[tid];   // 本轮已进入 inflight 快照：实时缓冲退役
      continue;
    }
    const el = document.createElement("div");
    el.id = `exec-live-${tid}`;
    el.className = "exec-msg-assistant inflight live";
    renderExecLive(el, buf);
    box.appendChild(el);
  }
}

function renderExecLive(el, buf) {
  el.textContent = "";
  if (buf.reasoning) {
    const details = document.createElement("details");
    details.className = "exec-msg-reasoning";
    const sum = document.createElement("summary");
    sum.textContent = "思考";
    const pre = document.createElement("pre");
    pre.textContent = buf.reasoning;
    details.append(sum, pre);
    el.appendChild(details);
  }
  const text = document.createElement("div");
  text.className = "exec-live-text";
  text.textContent = buf.text;
  el.appendChild(text);
}

function execAppendLive(taskId, kind, text) {
  if (taskId === undefined || taskId === null || !text) return;
  const buf = state.execLive[taskId]
    || (state.execLive[taskId] = {text: "", reasoning: ""});
  buf[kind] += text;
  if ($("tab-executor").classList.contains("hidden")) return;
  const box = $("exec-timeline");
  const stick = shouldStickToBottom(box.scrollHeight, box.scrollTop, box.clientHeight);
  let el = document.getElementById(`exec-live-${taskId}`);
  if (!el) {
    el = document.createElement("div");
    el.id = `exec-live-${taskId}`;
    el.className = "exec-msg-assistant inflight live";
    box.appendChild(el);
  }
  renderExecLive(el, buf);
  if (stick) box.scrollTop = box.scrollHeight;
}

function clearExecLive(taskId) {
  if (taskId === undefined || taskId === null) return;
  delete state.execLive[taskId];
  const el = document.getElementById(`exec-live-${taskId}`);
  if (el) el.remove();
}

function renderExecutorInflight(inflight) {
  // 进行中任务的消息：淡显（未持久化，任务结束后才进入正式历史）
  const box = $("exec-timeline");
  for (const raw of inflight || []) {
    const el = document.createElement("div");
    const m = executorMessageView(raw);
    el.className = (m.role === "user" ? "exec-msg-user" : "exec-msg-assistant") + " inflight";
    if (m.role === "user") {
      el.textContent = m.text;
      renderMath(el);
      box.appendChild(el);
      continue;
    }
    if (m.text) {
      const text = document.createElement("div");
      text.textContent = m.text;
      el.appendChild(text);
    }
    for (const call of m.calls) {
      const row = document.createElement("div");
      row.className = "exec-msg-tool";
      row.textContent = `🔧 ${call.name} ${JSON.stringify(call.arguments).slice(0, 120)}`;
      el.appendChild(row);
    }
    renderMath(el);
    if (el.childElementCount) box.appendChild(el);
  }
}

$("exec-send").onclick = () => {
  if (!state.current) return;
  if (state.execBusy) {
    send({type: "executor_stop", session: state.current});
    flash("正在中断执行者任务…");
    return;
  }
  const text = $("exec-input").value.trim();
  if (!text) return;
  send({type: "executor_message", session: state.current, text});
  $("exec-input").value = "";
};
$("exec-report").onclick = () => {
  if (!state.current) return;
  send({type: "set_executor_report", session: state.current, on: !state.execReport});
};
$("exec-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = $("exec-input").value.trim();
    if (!text || !state.current) return;
    send({type: "executor_message", session: state.current, text});
    $("exec-input").value = "";
  }
});

// 执行者相关事件 → 面板刷新（实时）
const EXEC_EVENT_TYPES = new Set([
  "subagent_queued", "subagent_started", "subagent_step", "subagent_text_delta",
  "subagent_tool_started", "subagent_tool_finished", "subagent_completed", "subagent_failed",
  "reasoning_delta",
]);

