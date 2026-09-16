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
  effortState,
  extractChanges,
  finalPreview,
  filterModels,
  groupActivityEvents,
  hasWork,
  inspectorLines,
  isSystemTurn,
  mergeTask,
  modelMenuState,
  modelSelectAction,
  pairToolEvents,
  parentPath,
  planLabel,
  planShort,
  planTaskCard,
  providerListState,
  roleModelOptions,
  reasoningTarget,
  ringLabel,
  routeActionEvent,
  sessionLabel,
  shouldDropShell,
  shortArg,
  taskCardAction,
  taskCardLines,
  taskStatusText,
  turnSummaryText,
} from "./timeline_logic.mjs";

const $ = (id) => document.getElementById(id);
const TOKEN = "__AGENT_TOKEN__";
const state = {
  projects: {}, sessions: {}, current: null, activity: new Set(),
  tasks: {}, liveTurn: null, planMode: false, wantNewSession: false,
  providerStatus: null, liveTaskProgress: {}, turnsById: {},
  taskCards: {}, taskCardEls: {}, fsPath: ".",
  modelCatalog: null,
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
  $("plan-label").textContent = planShort(state.planMode);
  $("plan-label").title = planLabel(state.planMode);
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
      $("plan-toggle").checked = ev.plan_mode;
      renderPlanLabel();
    }
    renderProviderStatus();
    return;
  }
  if (t === "model_catalog") {
    state.modelCatalog = ev;
    renderModelModal();
    if (!$("modal-mask").classList.contains("hidden")) renderRoleSettings();
    return;
  }
  if (t === "plan_mode") {
    state.planMode = ev.on;
    $("plan-toggle").checked = ev.on;
    renderPlanLabel();
    return;
  }
  if (t === "session_list" || t === "task_list") { if (t === "task_list") renderTaskList(ev.tasks); return; }

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
  if (t === "error") { note(`[错误] ${ev.message}`); return; }
  if (t === "stopped") { setStatus(""); note("[已中断]"); return; }
  if (t === "turn_end") { setStatus(""); return; }

  // 执行事件 → 时间线
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
  $("ctx-ring-wrap").title = info.unknown
    ? "上下文：未知模型，无可用上限"
    : `上下文：${info.sub}`;
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
  el.innerHTML = sanitize(marked.parse(preview.text));
  if (!preview.truncated) return;
  const wrap = el.closest(".final") || el;
  const btn = document.createElement("button");
  btn.className = "final-toggle"; btn.textContent = "展开全文";
  btn.onclick = () => { el.innerHTML = sanitize(marked.parse(text)); btn.remove(); };
  wrap.appendChild(btn);
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
  card.onclick = () => openInspector(taskId);
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
  refreshTasks();
}

/* ---------- 实时执行事件 ---------- */
function routeExecution(ev) {
  const t = ev.type;
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
    }
    upsertTask(ev.task_id, {title: action.title, status: action.status}); refreshTasks(); return;
  }
  if (t === "subagent_tool_started") {
    const prog = state.liveTaskProgress[ev.task_id] ||
      (state.liveTaskProgress[ev.task_id] = {steps: 0, actions: 0, last: ""});
    prog.actions++;
    if (lt) {
      ensureTaskCard(ev.task_id, "", "running", ensureShell(lt).body);
      routeActionEvent(state.turnsById, ev);  // 旧任务事件只加旧 turn
      markTurnTask(lt, ev.task_id, "running");
    }
    prog.last = `${displayToolName(ev.name)} ${compactArgs(ev.name, ev.arguments)}`.trim();
    upsertTask(ev.task_id, {last_action: prog.last});
    paintTaskCard(ev.task_id);
    refreshTasks();
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
    }
    refreshTasks(); return;
  }
  if (t === "subagent_completed") {
    const action = taskCardAction(ev);
    const hitLimit = action.output.startsWith("[已达最大步数上限");
    upsertTask(ev.task_id, {status: action.status, stop_reason: hitLimit ? "max_steps" : ""});
    if (lt) {
      ensureTaskCard(ev.task_id, state.tasks[ev.task_id]?.title || "", action.status, ensureShell(lt).body);
      updateTaskCard(ev.task_id, action.status, action.output);
      markTurnTask(lt, ev.task_id, action.status);
    }
    refreshTasks(); return;
  }
  if (t === "subagent_failed") {
    const action = taskCardAction(ev);
    upsertTask(ev.task_id, {status: action.status});
    if (lt) {
      ensureTaskCard(ev.task_id, state.tasks[ev.task_id]?.title || "", action.status, ensureShell(lt).body);
      updateTaskCard(ev.task_id, action.status, action.output);
      markTurnTask(lt, ev.task_id, action.status);
    }
    refreshTasks(); return;
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
  for (const turn of data.turns || []) renderHistoryTurn(turn);
  refreshTasks();
  scrollDown();
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
}

async function openInspector(taskId) {
  const page = $("tab-tasks");
  const data = await api(`/api/task?session=${state.current}&task=${taskId}`);
  if (data.error || !data.task) { note(data.error || `任务不存在: #${taskId}`); return; }
  const task = data.task, events = data.events || [];
  page.innerHTML = "";
  const wrap = document.createElement("div"); wrap.className = "inspector";
  const dur = task.status === "unconfigured" ? "未执行" : fmtDur(task.started_at || task.created_at, task.completed_at);
  const actions = events.filter(e => e.type === "subagent_tool_started").length;
  const [progressLine, metaLine] = inspectorLines({
    status: task.status, steps: task.steps_used, maxSteps: task.max_steps,
    actions, lastAction: task.last_action, stopReason: task.stop_reason,
    durationLabel: dur, model: task.model || "",
  });
  wrap.innerHTML = `
    <div class="inspector-head">
      <h4>Task #${task.id} · ${escapeHtml(task.title || "")}</h4>
      <div class="meta">${escapeHtml(progressLine)}</div>
      <div class="meta">${escapeHtml(metaLine)}</div>
    </div>
    <div class="inspector-tabs">
      <button data-v="activity" class="active">Activity</button>
      <button data-v="result">Result</button>
      <button data-v="changes">Changes</button>
      <button data-v="validation">Validation</button>
    </div>
    <div class="inspector-body"></div>`;
  const back = document.createElement("button"); back.textContent = "← 任务列表";
  back.onclick = () => { page.innerHTML = ""; refreshTasks(); };
  const body = wrap.querySelector(".inspector-body");
  const views = {
    activity: () => renderActivity(body, events),
    result: () => {
      body.innerHTML = "";
      if (task.status === "unconfigured") {
        const noteEl = document.createElement("div");
        noteEl.className = "act-row";
        noteEl.textContent = "executor 未配置：任务未执行。请先连接 executor 后重新派发（重派=从头执行）";
        body.appendChild(noteEl);
      } else if (task.stop_reason === "max_steps") {
        const noteEl = document.createElement("div");
        noteEl.className = "act-row";
        noteEl.textContent = "已达最大步数上限：任务可能未完成（需要时可重派一次，重派=从头执行）";
        body.appendChild(noteEl);
      } else if (task.status === "error") {
        const noteEl = document.createElement("div");
        noteEl.className = "act-row";
        noteEl.textContent = "任务执行失败：原因见下方输出";
        body.appendChild(noteEl);
      }
      const pre = document.createElement("pre"); pre.textContent = task.final_output || "(无输出)";
      body.appendChild(pre);
    },
    changes: () => renderChanges(body, events),
    validation: () => renderValidation(body, events),
  };
  wrap.querySelectorAll(".inspector-tabs button").forEach(btn => {
    btn.onclick = () => {
      wrap.querySelectorAll(".inspector-tabs button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      body.innerHTML = "";
      views[btn.dataset.v]();
    };
  });
  page.append(back, wrap);
  // 切到任务页签
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelector('.tab[data-tab="tasks"]').classList.add("active");
  $("tab-files").classList.add("hidden");
  page.classList.remove("hidden");
  views.activity();
}

function renderActivity(body, events) {
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

/* ---------- 右栏任务列表 ---------- */
function refreshTasks() {
  if (!state.current) return;
  const page = $("tab-tasks");
  if (page.querySelector(".inspector")) return; // 正在看 inspector 不刷
  page.innerHTML = "";
  const tasks = Object.values(state.tasks).sort((a, b) => (b.id || 0) - (a.id || 0));
  if (!tasks.length) { page.innerHTML = `<div class="act-row">（暂无任务）</div>`; return; }
  for (const t of tasks) {
    const el = document.createElement("div");
    el.className = "task-item";
    const steps = t.steps_used ? ` · ${t.steps_used}/${t.max_steps || "?"} 步` : "";
    const acts = t.actions_used ? ` · ${t.actions_used} 动作` : "";
    el.innerHTML = `<span>#${t.id} ${escapeHtml(t.title || "")}${steps}${acts}</span><span class="task-status ${t.status}">${taskStatusText(t.status)}</span>`;
    el.onclick = () => openInspector(t.id);
    page.appendChild(el);
  }
}
function renderTaskList(tasks) {
  for (const t of tasks) upsertTask(t.id, t);
  refreshTasks();
}

/* ---------- 左栏（项目/会话树） ---------- */
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
  state.taskCards = {}; state.taskCardEls = {};
  stream().innerHTML = ""; state.usage = null; state.context = null;
  $("ctx-badge").classList.add("hidden");
  renderUsageLine();
  renderTree(); renderFiles(); loadTimeline();
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
$("btn-send").onclick = sendInput;
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendInput(); }
});
function sendInput() {
  const text = $("input").value.trim();
  if (!text || !state.current) return;
  send({type: "user", session: state.current, text});
  userBubble(text); $("input").value = "";
}
$("btn-stop").onclick = () => send({type: "stop", session: state.current});
$("plan-toggle").onchange = (e) => send({type: "set_plan_mode", on: e.target.checked});

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
    } else {
      btn.textContent = "连接";
      btn.onclick = () => startConnectPreset(row);
    }
    actions.appendChild(btn);
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
  if (!st.configured) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "未配置"; o.disabled = true; sel.appendChild(o);
    return;
  }
  if (!stateInfo.selectable) {
    const hints = {fixed: "固定", none: "无推理", unknown: "未知", "unknown-levels": "档位未知"};
    const o = document.createElement("option");
    o.value = ""; o.textContent = hints[stateInfo.mode] || "不可调"; o.disabled = true;
    sel.appendChild(o);
    return;
  }
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
    const effortSel = document.createElement("select");
    effortSel.className = "role-effort";
    effortSel.title = `${role} 的思考强度`;
    fillEffortSelect(effortSel, role, st);
    row.append(name, modelSel, effortSel);
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
  renderRing({ used_tokens: used, exact, window, percent });
}

function renderProviderStatus() {
  const roles = state.providerStatus?.roles || {};
  renderModelButton(roles);
  renderEffortButton(roles.main);
  renderRingFromStatus(roles);  // 模型切换后窗口能力立即反映到环
  if (!$("modal-mask").classList.contains("hidden")) {
    renderProviders();
    renderRoleSettings();
    renderCapabilityEditor();
  }
}

function closeMenus() {
  for (const id of ["effort-menu"]) $(id).classList.add("hidden");
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
  $("btn-model").textContent = `模型：${main.model || "未配置"} ▾`;
  $("btn-model").title = main.configured
    ? `当前模型：${main.model}（点击从 provider 列表选择）`
    : "尚未配置 provider，点击进入设置";
}

function renderEffortButton(mainStatus) {
  const stateInfo = effortState(mainStatus && {
    reasoning_mode: mainStatus.reasoning_mode,
    levels: mainStatus.efforts,
  });
  const effort = (mainStatus && mainStatus.effort) || "off";
  const label = stateInfo.mode === "adjustable"
    ? effortLabel(effort)
    : stateInfo.label;
  $("btn-effort").textContent = `思考：${label} ▾`;
  const menu = $("effort-menu");
  menu.innerHTML = "";
  if (!stateInfo.selectable) {
    const hints = {
      fixed: "该模型思考固定开启，不支持调节",
      none: "该模型不支持推理",
      unknown: "模型能力未知（provider 未返回能力信息）",
      "unknown-levels": "档位未知（请在模型能力中声明）",
    };
    const item = document.createElement("button");
    item.className = "menu-item dim"; item.disabled = true;
    item.textContent = hints[stateInfo.mode] || "不可调节";
    menu.appendChild(item);
    return;
  }
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
    $("tab-tasks").classList.toggle("hidden", tab.dataset.tab !== "tasks");
    if (tab.dataset.tab === "tasks") refreshTasks();
  };
}
