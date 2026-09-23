import test from "node:test";
import assert from "node:assert/strict";

import {
  pendingBindPlan,
  breadcrumb,
  composerLayout,
  executorMessageView,
  connectRequestForRow,
  connectRowPlan,
  contextLabel,
  capabilityEditorState,
  capabilityFormState,
  capabilityOverrideFromForm,
  effortChipVisible,
  effortLabel,
  effortState,
  countActions,
  extractChanges,
  finalPreview,
  filterModels,
  groupActivityEvents,
  hasWork,
  historySummary,
  inspectorLines,
  isSystemTurn,
  liveSummary,
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
  ringLabel,
  ringVisible,
  roleEffortVisible,
  reasoningTarget,
  roleStatusText,
  routeActionEvent,
  sessionLabel,
  shouldDropShell,
  commandQuery,
  diffSummary,
  extractMath,
  insertMention,
  liveBufferStale,
  mentionQuery,
  shouldStickToBottom,
  subagentTaskCount,
  taskCardAction,
  taskCardLines,
  taskProgressText,
  taskStatusText,
  turnSummaryText,
  workingCounts,
} from "./timeline_logic.mjs";

test("动作事件分类：final/turn 事件不算 work", () => {
  assert.equal(hasWork([]), false);
  assert.equal(
    hasWork([
      { type: "turn_started" },
      { type: "final_started" },
      { type: "final_text_delta" },
      { type: "final_completed" },
      { type: "turn_completed" },
    ]),
    false,
  );
  for (const type of [
    "activity_text_delta",
    "tool_started",
    "subagent_queued",
    "subagent_started",
    "subagent_text_delta",
    "subagent_tool_started",
    "subagent_tool_finished",
    "subagent_completed",
    "subagent_failed",
  ]) {
    assert.equal(hasWork([{ type }]), true, type);
  }
});

test("系统事件 turn：无用户消息且无动作", () => {
  const eventTurn = { user_message: "", events: [{ type: "final_completed" }] };
  assert.equal(isSystemTurn(eventTurn), true);
  assert.equal(isSystemTurn({ ...eventTurn, user_message: "你好" }), false);
  assert.equal(
    isSystemTurn({ user_message: "", events: [{ type: "tool_started" }] }),
    false,
  );
});

test("计数：只数真实动作，不重复累计子代理事件", () => {
  const events = [
    { type: "tool_started" },
    { type: "tool_started" },
    { type: "subagent_queued", task_id: 1 },
    { type: "subagent_started", task_id: 1 },
    { type: "subagent_tool_started", task_id: 1 },
    { type: "subagent_tool_finished", task_id: 1 },
    { type: "subagent_text_delta", task_id: 1 },
    { type: "subagent_completed", task_id: 1 },
    { type: "final_completed" },
  ];
  assert.deepEqual(workingCounts(events), { actions: 3, tasks: 1, total: 3 });
  assert.equal(subagentTaskCount(events), 1);
});

test("299 类事件不再超数：30 步约 60 条事件仍算 30 动作", () => {
  const events = [];
  for (let i = 0; i < 30; i++) {
    events.push({ type: "subagent_tool_started", task_id: 1 });
    events.push({ type: "subagent_tool_finished", task_id: 1 });
  }
  events.push({ type: "subagent_text_delta", task_id: 1 });
  assert.equal(workingCounts(events).actions, 30);
});

test("摘要文案：历史与实时", () => {
  const events = [
    { type: "tool_started" },
    { type: "subagent_queued", task_id: 1 },
    { type: "subagent_tool_started", task_id: 1 },
    { type: "subagent_tool_finished", task_id: 1 },
    { type: "subagent_completed", task_id: 1 },
  ];
  assert.equal(historySummary("3s", events), "Worked 3s · 2 actions · 1 subagent");
  assert.equal(historySummary("0s", []), "Worked 0s · 0 actions · 0 subagent");
  assert.equal(liveSummary(1, 2), "Worked · 1 actions · 2 subagent");
});

test("任务进度文案：状态/步数/动作/最近动作/撞上限", () => {
  assert.equal(
    taskProgressText({ status: "running", steps: 3, maxSteps: 30, lastAction: "Write a.py" }),
    "执行中 · 步数 3/30 · 最近：Write a.py",
  );
  assert.equal(
    taskProgressText({
      status: "running", steps: 1, maxSteps: 30, actions: 3, lastAction: "Write a.py",
    }),
    "执行中 · 步数 1/30 · 动作 3 · 最近：Write a.py",
  );
  assert.equal(
    taskProgressText({ status: "done", steps: 30, maxSteps: 30, stopReason: "max_steps" }),
    "已达最大步数上限 · 步数 30/30",
  );
  assert.equal(
    taskProgressText({ status: "unconfigured", steps: 0, maxSteps: 30 }),
    "executor 未配置 · 任务未执行 · 步数 0/30",
  );
  assert.equal(taskProgressText({ steps: 0 }), "执行中 · 步数 0");
  assert.equal(
    taskProgressText({ status: "done", steps: 5, maxSteps: 30 }),
    "已完成 · 步数 5/30",
  );
  assert.equal(taskProgressText({ status: "cancelled" }), "已取消 · 步数 0");
});


test("任务合并：实时事件建条目必须带 id", () => {
  assert.deepEqual(mergeTask(undefined, 3, { status: "unconfigured" }), {
    id: 3, status: "unconfigured",
  });
  assert.deepEqual(mergeTask({ id: 1, title: "任务" }, 1, { status: "done" }), {
    id: 1, title: "任务", status: "done",
  });
  assert.equal(mergeTask({ id: 1 }, 1, {}).id, 1);  // patch 不含 id 也不丢
});

test("动作计数：一次工具调用只加一次", () => {
  assert.equal(countActions([{ type: "tool_started" }]), 1);
  assert.equal(countActions([{ type: "subagent_tool_started", task_id: 1 }]), 1);
  assert.equal(
    countActions([
      { type: "tool_started" },
      { type: "subagent_tool_started", task_id: 1 },
    ]),
    2,
  );
});

test("动作计数：30 次工具调用显示 30 actions", () => {
  const events = [];
  for (let i = 0; i < 30; i++) events.push({ type: "subagent_tool_started", task_id: 1 });
  assert.equal(countActions(events), 30);
  assert.equal(workingCounts(events).actions, 30);
});

test("动作计数：finished/completed/queued/文本等不计", () => {
  const events = [
    { type: "subagent_queued", task_id: 1 },
    { type: "subagent_started", task_id: 1 },
    { type: "subagent_tool_finished", task_id: 1 },
    { type: "subagent_text_delta", task_id: 1 },
    { type: "subagent_completed", task_id: 1 },
    { type: "subagent_failed", task_id: 1 },
    { type: "final_completed" },
  ];
  assert.equal(countActions(events), 0);
});

test("跨 turn 路由：动作只加所属 turn，非动作事件不计数", () => {
  const turns = { 5: { actions: 0 }, 6: { actions: 0 } };
  assert.equal(routeActionEvent(turns, { type: "subagent_tool_started", turn_id: 5 }), turns[5]);
  assert.equal(routeActionEvent(turns, { type: "tool_started", turn_id: 5 }), turns[5]);
  assert.equal(turns[5].actions, 2);   // 旧 turn 累加
  assert.equal(turns[6].actions, 0);   // 新 turn 不受影响
  assert.equal(routeActionEvent(turns, { type: "subagent_tool_finished", turn_id: 5 }), null);
  assert.equal(routeActionEvent(turns, { type: "subagent_completed", turn_id: 5 }), null);
  assert.equal(turns[5].actions, 2);   // finished/completed 不计数
  assert.equal(routeActionEvent(turns, { type: "tool_started", turn_id: 99 }), null);
});

test("实时与历史同口径：刷新前后 actions 一致", () => {
  const events = [
    { type: "tool_started" },
    { type: "subagent_queued", task_id: 1 },
    { type: "subagent_tool_started", task_id: 1 },
    { type: "subagent_tool_finished", task_id: 1 },
    { type: "subagent_tool_started", task_id: 1 },
    { type: "subagent_completed", task_id: 1 },
  ];
  const liveActions = countActions(events);
  const counts = workingCounts(events);
  assert.equal(counts.actions, liveActions);
  assert.ok(historySummary("1s", events).includes(`${liveActions} actions`));
  assert.equal(liveSummary(liveActions, counts.tasks), "Worked · 3 actions · 1 subagent");
});

test("空壳判定：0 动作且 body 为空才移除", () => {
  assert.equal(shouldDropShell(0, 0, 0), true);
  assert.equal(shouldDropShell(1, 0, 0), false);
  assert.equal(shouldDropShell(0, 1, 0), false);
  assert.equal(shouldDropShell(0, 0, 2), false);  // activity 文本：保留壳
});

test("上下文文案：token 单位且标「约」", () => {
  assert.equal(
    contextLabel({ messages: 12, tokens: 3450, percent: 2 }),
    "历史 12 条 · 上下文 约 3450 tokens (2%)",
  );
  assert.equal(contextLabel(null), "上下文 约 0 tokens");
});

test("会话标签：同名或空标题才补短 ID", () => {
  const s1 = { id: "ab12cd34", title: "对话" };
  const s2 = { id: "ef56ab78", title: "对话" };
  const unique = { id: "99998888", title: "唯一" };
  const untitled = { id: "77776666", title: "" };
  assert.equal(sessionLabel(unique, [unique, s1, s2]), "唯一");
  assert.equal(sessionLabel(s1, [s1, s2]), "对话 · #ab12");
  assert.equal(sessionLabel(s2, [s1, s2]), "对话 · #ef56");
  assert.equal(sessionLabel(untitled, [untitled]), "新会话 · #7777");
});


test("状态中文映射：六种内部状态", () => {
  assert.equal(taskStatusText("queued"), "排队中");
  assert.equal(taskStatusText("running"), "执行中");
  assert.equal(taskStatusText("done"), "已完成");
  assert.equal(taskStatusText("error"), "执行失败");
  assert.equal(taskStatusText("cancelled"), "已取消");
  assert.equal(taskStatusText("unconfigured"), "未执行");
  assert.equal(taskStatusText("incomplete"), "未完成");
  assert.equal(taskStatusText("weird"), "weird");
});

test("详情头部：中文状态 + 步数/动作，副行耗时/模型，无 turn", () => {
  const [line1, line2] = inspectorLines({
    status: "done", steps: 1, maxSteps: 30, actions: 2,
    durationLabel: "2s", model: "deepseek-chat",
  });
  assert.equal(line1, "已完成 · 步数 1/30 · 动作 2");
  assert.equal(line2, "2s · deepseek-chat");
  assert.ok(!line1.includes("turn") && !line2.includes("turn"));
  const [limit] = inspectorLines({ status: "done", steps: 30, maxSteps: 30, stopReason: "max_steps" });
  assert.equal(limit, "已达最大步数上限 · 步数 30/30");
  const [unconf] = inspectorLines({ status: "unconfigured", steps: 0, maxSteps: 30 });
  assert.equal(unconf, "executor 未配置 · 任务未执行 · 步数 0/30");
});

test("角色状态文案：未配置/已连接，不含内部来源", () => {
  assert.equal(roleStatusText(null), "当前：未配置");
  assert.equal(roleStatusText({ configured: false }), "当前：未配置");
  assert.equal(
    roleStatusText({ configured: true, preset: "deepseek", model: "deepseek-chat", source: "runtime" }),
    "当前：deepseek · deepseek-chat · 已连接",
  );
  assert.equal(
    roleStatusText({ configured: true, protocol: "openai", model: "m" }),
    "当前：openai · m · 已连接",
  );
});

test("plan 文案：开关两态", () => {
  assert.equal(planLabel(true), "plan 模式：已开启（不会修改文件）");
  assert.equal(planLabel(false), "plan 模式：已关闭");
});

test("Activity 分组：连续 delta 合并、工具分隔、单时间戳", () => {
  const groups = groupActivityEvents([
    { type: "subagent_text_delta", ts: 1, text: "1" },
    { type: "subagent_text_delta", ts: 2, text: "7" },
    { type: "subagent_text_delta", ts: 3, text: "×" },
    { type: "subagent_tool_started", ts: 4, name: "write_file", arguments: "{}" },
    { type: "subagent_text_delta", ts: 5, text: "完成" },
    { type: "subagent_text_delta", ts: 6, text: "。" },
    { type: "subagent_tool_finished", ts: 7, name: "write_file", status: "success" },
    { type: "subagent_completed", ts: 8, text: "完成。" },
  ]);
  assert.deepEqual(groups.map(g => g.kind), ["text", "tool_started", "text", "tool_finished", "status"]);
  assert.equal(groups[0].text, "17×");
  assert.equal(groups[0].ts, 1);          // 只保留首个时间戳
  assert.equal(groups[2].text, "完成。");
  assert.equal(groups[4].label, "完成");
});

test("Activity 分组：空 delta 忽略、失败文案", () => {
  const groups = groupActivityEvents([
    { type: "subagent_text_delta", ts: 1, text: "" },
    { type: "subagent_failed", ts: 2, status: "unconfigured" },
  ]);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].label, "失败（unconfigured）");
});

test("工具配对：重复命令按顺序 FIFO，不串结果", () => {
  const pairs = pairToolEvents([
    { type: "subagent_tool_started", ts: 1, name: "run_shell", arguments: '{"command":"pytest"}' },
    { type: "subagent_tool_started", ts: 2, name: "run_shell", arguments: '{"command":"pytest"}' },
    { type: "subagent_tool_finished", ts: 3, name: "run_shell", status: "success", result: "第一次" },
    { type: "subagent_tool_finished", ts: 4, name: "run_shell", status: "error", result: "第二次" },
  ]);
  assert.equal(pairs.length, 2);
  assert.equal(pairs[0].result, "第一次");
  assert.equal(pairs[1].result, "第二次");
  assert.equal(pairs[1].status, "error");
  assert.equal(pairs[0].ts, 1);  // FIFO：第一个 started 配第一个 finished
  assert.equal(pairs[1].ts, 2);  // LIFO 会得到 ts=2/1，被此断言杀死
});

test("Changes 提取：写文件 + Shell 变更启发", () => {
  const changes = extractChanges([
    { type: "subagent_tool_started", name: "write_file", arguments: '{"path":"a.py","content":"x"}' },
    { type: "subagent_tool_finished", name: "write_file", status: "success", result: "ok" },
    { type: "subagent_tool_started", name: "run_shell", arguments: '{"command":"echo hi > b.txt"}' },
    { type: "subagent_tool_finished", name: "run_shell", status: "success", result: "exit code: 0" },
    { type: "subagent_tool_started", name: "run_shell", arguments: '{"command":"ls"}' },
    { type: "subagent_tool_finished", name: "run_shell", status: "success", result: "exit code: 0" },
  ]);
  assert.deepEqual(changes.map(c => c.kind), ["file", "shell"]);
  assert.equal(changes[0].label, "a.py");
  assert.ok(changes[1].label.includes("echo hi > b.txt"));
});

test("文件路径：面包屑与返回上级", () => {
  assert.deepEqual(breadcrumb("."), ["项目"]);
  assert.deepEqual(breadcrumb("src/agent/core"), ["项目", "src", "agent", "core"]);
  assert.equal(parentPath("src/agent/core"), "src/agent");
  assert.equal(parentPath("src"), ".");
  assert.equal(parentPath("."), ".");
});

test("任务卡动作：去重与状态更新", () => {
  assert.deepEqual(taskCardAction({ type: "subagent_queued", text: "任务" }),
    { kind: "create", status: "queued", title: "任务" });
  assert.deepEqual(taskCardAction({ type: "subagent_started" }),
    { kind: "update", status: "running" });
  assert.equal(taskCardAction({ type: "subagent_completed", text: "ok" }).status, "done");
  assert.equal(taskCardAction({ type: "subagent_failed", status: "unconfigured" }).status, "unconfigured");
  assert.deepEqual(taskCardAction({ type: "subagent_step", steps_used: 2, max_steps: 30 }),
    { kind: "update", steps: 2, maxSteps: 30 });
  assert.deepEqual(taskCardAction({ type: "subagent_tool_started" }), { kind: "update", countAction: true });
  assert.equal(taskCardAction({ type: "subagent_tool_finished" }).kind, "none");
});

test("任务卡注册：同任务唯一，无处落位不占号", () => {
  const registry = {};
  assert.equal(planTaskCard(registry, 1, true), "create");
  assert.equal(planTaskCard(registry, 1, true), "update");   // 第二次不再建卡
  assert.equal(planTaskCard(registry, 1, false), "update");  // 已有卡：无容器也走更新
  assert.equal(planTaskCard(registry, 2, false), "skip");    // 无容器：不建也不占号
  assert.equal(registry[2], undefined);
  assert.equal(planTaskCard(registry, 2, true), "create");   // 之后仍可正常建卡
});

test("Working 摘要：折叠时任务状态仍可见（六态，中文）", () => {
  assert.equal(turnSummaryText(1, 1, [{ id: 1, status: "queued" }]),
    "任务 #1 排队中 · 本轮动作 1 次");
  assert.equal(turnSummaryText(1, 1, [{ id: 1, status: "running" }]),
    "任务 #1 执行中 · 本轮动作 1 次");
  const done = turnSummaryText(1, 1, [{ id: 1, status: "done" }], "2s");
  assert.equal(done, "用时 2s · 任务 #1 已完成 · 本轮动作 1 次");
  assert.ok(!done.includes("Worked") && !done.includes("actions"));
  assert.equal(turnSummaryText(0, 1, [{ id: 2, status: "error" }]).endsWith("任务 #2 执行失败"), true);
  assert.equal(turnSummaryText(0, 1, [{ id: 3, status: "cancelled" }]).endsWith("任务 #3 已取消"), true);
  assert.equal(turnSummaryText(0, 1, [{ id: 4, status: "unconfigured" }]).endsWith("任务 #4 未执行"), true);
  // 无任务时显示中文占位
  assert.equal(turnSummaryText(2, 0, []), "处理中 · 本轮动作 2 次");
  // 同一任务只出现一次
  const text = turnSummaryText(1, 1, [{ id: 1, status: "done" }]);
  assert.equal(text.split("任务 #1").length - 1, 1);
});

test("上下文环文案：精确/估算/未知/零值", () => {
  assert.deepEqual(ringLabel(null), { text: "?", sub: "未知", percent: 0, unknown: true });
  const zero = ringLabel({ used_tokens: 0, exact: true, window: 1000, percent: 0 });
  assert.equal(zero.text, "0%");
  assert.equal(zero.sub, "0 / 1k（精确）");
  const unknown = ringLabel({ used_tokens: 1200, exact: false, window: null });
  assert.equal(unknown.text, "?");
  assert.equal(unknown.sub, "约1k / ?");
  const exact = ringLabel({ used_tokens: 64000, exact: true, window: 128000, percent: 50 });
  assert.equal(exact.text, "50%");
  assert.equal(exact.sub, "64k / 128k（精确）");
  const estimated = ringLabel({ used_tokens: 64000, exact: false, window: 128000, percent: 50 });
  assert.equal(estimated.sub, "约64k / 128k（估算）");
  assert.equal(ringLabel({ used_tokens: 999, exact: false, window: 1000, percent: 120 }).percent, 100);
});

test("思考强度文案与窄窗布局判定", () => {
  assert.equal(effortLabel("off"), "关闭");
  assert.equal(effortLabel("high"), "高");
  assert.equal(effortLabel("weird"), "weird");
  assert.equal(composerLayout(1200), "wide");
  assert.equal(composerLayout(759), "narrow");
  assert.equal(composerLayout(760), "wide");
});

test("plan 短文案与完整提示", () => {
  assert.equal(planShort(true), "plan：开");
  assert.equal(planShort(false), "plan：关");
  assert.equal(planLabel(true), "plan 模式：已开启（不会修改文件）");
});

test("Activity 分组：reasoning 连续合并，与正文/工具分隔", () => {
  const groups = groupActivityEvents([
    { type: "reasoning_delta", ts: 1, text: "想" },
    { type: "reasoning_delta", ts: 2, text: "一下" },
    { type: "subagent_text_delta", ts: 3, text: "结果" },
    { type: "subagent_tool_started", ts: 4, name: "ls", arguments: "{}" },
    { type: "reasoning_delta", ts: 5, text: "再想" },
  ]);
  assert.deepEqual(groups.map(g => g.kind), ["reasoning", "text", "tool_started", "reasoning"]);
  assert.equal(groups[0].text, "想一下");
  assert.equal(groups[0].ts, 1);
  assert.equal(groups[3].text, "再想");
});

test("任务卡两行：状态只出现一次、六态通用", () => {
  const base = { id: 1, title: "计算 123+456", steps_used: 0, max_steps: 30, actions: 0 };
  for (const [status, label] of Object.entries({
    queued: "排队中", running: "执行中", done: "已完成",
    error: "执行失败", cancelled: "已取消", unconfigured: "未执行", incomplete: "未完成",
  })) {
    const lines = taskCardLines({ ...base, status });
    assert.equal(lines.title, "Task #1 · 计算 123+456");
    assert.ok(lines.line.startsWith(status === "unconfigured" ? "executor 未配置" : label), lines.line);
    assert.equal(lines.line.split(label).length - 1, 1);
  }
  const done = taskCardLines({ ...base, status: "done" });
  assert.equal(done.line, "已完成 · 步数 0/30 · 动作 0");
  assert.equal(done.line.split("已完成").length - 1, 1);  // 不再重复「已完成」
});

test("最终回复预览：短文本原样、长文本截断可展开", () => {
  const short = finalPreview("任务完成", 160);
  assert.deepEqual(short, { text: "任务完成", truncated: false });
  const long = finalPreview("长".repeat(400), 160);
  assert.equal(long.truncated, true);
  assert.equal(long.text.length, 161);  // 160 + 省略号
  assert.ok(long.text.endsWith("…"));
  assert.equal(finalPreview("x".repeat(160), 160).truncated, false);  // 边界不截断
  assert.deepEqual(finalPreview("", 160), { text: "", truncated: false });
});

test("会话标题：改名后与左侧同一标签（同名规则不变）", () => {
  const s1 = { id: "ab12cd34", title: "计算会话" };
  assert.equal(sessionLabel(s1, [s1]), "计算会话");
  const s2 = { id: "ef56ab78", title: "计算会话" };
  assert.equal(sessionLabel(s1, [s1, s2]), "计算会话 · #ab12");
  const renamed = { id: "ab12cd34", title: "计算会话（改名）" };
  assert.equal(sessionLabel(renamed, [renamed, s2]), "计算会话（改名）");  // 改名后去掉短 ID
  assert.equal(sessionLabel({ id: "99998888", title: "" }, [{ id: "99998888", title: "" }]),
    "新会话 · #9999");
});

/* ---------- 模型选择器与思考四态 ---------- */

test("模型菜单消费后端聚合结构（多 provider，单点失败隔离）", () => {
  assert.deepEqual(modelMenuState(null), { loading: true, groups: [], models: [], errors: [] });
  const catalog = { providers: [
    { id: "p-a", name: "Provider A", status: "ok", configured: true, models: [
      { id: "same", display_name: "A 同名", provider_id: "p-a" },
      { id: "a2", display_name: "A2", provider_id: "p-a" },
    ]},
    { id: "p-b", name: "Provider B", status: "error", error: "无法获取模型列表: HTTP 500",
      configured: true, models: [] },
  ]};
  const menu = modelMenuState(catalog);
  assert.equal(menu.groups.length, 2);
  assert.equal(menu.models.length, 2);       // 失败 provider 不影响正常 provider
  assert.equal(menu.errors.length, 1);
  assert.equal(menu.groups[0].id, "p-a");
  assert.equal(menu.groups[1].status, "error");
});

test("模型搜索：按 id/显示名过滤，多模型全部可选", () => {
  const models = [
    { id: "mock-basic", display_name: "基础模型" },
    { id: "mock-thinker", display_name: "思考模型" },
    { id: "mock-fixed", display_name: "固定思考" },
  ];
  assert.equal(filterModels(models, "").length, 3);
  assert.deepEqual(filterModels(models, "think").map(m => m.id), ["mock-thinker"]);
  assert.deepEqual(filterModels(models, "思考").map(m => m.id), ["mock-thinker", "mock-fixed"]);
  assert.deepEqual(filterModels(models, "不存在"), []);
});

test("模型动作：未配置进连接；已配置同/跨 provider 都直接切", () => {
  assert.equal(modelSelectAction("p-a", "p-a", true), "set_model");
  assert.equal(modelSelectAction("p-b", "p-a", true), "switch_provider");  // 已配置跨 provider
  assert.equal(modelSelectAction("p-c", "p-a", false), "connect");         // 未配置
  assert.equal(modelSelectAction("p-a", "", false), "connect");
});

test("能力表单：唯一数据源为角色状态，未知/空/False 均保持明确语义", () => {
  const unknown = capabilityFormState(null);
  assert.equal(unknown.mode, "unknown");
  assert.equal(unknown.source, "未知");
  assert.equal(unknown.levelsText, "");      // null=未知 → 空文本
  assert.equal(unknown.levelsUnknown, true);
  assert.equal(unknown.tools, "");
  assert.equal(unknown.window, "");

  const role = {
    window: 128000, window_source: "user", reasoning_mode: "adjustable",
    efforts: ["off", "high"], tools: false, max_output: 8192,
    provider_id: "p-a", model: "m",
  };
  const user = capabilityFormState(role);
  assert.equal(user.source, "用户设置");
  assert.equal(user.sourceKey, "user");
  assert.equal(user.tools, "no");            // False 不得被 || 吃掉
  assert.equal(user.window, 128000);
  assert.equal(user.levelsText, "off,high");
  assert.equal(user.levelsUnknown, false);

  const fixed = capabilityFormState({ reasoning_mode: "fixed", efforts: [] });
  assert.equal(fixed.levelsText, "");
  assert.equal(fixed.levelsUnknown, false);  // [] = 明确为空，不是未知
});

test("能力编辑器：按 #m-role 选择角色，main/executor 分别回填", () => {
  const roles = {
    main: { provider_id: "p-a", provider: "A", model: "m-main", window_source: "provider",
            window: 1000, reasoning_mode: "adjustable", efforts: ["off", "low"], tools: true },
    executor: { provider_id: "p-b", provider: "B", model: "m-exec", window_source: "user",
                window: 2000, reasoning_mode: "fixed", efforts: [], tools: false },
  };
  const main = capabilityEditorState(roles, "main");
  assert.equal(main.empty, false);
  assert.deepEqual(main.target, { provider: "p-a", model: "m-main" });
  assert.equal(main.form.window, 1000);
  assert.equal(main.form.source, "provider 返回");
  assert.deepEqual(main.form.levelsText, "off,low");
  assert.ok(main.title.includes("m-main"));

  const exec = capabilityEditorState(roles, "executor");
  assert.deepEqual(exec.target, { provider: "p-b", model: "m-exec" });
  assert.equal(exec.form.window, 2000);
  assert.equal(exec.form.source, "用户设置");
  assert.equal(exec.form.levelsUnknown, false);   // efforts=[] 明确为空
  assert.equal(exec.form.tools, "no");            // False 保持
  assert.ok(exec.title.includes("m-exec"));

  const empty = capabilityEditorState(roles, "executor2");
  assert.equal(empty.empty, true);
});

test("能力覆盖表单→override：未知=null、空=[]、False 不丢", () => {
  const adjustable = capabilityOverrideFromForm({
    window: "50000", maxOutput: "8192", mode: "adjustable",
    levelsText: "off,low,medium,high", levelsUnknown: false, tools: "no",
  });
  assert.deepEqual(adjustable, {
    window: 50000, max_output: 8192, reasoning_mode: "adjustable",
    levels: ["off", "low", "medium", "high"], tools: false,
  });

  const unknown = capabilityOverrideFromForm({
    window: "", maxOutput: "", mode: "unknown", levelsText: "", levelsUnknown: true, tools: "",
  });
  assert.deepEqual(unknown, {
    window: null, max_output: null, reasoning_mode: "unknown", levels: null, tools: null,
  });

  const fixed = capabilityOverrideFromForm({
    mode: "fixed", levelsText: "off", levelsUnknown: true, tools: "yes",
  });
  assert.deepEqual(fixed.levels, []);        // 固定模式 → 明确为空
  assert.equal(fixed.tools, true);

  // 原本未知、未输入 → 保持 null；用户清空了已知档位 → 明确 []
  const unknownKept = capabilityOverrideFromForm({
    mode: "adjustable", levelsText: "", levelsUnknown: true,
  });
  assert.equal(unknownKept.levels, null);
  const cleared = capabilityOverrideFromForm({
    mode: "adjustable", levelsText: "", levelsUnknown: false,
  });
  assert.deepEqual(cleared.levels, []);
});

test("思考四态：可调/固定/关闭/未知", () => {
  const adjustable = effortState({ reasoning_mode: "adjustable", levels: ["off", "low"] });
  assert.equal(adjustable.selectable, true);
  assert.deepEqual(adjustable.levels, ["off", "low"]);
  const fixed = effortState({ reasoning_mode: "fixed" });
  assert.deepEqual([fixed.selectable, fixed.label], [false, "固定"]);
  const none = effortState({ reasoning_mode: "none" });
  assert.deepEqual([none.selectable, none.label], [false, "关闭"]);
  const unknown = effortState(null);
  assert.deepEqual([unknown.selectable, unknown.label], [false, "未知"]);
  assert.equal(effortState({ reasoning_mode: "weird" }).mode, "unknown");
});

test("reasoning 归属：主 agent 与 subagent 分开", () => {
  assert.equal(reasoningTarget({ type: "reasoning_delta", text: "x" }), "main");
  assert.equal(reasoningTarget({ type: "reasoning_delta", task_id: 3 }), "task:3");
});

test("可调但档位未知/为空：禁止空菜单，提示先声明", () => {
  const unknownLevels = effortState({ reasoning_mode: "adjustable", levels: null });
  assert.equal(unknownLevels.selectable, false);
  assert.equal(unknownLevels.mode, "unknown-levels");
  assert.equal(unknownLevels.label, "档位未知");
  const emptyLevels = effortState({ reasoning_mode: "adjustable", levels: [] });
  assert.equal(emptyLevels.selectable, false);   // 明确为空也不得开空菜单
  const real = effortState({ reasoning_mode: "adjustable", levels: ["off", "low"] });
  assert.equal(real.selectable, true);
  assert.deepEqual(real.levels, ["off", "low"]);
});


test("服务商列表：预设行 + 自定义行，连接状态来自 provider_id", () => {
  const status = {
    presets: {
      deepseek: {label: "DeepSeek", protocol: "openai", base_url: "https://api.deepseek.com/v1",
                 model: "deepseek-chat", needs_key: true, provider_id: "p-d", configured: true},
      ollama: {label: "Ollama（本地）", protocol: "openai",
               base_url: "http://127.0.0.1:11434/v1", model: "qwen3", needs_key: false,
               provider_id: "p-o", configured: false},
    },
    providers: [
      {id: "p-d", name: "openai · api.deepseek.com/v1", configured: true, model_count: 2,
       protocol: "openai", base_url: "https://api.deepseek.com/v1"},
      {id: "p-x", name: "openai · 127.0.0.1:9/v1", configured: true, model_count: 0,
       protocol: "openai", base_url: "http://127.0.0.1:9/v1"},
    ],
  };
  const rows = providerListState(status);
  assert.equal(rows[0].kind, "preset");
  assert.equal(rows[0].label, "DeepSeek");
  assert.equal(rows[0].configured, true);
  assert.equal(rows[0].modelCount, 2);
  assert.equal(rows[1].configured, false);
  assert.equal(rows[1].needsKey, false);
  const custom = rows.find((r) => r.providerId === "p-x");
  assert.equal(custom.kind, "custom");
  assert.equal(custom.label, "openai · 127.0.0.1:9/v1");
  assert.equal(custom.needsKey, false);
  assert.equal(custom.protocol, "openai");     // 未配置的自定义行可重连
  assert.equal(custom.baseUrl, "http://127.0.0.1:9/v1");
});

test("角色模型选项：只列已连接 provider 的模型，按 provider 分组", () => {
  const status = {providers: [{id: "p-d", configured: true}, {id: "p-off", configured: false}]};
  const catalog = {providers: [
    {id: "p-d", name: "DeepSeek", models: [{id: "m1", display_name: "M1"}]},
    {id: "p-off", name: "Off", models: [{id: "m2"}]},
  ]};
  const groups = roleModelOptions(catalog, status);
  assert.deepEqual(groups.map((g) => g.providerId), ["p-d"]);
  assert.deepEqual(groups[0].models, [{id: "m1", name: "M1"}]);
  assert.deepEqual(roleModelOptions(null, status), []);
});


test("连接消息：自定义行带 protocol/base_url；预设行按需带 key", () => {
  assert.deepEqual(
    connectRequestForRow({kind: "custom", protocol: "anthropic",
                          baseUrl: "http://127.0.0.1:9999"}, ""),
    {type: "connect_provider", protocol: "anthropic",
     base_url: "http://127.0.0.1:9999", api_key: ""});
  assert.deepEqual(
    connectRequestForRow({kind: "preset", name: "deepseek", needsKey: true}, "sk-1"),
    {type: "connect_provider", preset: "deepseek", api_key: "sk-1"});
  assert.deepEqual(
    connectRequestForRow({kind: "preset", name: "ollama", needsKey: false}, "ignored"),
    {type: "connect_provider", preset: "ollama", api_key: ""});
  assert.equal(connectRequestForRow(null, "x"), null);
});


test("连接动作规划：缺协议的自定义行必须引导高级（不得直接发连接）", () => {
  assert.equal(connectRowPlan({kind: "custom", protocol: ""}).kind, "guide-advanced");
  assert.equal(connectRowPlan({kind: "custom", protocol: "anthropic"}).kind, "key");
  assert.equal(connectRowPlan({kind: "preset", needsKey: false}).kind, "direct");
  assert.equal(connectRowPlan({kind: "preset", needsKey: true}).kind, "key");
  assert.equal(connectRowPlan(null).kind, "none");
});

test("连接消息空值归一化：protocol 缺省为空串、baseUrl 缺省为 null", () => {
  assert.deepEqual(connectRequestForRow({kind: "custom"}, "k"), {
    type: "connect_provider", protocol: "", base_url: null, api_key: "k",
  });
  assert.deepEqual(connectRequestForRow({kind: "custom", protocol: "openai"}, ""), {
    type: "connect_provider", protocol: "openai", base_url: null, api_key: "",
  });
});


test("控件可见性：能力未知不显示思考控件，窗口未知不显示上下文环", () => {
  assert.equal(effortChipVisible(null), false);
  assert.equal(effortChipVisible({reasoning_mode: "unknown", efforts: null}), false);
  assert.equal(effortChipVisible({reasoning_mode: "fixed", efforts: []}), false);
  assert.equal(effortChipVisible({reasoning_mode: "none", efforts: []}), false);
  assert.equal(effortChipVisible({reasoning_mode: "adjustable", efforts: null}), false);
  assert.equal(effortChipVisible({reasoning_mode: "adjustable", efforts: ["off", "low"]}), true);

  assert.equal(ringVisible(null), false);
  assert.equal(ringVisible(undefined), false);
  assert.equal(ringVisible(0), false);
  assert.equal(ringVisible(128000), true);
  assert.equal(ringVisible(77777), true);
});


test("未配置角色不得显示强度控件（即使缓存了可调能力）", () => {
  const adjustable = {reasoning_mode: "adjustable", efforts: ["off", "low"]};
  assert.equal(roleEffortVisible({...adjustable, configured: false}), false);
  assert.equal(roleEffortVisible({...adjustable, configured: true}), true);
  assert.equal(roleEffortVisible({configured: true, reasoning_mode: "unknown", efforts: null}), false);
  assert.equal(roleEffortVisible(null), false);
});


test("effortLabel 支持 max；能力来源支持公共目录", () => {
  assert.equal(effortLabel("max"), "最高");
  const form = capabilityFormState({ window_source: "catalog" });
  assert.equal(form.source, "公共目录");
});


test("执行者消息视图：role/text/思考/工具调用/工具结果", () => {
  const v = executorMessageView({
    role: "assistant", content: "干活", reasoning: "想想",
    tool_calls: [{name: "read_file", arguments: {path: "a.txt"}}],
  });
  assert.deepEqual(v, {role: "assistant", text: "干活", reasoning: "想想",
                       calls: [{name: "read_file", arguments: {path: "a.txt"}}],
                       isToolResult: false});
  const tr = executorMessageView({role: "tool", content: "结果"});
  assert.equal(tr.isToolResult, true);
  assert.deepEqual(tr.calls, []);
  assert.equal(executorMessageView(null).text, "");
});

test("执行者时间线：在底部才跟随，上翻后不被拽回", () => {
  assert.equal(shouldStickToBottom(1000, 400, 600), true);    // 贴底
  assert.equal(shouldStickToBottom(1000, 350, 600), false);   // 上翻 50px
  assert.equal(shouldStickToBottom(1000, 300, 600), false);
  assert.equal(shouldStickToBottom(600, 0, 600), true);       // 不可滚动
  assert.equal(shouldStickToBottom(1000, 361, 600), true);    // 距底 <40 视为贴底
});

test("实时缓冲退役：按轮次身份判定（同文案/短前缀不误清）", () => {
  // 第一轮流式中：缓冲 round=0，inflight 里还没有 assistant → 保留
  assert.equal(liveBufferStale(0, 0), false);
  // 第一轮完成：inflight 出现 1 条 assistant → 退役
  assert.equal(liveBufferStale(0, 1), true);
  // 第二轮流式中：round=1，inflight 仍是 1 条（上一轮） → 保留（即使文案完全相同）
  assert.equal(liveBufferStale(1, 1), false);
  // 第二轮完成：inflight 2 条 → 退役
  assert.equal(liveBufferStale(1, 2), true);
  // 缺省/异常输入安全
  assert.equal(liveBufferStale(undefined, 0), false);
  assert.equal(liveBufferStale(undefined, 2), true);
});

test("公式提取：四种分隔符 + 代码段保护 + 单 $ 防误伤", () => {
  const r = extractMath("行内 \\(a^2\\) 展示 \\[\\int_0^1 x\\,dx\\] 美元 $$E=mc^2$$ 和 $b+c$");
  assert.equal(r.items.length, 4, JSON.stringify(r));
  assert.deepEqual(r.items[0], {tex: "a^2", display: false});
  assert.deepEqual(r.items[1], {tex: "\\int_0^1 x\\,dx", display: true});
  assert.deepEqual(r.items[2], {tex: "E=mc^2", display: true});
  assert.deepEqual(r.items[3], {tex: "b+c", display: false});
  assert.equal(r.text.includes("\\("), false);       // 原文分隔符已被占位
  assert.equal(r.text.includes("@@KATEX0@@"), true);

  const code = extractMath("`\\(not math\\)` 与 ```\n$$x$$\n``` 之后 \\(yes\\)");
  assert.equal(code.items.length, 1);
  assert.equal(code.items[0].tex, "yes");
  assert.equal(code.text.includes("`\\(not math\\)`"), true);   // 行内代码原样

  const price = extractMath("价格是 $5 一个");
  assert.equal(price.items.length, 0);                // 单 $ 后紧跟空格/数字：不误伤

  const escaped = extractMath("价格 \\$5 到 \\$10 之间有 \\(x\\)");
  assert.equal(escaped.items.length, 1);              // 转义美元不算公式
  assert.equal(escaped.items[0].tex, "x");

  const pair = extractMath("从 $5 到 $10 的区间");
  assert.equal(pair.items.length, 0);                 // 成对价格不误伤（$ 前有空格）

  const tilde = extractMath("~~~\n$$x$$\n~~~ 之后 \\(y\\)");
  assert.equal(tilde.items.length, 1);
  assert.equal(tilde.items[0].tex, "y");              // ~~~ 围栏内不处理

  const indented = extractMath("    $$x$$\n\\(y\\)");
  assert.equal(indented.items.length, 1);
  assert.equal(indented.items[0].tex, "y");           // 段首缩进代码不处理

  const afterBlank = extractMath("段落\n\n    $$x$$\n\\(y\\)");
  assert.equal(afterBlank.items.length, 1);
  assert.equal(afterBlank.items[0].tex, "y");         // 空行后的缩进块是代码（x 不提取）

  const listCont = extractMath("- 项目\n    $x$");
  assert.equal(listCont.items.length, 1);             // 列表缩进续行是正文
  assert.equal(listCont.items[0].tex, "x");

  const paraCont = extractMath("段落文字\n    $y$ 续行");
  assert.equal(paraCont.items.length, 1);             // 段落缩进续行是正文
  assert.equal(paraCont.items[0].tex, "y");

  const multi = extractMath("$$\nE = mc^2\n$$ 与 \\[\n\\int_0^1 x\\,dx\n\\] 与 \\(a +\nb\\)");
  assert.equal(multi.items.length, 3);                // 跨行公式不受缩进处理影响
  assert.deepEqual(multi.items[0], {tex: "E = mc^2", display: true});
  assert.equal(multi.items[1].display, true);
  assert.equal(multi.items[2].display, false);

  const multiAfterCode = extractMath("    $$x$$\n$$\nE=mc^2\n$$");
  assert.equal(multiAfterCode.items.length, 1);       // 代码块跳过、跨行公式仍提取
  assert.equal(multiAfterCode.items[0].tex, "E=mc^2");
});

test("改动汇总：累计 task_diff、忽略其它事件与坏 JSON、按路径排序", () => {
  const events = [
    {type: "task_diff", text: JSON.stringify({files: [{path: "b.js", added: 2, removed: 1}]})},
    {type: "tool_started", name: "read_file"},
    {type: "task_diff", text: "not json"},
    {type: "task_diff", text: JSON.stringify({files: [
      {path: "a.js", added: 3, removed: 0},
      {path: "b.js", added: 1, removed: 4},
    ]})},
  ];
  const s = diffSummary(events);
  assert.equal(s.label, "2 个文件已更改");
  assert.equal(s.added, 6);
  assert.equal(s.removed, 5);
  assert.deepEqual(s.files.map((f) => f.path), ["a.js", "b.js"]);
  assert.deepEqual(s.files[1], {path: "b.js", added: 3, removed: 5});
  assert.equal(diffSummary([]).label, "");
});

test("@提及：触发条件、光标定位与插入", () => {
  assert.deepEqual(mentionQuery("看看 @app", 8), {start: 3, query: "app"});
  assert.deepEqual(mentionQuery("@a", 2), {start: 0, query: "a"});
  assert.deepEqual(mentionQuery("x@y", 3), null);          // 词中 @ 不触发
  assert.deepEqual(mentionQuery("看看 @app 再说", 12), null); // 片段已结束
  assert.deepEqual(mentionQuery("看看 @src/ma", 11), {start: 3, query: "src/ma"});

  assert.deepEqual(insertMention("看看 @app", 3, 7, "web/app.js"),
                   {text: "看看 @web/app.js", caret: 14});
  assert.deepEqual(insertMention("@a b", 0, 2, "a.txt"),
                   {text: "@a.txt b", caret: 6});              // 原有空格不重复补
  assert.deepEqual(insertMention("@a(x)", 0, 2, "a.txt"),
                   {text: "@a.txt (x)", caret: 7});            // 括号前补空格
});

test("/命令：触发条件与片段", () => {
  assert.deepEqual(commandQuery("/ski", 4), {start: 0, query: "/ski"});
  assert.deepEqual(commandQuery("你好 /skill", 10), {start: 3, query: "/skill"});
  assert.deepEqual(commandQuery("看看 /a", 6), {start: 3, query: "/a"});
  assert.deepEqual(commandQuery("/skill demo", 11), null);   // 参数开始后不再匹配
  assert.deepEqual(commandQuery("a/b", 3), null);            // 词中斜杠不触发
});

test("待绑定气泡：前缀匹配、合并消息、被并入的旧消息跳过", () => {
  assert.deepEqual(pendingBindPlan("第一问", ["第一问"]), {skip: 0, bind: 1});
  assert.deepEqual(pendingBindPlan("第一问\n第二问", ["第一问", "第二问"]), {skip: 0, bind: 2});
  // 批次以非气泡文本开头（如 /skill 注入的提示）时：为保证行边界防碰撞，本批不绑定
  assert.deepEqual(pendingBindPlan("说话\n第一问\n第二问", ["第一问", "第二问"]),
                   {skip: 2, bind: 0});
  // 一条被并入上一回合的旧消息（不在本回合文本里）：跳过它，绑定后面的
  assert.deepEqual(pendingBindPlan("第三问", ["第二问", "第三问"]), {skip: 1, bind: 1});
  assert.deepEqual(pendingBindPlan("新话", ["旧话"]), {skip: 1, bind: 0});
  assert.deepEqual(pendingBindPlan("", ["第一问"]), {skip: 1, bind: 0});   // 合成回合
  assert.deepEqual(pendingBindPlan("第一问", []), {skip: 0, bind: 0});
  // 前缀碰撞：旧消息是本次新消息的前缀时，旧气泡不能被误绑（行边界锚定）
  assert.deepEqual(pendingBindPlan("继续做这件事", ["继续", "继续做这件事"]),
                   {skip: 1, bind: 1});
  assert.deepEqual(pendingBindPlan("测试", ["测试一下这个功能"]), {skip: 1, bind: 0});
});
