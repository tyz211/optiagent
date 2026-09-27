const providers = {
  openai: {
    baseUrl: "https://api.openai.com/v1",
    models: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
  },
  deepseek: {
    baseUrl: "https://api.deepseek.com",
    models: ["deepseek-v4-pro", "deepseek-v4-flash"],
  },
  custom: {
    baseUrl: "",
    models: ["custom-model"],
  },
};

// 前端预先绘制完整工作流，SSE 到达后只更新对应节点状态。
const agentNodeBlueprint = [
  { node_id: "requirements", label: "Requirement Analyst", description: "汇总多轮需求并判断是否需要澄清", sequence: 1 },
  { node_id: "planner", label: "Planner", description: "识别意图与规划工具链", sequence: 2 },
  { node_id: "data", label: "Data Agent", description: "定位并检查当前数据上下文", sequence: 3 },
  { node_id: "modeler", label: "Modeler", description: "生成结构化问题定义", sequence: 4 },
  { node_id: "solver", label: "Solver", description: "通过 MCP Gateway 执行求解", sequence: 5 },
  { node_id: "verifier", label: "Verifier", description: "检查响应合同与求解状态", sequence: 6 },
  { node_id: "policy", label: "Policy", description: "根据验证反馈选择接受、恢复或终止", sequence: 7 },
  { node_id: "explainer", label: "Explainer", description: "组织业务结论与可审计轨迹", sequence: 8 },
];

const state = {
  activeDatasetId: null,
  activeConversationId: Number(localStorage.getItem("optiagent_active_conversation_id") || 0) || null,
  hasData: false,
  hasRuns: false,
  token: localStorage.getItem("optiagent_session_token") || "",
  conversations: [],
  conversationSearch: "",
  asking: false,
};

const fmt = (value) => {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value);
  }
  return numeric.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
};

const fmtMetric = (item) => {
  if (item.format === "percent" && item.value !== null && item.value !== undefined) {
    const numeric = Number(item.value);
    return Number.isNaN(numeric) ? String(item.value) : `${(numeric * 100).toFixed(2)}%`;
  }
  return fmt(item.value);
};

const formatMetricItem = (item, result) => {
  if (item.label === "最优性证明") {
    if (item.value === true) {
      return "已证明";
    }
    if (item.value === false) {
      return "未证明";
    }
    if (String(item.value) === "NaN") {
      return result?.status === "OPTIMAL" ? "已证明" : "-";
    }
  }
  return fmtMetric(item);
};

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function api(path, options = {}) {
  options.headers = {
    ...(options.headers || {}),
    ...(state.token ? { "X-Session-Token": state.token } : {}),
  };
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    try {
      const payload = JSON.parse(text);
      const detail = payload.detail;
      const message = Array.isArray(detail?.messages)
        ? detail.messages.join("；")
        : typeof detail === "string"
          ? detail
          : text;
      throw new Error(message);
    } catch (err) {
      if (err instanceof SyntaxError) {
        throw new Error(text || res.statusText);
      }
      throw err;
    }
  }
  return res.json();
}

function byId(id) {
  return document.getElementById(id);
}

function on(id, event, handler) {
  const el = byId(id);
  if (el) {
    el.addEventListener(event, handler);
  }
}

function setText(id, value) {
  const el = byId(id);
  if (el) {
    el.textContent = value;
  }
}

function togglePanel(id) {
  const panel = byId(id);
  if (panel) {
    panel.classList.toggle("hidden");
  }
}

async function loadAll() {
  const [me, llm, conversationPayload] = await Promise.all([
    api("/api/me"),
    api("/api/llm-config"),
    api("/api/conversations"),
  ]);
  setText("userState", me.logged_in ? `已登录：${me.user.username}` : "未登录");
  await ensureActiveConversation(conversationPayload.conversations || []);
  const cid = state.activeConversationId;
  const query = cid ? `?conversation_id=${encodeURIComponent(cid)}` : "";
  const [datasets, summaryResult, runs] = await Promise.all([
    api(`/api/datasets${query}`),
    api(`/api/data/summary${query}`).catch(() => ({ has_data: false, warehouses: [], customers: [] })),
    api(`/api/runs${query}`),
  ]);
  if (datasets.conversation_id && datasets.conversation_id !== state.activeConversationId) {
    setActiveConversation(datasets.conversation_id);
  }
  const summary = summaryResult || { has_data: false, warehouses: [], customers: [] };
  state.activeDatasetId = datasets.active_dataset_id;
  state.hasData = Boolean(summary.has_data || (datasets.uploaded_files || []).length);
  state.hasRuns = Boolean((runs.runs || []).length);
  renderDataVisibility();
  renderConversations();
  renderDatasets(datasets.datasets, datasets.active_dataset_id);
  renderUploadedFiles(datasets.uploaded_files || []);
  renderDatasetHeader(summary);
  renderConversationMessages(runs.runs || []);
  setText("llmState", llm.configured ? llm.model : "未配置");
}

async function ensureActiveConversation(conversations) {
  state.conversations = conversations;
  const activeExists = state.activeConversationId
    && conversations.some((item) => item.id === state.activeConversationId);
  if (activeExists) {
    return;
  }
  if (conversations.length) {
    setActiveConversation(conversations[0].id);
    return;
  }
  const created = await api("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "新对话" }),
  });
  state.conversations = [created.conversation];
  setActiveConversation(created.conversation.id);
}

function setActiveConversation(conversationId) {
  state.activeConversationId = conversationId;
  localStorage.setItem("optiagent_active_conversation_id", String(conversationId));
}

function renderDataVisibility() {
  byId("askPanel")?.classList.remove("hidden");
  byId("emptyState")?.classList.toggle("hidden", state.hasData || state.hasRuns);
}

function renderDatasetHeader(summary) {
  if (!summary.has_data && state.hasData) {
    setText("datasetName", "已上传文件，可直接提问");
    return;
  }
  if (!summary.has_data) {
    setText("datasetName", "未选择");
    return;
  }
  setText("datasetName", `当前数据：${summary.warehouses.length} 仓 / ${summary.customers.length} 客户`);
}

function renderDatasets(datasets, activeId) {
  const root = byId("datasetList");
  if (!root) {
    return;
  }
  root.innerHTML = "";
  const visibleDatasets = (datasets || []).filter((dataset) => !isLegacyDatasetName(dataset.name));
  if (!visibleDatasets.length) {
    root.innerHTML = '<div class="item muted-item">暂无结构化数据集</div>';
    return;
  }
  visibleDatasets.forEach((dataset) => {
    const button = document.createElement("button");
    button.textContent = `${dataset.name}${dataset.id === activeId ? " / 当前" : ""}`;
    button.onclick = async () => {
      const query = state.activeConversationId
        ? `?conversation_id=${encodeURIComponent(state.activeConversationId)}`
        : "";
      await api(`/api/datasets/active/${dataset.id}${query}`, { method: "POST" });
      await loadAll();
    };
    root.appendChild(button);
  });
}

function renderUploadedFiles(files) {
  const existing = byId("uploadedFileList");
  if (!existing) {
    const root = byId("datasetList")?.parentElement;
    if (!root) {
      return;
    }
    const block = document.createElement("div");
    block.className = "uploaded-files";
    block.innerHTML = '<div class="side-title">最近上传</div><div id="uploadedFileList" class="list"></div>';
    root.appendChild(block);
  }
  const list = byId("uploadedFileList");
  if (!list) {
    return;
  }
  if (!files.length) {
    list.innerHTML = '<div class="item">暂无上传文件</div>';
    return;
  }
  list.innerHTML = files.map((file) => `<div class="item">${escapeHtml(file.filename)}</div>`).join("");
}

function isLegacyDatasetName(name) {
  const normalized = String(name || "").trim().toLowerCase();
  return ["示例数据", "默认数据", "结构化上传测试", "sample data", "demo data"].includes(normalized);
}

function renderSelectedFiles() {
  const files = Array.from(byId("datasetFiles")?.files || []);
  const root = byId("selectedFiles");
  if (!root) {
    return;
  }
  if (!files.length) {
    root.textContent = "尚未选择文件";
    return;
  }
  root.innerHTML = files.map((file) => `
    <div class="file-chip pending">
      <strong>${escapeHtml(file.name)}</strong>
      <span>准备上传 · ${formatBytes(file.size)}</span>
    </div>
  `).join("");
}

function renderConversations() {
  const root = byId("historyList");
  if (!root) {
    return;
  }
  root.innerHTML = "";
  const keyword = state.conversationSearch.trim().toLowerCase();
  const conversations = keyword
    ? state.conversations.filter((conversation) => String(conversation.title || "").toLowerCase().includes(keyword))
    : state.conversations;
  if (!conversations.length) {
    root.innerHTML = `<div class="item">${keyword ? "未找到匹配对话" : "暂无对话"}</div>`;
    return;
  }
  conversations.forEach((conversation) => {
    const item = document.createElement("button");
    item.className = `item${conversation.id === state.activeConversationId ? " active" : ""}`;
    item.textContent = conversation.title || "新对话";
    item.title = conversation.title || "新对话";
    item.onclick = async () => {
      if (conversation.id === state.activeConversationId) {
        return;
      }
      setActiveConversation(conversation.id);
      clearChatStream();
      await loadAll();
    };
    root.appendChild(item);
  });
}

function clearChatStream() {
  const stream = byId("chatStream");
  if (!stream) {
    return;
  }
  stream.innerHTML = "";
}

function renderConversationMessages(runs) {
  clearChatStream();
  if (!runs.length) {
    appendWelcomeMessage();
    return;
  }
  runs.forEach((run) => {
    appendUserMessage(run.question);
    const result = parseRunResult(run);
    appendAssistantMessage(result);
  });
}

function parseRunResult(run) {
  try {
    const result = JSON.parse(run.result_json || "{}");
    return {
      ...result,
      answer: result.answer || run.answer,
      status: result.status || run.status,
      objective_value: result.objective_value ?? run.objective_value,
      transport_cost: result.transport_cost ?? run.transport_cost,
      fixed_cost: result.fixed_cost ?? run.fixed_cost,
      question: result.question || run.question,
    };
  } catch {
    return {
      answer: run.answer,
      structured_answer: {
        conclusion: run.answer,
        metrics: {},
        recommendations: [],
        risks: [],
        evidence: [],
        raw_answer: run.answer,
      },
      question: run.question,
      status: run.status,
      objective_value: run.objective_value,
      transport_cost: run.transport_cost,
      fixed_cost: run.fixed_cost,
      open_warehouses: [],
      scenario_changes: [],
      warnings: [],
      explanation: [],
      rag_notes: [],
      rag_docs: [],
      tool_names: [],
      warehouse_summary: [],
      allocations: [],
    };
  }
}

function appendWelcomeMessage() {
  const stream = byId("chatStream");
  if (!stream) {
    return;
  }
  const article = document.createElement("article");
  article.className = "message assistant-message";
  article.innerHTML = `
    <div class="avatar">OA</div>
    <div class="message-body">
      <p>你好，你可以先描述业务目标。我会在当前对话中持续整理目标、约束和数据缺口，信息足够后再建模求解。</p>
      <div class="suggestion-row">
        <button class="suggestion" data-example="knapsack">载入背包演示数据</button>
        <button class="suggestion">分析当前供应链数据</button>
        <button class="suggestion">求解一个背包问题</button>
        <button class="suggestion">做员工班次指派</button>
      </div>
    </div>
  `;
  stream.appendChild(article);
  bindSuggestionButtons(article);
}

async function ask() {
  const status = byId("runStatus");
  const input = byId("questionInput");
  const question = input?.value.trim() || "";
  if (!question || state.asking) {
    return;
  }
  state.asking = true;
  byId("askBtn").disabled = true;
  // 求解期间保持会话与数据来源稳定，防止结果被插入刚切换的其他对话。
  document.querySelectorAll(".sidebar,.top-actions,.config-drawer").forEach(el => { el.inert = true; });
  setText("runStatus", "连接中...");
  appendUserMessage(question);
  const streamingMessage = appendStreamingAssistantMessage();
  input.value = "";
  try {
    const result = await streamAsk({
      question,
      dataset_id: state.activeDatasetId,
      conversation_id: state.activeConversationId,
      mcp_config: byId("mcpInput")?.value || "",
    }, {
      onStatus(message) {
        setText("runStatus", message || "运行中...");
        streamingMessage.setStatus(message || "运行中...");
      },
      onDelta(text) {
        streamingMessage.append(text);
      },
      onStep(event) {
        streamingMessage.updateStep(event);
      },
    });
    if (result.conversation_id) {
      setActiveConversation(result.conversation_id);
    }
    streamingMessage.remove();
    appendAssistantMessage(result);
    setText("runStatus", "完成");
    await loadAll();
  } catch (err) {
    setText("runStatus", "失败");
    streamingMessage.fail(err.message);
  } finally {
    state.asking = false;
    byId("askBtn").disabled = false;
    document.querySelectorAll(".sidebar,.top-actions,.config-drawer").forEach(el => { el.inert = false; });
  }
}

async function streamAsk(payload, handlers = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(state.token ? { "X-Session-Token": state.token } : {}),
  };
  const res = await fetch("/api/ask/stream", {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }

  const decoder = new TextDecoder();
  const reader = res.body.getReader();
  let buffer = "";
  let finalResult = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      if (!part.trim()) {
        continue;
      }
      const event = parseSseEvent(part);
      if (event.type === "status") {
        handlers.onStatus?.(event.data.message);
      } else if (event.type === "agent_step") {
        handlers.onStep?.(event.data);
      } else if (event.type === "answer_delta") {
        handlers.onDelta?.(event.data.text || "");
      } else if (event.type === "final") {
        finalResult = event.data;
      } else if (event.type === "error") {
        throw new Error(event.data.message || "流式回答失败");
      }
    }
  }
  if (buffer.trim()) {
    const event = parseSseEvent(buffer);
    if (event.type === "final") {
      finalResult = event.data;
    } else if (event.type === "agent_step") {
      handlers.onStep?.(event.data);
    } else if (event.type === "error") {
      throw new Error(event.data.message || "流式回答失败");
    }
  }
  if (!finalResult) {
    throw new Error("流式回答未返回最终结果。");
  }
  return finalResult;
}

function parseSseEvent(raw) {
  const lines = raw.split("\n");
  const typeLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));
  const type = typeLine ? typeLine.slice(6).trim() : "message";
  const dataText = dataLines.map((line) => line.slice(5).trimStart()).join("\n");
  let data = {};
  try {
    data = dataText ? JSON.parse(dataText) : {};
  } catch (err) {
    data = { message: dataText || String(err) };
  }
  return { type, data };
}

async function uploadDataset() {
  const fileInput = byId("datasetFiles");
  const files = Array.from(fileInput?.files || []);
  if (!files.length) {
    return;
  }
  renderSelectedFiles();
  setText("runStatus", "上传中...");
  const form = new FormData();
  form.append("name", "上传数据集");
  if (state.activeConversationId) {
    form.append("conversation_id", String(state.activeConversationId));
  }
  files.forEach((file) => form.append("files", file));
  try {
    const result = await api("/api/upload", { method: "POST", body: form });
    if (result.conversation_id) {
      setActiveConversation(result.conversation_id);
    }
    renderUploadCheck(result.check, result.files || []);
    if (result.dataset_id) {
      state.activeDatasetId = result.dataset_id;
      state.hasData = true;
    } else {
      state.activeDatasetId = null;
      state.hasData = true;
    }
    setText("runStatus", "上传完成");
    if (fileInput) {
      fileInput.value = "";
    }
    await loadAll();
  } catch (err) {
    renderUploadCheck({ status: "error", messages: [err.message] });
    setText("runStatus", "上传失败");
    if (fileInput) {
      fileInput.value = "";
    }
  }
}

async function clearHistory() {
  if (!state.activeConversationId) {
    return;
  }
  if (!confirm("删除当前对话及其消息、上传文件和数据集？")) {
    return;
  }
  await api(`/api/conversations/${encodeURIComponent(state.activeConversationId)}`, { method: "DELETE" });
  state.activeConversationId = null;
  state.activeDatasetId = null;
  state.hasData = false;
  localStorage.removeItem("optiagent_active_conversation_id");
  clearChatStream();
  await loadAll();
}

async function newConversation() {
  const created = await api("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "新对话" }),
  });
  setActiveConversation(created.conversation.id);
  state.activeDatasetId = null;
  state.hasData = false;
  clearChatStream();
  await loadAll();
}

function renderUploadCheck(check, files = []) {
  const box = byId("uploadCheck");
  if (!box) {
    return;
  }
  box.className = `check-box ${check.status}`;
  const fileRows = files.map((file) => `
    <div class="upload-file-row ${file.status === "ok" ? "ok" : "error"}">
      <div>
        <strong>${escapeHtml(file.filename)}</strong>
        <span>${fmt(file.rows || 0)} 行 · ${(file.columns || []).length} 列</span>
      </div>
      <em>${escapeHtml(file.message || (file.status === "ok" ? "上传成功" : "上传失败"))}</em>
    </div>
  `).join("");
  const messages = (check.messages || []).map((message) => `<div>${escapeHtml(message)}</div>`).join("");
  box.innerHTML = `${fileRows}${messages ? `<div class="upload-messages">${messages}</div>` : ""}`;
}

async function refreshAll() {
  setText("runStatus", "刷新中...");
  try {
    await loadAll();
    setText("runStatus", "已刷新");
  } catch (err) {
    setText("runStatus", "刷新失败");
    appendErrorMessage(`刷新失败：${err.message}`);
  }
}

async function saveLlm() {
  await api("/api/llm-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
      name: byId("providerSelect")?.value || "openai",
      base_url: byId("baseUrlInput")?.value || "",
      model: byId("modelSelect")?.value || "",
      api_key: byId("apiKeyInput")?.value || "",
      temperature: Number(byId("temperatureInput")?.value || 0.2),
    }),
  });
  await loadAll();
}

async function login() {
  const username = byId("usernameInput")?.value.trim() || "";
  if (!username) {
    alert("请输入用户名。");
    return;
  }
  const result = await api("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  state.token = result.session_token;
  localStorage.setItem("optiagent_session_token", state.token);
  state.activeConversationId = null;
  localStorage.removeItem("optiagent_active_conversation_id");
  await loadAll();
}

function updateModelOptions() {
  const providerName = byId("providerSelect")?.value || "openai";
  const provider = providers[providerName] || providers.openai;
  const modelSelect = byId("modelSelect");
  if (!modelSelect) {
    return;
  }
  modelSelect.innerHTML = provider.models.map((model) => `<option value="${model}">${model}</option>`).join("");
  const baseUrlInput = byId("baseUrlInput");
  if (baseUrlInput) {
    baseUrlInput.value = provider.baseUrl;
  }
}

function appendUserMessage(text) {
  const stream = byId("chatStream");
  if (!stream) {
    return;
  }
  const article = document.createElement("article");
  article.className = "message user-message";
  article.innerHTML = `
    <div class="avatar">你</div>
    <div class="message-body"><p>${escapeHtml(text)}</p></div>
  `;
  stream.appendChild(article);
  scrollChatToBottom();
}

function appendStreamingAssistantMessage() {
  const stream = byId("chatStream");
  if (!stream) {
    return {
      append() {},
      setStatus() {},
      updateStep() {},
      fail(message) {
        appendErrorMessage(message);
      },
      remove() {},
    };
  }
  const article = document.createElement("article");
  article.className = "message assistant-message streaming-message";
  const nodeStates = new Map(agentNodeBlueprint.map((node) => [node.node_id, { ...node, status: "pending" }]));
  article.innerHTML = `
    <div class="avatar">OA</div>
    <div class="message-body">
      <div class="result-card">
        <div class="card-title"><span>正在生成回答</span><span class="stream-status">连接中...</span></div>
        <div class="agent-graph-live" aria-live="polite">
          ${buildAgentGraphHtml(Array.from(nodeStates.values()), true)}
        </div>
        <div class="answer streaming-answer"></div>
      </div>
    </div>
  `;
  stream.appendChild(article);
  const answer = article.querySelector(".streaming-answer");
  const status = article.querySelector(".stream-status");
  const graph = article.querySelector(".agent-graph-live");
  return {
    append(text) {
      if (!answer || !text) {
        return;
      }
      answer.textContent += text;
      scrollChatToBottom();
    },
    setStatus(message) {
      if (status) {
        status.textContent = message || "运行中...";
      }
    },
    updateStep(event) {
      if (!event?.node_id) {
        return;
      }
      const current = nodeStates.get(event.node_id) || {};
      nodeStates.set(event.node_id, { ...current, ...event });
      if (graph) {
        graph.innerHTML = buildAgentGraphHtml(Array.from(nodeStates.values()), true);
      }
      scrollChatToBottom();
    },
    fail(message) {
      if (status) {
        status.textContent = "失败";
      }
      if (answer) {
        answer.textContent = message || "回答失败。";
      }
      article.classList.add("streaming-error");
      scrollChatToBottom();
    },
    remove() {
      article.remove();
    },
  };
}

function appendAssistantMessage(result) {
  const stream = byId("chatStream");
  if (!stream) {
    return;
  }
  const article = document.createElement("article");
  article.className = "message assistant-message";
  const resultSummary = buildResultSummaryHtml(result);
  const analysis = buildStructuredHtml(result.structured_answer);
  const spec = buildProblemSpecHtml(result.problem_spec, result.rag_context || {});
  const tablePayload = buildDecisionTableHtml(result);
  const table = tablePayload.html;
  const tableTitle = tablePayload.title;
  const answer = result.structured_answer?.raw_answer || result.answer || "";
  const answerBlock = result.generic_result || result.structured_answer ? "" : `<div class="answer">${escapeHtml(answer)}</div>`;
  const requirementAnalysis = buildRequirementAnalysisHtml(result.requirement_analysis);
  const resultTitle = ({NEEDS_CLARIFICATION: "需求分析", REQUIREMENT_UPDATED: "需求已更新", COMPARISON: "方案比较"})[result.status] || "优化结论";
  const toolNames = (result.tool_names || []).map((name) => `<span>${escapeHtml(name)}</span>`).join("");
  const ragDocs = (result.rag_docs || []).map((name) => `<span>${escapeHtml(name)}</span>`).join("");
  const graphNodes = result.agent_graph?.nodes || [];
  const graphPassed = result.workflow_verification?.passed;
  const graphCard = graphNodes.length
    ? `<div class="agent-graph-card">
        <div class="agent-graph-head">
          <div><span>Agent 执行图</span><small>LangGraph · ${graphNodes.length} 个节点</small></div>
          <em class="${graphPassed === false ? "failed" : "passed"}">${graphPassed === false ? "检查异常" : "运行完成"}</em>
        </div>
        ${buildAgentGraphHtml(graphNodes)}
      </div>`
    : "";
  const agentSteps = (result.agent_steps || []).map((step) => `
    <tr>
      <td>${escapeHtml(step.step)}</td>
      <td>${escapeHtml(step.tool)}</td>
      <td>${escapeHtml(step.output)}</td>
    </tr>
  `).join("");
  const traceCard = (toolNames || ragDocs || result.mcp_status)
    ? `<div class="model-card">
        <details class="trace-details">
          <summary class="card-title"><span>Agent 工作过程</span><span>${escapeHtml(result.mcp_status || "")}</span></summary>
          ${agentSteps ? `<div class="trace-note">展示的是可审计的工具调用和校验摘要，不包含模型隐藏推理链。</div>
          <div class="table-wrap compact-table">
            <table>
              <thead><tr><th>步骤</th><th>工具</th><th>输出</th></tr></thead>
              <tbody>${agentSteps}</tbody>
            </table>
          </div>` : ""}
        </details>
        ${toolNames ? `<div class="trace-row"><strong>Tools</strong><div>${toolNames}</div></div>` : ""}
        ${ragDocs ? `<div class="trace-row"><strong>RAG</strong><div>${ragDocs}</div></div>` : ""}
      </div>`
    : "";
  const modelCard = spec.trim()
    ? `<div class="model-card">
        <details class="trace-details">
          <summary class="card-title"><span>建模方案</span><span>${escapeHtml(result.problem_spec?.problem_type || "")}</span></summary>
          <div class="spec-box">${spec}</div>
        </details>
      </div>`
    : "";
  const tableCard = table.trim()
    ? `<div class="table-card">
        <div class="card-title"><span>${escapeHtml(tableTitle || "数据结果")}</span><span>${escapeHtml(fmt(result.objective_value))}</span></div>
        <div class="table-wrap">${table}</div>
      </div>`
    : "";
  article.innerHTML = `
    <div class="avatar">OA</div>
    <div class="message-body">
      <div class="result-card">
        <div class="card-title"><span>${resultTitle}</span><span>${escapeHtml(result.status || "-")}</span></div>
        <div class="result-summary">${resultSummary}</div>
        ${answerBlock}
      </div>
      ${requirementAnalysis ? `<div class="requirement-card">${requirementAnalysis}</div>` : ""}
      ${buildPlanComparisonHtml(result.plan_comparison)}
      ${buildDialogueControlsHtml(result)}
      ${graphCard}
      ${tableCard}
      ${analysis.trim() ? `<div class="model-card">
        <details class="trace-details">
          <summary class="card-title"><span>分析说明</span><span>建议 / 风险 / 依据</span></summary>
          <div class="structured">${analysis}</div>
        </details>
      </div>` : ""}
      ${traceCard}
      ${modelCard}
    </div>
  `;
  stream.appendChild(article);
  bindSuggestionButtons(article);
  article.querySelector(".download-plan")?.addEventListener("click", () => {
    // 下载当前这条消息的完整记录，后续编辑不会改变该方案内容。
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = `optiagent-plan-${result.run_id || "result"}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  scrollChatToBottom();
}

function buildPlanComparisonHtml(comparison) {
  // 比较的是两次已验算结果，不把不同业务条件下的增减标成算法优劣。
  if (!comparison?.available) return "";
  const labels = {capacity: "容量上限", items: "物品数据", resources: "资源", tasks: "任务", costs: "成本", products: "产品", capacities: "资源容量", distance_matrix: "距离矩阵", distances: "距离", nodes: "地点", integer: "整数要求"};
  const changeText = comparison.input_diff_available === false ? "旧方案未记录结构化版本，无法确定输入变化。"
    : comparison.changed_fields.length ? `变化内容：${comparison.changed_fields.map(field => labels[field] || field).join("、")}` : "有效输入未变化";
  return `<section class="plan-comparison"><div class="card-title"><span>方案变化</span><span>同一对话 · 已验算</span></div>
    <div class="plan-values"><div><small>上次方案 #${escapeHtml(comparison.before_run_id)}</small><strong>${escapeHtml(fmt(comparison.before_objective))}</strong></div>
    <span aria-hidden="true">→</span><div><small>本次方案 #${escapeHtml(comparison.after_run_id)}</small><strong>${escapeHtml(fmt(comparison.after_objective))}</strong></div>
    <div><small>目标值变化</small><strong>${comparison.delta > 0 ? "+" : ""}${escapeHtml(fmt(comparison.delta))}</strong></div></div>
    <p>${escapeHtml(changeText)}</p>
    <small>${escapeHtml(comparison.note)}</small></section>`;
}

function buildDialogueControlsHtml(result) {
  const contract = result.requirement_analysis?.dialogue_contract;
  if (!contract?.data) return "";
  const changes = (contract.changes || []).map(change => change.field === "capacity"
    ? `容量：${change.before} → ${change.after}` : ({initialize: "载入初始数据", replace_data: "替换完整数据", undo: "撤销修改并恢复数据"})[change.operation]).filter(Boolean);
  return `<section class="dialogue-revision"><div class="card-title"><span>本次使用的数据</span><span>版本 ${escapeHtml(contract.revision)}</span></div>
    ${changes.length ? `<p>${changes.map(escapeHtml).join("；")}</p>` : ""}
    <details><summary>查看有效数据</summary><pre>${escapeHtml(JSON.stringify(contract.data, null, 2))}</pre></details>
    <div class="suggestion-row">${contract.template_id === "knapsack" ? '<button class="suggestion">把容量改成 3</button>' : ""}
      <button class="suggestion">继续求解</button><button class="suggestion">撤销上次修改</button>
      <button class="suggestion">比较最近两个方案</button><button class="download-plan">下载本次记录</button></div>
    <small>快捷操作会填入输入框，发送后作用于当前会话的最新版本。</small></section>`;
}

function buildRequirementAnalysisHtml(brief) {
  // 需求面板只展示结构化摘要，不暴露模型隐藏推理或运行时密钥。
  if (!brief || !brief.summary) {
    return "";
  }
  const readinessLabels = {
    needs_clarification: "等待补充",
    ready_for_analysis: "可分析",
    ready_to_solve: "可求解",
  };
  const constraints = (brief.constraints || []).slice(0, 5);
  const missing = (brief.missing_information || []).slice(0, 4);
  const questions = (brief.clarification_questions || []).slice(0, 4);
  return `
    <div class="requirement-head">
      <div><span>需求理解</span><small>第 ${escapeHtml(brief.turn_count || 1)} 轮累计</small></div>
      <em class="${escapeHtml(brief.readiness || "needs_clarification")}">${escapeHtml(readinessLabels[brief.readiness] || brief.readiness)}</em>
    </div>
    <p>${escapeHtml(brief.summary)}</p>
    <div class="requirement-facts">
      <div><span>问题</span><strong>${escapeHtml(brief.problem_type || "待确认")}</strong></div>
      <div><span>目标</span><strong>${escapeHtml(brief.objective || "待确认")}</strong></div>
    </div>
    ${constraints.length ? `<div class="requirement-list"><span>已确认约束</span><ul>${constraints.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
    ${missing.length ? `<div class="requirement-list warning"><span>仍缺少</span><ul>${missing.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
    ${questions.length ? `<div class="requirement-list questions"><span>请继续回答</span><ol>${questions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol></div>` : ""}
  `;
}

function buildAgentGraphHtml(nodes, live = false) {
  // 实时轨道保持固定职责顺序，历史结果则按真实 transition 顺序展示重试。
  const orderOf = (node) => live ? node.sequence : (node.transition_sequence || node.sequence);
  const normalized = [...nodes].sort((left, right) => Number(orderOf(left) || 0) - Number(orderOf(right) || 0));
  const stateLabels = {
    pending: "等待",
    running: "运行中",
    completed: "完成",
    failed: "失败",
  };
  return `
    <div class="agent-flow ${live ? "is-live" : ""}" role="list" aria-label="Agent 执行流程">
      ${normalized.map((node) => {
        const nodeStatus = node.status || "pending";
        const elapsed = node.elapsed_ms == null ? "" : `${fmt(node.elapsed_ms)} ms`;
        return `
          <div class="agent-node ${escapeHtml(nodeStatus)}" role="listitem">
            <div class="agent-node-marker"><span></span></div>
            <div class="agent-node-copy">
              <div><strong>${escapeHtml(node.label || node.node_id)}${Number(node.attempt || 1) > 1 ? ` #${escapeHtml(node.attempt)}` : ""}</strong><em>${escapeHtml(stateLabels[nodeStatus] || nodeStatus)}</em></div>
              <small>${escapeHtml(node.detail || node.description || "等待上游节点")}</small>
              ${elapsed ? `<time>${escapeHtml(elapsed)}</time>` : ""}
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function buildResultSummaryHtml(result) {
  const structured = result.structured_answer || {};
  const metricItems = buildPrimaryMetrics(result);
  return `
    <div class="result-conclusion">${escapeHtml(structured.conclusion || result.answer || "")}</div>
    ${metricItems.length ? `<div class="result-metric-grid">
      ${metricItems.map((item) => `
        <div class="${item.primary ? "primary" : ""}">
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(item.value)}${escapeHtml(item.suffix || "")}</strong>
        </div>
      `).join("")}
    </div>` : ""}
    ${buildOpenWarehouseChips(result)}
  `;
}

function buildPrimaryMetrics(result) {
  const structured = result.structured_answer || {};
  const metrics = structured.metrics || {};
  const items = [];
  const objectiveLabel = metrics.objective_label || result.generic_result?.objective_label || "总成本";
  if (result.objective_value !== null && result.objective_value !== undefined) {
    items.push({ label: objectiveLabel, value: fmt(result.objective_value), primary: true });
  }
  if (result.fixed_cost !== null && result.fixed_cost !== undefined) {
    items.push({ label: "固定成本", value: fmt(result.fixed_cost) });
  }
  if (result.transport_cost !== null && result.transport_cost !== undefined) {
    items.push({ label: "运输成本", value: fmt(result.transport_cost) });
  }
  (metrics.extra || []).slice(0, 6).forEach((item, index) => {
    items.push({ label: item.label, value: formatMetricItem(item, result), suffix: item.suffix || "", primary: !items.length && index === 0 });
  });
  return items;
}

function buildOpenWarehouseChips(result) {
  const names = result.open_warehouses || [];
  if (!names.length) {
    return "";
  }
  return `
    <div class="result-chip-row">
      <span>开启仓库</span>
      <div>${names.map((name) => `<strong>${escapeHtml(name)}</strong>`).join("")}</div>
    </div>
  `;
}

function buildStructuredHtml(structured) {
  if (!structured) {
    return "";
  }
  return `
    <h3>建议</h3>
    <ul>${(structured.recommendations || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <h3>风险</h3>
    <ul>${(structured.risks || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <h3>依据</h3>
    <ul>${(structured.evidence || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
  `;
}

function buildProblemSpecHtml(spec, ragContext = {}) {
  if (!spec) {
    return "";
  }
  const requirements = (spec.data_requirements || [])
    .map((item) => `<li><strong>${escapeHtml(item.table)}</strong>：${(item.columns || []).map(escapeHtml).join("、")}；${escapeHtml(item.description || "")}</li>`)
    .join("");
  const docs = Object.entries(ragContext)
    .map(([category, items]) => {
      const names = (items || []).map((doc) => escapeHtml(doc.title)).join("、") || "暂无命中";
      return `<div><span>${escapeHtml(category)}</span><strong>${names}</strong></div>`;
    })
    .join("");
  return `
    <div class="spec-head">
      <div>
        <span>问题类型</span>
        <strong>${escapeHtml(spec.display_name)} / ${escapeHtml(spec.problem_type)}</strong>
      </div>
      <div>
        <span>推荐求解器</span>
        <strong>${escapeHtml(spec.recommended_solver)}</strong>
      </div>
      <div>
        <span>识别置信度</span>
        <strong>${Math.round((spec.confidence || 0) * 100)}%</strong>
      </div>
    </div>
    <div class="spec-section"><span>目标</span><p>${escapeHtml(spec.objective || "")}</p></div>
    <div class="spec-columns">
      <div><span>决策变量</span><ul>${(spec.decision_variables || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>
      <div><span>关键约束</span><ul>${(spec.constraints || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>
    </div>
    <div class="spec-section"><span>数据要求</span><ul>${requirements}</ul></div>
    <div class="spec-section"><span>RAG 命中</span><div class="rag-hit-grid">${docs}</div></div>
  `;
}

function buildDecisionTableHtml(result) {
  if (result.generic_result) {
    return buildGenericDecisionTableHtml(result.generic_result);
  }
  const warehouseRows = result.warehouse_summary || [];
  const openRows = warehouseRows.filter((row) => row.is_open === 1);
  if (!openRows.length) {
    return { title: "", html: "" };
  }
  return {
    title: "推荐启用仓库",
    html: `
      <table>
        <thead><tr><th>仓库</th><th>区域</th><th>使用量</th><th>利用率</th><th>固定成本</th></tr></thead>
        <tbody>
          ${openRows.map((row) => `
            <tr>
              <td>${escapeHtml(row.warehouse)}</td>
              <td>${escapeHtml(row.region)}</td>
              <td>${fmt(row.used_capacity)}</td>
              <td>${row.utilization == null ? "-" : (row.utilization * 100).toFixed(1) + "%"}</td>
              <td>${fmt(row.active_fixed_cost)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `,
  };
}

function buildGenericDecisionTableHtml(generic) {
  const decisions = generic.decisions || [];
  if (!decisions.length) {
    return { title: generic.display_name || "", html: escapeHtml(generic.summary || "") };
  }
  const tableByType = {
    knapsack: {
      headers: ["项目", "是否选择", "价值", "资源消耗"],
      rows: decisions.map((row) => [displayItemName(row.item), row.selected === 1 ? "选择" : "不选", fmt(row.value), fmt(row.weight)]),
    },
    assignment: {
      headers: ["资源", "任务", "成本"],
      rows: decisions.map((row) => [row.resource, row.task, fmt(row.cost)]),
    },
    tsp: {
      headers: ["从", "到", "距离"],
      rows: decisions.map((row) => [row.from, row.to, fmt(row.distance)]),
    },
    job_shop_scheduling: {
      headers: ["作业", "机器", "顺序", "开始", "结束", "时长"],
      rows: decisions.map((row) => [row.job, row.machine, fmt(row.order), fmt(row.start), fmt(row.end), fmt(row.duration)]),
    },
    production_mix: {
      headers: ["产品", "产量", "单位利润", "利润贡献"],
      rows: decisions.map((row) => [row.product, fmt(row.quantity), fmt(row.profit), fmt(row.total_profit)]),
    },
  };
  const preset = tableByType[generic.template_id];
  if (preset) {
    return { title: generic.display_name || "数据结果", html: buildTableHtml(preset.headers, preset.rows) };
  }
  const columns = Array.from(new Set(decisions.flatMap((row) => Object.keys(row))));
  return {
    title: generic.display_name || "数据结果",
    html: buildTableHtml(columns, decisions.map((row) => columns.map((column) => row[column] ?? ""))),
  };
}

function buildTableHtml(headers, rows) {
  return `
    <table>
      <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${row.map((value) => `<td>${escapeHtml(value)}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function appendErrorMessage(message) {
  const stream = byId("chatStream");
  if (!stream) {
    return;
  }
  const article = document.createElement("article");
  article.className = "message assistant-message";
  article.innerHTML = `
    <div class="avatar">OA</div>
    <div class="message-body">
      <div class="result-card">
        <div class="card-title"><span>运行失败</span><span>ERROR</span></div>
        <div class="answer">${escapeHtml(message)}</div>
      </div>
    </div>
  `;
  stream.appendChild(article);
  scrollChatToBottom();
}

function bindSuggestionButtons(root = document) {
  root.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      const input = byId("questionInput");
      if (input) {
        input.value = button.dataset.example === "knapsack"
          ? '求解背包问题，最大化总价值。以下为演示数据：\n' + JSON.stringify({capacity: 5, items: [{item: "A", value: 8, weight: 3}, {item: "B", value: 5, weight: 2}]}, null, 2)
          : button.textContent;
        input.focus();
      }
    });
  });
}

function scrollChatToBottom() {
  const stream = byId("chatStream");
  if (stream) {
    stream.scrollTop = stream.scrollHeight;
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayItemName(value) {
  const text = String(value ?? "");
  return /^\d+$/.test(text) ? `物品 ${text}` : text;
}

document.querySelectorAll("button[data-panel]").forEach((button) => {
  button.addEventListener("click", () => togglePanel(button.dataset.panel));
});
on("askBtn", "click", ask);
on("saveLlmBtn", "click", saveLlm);
on("loginBtn", "click", login);
on("refreshBtn", "click", refreshAll);
on("clearHistoryBtn", "click", clearHistory);
on("newConversationBtn", "click", newConversation);
on("newChatBtn", "click", newConversation);
on("providerSelect", "change", updateModelOptions);
on("datasetFiles", "change", uploadDataset);
on("conversationSearch", "input", (event) => {
  state.conversationSearch = event.target.value || "";
  renderConversations();
});
bindSuggestionButtons();
on("questionInput", "keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
});

updateModelOptions();
loadAll().catch((err) => {
  setText("datasetName", "加载失败");
  byId("askPanel")?.classList.remove("hidden");
  console.error(err);
});
