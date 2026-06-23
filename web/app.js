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

const state = {
  activeDatasetId: null,
  activeConversationId: Number(localStorage.getItem("optiagent_active_conversation_id") || 0) || null,
  hasData: false,
  token: localStorage.getItem("optiagent_session_token") || "",
  lastResult: null,
  conversations: [],
  conversationSearch: "",
};

const fmt = (value) => value === null || value === undefined ? "-" : Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 0 });

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
  byId("emptyState")?.classList.toggle("hidden", state.hasData);
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
      <p>你好，上传 CSV 后我会先读取当前对话内的文件，再根据你的问题进行建模、分析或求解。</p>
      <div class="suggestion-row">
        <button class="suggestion">分析当前供应链数据</button>
        <button class="suggestion">求解一个背包问题</button>
        <button class="suggestion">做员工班次指派</button>
      </div>
    </div>
  `;
  stream.appendChild(article);
  bindSuggestionButtons(article);
}

function renderResult(result) {
  state.lastResult = result;
  renderStructured(result.structured_answer);
  renderProblemSpec(result.problem_spec, result.rag_context || {});
  renderDecisionTable(result);
  setText("answerBox", result.structured_answer?.raw_answer || result.answer);
  setText("objectiveMetric", fmt(result.objective_value));
  setText("transportMetric", fmt(result.transport_cost));
  setText("fixedMetric", fmt(result.fixed_cost));
  setText("statusMetric", result.status || "-");
  appendAssistantMessage(result);
}

function renderDecisionTable(result) {
  const warehouseRows = result.warehouse_summary || [];
  if (!result.generic_result && !warehouseRows.length) {
    setText("resultTableTitle", "");
    const table = byId("warehouseTable");
    if (table) {
      table.innerHTML = "";
    }
    return;
  }
  if (result.generic_result) {
    renderGenericDecisionTable(result.generic_result);
    return;
  }
  setText("resultTableTitle", "推荐启用仓库");
  const openRows = warehouseRows.filter((row) => row.is_open === 1);
  const table = byId("warehouseTable");
  if (!table) {
    return;
  }
  if (!openRows.length) {
    table.textContent = "暂无可用方案。";
    return;
  }
  table.innerHTML = `
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
  `;
}

function renderGenericDecisionTable(generic) {
  const table = byId("warehouseTable");
  if (!table) {
    return;
  }
  setText("resultTableTitle", generic.display_name);
  const decisions = generic.decisions || [];
  if (!decisions.length) {
    table.textContent = generic.summary || "暂无可用方案。";
    return;
  }
  if (generic.template_id === "knapsack") {
    table.innerHTML = `
      <table>
        <thead><tr><th>项目</th><th>是否选择</th><th>价值</th><th>资源消耗</th></tr></thead>
        <tbody>
          ${decisions.map((row) => `
            <tr>
              <td>${escapeHtml(displayItemName(row.item))}</td>
              <td>${row.selected === 1 ? "选择" : "不选"}</td>
              <td>${fmt(row.value)}</td>
              <td>${fmt(row.weight)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    return;
  }
  if (generic.template_id === "assignment") {
    table.innerHTML = `
      <table>
        <thead><tr><th>资源</th><th>任务</th><th>成本</th></tr></thead>
        <tbody>
          ${decisions.map((row) => `
            <tr>
              <td>${escapeHtml(row.resource)}</td>
              <td>${escapeHtml(row.task)}</td>
              <td>${fmt(row.cost)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    return;
  }
  if (generic.template_id === "tsp") {
    table.innerHTML = `
      <table>
        <thead><tr><th>从</th><th>到</th><th>距离</th></tr></thead>
        <tbody>
          ${decisions.map((row) => `
            <tr>
              <td>${escapeHtml(row.from)}</td>
              <td>${escapeHtml(row.to)}</td>
              <td>${fmt(row.distance)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    return;
  }
  if (generic.template_id === "job_shop_scheduling") {
    table.innerHTML = `
      <table>
        <thead><tr><th>作业</th><th>机器</th><th>顺序</th><th>开始</th><th>结束</th><th>时长</th></tr></thead>
        <tbody>
          ${decisions.map((row) => `
            <tr>
              <td>${escapeHtml(row.job)}</td>
              <td>${escapeHtml(row.machine)}</td>
              <td>${fmt(row.order)}</td>
              <td>${fmt(row.start)}</td>
              <td>${fmt(row.end)}</td>
              <td>${fmt(row.duration)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    return;
  }
  if (generic.template_id === "production_mix") {
    table.innerHTML = `
      <table>
        <thead><tr><th>产品</th><th>产量</th><th>单位利润</th><th>利润贡献</th></tr></thead>
        <tbody>
          ${decisions.map((row) => `
            <tr>
              <td>${escapeHtml(row.product)}</td>
              <td>${fmt(row.quantity)}</td>
              <td>${fmt(row.profit)}</td>
              <td>${fmt(row.total_profit)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    return;
  }
  renderDynamicDecisionTable(table, decisions);
}

function renderDynamicDecisionTable(table, decisions) {
  const columns = Array.from(new Set(decisions.flatMap((row) => Object.keys(row))));
  table.innerHTML = `
    <table>
      <thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
      <tbody>
        ${decisions.map((row) => `
          <tr>${columns.map((column) => `<td>${escapeHtml(row[column] ?? "")}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderProblemSpec(spec, ragContext = {}) {
  const root = byId("problemSpecBox");
  if (!root) {
    return;
  }
  root.innerHTML = buildProblemSpecHtml(spec, ragContext);
}

function renderStructured(structured) {
  const root = byId("structuredBox");
  if (!root) {
    return;
  }
  root.innerHTML = buildStructuredHtml(structured);
}

async function ask() {
  const status = byId("runStatus");
  const input = byId("questionInput");
  const question = input?.value.trim() || "";
  if (!question) {
    return;
  }
  setText("runStatus", "运行中...");
  appendUserMessage(question);
  input.value = "";
  try {
    const result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        dataset_id: state.activeDatasetId,
        conversation_id: state.activeConversationId,
        mcp_config: byId("mcpInput")?.value || "",
      }),
    });
    if (result.conversation_id) {
      setActiveConversation(result.conversation_id);
    }
    renderResult(result);
    setText("runStatus", "完成");
    loadAll().catch((err) => console.error(err));
  } catch (err) {
    setText("runStatus", "失败");
    setText("answerBox", err.message);
    appendErrorMessage(err.message);
  }
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
  state.lastResult = null;
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
  state.lastResult = null;
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
  const toolNames = (result.tool_names || []).map((name) => `<span>${escapeHtml(name)}</span>`).join("");
  const ragDocs = (result.rag_docs || []).map((name) => `<span>${escapeHtml(name)}</span>`).join("");
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
        <div class="card-title"><span>优化结论</span><span>${escapeHtml(result.status || "-")}</span></div>
        <div class="result-summary">${resultSummary}</div>
        ${answerBlock}
      </div>
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
  scrollChatToBottom();
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
  (metrics.extra || []).slice(0, 3).forEach((item) => {
    items.push({ label: item.label, value: fmt(item.value), suffix: item.suffix || "" });
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
        input.value = button.textContent;
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
