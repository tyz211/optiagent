'use strict';
// 页面只发送有限选项；所有执行记录均来自当前请求。
const $ = id => document.getElementById(id);
const actions = {accept_solution: '接受', retry_solver: '重试求解', rebuild_model: '重建映射', terminate: '终止'};
const nodeLabels = {requirements: '需求分析', planner: '规划', data: '数据准备', modeler: '建模', solver: '求解', verifier: '验算', policy: '策略决策', explainer: '结果说明'};
let catalog, controller = null, lastResult = null, currentInput = null;
let history = [], nodeElements = new Map(), running = false;

function status(label, message, tone = '') {
  $('state').textContent = label;
  $('state').className = 'badge ' + tone;
  $('message').textContent = message;
}
function options(id, entries) {
  $(id).replaceChildren(...entries.map(([value, label, disabled = false]) => {
    const option = document.createElement('option');
    Object.assign(option, {value, textContent: label, disabled});
    return option;
  }));
}
function busy(value) {
  running = value;
  for (const id of ['template', 'scenario', 'policy', 'seed', 'instance', 'run']) $(id).disabled = value || !catalog;
  $('cancel').hidden = !value;
}
function request() {
  return {template_id: $('template').value, scenario: $('scenario').value, policy: $('policy').value,
    seed: Number($('seed').value), instance_index: Number($('instance').value)};
}
function key(value) { return JSON.stringify([value.template_id, value.scenario, value.seed, value.instance_index]); }
function safeStop(c) { return !c.unverified_accept && c.valid_actions && c.actions.at(-1) === 'terminate'; }
function outcome(c) {
  if (c.unverified_accept || !c.valid_actions) return '决策异常';
  return c.success ? '验算通过' : !c.recoverable && safeStop(c) ? '安全终止' : '未能恢复';
}
function paragraphs(parent, lines) {
  for (const line of lines) {
    const p = document.createElement('p');
    p.textContent = line;
    parent.append(p);
  }
}
function compare() {
  const selected = key(request()), latest = new Map();
  for (const item of history) if (key(item.request) === selected) latest.set(item.request.policy, item);
  $('comparison').replaceChildren();
  if (!latest.size) {
    const tr = document.createElement('tr'), td = document.createElement('td');
    td.colSpan = 4;
    td.textContent = '此案例尚未运行。';
    tr.append(td);
    $('comparison').append(tr);
    return;
  }
  for (const [policy, item] of latest) {
    const tr = document.createElement('tr');
    for (const text of [catalog.policies.find(p => p.id === policy).label, outcome(item.case),
      `${item.case.model_calls} / ${item.case.solver_calls}`, item.case.actions.map(a => actions[a]).join(' → ')]) {
      const td = document.createElement('td');
      td.textContent = text;
      tr.append(td);
    }
    $('comparison').append(tr);
  }
}
function evidence(event) {
  if ($('evidence').querySelector('.empty')) $('evidence').replaceChildren();
  const card = document.createElement('div'), title = document.createElement('strong');
  card.className = 'event' + (event.event === 'solve' ? (event.source_verified ? ' pass' : ' fail') : '');
  title.textContent = event.event === 'rebind' ? (event.rebuilding ? '重新绑定映射' : '首次绑定映射')
    : event.event === 'solve' ? '实际求解与独立验算' : '传输失败';
  card.append(title);
  const short = v => v.slice(0, 12), number = v => v == null ? '不可复算' : Number(v).toFixed(2);
  const lines = event.event === 'solve' ? [
    event.source_verified ? '原始数据验算通过' : '原始数据验算未通过',
    `返回目标：${number(event.reported_objective)} · 复算目标：${number(event.recomputed_objective)}`,
    `输入 ${short(event.input)} / 原始 ${short(event.source)}`,
  ] : event.event === 'rebind' ? [
    event.after === event.source ? '当前映射与原始数据一致' : '当前映射与原始数据不同',
    `之前 ${short(event.before)} → 之后 ${short(event.after)}`,
  ] : ['本次未调用底层求解器，等待策略决定是否重试。'];
  paragraphs(card, lines);
  $('evidence').append(card);
}
function result(payload) {
  // 导出时保留本次输入；页面上的新选项不会覆盖已完成记录的身份。
  lastResult = {...payload, input: currentInput};
  history.push(lastResult);
  const c = payload.case, safe = !c.recoverable && safeStop(c);
  const abnormal = c.unverified_accept || !c.valid_actions;
  status(abnormal ? '决策异常' : c.success ? '运行完成' : safe ? '安全终止' : '未能恢复',
    abnormal ? '本次决策未通过安全检查，请查看导出记录。'
      : c.success ? '结果已通过原始数据的独立验算。'
      : safe ? '此场景被设为不可修复，策略已拒绝接受错误解。'
      : '该策略未恢复此次故障，可更换策略比较。',
    !abnormal && (c.success || safe) ? 'good' : 'bad');
  $('result').replaceChildren();
  paragraphs($('result'), [
    `恢复动作：${c.actions.map(a => actions[a]).join(' → ')}`,
    `建模 ${c.model_calls} 次 · 求解尝试 ${c.solver_calls} 次 · 推理回退 ${c.fallbacks} 次`,
    `本次运行 ${payload.run_id.slice(0, 12)} · 工作流 ${c.elapsed_seconds.toFixed(2)} 秒（不含进程启动）`,
    payload.checkpoint_sha256 ? `模型摘要 ${payload.checkpoint_sha256.slice(0, 16)}` : '使用确定性规则策略',
  ]);
  $('result').hidden = false;
  $('download').hidden = false;
  compare();
}
function handle(kind, data) {
  if (kind === 'started') {
    status('正在启动', '正在启动隔离工作流…');
  } else if (kind === 'input') {
    currentInput = data;
    $('sourceData').textContent = JSON.stringify(data.source_data, null, 2);
    $('corruptedData').textContent = JSON.stringify(data.corrupted_data, null, 2);
  } else if (kind === 'node') {
    const id = `${data.node_id}:${data.attempt}`;
    let li = nodeElements.get(id);
    if (!li) { li = document.createElement('li'); nodeElements.set(id, li); $('nodes').append(li); }
    li.textContent = `${nodeLabels[data.node_id] || data.label} · 第 ${data.attempt} 次`;
    li.className = data.status;
    li.title = data.detail || data.description || '';
    if (data.status === 'running') status('正在执行', data.detail || data.label);
  } else if (kind === 'evidence') evidence(data);
  else if (kind === 'final') result(data);
  else if (kind === 'error') throw new Error(data.message || '本次运行失败。');
}
async function run(event) {
  event.preventDefault();
  if (running || !catalog) return;
  const body = request();
  controller = new AbortController();
  lastResult = null;
  currentInput = null;
  nodeElements = new Map();
  $('nodes').replaceChildren();
  $('evidence').replaceChildren();
  $('result').hidden = true;
  $('download').hidden = true;
  $('sourceData').textContent = '生成中…';
  $('corruptedData').textContent = '生成中…';
  $('runContext').textContent = `${catalog.templates[body.template_id]} · ${catalog.scenarios[body.scenario].label} · ${catalog.policies.find(p => p.id === body.policy).label} · 种子 ${body.seed} / 实例 ${body.instance_index}`;
  busy(true);
  status('正在连接', '准备本次实时演示…');
  let reader, terminal = false;
  try {
    const response = await fetch('/api/demo/repair/stream', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body), signal: controller.signal});
    if (!response.ok) {
      const error = await response.json();
      throw new Error(typeof error.detail === 'string' ? error.detail : '请求参数或服务状态不正确。');
    }
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    // 网络分块可能截断中文或 SSE 事件，必须累计到完整边界再解析。
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      let end;
      while ((end = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        let kind = '', json = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) kind = line.slice(6).trim();
          if (line.startsWith('data:')) json += line.slice(5).trim();
        }
        if (json) { handle(kind, JSON.parse(json)); if (kind === 'final') terminal = true; }
      }
      if (done) break;
    }
    if (!terminal) throw new Error('连接结束但未收到完整结果，请重试。');
  } catch (error) {
    if (error.name === 'AbortError') status('已停止', '本次运行已取消，服务将回收演示进程。');
    else status('运行失败', error.message, 'bad');
  } finally {
    if (reader) { try { await reader.cancel(); } catch {} reader.releaseLock(); }
    controller = null;
    busy(false);
  }
}
async function load() {
  try {
    const response = await fetch('/api/demo/repair/catalog');
    if (!response.ok) throw new Error('无法读取演示配置。');
    catalog = await response.json();
    options('template', Object.entries(catalog.templates));
    options('scenario', Object.entries(catalog.scenarios).map(([id, item]) => [id, item.label]));
    options('policy', catalog.policies.map(item => [item.id, item.label + (item.available ? '' : '（未配置）'), !item.available]));
    $('scenario').value = 'model_error';
    $('policy').value = catalog.policies.find(p => p.id === 'learned' && p.available) ? 'learned' : 'rule';
    $('availability').textContent = catalog.policies.some(p => !p.available)
      ? '部分模型未配置。可明确选择规则策略运行，也可在服务器配置检查点。'
      : '三种策略均可用。每次只执行当前选中的策略。';
    $('scenarioDescription').textContent = catalog.scenarios[$('scenario').value].description;
    $('reload').hidden = true;
    status('准备就绪', '选择场景和策略，开始一次新的实时运行。');
    busy(false);
  } catch (error) {
    catalog = null;
    status('配置不可用', error.message, 'bad');
    $('reload').hidden = false;
    busy(false);
  }
}
$('runForm').addEventListener('submit', run);
$('cancel').onclick = () => controller?.abort();
$('reload').onclick = load;
for (const id of ['template', 'scenario', 'seed', 'instance']) $(id).addEventListener('change', () => {
  if (catalog) { $('scenarioDescription').textContent = catalog.scenarios[$('scenario').value].description; compare(); }
});
$('download').onclick = () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = `repair-${lastResult.run_id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
load();
