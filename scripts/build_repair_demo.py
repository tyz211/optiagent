from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    """生成可离线打开的真实评测回放页；页面不伪装成在线训练或即时求解。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.study / 'summary.json').read_text())
    if summary.get('benchmark_profile') != 'data_repair_v1':
        raise ValueError('演示只接受实际映射修复实验。')
    cases = []
    for seed, expected in summary['evaluation_sha256'].items():
        content = (args.study / f'evaluation-{seed}.json').read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('评测文件摘要不符。')
        cases.extend(dict(case, seed=int(seed)) for case in json.loads(content)['cases'])
    # 转义标签起始符，避免数据内容被浏览器当作脚本结束标签。
    payload = json.dumps({'cases': cases, 'summary': summary}, ensure_ascii=False).replace('<', '\\u003c')
    html = HTML.replace('__DATA__', payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(html)
    print(args.output.resolve())


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OptiAgent · 修复实验室</title><style>
/* 采用实验记录纸的冷白、深蓝与状态绿红，颜色仅表达验证结果。 */
:root{--paper:#f2f6fb;--ink:#17304d;--blue:#2258a3;--muted:#61758c;--green:#187758;--red:#b34745;--line:#d7e2ef}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,'PingFang SC',sans-serif}main{max-width:1160px;margin:auto;padding:36px 28px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid var(--ink);padding-bottom:18px}.brand{font:700 23px Georgia,serif;letter-spacing:1px}small,.muted{color:var(--muted)}h1{font-size:36px;line-height:1.3;margin:30px 0 12px}h2{font-size:20px}p{max-width:900px}.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}label{font-size:13px;color:var(--muted)}select{display:block;width:100%;margin-top:6px;border:1px solid var(--line);border-radius:6px;padding:11px;color:var(--ink);background:white;font:inherit}section{background:white;border:1px solid var(--line);padding:24px;margin:20px 0;border-radius:10px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:580px}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid var(--line);font-size:14px}th{color:var(--muted);font-weight:500}.good{color:var(--green)}.bad{color:var(--red)}button{font:inherit;border:1px solid var(--line);background:white;border-radius:5px;color:var(--blue);padding:6px 12px;cursor:pointer}button:hover{background:var(--paper)}:focus-visible{outline:3px solid #769ac8;outline-offset:3px}.rail{display:flex;gap:12px;overflow:auto;padding:12px 0}.event{min-width:215px;max-width:280px;border-top:4px solid var(--blue);background:var(--paper);padding:16px;border-radius:4px}.event.fail{border-color:var(--red)}.event.pass{border-color:var(--green)}.event b{display:block;margin-bottom:10px}.event code{font:12px ui-monospace,monospace}.note{border-left:3px solid var(--blue);padding-left:16px}footer{font-size:13px;color:var(--muted);margin:30px 0}.title-line{display:flex;justify-content:space-between;align-items:center;gap:12px}#status{font-size:13px}.selected{background:#edf4fd}@media(max-width:700px){main{padding:22px 16px}.controls{grid-template-columns:1fr 1fr}h1{font-size:28px}header{align-items:start;gap:15px}section{padding:16px}.title-line{display:block}}</style>
<main><header><span class="brand">OptiAgent / 修复实验室</span><small>生成数据 · 真实求解 · 受控故障</small></header>
<h1>一次重试，还是一次修复？</h1><p>查看错误映射如何影响求解，以及策略何时恢复正确数据。这里回放已完成、经过校验的实验记录，切换选项不会启动新训练或实时求解。</p>
<div class="controls"><label>优化问题<select id="template"></select></label><label>错误场景<select id="scenario"></select></label><label>训练种子<select id="seed"></select></label><label>独立实例<select id="instance"></select></label></div>
<p class="note" id="description"></p><section><h2>同一问题，五种策略</h2><div class="table-wrap"><table><thead><tr><th>策略</th><th>结果</th><th>建模 / 求解</th><th>恢复动作</th><th>记录</th></tr></thead><tbody id="comparison"></tbody></table></div></section>
<section><div class="title-line"><h2 id="trace-title">修复证据</h2><button id="step">逐步查看</button></div><div class="rail" id="trace" aria-live="polite"></div><small>每次实际求解都会记录输入指纹，并针对未被改动的原始数据重新验算。指纹不同表示求解输入确实发生变化。</small></section>
<footer><div id="status"></div><p>本 demo 中的重建是确定性的系数重新绑定，不代表 LLM 自由修复模型。持续错误场景的正确结果是拒绝接受错误解。种子共享实例，重复评测次数不等于独立样本数。</p></footer></main>
<script id="payload" type="application/json">__DATA__</script><script>
// 只渲染本地实验数据，不访问外部服务；文本通过 textContent 写入。
const data=JSON.parse(document.querySelector('#payload').textContent),$=id=>document.getElementById(id);
const names={knapsack:'背包选择',assignment:'资源指派',tsp:'路线规划',job_shop_scheduling:'车间调度',production_mix:'产品组合',facility_location:'仓库选址'};
const scenarios={clean:['正常输入','输入正确，独立验算通过后接受结果。'],transient_failure:['短暂传输失败','第一次调用未到达求解器，重试可恢复。'],tampered_objective:['短暂映射错误','第一次求解收到错误系数；后续调用可以重新取得正确数据。'],model_error:['持续映射错误','错误映射会在重试时保留；必须重建并恢复正确映射。'],persistent_failure:['无法修复的映射','重新绑定后仍无法取得正确输入，策略应拒绝接受错误解。']};
const policies={rule_policy:'规则策略',previous_policy:'上一轮模型',bc_policy:'本轮 BC',cql_final_policy:'末轮 CQL',selected_policy:'验证选中 CQL'};
const actions={accept_solution:'接受',retry_solver:'重试求解',rebuild_model:'重建映射',terminate:'终止'};
let active='selected_policy',shown=Infinity,current=[];
function options(id,items){$(id).replaceChildren(...items.map(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;return o}));}
options('template',Object.entries(names));options('scenario',Object.entries(scenarios).map(([v,t])=>[v,t[0]]));options('seed',[...new Set(data.cases.map(c=>c.seed))].sort((a,b)=>a-b).map(v=>[v,v]));$('scenario').value='model_error';
function instances(){options('instance',[...new Set(data.cases.filter(c=>c.template_id===$('template').value).map(c=>c.instance_fingerprint))].sort().map((v,i)=>[v,`实例 ${i+1} · ${v.slice(0,7)}`]));render();}
function render(){current=data.cases.filter(c=>c.template_id===$('template').value&&c.scenario===$('scenario').value&&c.seed===Number($('seed').value)&&c.instance_fingerprint===$('instance').value);$('description').textContent=scenarios[$('scenario').value][1];$('comparison').replaceChildren();for(const [key,label]of Object.entries(policies)){const c=current.find(c=>c.policy===key);if(!c)continue;const tr=document.createElement('tr');if(key===active)tr.className='selected';for(const text of [label,c.success?'验算通过':'未接受结果',`${c.model_calls} / ${c.solver_calls}`,c.actions.map(a=>actions[a]).join(' → ')]){const td=document.createElement('td');td.textContent=text;tr.append(td)}const td=document.createElement('td'),b=document.createElement('button');b.textContent='查看';b.onclick=()=>{active=key;shown=Infinity;render()};td.append(b);tr.append(td);$('comparison').append(tr)}trace();}
function trace(){const c=current.find(c=>c.policy===active);$('trace-title').textContent=`${policies[active]} · 执行证据`;$('trace').replaceChildren();for(const [i,e]of(c?.repair_events||[]).entries()){if(i>=shown)break;const div=document.createElement('div');div.className='event'+(e.event==='solve'?(e.source_verified?' pass':' fail'):'');const title=document.createElement('b');title.textContent=`${i+1}. `+(e.event==='rebind'?(e.rebuilding?'重新绑定':'首次绑定'):e.event==='solve'?'实际求解与验算':'传输失败');div.append(title);const lines=e.event==='solve'?[e.source_verified?'原始数据验算：通过':'原始数据验算：未通过',`输入 ${e.input.slice(0,12)}`,`原始 ${e.source.slice(0,12)}`,`返回目标 ${Number(e.reported_objective).toFixed(2)}`,`复算目标 ${Number(e.recomputed_objective).toFixed(2)}`]:e.event==='rebind'?[`之前 ${e.before.slice(0,12)}`,`之后 ${e.after.slice(0,12)}`,e.after===e.source?'映射与原始数据一致':'映射与原始数据不同']:['本次未调用底层求解器'];for(const text of lines){const p=document.createElement('div');p.textContent=text;div.append(p)}$('trace').append(div)}}
$('template').onchange=instances;for(const key of ['scenario','seed','instance'])$(key).onchange=()=>{shown=Infinity;render()};$('step').onclick=()=>{const total=current.find(c=>c.policy===active)?.repair_events.length||0;shown=!Number.isFinite(shown)||shown>=total?1:shown+1;trace()};$('status').textContent=`${data.summary.external_instances} 个独立外部实例 · ${data.summary.runs.length} 个训练种子 · 修复可靠性门槛${data.summary.repair_reliability_gate_passed?'通过':'未通过'} · 未切换线上策略`;instances();
</script></html>'''

if __name__ == '__main__':
    main()
