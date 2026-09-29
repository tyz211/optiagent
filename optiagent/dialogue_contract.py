from __future__ import annotations

from copy import deepcopy
import json
import math
import re

from optiagent.generic_solvers import _extract_json_payload
from optiagent.mcp_validation import validate_problem_data
from optiagent.templates.registry import get_template


# 对话编辑只改变已经能被 Gateway 消费的字段，不把自然语言约束冒充可执行约束。
ALIASES = {'knapsack': 'knapsack', 'assignment': 'assignment', 'tsp': 'tsp',
           'linear_program': 'linear_program',
           'job_shop': 'job_shop_scheduling', 'scheduling': 'job_shop_scheduling',
           'production': 'production_mix', 'production_mix': 'production_mix'}
SHAPES = {'knapsack': {'items', 'capacity'}, 'assignment': {'resources', 'tasks', 'costs'},
          'linear_program': {'variables', 'objective', 'constraints'},
          'job_shop_scheduling': {'tasks'}, 'production_mix': {'products', 'capacities'}}


def identify(payload: dict) -> tuple[str | None, dict]:
    """优先显式模板包装，再根据必需字段识别完整输入。"""
    for alias, template in ALIASES.items():
        if isinstance(payload.get(alias), dict):
            return template, payload[alias]
    for template, keys in SHAPES.items():
        if keys <= payload.keys():
            return template, payload
    if 'distance_matrix' in payload or 'distances' in payload:
        return 'tsp', payload
    return None, payload


def update_contract(question: str, previous: dict | None) -> dict | None:
    """保存不可变版本；非法或不支持的编辑不改变当前有效数据。"""
    old = deepcopy((previous or {}).get('dialogue_contract') or {})
    text = question.strip()
    payload = _extract_json_payload(text)
    template, data = identify(payload) if payload is not None else (None, {})
    if payload is None:
        # 数学文本和 JSON 形成相同版本合同；解析失败不能覆盖已有有效输入。
        from optiagent.linear_model import parse_linear_text
        try:
            parsed = parse_linear_text(text)
        except ValueError as exc:
            state = deepcopy(old)
            state.update(action='clarify', changes=[], error=f'数学模型尚不能完整解析：{str(exc)[:1800]}')
            return state
        if parsed is not None:
            template, data = 'linear_program', parsed
    if not template and not old:
        return None
    state = deepcopy(old)
    state.update(action='solve', changes=[], error=None)
    hold_words = r'先不求解|暂不求解|只修改|先不算|先分析|只分析|不要求解'
    hold = bool(re.search(hold_words, text))
    if template:
        # 新完整 JSON 原子替换旧输入；不把新旧 JSON 拼接后交给提取器猜测。
        state.update(template_id=template)
        candidate = deepcopy(data)
        operation = 'replace_data' if old else 'initialize'
    elif payload is not None or '{' in text:
        state.update(action='clarify', error='数据格式不完整，请提供当前模板所需的完整 JSON；旧版本未修改。')
        return state
    elif re.fullmatch(r'(?:请)?(?:和|与)?(?:上一(?:个|次)?方案)?(?:比较|对比)(?:一下)?[。！!？?\s]*', text) or text in {'和上一个方案比较', '比较最近两个方案'}:
        state['action'] = 'compare'
        return state
    elif re.fullmatch(r'(?:请)?(?:撤销|撤回)(?:上一次|上次|刚才的?|最近一次)?(?:修改|变更)?[。！!\s]*', text):
        active = next(v for v in old['versions'] if v['revision'] == old['revision'])
        parent = active.get('parent_revision')
        target = next((v for v in old['versions'] if v['revision'] == parent), None)
        if not target:
            state.update(action='clarify', error='当前已是初始版本，没有可撤销的修改。')
            return state
        candidate = deepcopy(target['data'])
        state['template_id'] = target['template_id']
        operation = 'undo'
    else:
        candidate = deepcopy(old['data'])
        # 只接受完整匹配的容量编辑，防止忽略同一句中的其他业务限制或未知单位。
        command = re.sub(r'[，,。；;\s]*(?:' + hold_words + r')[，,。；;\s]*', '', text)
        match = re.fullmatch(r'(?:请)?(?:把|将)?(?:背包)?容量(?:从\s*([0-9.]+)\s*)?(?:改为|改成|调整为|设为|设置为)\s*([+-]?[0-9.]+)\s*(?:并(?:重新)?求解|后(?:重新)?求解)?[。！!\s]*', command)
        if match and old['template_id'] == 'knapsack':
            try:
                value = float(match[2])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
                if match[1] is not None and float(match[1]) != float(candidate['capacity']):
                    raise ValueError
            except ValueError:
                state.update(action='clarify', error='容量必须为正数，且“从”的旧值必须与当前容量一致；旧版本未修改。')
                return state
            candidate['capacity'] = value
            operation = 'replace_capacity'
        elif re.fullmatch(r'(?:请)?(?:继续|重新|再次)?(?:求解|计算)|确认(?:并求解)?|开始求解', command):
            state['action'] = 'hold' if hold else 'solve'
            return state
        elif hold and not command.strip('，,。；; '):
            state['action'] = 'hold'
            return state
        else:
            state.update(action='clarify', error='这条修改尚不能可靠执行。可输入“把容量改成 6”（背包）、“撤销上次修改”、“比较最近两个方案”，或提交完整 JSON 替换数据；旧版本未修改。')
            return state
    try:
        json.dumps(candidate, allow_nan=False)
        report = validate_problem_data(state['template_id'], candidate)
        if not report.valid:
            raise ValueError('；'.join(report.errors))
    except (ValueError, TypeError, KeyError) as exc:
        # 保留旧合同，防止无效输入污染后续“继续求解”。
        state = deepcopy(old)
        state.update(action='clarify', changes=[], error=f'输入未通过校验：{exc}')
        return state
    if old and candidate == old.get('data') and state['template_id'] == old.get('template_id'):
        state['action'] = 'hold' if hold else 'solve'
        return state
    versions = deepcopy(old.get('versions', []))
    if len(versions) >= 100:
        state = deepcopy(old)
        state.update(action='clarify', changes=[], error='当前会话已达到 100 个需求版本，请新建对话继续。')
        return state
    revision = len(versions) + 1
    parent = old.get('revision')
    # 撤销沿恢复目标的父链继续，连续撤销不会在两个版本之间来回切换。
    if operation == 'undo':
        parent = target.get('parent_revision')
    change = {'operation': operation, 'turn': int((previous or {}).get('turn_count', 0)) + 1,
              'before_revision': old.get('revision'), 'after_revision': revision}
    if operation == 'replace_capacity':
        change.update(field='capacity', before=old['data']['capacity'], after=candidate['capacity'])
    versions.append({'revision': revision, 'parent_revision': parent, 'template_id': state['template_id'],
                     'data': deepcopy(candidate), 'change': change})
    state.update(data=candidate, revision=revision, versions=versions, changes=[change], action='hold' if hold else 'solve')
    return state


def apply_contract(brief, state: dict):
    """用唯一有效输入构造求解请求，不将历史错误指令送入求解器。"""
    brief.dialogue_contract = state
    if state.get('data'):
        template = get_template(state['template_id'])
        brief.template_id = state['template_id']
        brief.problem_type = template.display_name
        brief.resolved_request = f'请进行求解：{template.display_name}。\n数据：\n' + json.dumps(state['data'], ensure_ascii=False)
        spec = template.build_spec(brief.resolved_request, None)
        brief.objective = spec.objective
        # 已识别的标准容量约束保留用户原话，具体数值只由有效输入决定。
        retained = [text for text in brief.constraints if text in {'总重量不能超过容量', '总重量不超过容量'}]
        brief.constraints = list(spec.constraints)
        brief.constraints.extend(retained)
        if brief.template_id == 'linear_program':
            # 在需求摘要中展示真正执行的目标和约束，而不是空泛的模板介绍。
            from optiagent.linear_solver import describe_linear_model
            brief.objective, _, brief.constraints = describe_linear_model(state['data'])
        if brief.template_id == 'knapsack':
            brief.constraints.append(f"当前容量上限：{float(state['data']['capacity']):g}")
        brief.data_sources = [f"会话内结构化数据 · 版本 {state['revision']}"]
        brief.known_facts = []
        brief.assumptions = []
        brief.missing_information = []
        brief.clarification_questions = []
        brief.intent = 'solve'
        brief.readiness = 'ready_to_solve'
        brief.summary = f"{template.display_name}，使用当前有效数据版本 {state['revision']}。"
    if state.get('error'):
        brief.readiness = 'needs_clarification'
        brief.missing_information = [state['error']]
        brief.clarification_questions = [state['error']]
    elif state['action'] in {'hold', 'compare'}:
        brief.readiness = 'ready_for_analysis'
        brief.intent = 'analyze'
    return brief
