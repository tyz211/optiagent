"""本地小模型的运输需求草稿接口：结构通过后仍需用户核对语义。"""

import json
from urllib.parse import urlparse

from pydantic import Field

from optiagent.llm import LLMConfig, call_openai_compatible_chat, parse_json_object
from optiagent.transportation import TransportationData, TransportRecord


class TransportationDraft(TransportRecord):
    """缺失或不支持的需求单独返回，模型不得编造参数补齐。"""

    data: TransportationData | None
    missing_information: list[str] = Field(default_factory=list)
    unsupported_requirements: list[str] = Field(default_factory=list)
    source_quotes: list[str] = Field(default_factory=list)


def propose_transportation(question: str, config: LLMConfig) -> TransportationDraft:
    """首版仅调用显式配置的本机服务，不使用云端模型兜底。"""

    if not config.enabled or urlparse(config.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("运输草稿解析需要启用本机模型服务")
    if len(question) > 8000:
        raise ValueError("运输草稿输入过长，请缩短描述或使用结构化数据")
    schema = TransportationDraft.model_json_schema()
    raw = call_openai_compatible_chat(config, [
        {"role": "system", "content": (
            "将用户描述提取为单周期单商品运输分配草稿，只输出符合 Schema 的 JSON。"
            "只支持供给上限、需求恰好满足、显式禁运、非负连续运输量和线性运费最小化。"
            "必须明确数量单位和币种，每对供需点必须有运费或禁运声明。"
            "不得编造缺失成本、节点、供给、需求、单位或禁运线路。"
            "存在缺失信息时 data=null，并填写 missing_information。"
            "车辆、时间窗、整数、线路容量、多商品、缺货或其他目标等未支持条件必须列入 unsupported_requirements。"
            "source_quotes 使用用户原文片段，不得声称已求解；草稿将由用户核对。"
            "输出 Schema：" + json.dumps(schema, ensure_ascii=False)
        )},
        {"role": "user", "content": question},
    ], json_schema=schema, timeout=60, max_tokens=3000)
    draft = TransportationDraft.model_validate(parse_json_object(raw))
    if not draft.source_quotes or any(not quote.strip() or quote not in question for quote in draft.source_quotes):
        raise ValueError("模型没有提供可追溯到用户原文的依据")
    return draft


def add_transportation_draft(state: dict, question: str, config: LLMConfig) -> dict:
    """保存候选与运行方式；结构化解码失败时保留澄清，不执行求解。"""

    output = dict(state)
    output["parser_attempt"] = {"source": "local_llm", "model": config.model, "structured_output": True}
    try:
        draft = propose_transportation(question, config)
    except Exception as exc:
        # 不回显服务原始响应或密钥，只保存故障类型和原有的澄清说明。
        output["parser_attempt"]["status"] = "failed"
        output["parser_attempt"]["error_type"] = type(exc).__name__
        return output
    issues = [*draft.missing_information, *draft.unsupported_requirements]
    if issues or draft.data is None:
        output["parser_attempt"]["status"] = "needs_clarification"
        output["error"] = "运输需求需要澄清：" + "；".join(issues or ["尚未获得完整运输参数"])
        return output
    output["parser_attempt"]["status"] = "awaiting_confirmation"
    output["pending_transportation"] = draft.data.model_dump()
    output["pending_source_quotes"] = draft.source_quotes
    output["error"] = (
        "本地模型已生成候选运输数据，尚未求解。请核对节点、供需、每条运费、禁运线路与单位；"
        "确认无误可输入“确认运输草稿并求解”，或提交修正后的完整数据。草稿：\n"
        + json.dumps(draft.data.model_dump(), ensure_ascii=False, indent=2)
    )
    return output
