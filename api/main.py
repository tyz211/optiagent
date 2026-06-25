from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from io import StringIO
import json
import math
from pathlib import Path
import time

import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.database import (
    clear_runs,
    clear_active_dataset,
    create_conversation,
    delete_conversation,
    ensure_conversation,
    get_active_dataset_id_or_none,
    get_active_llm_config,
    get_conversation,
    get_user_by_token,
    init_db,
    list_conversations,
    list_datasets,
    list_uploaded_files,
    list_runs,
    load_dataset,
    login_user,
    save_dataset,
    save_llm_config,
    save_uploaded_files,
    set_active_dataset,
)
from api.services.ask_service import handle_ask
from optiagent.data import SupplyChainData, normalize_data, validate_data
from optiagent.llm import DataProfile, LLMConfig
from optiagent.schema_mapping import assemble_facility_data, apply_table_mapping, infer_facility_table, mapping_summary


app = FastAPI(title="OptiAgent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path("web")
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class LLMConfigIn(BaseModel):
    name: str = "default"
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.2


class AskRequest(BaseModel):
    question: str
    dataset_id: int | None = None
    conversation_id: int | None = None
    mcp_config: str = ""


class LoginRequest(BaseModel):
    username: str


class ConversationRequest(BaseModel):
    title: str = "新对话"


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "OptiAgent API is running."}


@app.head("/")
def index_head():
    return {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/login")
def login(request: LoginRequest):
    user = login_user(request.username)
    return {"user": {"id": user["id"], "username": user["username"]}, "session_token": user["session_token"]}


@app.get("/api/me")
def me(x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    if not user:
        return {"logged_in": False}
    return {"logged_in": True, "user": {"id": user["id"], "username": user["username"]}}


@app.get("/api/conversations")
def conversations(limit: int = 30, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    return {"conversations": list_conversations(user_id=user["id"] if user else None, limit=limit)}


@app.post("/api/conversations")
def new_conversation(request: ConversationRequest, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    conversation = create_conversation(request.title, user_id=user["id"] if user else None)
    return {"conversation": conversation}


@app.get("/api/datasets")
def datasets(conversation_id: int | None = None, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = ensure_conversation(conversation_id, user_id=uid)
    cid = conversation["id"]
    files = list_uploaded_files(limit=6, user_id=uid, conversation_id=cid)
    datasets = list_datasets(user_id=uid, conversation_id=cid)
    active_dataset_id = get_active_dataset_id_or_none(user_id=uid, conversation_id=cid)
    if active_dataset_id and not any(item["id"] == active_dataset_id for item in datasets):
        clear_active_dataset(user_id=uid, conversation_id=cid)
        active_dataset_id = None
    return {
        "conversation_id": cid,
        "datasets": datasets,
        "active_dataset_id": active_dataset_id,
        "uploaded_files": [
            {
                "id": item["id"],
                "filename": item["filename"],
                "role": item["role"],
                "created_at": item["created_at"],
            }
            for item in files
        ],
    }


@app.post("/api/datasets/active/{dataset_id}")
def activate_dataset(
    dataset_id: int,
    conversation_id: int | None = None,
    x_session_token: str | None = Header(default=None),
):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = ensure_conversation(conversation_id, user_id=uid)
    set_active_dataset(dataset_id, user_id=uid, conversation_id=conversation["id"])
    return {"ok": True, "active_dataset_id": dataset_id}


@app.post("/api/upload")
async def upload_dataset(
    name: str = Form("上传数据集"),
    conversation_id: int | None = Form(None),
    files: list[UploadFile] = File(...),
    x_session_token: str | None = Header(default=None),
):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = ensure_conversation(conversation_id, user_id=uid)
    cid = conversation["id"]
    parsed_files = []
    file_results = []
    for file in files:
        if not file.filename:
            continue
        try:
            frame = _read_csv(await file.read())
        except Exception as exc:
            file_results.append(
                {
                    "filename": file.filename,
                    "status": "error",
                    "message": f"上传失败：CSV 无法读取（{type(exc).__name__}）。",
                    "role": None,
                    "rows": 0,
                    "columns": [],
                }
            )
            continue
        mapping = infer_facility_table(frame, file.filename)
        role = mapping.role or _infer_upload_role(frame, file.filename)
        mapped_frame = apply_table_mapping(frame, mapping) if mapping.role else frame
        parsed_files.append(
            {
                "filename": file.filename,
                "frame": mapped_frame,
                "raw_frame": frame,
                "role": role,
                "mapping": mapping,
            }
        )
        file_results.append(
            {
                "filename": file.filename,
                "status": "ok",
                "message": "上传成功。",
                "role": role,
                "role_label": _role_label(role),
                "mapping": mapping_summary(mapping),
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
            }
        )

    if not parsed_files:
        return {
            "ok": False,
            "conversation_id": cid,
            "dataset_id": None,
            "files": file_results,
            "check": {
                "status": "error",
                "messages": ["没有可用 CSV 文件被成功读取。"],
            },
        }

    saved_count = save_uploaded_files(name, parsed_files, user_id=uid, conversation_id=cid)
    selected: dict[str, dict] = {}
    conflicts: list[str] = []
    for item in parsed_files:
        role = item["role"]
        if role not in {"warehouses", "customers", "costs"}:
            continue
        if role in selected:
            conflicts.append(f"{role} 识别到多个候选文件：{selected[role]['filename']}、{item['filename']}")
            continue
        selected[role] = item

    missing = [key for key in ["warehouses", "customers", "costs"] if key not in selected]
    dataset_id = None
    messages = [f"已成功保存 {saved_count} 个 CSV 文件。"]
    failed_messages = [item["message"] for item in file_results if item["status"] != "ok"]
    messages.extend(failed_messages)
    assembly = assemble_facility_data(
        [
            (item.get("raw_frame", item["frame"]), item["filename"], item.get("role"))
            for item in parsed_files
        ]
    )
    if assembly.data is not None and not conflicts:
        data = assembly.data
        check = _check_dataset(data)
        if check["status"] == "error":
            clear_active_dataset(user_id=uid, conversation_id=cid)
            messages.extend(check["messages"])
        else:
            dataset_id = save_dataset(name, data, make_active=True, user_id=uid, conversation_id=cid)
            messages.append("已识别为完整仓库选址数据集，可直接提问求解。")
            messages.extend(f"{filename}: {mapping_summary(mapping)}" for filename, mapping in assembly.mappings.items())
            messages.extend(assembly.warnings)
    else:
        clear_active_dataset(user_id=uid, conversation_id=cid)
        if conflicts:
            messages.extend(conflicts)
        if assembly.warnings:
            messages.extend(assembly.warnings)
        if assembly.confirmation_message:
            messages.append(assembly.confirmation_message)
        messages.append("文件已保存，可在提问时由 Agent 根据当前对话上传的数据进行判断和回答。")

    return {
        "ok": True,
        "conversation_id": cid,
        "dataset_id": dataset_id,
        "files": file_results,
        "check": {"status": "ok", "messages": messages},
    }


@app.get("/api/data/summary")
def data_summary(
    dataset_id: int | None = None,
    conversation_id: int | None = None,
    x_session_token: str | None = Header(default=None),
):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = ensure_conversation(conversation_id, user_id=uid)
    active_id = dataset_id or get_active_dataset_id_or_none(user_id=uid, conversation_id=conversation["id"])
    if active_id is None:
        return {"has_data": False, "profile": "", "profile_source": "", "warnings": [], "warehouses": [], "customers": []}
    data = load_dataset(active_id)
    profile = _local_supply_chain_profile(data)
    return {
        "has_data": True,
        "profile": profile.summary,
        "profile_source": profile.source,
        "warnings": profile.warnings,
        "warehouses": data.warehouses.to_dict(orient="records"),
        "customers": data.customers.to_dict(orient="records"),
    }


@app.post("/api/llm-config")
def configure_llm(config: LLMConfigIn, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    config_id = save_llm_config(config.model_dump(), user_id=user["id"] if user else None)
    return {"ok": True, "config_id": config_id}


@app.get("/api/llm-config")
def get_llm_config(x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    config = get_active_llm_config(user["id"] if user else None)
    if not config:
        return {"configured": False}
    return {
        "configured": True,
        "name": config["name"],
        "base_url": config["base_url"],
        "model": config["model"],
        "temperature": config["temperature"],
    }


@app.post("/api/ask")
def ask(request: AskRequest, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = ensure_conversation(
        request.conversation_id,
        user_id=uid,
        title=_conversation_title_from_question(request.question),
    )
    return handle_ask(
        question=request.question,
        requested_dataset_id=request.dataset_id,
        mcp_config=request.mcp_config,
        user_id=uid,
        conversation_id=conversation["id"],
    )


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = ensure_conversation(
        request.conversation_id,
        user_id=uid,
        title=_conversation_title_from_question(request.question),
    )

    def event_stream():
        try:
            yield _sse("status", {"message": "正在识别问题类型与可用数据..."})
            yield _sse("status", {"message": "正在选择 RAG、工具与求解器..."})
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    handle_ask,
                    question=request.question,
                    requested_dataset_id=request.dataset_id,
                    mcp_config=request.mcp_config,
                    user_id=uid,
                    conversation_id=conversation["id"],
                )
                waiting_messages = [
                    "正在解析上传数据与约束...",
                    "正在执行工具调用或优化求解...",
                    "求解仍在进行，正在等待结果...",
                ]
                wait_index = 0
                while True:
                    try:
                        result = future.result(timeout=1.2)
                        break
                    except TimeoutError:
                        yield _sse("status", {"message": waiting_messages[wait_index % len(waiting_messages)]})
                        wait_index += 1
            result["conversation_id"] = result.get("conversation_id") or conversation["id"]
            yield _sse("status", {"message": "正在生成结构化回答..."})
            for chunk in _stream_answer_text(result):
                yield _sse("answer_delta", {"text": chunk})
                time.sleep(0.015)
            yield _sse("final", result)
        except HTTPException as exc:
            yield _sse("error", {"message": str(exc.detail)})
        except Exception as exc:
            yield _sse("error", {"message": f"流式回答失败：{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, payload: dict) -> str:
    data = json.dumps(_json_safe(jsonable_encoder(payload)), ensure_ascii=False, allow_nan=False)
    return f"event: {event}\ndata: {data}\n\n"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _stream_answer_text(result: dict) -> list[str]:
    structured = result.get("structured_answer") or {}
    lines: list[str] = []
    conclusion = structured.get("conclusion") or result.get("answer")
    if conclusion:
        lines.append(str(conclusion))

    metrics = structured.get("metrics") or {}
    metric_lines = []
    objective = result.get("objective_value")
    objective_label = metrics.get("objective_label") or (result.get("generic_result") or {}).get("objective_label") or "目标值"
    if objective is not None:
        metric_lines.append(f"{objective_label}：{_format_stream_value(objective)}")
    if result.get("fixed_cost") is not None:
        metric_lines.append(f"固定成本：{_format_stream_value(result.get('fixed_cost'))}")
    if result.get("transport_cost") is not None:
        metric_lines.append(f"运输成本：{_format_stream_value(result.get('transport_cost'))}")
    for item in (metrics.get("extra") or [])[:4]:
        label = item.get("label")
        value = item.get("value")
        if label and value is not None:
            metric_lines.append(f"{label}：{_format_stream_metric(item)}")
    if metric_lines:
        lines.append("\n".join(metric_lines))

    recommendations = [str(item) for item in structured.get("recommendations") or [] if item]
    if recommendations:
        lines.append("建议：\n" + "\n".join(f"- {item}" for item in recommendations[:5]))
    risks = [str(item) for item in structured.get("risks") or [] if item]
    if risks:
        lines.append("风险：\n" + "\n".join(f"- {item}" for item in risks[:3]))

    text = "\n\n".join(lines).strip() or str(result.get("answer") or "已完成。")
    return _chunk_text(text)


def _chunk_text(text: str, size: int = 24) -> list[str]:
    chunks = []
    current = ""
    for char in text:
        current += char
        if len(current) >= size or char in "\n。；.!?":
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _format_stream_metric(item: dict) -> str:
    value = item.get("value")
    if item.get("format") == "percent":
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return str(value)
    suffix = item.get("suffix") or ""
    return f"{_format_stream_value(value)}{suffix}"


def _format_stream_value(value) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


@app.get("/api/runs")
def runs(
    limit: int = 20,
    conversation_id: int | None = None,
    x_session_token: str | None = Header(default=None),
):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = get_conversation(conversation_id, user_id=uid)
    if not conversation:
        return {"runs": []}
    return {"runs": list_runs(limit, user_id=uid, conversation_id=conversation["id"])}


@app.delete("/api/runs")
def clear_history(conversation_id: int | None = None, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    conversation = get_conversation(conversation_id, user_id=uid)
    deleted = clear_runs(user_id=uid, conversation_id=conversation["id"] if conversation else None)
    return {"ok": True, "deleted": deleted}


@app.delete("/api/conversations/{conversation_id}")
def remove_conversation(conversation_id: int, x_session_token: str | None = Header(default=None)):
    user = get_user_by_token(x_session_token)
    uid = user["id"] if user else None
    deleted = delete_conversation(conversation_id, user_id=uid)
    if not deleted:
        raise HTTPException(status_code=404, detail="对话不存在或无权删除。")
    return {"ok": True, "deleted_conversation_id": conversation_id}


def _conversation_title_from_question(question: str) -> str:
    cleaned = " ".join(question.strip().split())
    return cleaned[:32] or "新对话"


def _read_csv(content: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(StringIO(content.decode(encoding)))
        except Exception as exc:
            errors.append(f"{encoding}: {type(exc).__name__}")
    raise ValueError("; ".join(errors))


def _infer_upload_role(frame: pd.DataFrame, filename: str) -> str | None:
    columns = {str(column).strip().lower() for column in frame.columns}
    aliases = {
        "warehouse": {"warehouse", "warehouses", "仓库", "仓库id", "warehouse_id", "warehouse_name", "facility", "site", "候选仓库", "设施点", "from", "source", "起点"},
        "customer": {"customer", "customers", "客户", "客户id", "customer_id", "customer_name", "demand_point", "需求点", "to", "destination", "终点"},
        "capacity": {"capacity", "cap", "supply", "容量", "产能", "供应量", "可用容量"},
        "fixed_cost": {"fixed_cost", "fixed cost", "fixedcost", "open_cost", "opening_cost", "setup_cost", "固定成本", "开仓成本", "启用成本", "建设成本"},
        "demand": {"demand", "qty", "quantity", "需求", "需求量", "订单量"},
        "cost": {"cost", "运输成本", "单位成本", "unit_cost", "transport_cost", "shipping_cost", "费用"},
    }

    def has_any(names: set[str]) -> bool:
        return bool(columns & names)

    has_warehouse = has_any(aliases["warehouse"])
    has_customer = has_any(aliases["customer"])
    has_capacity = has_any(aliases["capacity"])
    has_fixed = has_any(aliases["fixed_cost"])
    has_demand = has_any(aliases["demand"])
    has_cost = has_any(aliases["cost"])
    lowered_name = filename.lower()

    generic_role = _infer_generic_upload_role(columns, lowered_name)
    if generic_role:
        return generic_role
    if has_warehouse and has_customer and has_cost:
        return "costs"
    if has_warehouse and (has_capacity or has_fixed):
        return "warehouses"
    if has_customer and has_demand:
        return "customers"
    if ("warehouses" in lowered_name or "warehouse" in lowered_name or "仓库" in lowered_name) and (has_warehouse or has_capacity or has_fixed):
        return "warehouses"
    if ("customers" in lowered_name or "customer" in lowered_name or "客户" in lowered_name) and (has_customer or has_demand):
        return "customers"
    if ("costs" in lowered_name or "cost" in lowered_name or "成本" in lowered_name) and (has_cost or (has_warehouse and has_customer)):
        return "costs"
    return None


def _infer_generic_upload_role(columns: set[str], lowered_name: str) -> str | None:
    aliases = {
        "item": {"item", "物品", "物品编号", "编号", "id", "name", "项目"},
        "value": {"value", "价值", "收益", "效益", "profit"},
        "weight": {"weight", "重量", "资源消耗", "消耗", "cost_weight"},
        "resource": {"resource", "资源", "员工", "人员", "worker"},
        "task": {"task", "任务", "班次", "岗位", "job"},
        "cost": {"cost", "成本", "匹配成本", "费用"},
        "from": {"from", "起点", "出发", "源点", "source"},
        "to": {"to", "终点", "到达", "目的地", "destination"},
        "distance": {"distance", "距离", "里程", "时间", "时长", "travel_time"},
        "city": {"city", "城市", "node", "节点"},
        "x": {"x", "lng", "lon", "longitude", "经度"},
        "y": {"y", "lat", "latitude", "纬度"},
        "job": {"job", "作业", "订单", "工件"},
        "machine": {"machine", "机器", "设备", "产线"},
        "duration": {"duration", "加工时长", "处理时间", "工时", "时长"},
        "product": {"product", "产品", "品类"},
        "profit": {"profit", "利润", "收益", "单位利润"},
    }

    def has(key: str) -> bool:
        return bool(columns & {item.lower() for item in aliases[key]})

    if has("item") and has("value") and has("weight"):
        return "knapsack"
    if has("resource") and has("task") and has("cost"):
        return "assignment"
    if (has("from") and has("to") and has("distance")) or (has("city") and has("x") and has("y")):
        return "tsp"
    if has("job") and has("machine") and has("duration"):
        return "job_shop_scheduling"
    if has("product") and has("profit"):
        return "production_mix"
    if "tsp" in lowered_name and has("city") and has("x") and has("y"):
        return "tsp"
    return None


def _role_label(role: str | None) -> str:
    return {
        "warehouses": "仓库候选表",
        "customers": "客户需求表",
        "costs": "运输成本表",
        "knapsack": "0-1 背包数据",
        "assignment": "指派匹配数据",
        "tsp": "TSP 路径数据",
        "job_shop_scheduling": "作业车间调度数据",
        "production_mix": "产品组合数据",
    }.get(role or "", "通用数据")


def _active_llm_config(user_id: int | None) -> LLMConfig | None:
    config = get_active_llm_config(user_id)
    if not config:
        return None
    return LLMConfig(
        enabled=True,
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        temperature=float(config["temperature"]),
    )


def _check_dataset(data: SupplyChainData) -> dict:
    try:
        normalized = normalize_data(data)
    except Exception as exc:
        return {"status": "error", "messages": [f"数据字段或数值格式无法解析：{exc}"]}

    errors = validate_data(normalized)
    messages = []
    if errors:
        return {"status": "error", "messages": errors}

    messages.append("Agent 已识别出仓库、客户需求与运输成本三类数据，并完成结构化入库。")
    messages.append("规则检查通过：字段、非负数、成本矩阵覆盖和容量需求关系均可用。")
    messages.append("上传阶段未调用 LLM；用户提问时再由 Agent 基于当前对话文件回答。")
    return {"status": "ok", "messages": messages}


def _local_supply_chain_profile(data: SupplyChainData) -> DataProfile:
    total_capacity = data.warehouses["capacity"].sum()
    total_demand = data.customers["demand"].sum()
    ratio = total_demand / total_capacity if total_capacity else 0
    summary = (
        f"当前结构化数据包含 {len(data.warehouses)} 个候选仓库、{len(data.customers)} 个客户点，"
        f"总容量 {total_capacity:,.0f}，总需求 {total_demand:,.0f}，需求/容量比 {ratio:.1%}。"
    )
    return DataProfile(source="本地规则解析", summary=summary, warnings=[], llm_used=False)
