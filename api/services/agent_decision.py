"""主控共享行动合同与各工具参数模型；新增工具复用公开参数字段。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from optiagent.requirement_patch import RequirementPatch


class NoParameters(BaseModel):
    """无参数工具仍拒绝未知参数。"""

    model_config = ConfigDict(extra="forbid")


class QueryParameters(NoParameters):
    """数据筛选与检索统一使用有限长度 query。"""

    query: str = Field(default="", max_length=500)


class PatchParameters(NoParameters):
    """需求修改必须携带可复核的原文证据。"""

    patch: RequirementPatch


class ResultReferenceParameters(QueryParameters):
    """方案编号仅作为当前用户明确指令的引用。"""

    run_id: int | None = Field(default=None, strict=True, ge=1)


class ExplanationParameters(NoParameters):
    """解释只能选取来源工具返回的事实编号。"""

    fact_ids: list[str] | None = Field(default=None, max_length=12)


class ClarificationParameters(NoParameters):
    """追问文本不能超出共享行动合同的长度。"""

    message: str = Field(default="", max_length=2000)


class ControllerDecision(BaseModel):
    """接口字段全部必填，无对应参数使用 null；本地构造保留兼容默认值。"""

    model_config = ConfigDict(extra="forbid")
    action: str
    reason: str = Field(max_length=500)
    query: str = Field(max_length=500)
    message: str = Field(max_length=2000)
    patch: RequirementPatch | None = None
    run_id: int | None = Field(default=None, strict=True, ge=1)
    fact_ids: list[str] | None = Field(default=None, max_length=12)

    @field_validator("action")
    @classmethod
    def registered_action(cls, value: str) -> str:
        """只接受已完整注册的行动，新工具无需修改固定 Literal。"""
        from api.services.agent_tool_registry import get_agent_tool
        get_agent_tool(value)
        return value

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict:
        """将可空参数转成严格接口的必填可空字段，递归关闭嵌套对象。"""
        schema = super().model_json_schema(*args, **kwargs)

        def close(value) -> None:
            if isinstance(value, dict):
                if value.get("type") == "object":
                    value["required"] = list(value.get("properties", {}))
                    value["additionalProperties"] = False
                value.pop("default", None)
                for child in value.values():
                    close(child)
            elif isinstance(value, list):
                for child in value:
                    close(child)

        close(schema)
        from api.services.agent_tool_registry import list_agent_tools
        schema["properties"]["action"]["enum"] = [tool.name for tool in list_agent_tools()]
        return schema
