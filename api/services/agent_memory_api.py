"""显式记忆与任务状态接口，供用户查看、录入和删除本地记忆。"""

from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from api.database import get_user_by_token
from api.services.agent_memory import add_memory, delete_memory, load_task, search_memories


router = APIRouter(prefix="/api/agent", tags=["agent_memory"])


class MemoryRequest(BaseModel):
    """记忆由用户显式提交，未提供会话时仅登录用户可保存个人偏好。"""

    content: str = Field(min_length=1, max_length=2000)
    kind: Literal["preference", "business_rule", "terminology"] = "preference"
    conversation_id: int | None = None


def _user_id(token: str | None) -> int | None:
    """沿用现有会话令牌，但无效令牌不能退化成匿名记忆访问。"""

    user = get_user_by_token(token)
    if token and user is None:
        raise HTTPException(status_code=401, detail="会话令牌无效")
    return user["id"] if user else None


@router.get("/task")
def task_state(conversation_id: int, x_session_token: str | None = Header(default=None)):
    """返回当前任务快照，不暴露模型配置或密钥。"""
    try:
        return {"task": load_task(_user_id(x_session_token), conversation_id)}
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/memories")
def memories(conversation_id: int | None = None, query: str = "",
             x_session_token: str | None = Header(default=None)):
    """按当前作用域查看或搜索显式记忆。"""
    try:
        return {"memories": search_memories(user_id=_user_id(x_session_token),
                    conversation_id=conversation_id, query=query, limit=20)}
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/memories")
def remember(request: MemoryRequest, x_session_token: str | None = Header(default=None)):
    """明确记下用户提交的信息，不从模型推测生成长期记忆。"""
    try:
        return {"memory": add_memory(user_id=_user_id(x_session_token),
                    conversation_id=request.conversation_id, content=request.content, kind=request.kind)}
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/memories/{memory_id}")
def forget(memory_id: int, conversation_id: int | None = None,
           x_session_token: str | None = Header(default=None)):
    """只删除指定作用域内的记忆，删除后后续上下文不再检索该项。"""
    try:
        deleted = delete_memory(memory_id, _user_id(x_session_token), conversation_id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="记忆不存在或无权访问")
    return {"deleted": True}
