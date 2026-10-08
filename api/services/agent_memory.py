"""本地任务快照与显式长期记忆，查询和写入均验证用户及会话作用域。"""

from __future__ import annotations

import json
import re
from typing import Any

from api.database import connect, get_conversation, init_db


def require_scope(user_id: int | None, conversation_id: int | None) -> None:
    """匿名长期记忆必须绑定会话，禁止所有匿名用户共享偏好。"""

    if conversation_id is not None:
        if get_conversation(conversation_id, user_id=user_id) is None:
            raise PermissionError("对话不存在或无权访问")
    elif user_id is None:
        raise PermissionError("匿名记忆必须绑定一个有效对话")


def load_task(user_id: int | None, conversation_id: int | None) -> dict[str, Any]:
    """读取任务快照；无持久化会话的独立运行使用空状态。"""

    if conversation_id is None:
        return {}
    require_scope(user_id, conversation_id)
    with connect() as conn:
        row = conn.execute("SELECT state_json FROM agent_task_states WHERE conversation_id = ?", (conversation_id,)).fetchone()
    return json.loads(row["state_json"]) if row else {}


def save_task(user_id: int | None, conversation_id: int | None, state: dict[str, Any]) -> None:
    """保存任务状态；密钥和模型配置不能进入快照。"""

    if conversation_id is None:
        return
    require_scope(user_id, conversation_id)
    encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
    with connect() as conn:
        conn.execute("""INSERT INTO agent_task_states (conversation_id, user_id, state_json)
                        VALUES (?, ?, ?) ON CONFLICT(conversation_id) DO UPDATE SET
                        state_json = excluded.state_json, updated_at = CURRENT_TIMESTAMP""",
                     (conversation_id, user_id, encoded))


def add_memory(*, user_id: int | None, conversation_id: int | None,
               content: str, kind: str, source: str = "user_input") -> dict[str, Any]:
    """仅接收用户明确提交的记忆；假设不会由模型自动写入。"""

    init_db()
    require_scope(user_id, conversation_id)
    if kind not in {"preference", "business_rule", "terminology"}:
        raise ValueError("记忆类型不支持")
    if not content.strip() or len(content) > 2000:
        raise ValueError("记忆内容必须为 1 到 2000 字符")
    with connect() as conn:
        cursor = conn.execute("""INSERT INTO agent_memories
            (user_id, conversation_id, kind, content, source) VALUES (?, ?, ?, ?, ?)""",
            (user_id, conversation_id, kind, content.strip(), source))
        row = conn.execute("SELECT * FROM agent_memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def search_memories(*, user_id: int | None, conversation_id: int | None,
                    query: str = "", limit: int = 6) -> list[dict[str, Any]]:
    """先隔离作用域，再对显式记忆进行轻量关键词相关性排序。"""

    init_db()
    require_scope(user_id, conversation_id)
    user_clause = "user_id IS NULL" if user_id is None else "user_id = ?"
    params = [] if user_id is None else [user_id]
    if conversation_id is None:
        scope_clause = "conversation_id IS NULL"
    elif user_id is None:
        scope_clause = "conversation_id = ?"
        params.append(conversation_id)
    else:
        scope_clause = "(conversation_id IS NULL OR conversation_id = ?)"
        params.append(conversation_id)
    with connect() as conn:
        rows = conn.execute(f"SELECT * FROM agent_memories WHERE {user_clause} AND {scope_clause} ORDER BY id DESC LIMIT 200", params).fetchall()
    # 中文使用二字片段，英文按词；首版不宣称语义向量检索能力。
    tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    tokens.update(query[index:index + 2] for index in range(len(query) - 1)
                  if all('\u4e00' <= char <= '\u9fff' for char in query[index:index + 2]))
    ranked = [(sum(token in row["content"].lower() for token in tokens), dict(row)) for row in rows]
    if query.strip():
        ranked = [item for item in ranked if item[0] > 0]
    ranked.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
    return [row for _, row in ranked[:max(1, min(limit, 20))]]


def delete_memory(memory_id: int, user_id: int | None, conversation_id: int | None) -> bool:
    """删除时要求精确作用域，不能借当前会话删除别的会话记忆。"""

    init_db()
    require_scope(user_id, conversation_id)
    clause = "user_id IS NULL" if user_id is None else "user_id = ?"
    params = [memory_id] if user_id is None else [memory_id, user_id]
    if conversation_id is None:
        clause += " AND conversation_id IS NULL"
    else:
        clause += " AND conversation_id = ?"
        params.append(conversation_id)
    with connect() as conn:
        cursor = conn.execute(f"DELETE FROM agent_memories WHERE id = ? AND {clause}", params)
    return bool(cursor.rowcount)
