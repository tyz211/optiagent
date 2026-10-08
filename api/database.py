from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import secrets
from typing import Iterator
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd

from optiagent.data import SupplyChainData, normalize_data


# 数据库位置固定在项目目录，重启或更换启动目录不会切换用户与模型配置。
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "optiagent.sqlite3"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id INTEGER,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS warehouses (
                dataset_id INTEGER NOT NULL,
                warehouse TEXT NOT NULL,
                region TEXT NOT NULL,
                capacity REAL NOT NULL,
                fixed_cost REAL NOT NULL,
                min_open_ratio REAL NOT NULL,
                force_open INTEGER NOT NULL,
                force_closed INTEGER NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );

            CREATE TABLE IF NOT EXISTS customers (
                dataset_id INTEGER NOT NULL,
                customer TEXT NOT NULL,
                demand REAL NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );

            CREATE TABLE IF NOT EXISTS costs (
                dataset_id INTEGER NOT NULL,
                warehouse TEXT NOT NULL,
                customer TEXT NOT NULL,
                cost REAL NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );

            CREATE TABLE IF NOT EXISTS llm_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                model TEXT NOT NULL,
                api_key TEXT NOT NULL,
                temperature REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                session_token TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id INTEGER,
                dataset_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                objective_value REAL,
                transport_cost REAL,
                fixed_cost REAL,
                status TEXT NOT NULL,
                open_warehouses TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id),
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );

            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id INTEGER,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                role TEXT,
                columns_json TEXT NOT NULL,
                content_csv TEXT NOT NULL DEFAULT '',
                preview_csv TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS agent_episodes (
                episode_id TEXT PRIMARY KEY,
                user_id INTEGER,
                conversation_id INTEGER,
                run_id INTEGER,
                question TEXT NOT NULL,
                policy_name TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                status TEXT NOT NULL,
                template_id TEXT,
                total_reward REAL,
                reward_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id),
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS agent_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                node_id TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                state_json TEXT NOT NULL DEFAULT '{}',
                action_json TEXT NOT NULL DEFAULT '{}',
                observation_json TEXT NOT NULL DEFAULT '{}',
                reward_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                elapsed_ms REAL,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                FOREIGN KEY(episode_id) REFERENCES agent_episodes(episode_id),
                UNIQUE(episode_id, node_id, attempt)
            );

            CREATE TABLE IF NOT EXISTS conversation_requirements (
                conversation_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                state_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );

            -- 主控任务状态按会话持久化，长期记忆仅由用户明确录入。
            CREATE TABLE IF NOT EXISTS agent_task_states (
                conversation_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                state_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS agent_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id INTEGER,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_agent_memories_scope
            ON agent_memories(user_id, conversation_id);

            CREATE INDEX IF NOT EXISTS idx_agent_episodes_scope
            ON agent_episodes(user_id, conversation_id, started_at);

            CREATE INDEX IF NOT EXISTS idx_agent_steps_episode
            ON agent_steps(episode_id, sequence, attempt);
            """
        )
        _ensure_column(conn, "llm_configs", "user_id", "INTEGER")
        _ensure_column(conn, "datasets", "user_id", "INTEGER")
        _ensure_column(conn, "datasets", "conversation_id", "INTEGER")
        _ensure_column(conn, "runs", "user_id", "INTEGER")
        _ensure_column(conn, "runs", "conversation_id", "INTEGER")
        _ensure_column(conn, "uploaded_files", "conversation_id", "INTEGER")
        _ensure_column(conn, "uploaded_files", "content_csv", "TEXT NOT NULL DEFAULT ''")
        _relax_runs_dataset_id(conn)
        _remove_legacy_demo_datasets(conn)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _relax_runs_dataset_id(conn: sqlite3.Connection) -> None:
    columns = conn.execute("PRAGMA table_info(runs)").fetchall()
    dataset_col = next((row for row in columns if row["name"] == "dataset_id"), None)
    if not dataset_col or int(dataset_col["notnull"]) == 0:
        return

    conn.executescript(
        """
        ALTER TABLE runs RENAME TO runs_old;
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            conversation_id INTEGER,
            dataset_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            objective_value REAL,
            transport_cost REAL,
            fixed_cost REAL,
            status TEXT NOT NULL,
            open_warehouses TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id),
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );
        INSERT INTO runs (
            id, user_id, conversation_id, dataset_id, question, answer, objective_value, transport_cost,
            fixed_cost, status, open_warehouses, result_json, created_at
        )
        SELECT
            id, user_id, conversation_id, dataset_id, question, answer, objective_value, transport_cost,
            fixed_cost, status, open_warehouses, result_json, created_at
        FROM runs_old;
        DROP TABLE runs_old;
        """
    )


def _remove_legacy_demo_datasets(conn: sqlite3.Connection) -> None:
    legacy_names = ("示例数据", "默认数据", "结构化上传测试", "sample data", "demo data")
    rows = conn.execute(
        f"SELECT id FROM datasets WHERE lower(name) IN ({','.join(['lower(?)'] * len(legacy_names))})",
        legacy_names,
    ).fetchall()
    dataset_ids = [int(row["id"]) for row in rows]
    if not dataset_ids:
        return

    placeholders = ",".join("?" for _ in dataset_ids)
    for table in ("warehouses", "customers", "costs"):
        conn.execute(f"DELETE FROM {table} WHERE dataset_id IN ({placeholders})", dataset_ids)
    conn.execute(f"UPDATE runs SET dataset_id = NULL WHERE dataset_id IN ({placeholders})", dataset_ids)
    conn.execute(f"DELETE FROM datasets WHERE id IN ({placeholders})", dataset_ids)


def _scope_clause(user_id: int | None, conversation_id: int | None, prefix: str = "") -> tuple[str, list]:
    name = f"{prefix}." if prefix else ""
    clauses = [f"{name}user_id IS NULL" if user_id is None else f"{name}user_id = ?"]
    params: list = [] if user_id is None else [user_id]
    if conversation_id is None:
        clauses.append(f"{name}conversation_id IS NULL")
    else:
        clauses.append(f"{name}conversation_id = ?")
        params.append(conversation_id)
    return " AND ".join(clauses), params


def create_conversation(title: str, user_id: int | None = None) -> dict:
    init_db()
    clean = title.strip() or "新对话"
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
            (user_id, clean[:80]),
        )
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return dict(row)


def list_conversations(user_id: int | None = None, limit: int = 30) -> list[dict]:
    init_db()
    with connect() as conn:
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE user_id IS NULL ORDER BY updated_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC, id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]


def get_conversation(conversation_id: int | None, user_id: int | None = None) -> dict | None:
    init_db()
    if conversation_id is None:
        return None
    with connect() as conn:
        if user_id is None:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id IS NULL",
                (conversation_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        return dict(row) if row else None


def ensure_conversation(conversation_id: int | None, user_id: int | None = None, title: str = "新对话") -> dict:
    existing = get_conversation(conversation_id, user_id)
    if existing:
        return existing
    return create_conversation(title, user_id=user_id)


def touch_conversation(conversation_id: int | None, title: str | None = None) -> None:
    if conversation_id is None:
        return
    clean_title = (title or "").strip()
    with connect() as conn:
        if clean_title:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            current_title = str(row["title"]) if row else ""
            if current_title == "新对话":
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (clean_title[:80], conversation_id),
                )
                return
        conn.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))


def get_conversation_requirement(
    conversation_id: int | None,
    user_id: int | None = None,
) -> dict | None:
    """读取当前会话累计的需求摘要，并保持用户作用域隔离。"""

    if conversation_id is None or get_conversation(conversation_id, user_id=user_id) is None:
        return None
    with connect() as conn:
        if user_id is None:
            row = conn.execute(
                "SELECT state_json FROM conversation_requirements WHERE conversation_id = ? AND user_id IS NULL",
                (conversation_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT state_json FROM conversation_requirements WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
    payload = _json_load(row["state_json"]) if row else None
    return payload if isinstance(payload, dict) else None


def save_conversation_requirement(
    conversation_id: int | None,
    state: dict,
    user_id: int | None = None,
) -> None:
    """原子更新多轮需求状态；无会话的离线图测试不写入数据库。"""

    if conversation_id is None or get_conversation(conversation_id, user_id=user_id) is None:
        return
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_requirements (conversation_id, user_id, state_json)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                user_id = excluded.user_id,
                state_json = excluded.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (conversation_id, user_id, _json_dump(state)),
        )


def compare_and_save_requirement(conversation_id: int, *, user_id: int | None,
                                 expected: dict, replacement: dict) -> None:
    """在同一事务中比较完整旧状态并提交，防止相同版本号下的更新被覆盖。"""

    if get_conversation(conversation_id, user_id=user_id) is None:
        raise PermissionError("会话不属于当前用户")
    with connect() as conn:
        # 写锁覆盖读取和更新；冲突时不保存任何候选数据或版本。
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT state_json FROM conversation_requirements WHERE conversation_id = ? AND user_id IS ?",
            (conversation_id, user_id),
        ).fetchone()
        if row is None or _json_load(row["state_json"]) != expected:
            raise ValueError("需求状态已变化，请重新读取有效版本")
        conn.execute(
            "UPDATE conversation_requirements SET state_json = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE conversation_id = ? AND user_id IS ?",
            (_json_dump(replacement), conversation_id, user_id),
        )


def get_scoped_run(run_id: int, *, user_id: int | None, conversation_id: int) -> dict | None:
    """按运行、用户和会话三重作用域读取，不能跨会话借用历史方案。"""

    if get_conversation(conversation_id, user_id=user_id) is None:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE id = ? AND user_id IS ? AND conversation_id = ?",
            (run_id, user_id, conversation_id),
        ).fetchone()
    return dict(row) if row else None


def save_dataset(
    name: str,
    data: SupplyChainData,
    make_active: bool = True,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> int:
    normalized = normalize_data(data)
    with connect() as conn:
        if make_active:
            clause, params = _scope_clause(user_id, conversation_id)
            conn.execute(f"UPDATE datasets SET is_active = 0 WHERE {clause}", params)
        cursor = conn.execute(
            "INSERT INTO datasets (user_id, conversation_id, name, is_active) VALUES (?, ?, ?, ?)",
            (user_id, conversation_id, name, 1 if make_active else 0),
        )
        dataset_id = int(cursor.lastrowid)
        _insert_frame(conn, "warehouses", dataset_id, normalized.warehouses)
        _insert_frame(conn, "customers", dataset_id, normalized.customers)
        _insert_frame(conn, "costs", dataset_id, normalized.costs)
        return dataset_id


def _insert_frame(conn: sqlite3.Connection, table: str, dataset_id: int, frame: pd.DataFrame) -> None:
    rows = frame.copy()
    rows.insert(0, "dataset_id", dataset_id)
    rows.to_sql(table, conn, if_exists="append", index=False)


def get_active_dataset_id(user_id: int | None = None, conversation_id: int | None = None) -> int:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        row = conn.execute(f"SELECT id FROM datasets WHERE is_active = 1 AND {clause} ORDER BY id DESC LIMIT 1", params).fetchone()
        if row:
            return int(row["id"])
        raise ValueError("尚未选择数据集。")


def get_active_dataset_id_or_none(user_id: int | None = None, conversation_id: int | None = None) -> int | None:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        row = conn.execute(f"SELECT id FROM datasets WHERE is_active = 1 AND {clause} ORDER BY id DESC LIMIT 1", params).fetchone()
        return int(row["id"]) if row else None


def load_dataset(dataset_id: int | None = None) -> SupplyChainData:
    init_db()
    dataset_id = dataset_id or get_active_dataset_id()
    with connect() as conn:
        warehouses = pd.read_sql_query("SELECT warehouse, region, capacity, fixed_cost, min_open_ratio, force_open, force_closed FROM warehouses WHERE dataset_id = ?", conn, params=(dataset_id,))
        customers = pd.read_sql_query("SELECT customer, demand FROM customers WHERE dataset_id = ?", conn, params=(dataset_id,))
        costs = pd.read_sql_query("SELECT warehouse, customer, cost FROM costs WHERE dataset_id = ?", conn, params=(dataset_id,))
    return normalize_data(SupplyChainData(warehouses=warehouses, customers=customers, costs=costs))


def list_datasets(user_id: int | None = None, conversation_id: int | None = None) -> list[dict]:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        rows = conn.execute(
            """
            SELECT * FROM datasets
            WHERE {scope}
            AND lower(name) NOT IN (
                lower('示例数据'),
                lower('默认数据'),
                lower('结构化上传测试'),
                lower('sample data'),
                lower('demo data')
            )
            ORDER BY id DESC
            """.format(scope=clause),
            params,
        ).fetchall()
        return [dict(row) for row in rows]


def set_active_dataset(dataset_id: int, user_id: int | None = None, conversation_id: int | None = None) -> None:
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        conn.execute(f"UPDATE datasets SET is_active = 0 WHERE {clause}", params)
        conn.execute(f"UPDATE datasets SET is_active = 1 WHERE id = ? AND {clause}", [dataset_id, *params])


def clear_active_dataset(user_id: int | None = None, conversation_id: int | None = None) -> None:
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        conn.execute(f"UPDATE datasets SET is_active = 0 WHERE {clause}", params)


def login_user(username: str) -> dict:
    init_db()
    clean = username.strip()
    if not clean:
        raise ValueError("用户名不能为空。")
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (clean,)).fetchone()
        if row:
            return dict(row)
        token = secrets.token_urlsafe(32)
        cursor = conn.execute(
            "INSERT INTO users (username, session_token) VALUES (?, ?)",
            (clean, token),
        )
        return {
            "id": int(cursor.lastrowid),
            "username": clean,
            "session_token": token,
        }


def get_user_by_token(token: str | None) -> dict | None:
    init_db()
    if not token:
        return None
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE session_token = ?", (token,)).fetchone()
        return dict(row) if row else None


def save_llm_config(config: dict, user_id: int | None = None) -> int:
    """按账户持久化配置；空密钥保留原值，校验失败不覆盖有效配置。"""

    init_db()
    base_url = str(config.get("base_url") or "").strip().rstrip("/")
    model = str(config.get("model") or "").strip()
    if not base_url or not model:
        raise ValueError("请填写模型服务地址和模型名称。")
    with connect() as conn:
        # 同一事务读取旧密钥并切换配置，避免空表单或并发保存造成凭据丢失。
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT * FROM llm_configs WHERE is_active = 1 AND user_id IS ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        api_key = str(config.get("api_key") or "").strip()
        if not api_key and previous and previous["api_key"].strip():
            if previous["base_url"].strip().rstrip("/") != base_url:
                raise ValueError("更换模型服务地址时，请填写该服务的 API Key。")
            api_key = previous["api_key"]
        # 本机兼容服务允许无需密钥；远程服务首次保存必须提供密钥。
        if not api_key and urlparse(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("首次配置该模型服务时，请填写 API Key。")
        temperature = float(config.get("temperature", 0.2))
        conn.execute("UPDATE llm_configs SET is_active = 0 WHERE user_id IS ?", (user_id,))
        cursor = conn.execute(
            """
            INSERT INTO llm_configs (user_id, name, base_url, model, api_key, temperature, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                user_id,
                config.get("name") or "default",
                base_url,
                model,
                api_key,
                temperature,
            ),
        )
        return int(cursor.lastrowid)


def get_active_llm_config(user_id: int | None = None) -> dict | None:
    init_db()
    with connect() as conn:
        if user_id is None:
            row = conn.execute(
                "SELECT * FROM llm_configs WHERE is_active = 1 AND user_id IS NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM llm_configs WHERE is_active = 1 AND user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None


def save_run(
    dataset_id: int | None,
    question: str,
    answer: str,
    result: dict,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                user_id, conversation_id, dataset_id, question, answer, objective_value, transport_cost, fixed_cost,
                status, open_warehouses, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                conversation_id,
                dataset_id,
                question,
                answer,
                result.get("objective_value"),
                result.get("transport_cost"),
                result.get("fixed_cost"),
                result.get("status", ""),
                json.dumps(result.get("open_warehouses", []), ensure_ascii=False),
                json.dumps(result, ensure_ascii=False, default=str),
            ),
        )
        return int(cursor.lastrowid)


def update_run_result(run_id: int, result: dict) -> None:
    """在 Agent 状态图完成后补充最终轨迹和校验信息。"""

    with connect() as conn:
        conn.execute(
            """
            UPDATE runs
            SET question = ?, answer = ?, objective_value = ?, transport_cost = ?, fixed_cost = ?,
                status = ?, open_warehouses = ?, result_json = ?
            WHERE id = ?
            """,
            (
                result.get("question", ""),
                result.get("answer", ""),
                result.get("objective_value"),
                result.get("transport_cost"),
                result.get("fixed_cost"),
                result.get("status", ""),
                json.dumps(result.get("open_warehouses", []), ensure_ascii=False),
                json.dumps(result, ensure_ascii=False, default=str),
                run_id,
            ),
        )


def create_agent_episode(
    question: str,
    user_id: int | None,
    conversation_id: int | None,
    *,
    policy_name: str = "langgraph_baseline",
    policy_version: str = "1.0",
) -> str:
    """创建一次 Agent episode，并返回跨 API 稳定的字符串标识。"""

    init_db()
    episode_id = uuid4().hex
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_episodes (
                episode_id, user_id, conversation_id, question,
                policy_name, policy_version, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'running')
            """,
            (episode_id, user_id, conversation_id, question, policy_name, policy_version),
        )
    return episode_id


def start_agent_step(
    episode_id: str,
    sequence: int,
    node_id: str,
    state: dict,
    action: dict,
    *,
    attempt: int = 1,
) -> None:
    """记录 policy 在给定状态下选择的节点动作。"""

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_steps (
                episode_id, sequence, node_id, attempt, status, state_json, action_json
            ) VALUES (?, ?, ?, ?, 'running', ?, ?)
            ON CONFLICT(episode_id, node_id, attempt) DO UPDATE SET
                sequence = excluded.sequence,
                status = 'running',
                state_json = excluded.state_json,
                action_json = excluded.action_json,
                observation_json = '{}',
                reward_json = '{}',
                error = NULL,
                elapsed_ms = NULL,
                started_at = CURRENT_TIMESTAMP,
                completed_at = NULL
            """,
            (episode_id, sequence, node_id, attempt, _json_dump(state), _json_dump(action)),
        )


def finish_agent_step(
    episode_id: str,
    node_id: str,
    observation: dict,
    *,
    status: str,
    elapsed_ms: float,
    reward: dict | None = None,
    error: str | None = None,
    attempt: int = 1,
    action: dict | None = None,
) -> None:
    """补充工具观察、节点奖励、耗时和成功或失败状态。"""

    with connect() as conn:
        conn.execute(
            """
            UPDATE agent_steps
            SET status = ?, observation_json = ?, reward_json = ?, error = ?,
                elapsed_ms = ?, action_json = COALESCE(?, action_json), completed_at = CURRENT_TIMESTAMP
            WHERE episode_id = ? AND node_id = ? AND attempt = ?
            """,
            (
                status,
                _json_dump(observation),
                _json_dump(reward or {}),
                error,
                elapsed_ms,
                _json_dump(action) if action is not None else None,
                episode_id,
                node_id,
                attempt,
            ),
        )


def set_agent_episode_template(episode_id: str, template_id: str) -> None:
    """Modeler 完成后尽早标注任务类型，保留后续失败轨迹的语义。"""

    with connect() as conn:
        conn.execute(
            "UPDATE agent_episodes SET template_id = ? WHERE episode_id = ?",
            (template_id, episode_id),
        )


def complete_agent_episode(
    episode_id: str,
    *,
    status: str,
    run_id: int | None = None,
    template_id: str | None = None,
    reward: dict | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """结束 episode，并保存训练所需的终局奖励和最终结果。"""

    reward_payload = reward or {}
    with connect() as conn:
        conn.execute(
            """
            UPDATE agent_episodes
            SET run_id = ?, status = ?, template_id = COALESCE(?, template_id), total_reward = ?,
                reward_json = ?, result_json = ?, error = ?, completed_at = CURRENT_TIMESTAMP
            WHERE episode_id = ?
            """,
            (
                run_id,
                status,
                template_id,
                reward_payload.get("total"),
                _json_dump(reward_payload),
                _json_dump(result or {}),
                error,
                episode_id,
            ),
        )


def list_agent_episodes(
    *,
    user_id: int | None,
    conversation_id: int | None = None,
    limit: int = 50,
) -> list[dict]:
    """按用户与可选会话列出 episode 摘要。"""

    init_db()
    clauses = ["user_id IS NULL" if user_id is None else "user_id = ?"]
    params: list = [] if user_id is None else [user_id]
    if conversation_id is not None:
        clauses.append("conversation_id = ?")
        params.append(conversation_id)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT episode_id, user_id, conversation_id, run_id, question,
                   policy_name, policy_version, status, template_id, total_reward,
                   error, started_at, completed_at
            FROM agent_episodes
            WHERE {' AND '.join(clauses)}
            ORDER BY started_at DESC, episode_id DESC
            LIMIT ?
            """,
            [*params, max(1, min(int(limit), 500))],
        ).fetchall()
        return [dict(row) for row in rows]


def get_agent_episode(episode_id: str, *, user_id: int | None) -> dict | None:
    """读取完整 episode 与有序 step，供审计和回放使用。"""

    init_db()
    user_clause = "user_id IS NULL" if user_id is None else "user_id = ?"
    params = [episode_id] if user_id is None else [episode_id, user_id]
    with connect() as conn:
        episode_row = conn.execute(
            f"SELECT * FROM agent_episodes WHERE episode_id = ? AND {user_clause}",
            params,
        ).fetchone()
        if not episode_row:
            return None
        step_rows = conn.execute(
            "SELECT * FROM agent_steps WHERE episode_id = ? ORDER BY sequence ASC, attempt ASC, id ASC",
            (episode_id,),
        ).fetchall()
    episode = dict(episode_row)
    for field in ("reward_json", "result_json"):
        episode[field.removesuffix("_json")] = _json_load(episode.pop(field))
    steps = []
    for row in step_rows:
        step = dict(row)
        for field in ("state_json", "action_json", "observation_json", "reward_json"):
            step[field.removesuffix("_json")] = _json_load(step.pop(field))
        steps.append(step)
    episode["steps"] = steps
    return episode


def get_training_episode(episode_id: str, *, user_id: int | None) -> dict | None:
    """转换为离线 RL/行为克隆可直接消费的 transition 视图。"""

    episode = get_agent_episode(episode_id, user_id=user_id)
    if episode is None:
        return None
    return build_training_episode(episode)


def build_training_episode(episode: dict) -> dict:
    """将已读取的 episode 转为训练视图，供在线 API 与只读批量导出共用。"""

    episode_id = episode["episode_id"]
    steps = episode["steps"]
    terminal_episode = episode["status"] in {"completed", "failed"}
    transitions = []
    for index, step in enumerate(steps):
        is_terminal = terminal_episode and index == len(steps) - 1
        next_state = steps[index + 1]["state"] if index + 1 < len(steps) else {"terminal": terminal_episode}
        transitions.append(
            {
                "t": index,
                "node_id": step["node_id"],
                "attempt": step["attempt"],
                "state": step["state"],
                "action": step["action"],
                "observation": step["observation"],
                "next_state": next_state,
                "reward": float(episode.get("total_reward") or 0.0) if is_terminal else 0.0,
                "done": is_terminal,
                "status": step["status"],
            }
        )
    decision_steps = [step for step in steps if step["status"] == "completed" and (step.get("action") or {}).get("candidate_ids")]
    decision_transitions = []
    for index, step in enumerate(decision_steps):
        action = step["action"]
        is_terminal_decision = terminal_episode and index == len(decision_steps) - 1
        next_step = decision_steps[index + 1] if index + 1 < len(decision_steps) else None
        decision_transitions.append(
            {
                "t": index,
                "node_id": step["node_id"],
                "attempt": step["attempt"],
                "state": step["state"],
                "policy_observation": step["state"].get("recovery_observation"),
                "candidate_actions": action.get("candidate_ids", []),
                "action_mask": action.get("action_mask", []),
                "action": action.get("selected_action"),
                "observation": step["observation"],
                "next_state": next_step["state"] if next_step else {"terminal": terminal_episode},
                "next_policy_observation": next_step["state"].get("recovery_observation") if next_step else None,
                "policy": {"name": action.get("policy_name"), "version": action.get("policy_version")},
                "reward": float(episode.get("total_reward") or 0.0) if is_terminal_decision else 0.0,
                "done": is_terminal_decision,
                "status": step["status"],
            }
        )
    template_id = episode["template_id"] or _template_from_steps(steps)
    observations = [item.get("policy_observation") or {} for item in decision_transitions]
    failed_checks = [item for item in observations if not (item.get("verification") or {}).get("passed")]
    final_action = decision_transitions[-1]["action"] if decision_transitions else None
    outcome = ("execution_failed" if episode["status"] == "failed" else
               "terminated_failure" if final_action == "terminate" else
               "recovered_success" if final_action == "accept_solution" and failed_checks else
               "verified_success" if final_action == "accept_solution" else "no_recovery_decisions")
    return {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "policy": {"name": episode["policy_name"], "version": episode["policy_version"]},
        "task": {"question": episode["question"], "template_id": template_id,
                 "instance_fingerprint": observations[0].get("instance_fingerprint") if observations else None},
        "trajectory_source": steps[0]["state"].get("trajectory_source", "unknown_legacy") if steps else "unknown_legacy",
        "outcome": outcome,
        "failed_verification_count": len(failed_checks),
        "status": episode["status"],
        "total_reward": episode["total_reward"],
        "transitions": transitions,
        "decision_transitions": decision_transitions,
    }


def _template_from_steps(steps: list[dict]) -> str | None:
    """失败 episode 可从 Modeler observation 恢复模板标签。"""

    for step in steps:
        problem_spec = (step.get("observation") or {}).get("problem_spec") or {}
        if problem_spec.get("template_id"):
            return str(problem_spec["template_id"])
    return None


def _json_dump(value: object) -> str:
    """统一数据库 JSON 编码，保留中文并兼容模型对象。"""

    return json.dumps(value, ensure_ascii=False, default=str)


def _json_load(value: str | None) -> object:
    """兼容旧记录或空字段，避免查询接口因单条坏数据失败。"""

    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}


def list_runs(limit: int = 20, user_id: int | None = None, conversation_id: int | None = None) -> list[dict]:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        rows = conn.execute(
            f"SELECT * FROM (SELECT * FROM runs WHERE {clause} ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            [*params, limit],
        ).fetchall()
        return [dict(row) for row in rows]


def clear_runs(user_id: int | None = None, conversation_id: int | None = None) -> int:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        _delete_agent_episodes(conn, clause, params)
        conn.execute(f"DELETE FROM conversation_requirements WHERE {clause}", params)
        conn.execute(f"DELETE FROM agent_task_states WHERE {clause}", params)
        cursor = conn.execute(f"DELETE FROM runs WHERE {clause}", params)
        return int(cursor.rowcount or 0)


def delete_conversation(conversation_id: int, user_id: int | None = None) -> bool:
    init_db()
    conversation = get_conversation(conversation_id, user_id=user_id)
    if not conversation:
        return False
    with connect() as conn:
        dataset_rows = conn.execute(
            "SELECT id FROM datasets WHERE conversation_id = ? AND " + ("user_id IS NULL" if user_id is None else "user_id = ?"),
            (conversation_id,) if user_id is None else (conversation_id, user_id),
        ).fetchall()
        dataset_ids = [int(row["id"]) for row in dataset_rows]
        if dataset_ids:
            placeholders = ",".join("?" for _ in dataset_ids)
            for table in ("warehouses", "customers", "costs"):
                conn.execute(f"DELETE FROM {table} WHERE dataset_id IN ({placeholders})", dataset_ids)
            conn.execute(f"DELETE FROM datasets WHERE id IN ({placeholders})", dataset_ids)
        clause, params = _scope_clause(user_id, conversation_id)
        _delete_agent_episodes(conn, clause, params)
        conn.execute(f"DELETE FROM conversation_requirements WHERE {clause}", params)
        conn.execute(f"DELETE FROM agent_task_states WHERE {clause}", params)
        conn.execute(f"DELETE FROM agent_memories WHERE {clause}", params)
        conn.execute(f"DELETE FROM runs WHERE {clause}", params)
        conn.execute(f"DELETE FROM uploaded_files WHERE {clause}", params)
        if user_id is None:
            conn.execute("DELETE FROM conversations WHERE id = ? AND user_id IS NULL", (conversation_id,))
        else:
            conn.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
        return True


def _delete_agent_episodes(conn: sqlite3.Connection, clause: str, params: list) -> None:
    """在清理对话或运行记录时同步移除对应训练轨迹。"""

    rows = conn.execute(f"SELECT episode_id FROM agent_episodes WHERE {clause}", params).fetchall()
    episode_ids = [str(row["episode_id"]) for row in rows]
    if not episode_ids:
        return
    placeholders = ",".join("?" for _ in episode_ids)
    conn.execute(f"DELETE FROM agent_steps WHERE episode_id IN ({placeholders})", episode_ids)
    conn.execute(f"DELETE FROM agent_episodes WHERE episode_id IN ({placeholders})", episode_ids)


def save_uploaded_files(
    name: str,
    files: list[dict],
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> int:
    init_db()
    with connect() as conn:
        count = 0
        for item in files:
            frame = item["frame"]
            content_csv = frame.to_csv(index=False)
            conn.execute(
                """
                INSERT INTO uploaded_files (user_id, conversation_id, name, filename, role, columns_json, content_csv, preview_csv)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    conversation_id,
                    name,
                    item["filename"],
                    item.get("role"),
                    json.dumps([str(column) for column in frame.columns], ensure_ascii=False),
                    content_csv,
                    frame.head(20).to_csv(index=False),
                ),
            )
            count += 1
        return count


def list_uploaded_files(
    limit: int = 10,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> list[dict]:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        rows = conn.execute(
            f"SELECT * FROM uploaded_files WHERE {clause} ORDER BY id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [dict(row) for row in rows]
