from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import secrets
from typing import Iterator

import pandas as pd

from optiagent.data import SupplyChainData, normalize_data


DB_PATH = Path("data/optiagent.sqlite3")


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
    init_db()
    with connect() as conn:
        if user_id is None:
            conn.execute("UPDATE llm_configs SET is_active = 0 WHERE user_id IS NULL")
        else:
            conn.execute("UPDATE llm_configs SET is_active = 0 WHERE user_id = ?", (user_id,))
        cursor = conn.execute(
            """
            INSERT INTO llm_configs (user_id, name, base_url, model, api_key, temperature, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                user_id,
                config.get("name") or "default",
                config["base_url"],
                config["model"],
                config["api_key"],
                float(config.get("temperature", 0.2)),
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


def list_runs(limit: int = 20, user_id: int | None = None, conversation_id: int | None = None) -> list[dict]:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
        rows = conn.execute(
            f"SELECT * FROM runs WHERE {clause} ORDER BY id ASC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [dict(row) for row in rows]


def clear_runs(user_id: int | None = None, conversation_id: int | None = None) -> int:
    init_db()
    with connect() as conn:
        clause, params = _scope_clause(user_id, conversation_id)
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
        conn.execute(f"DELETE FROM runs WHERE {clause}", params)
        conn.execute(f"DELETE FROM uploaded_files WHERE {clause}", params)
        if user_id is None:
            conn.execute("DELETE FROM conversations WHERE id = ? AND user_id IS NULL", (conversation_id,))
        else:
            conn.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
        return True


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
