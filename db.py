import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

APP_DATABASE_URL = os.environ.get("APP_DATABASE_URL")


@contextmanager
def get_conn():
    if not APP_DATABASE_URL:
        raise RuntimeError("APP_DATABASE_URL is not set.")
    with connect(APP_DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


def list_conversations(owner_key: str) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, last_message_preview, updated_at, last_message_at
            FROM app.conversations
            WHERE owner_key = %s AND archived_at IS NULL
            ORDER BY updated_at DESC
            """,
            (owner_key,),
        )
        return cur.fetchall()


def create_conversation(owner_key: str, title: Optional[str] = None) -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO app.conversations (owner_key, title)
            VALUES (%s, %s)
            RETURNING *
            """,
            (owner_key, title),
        )
        row = cur.fetchone()
        conn.commit()
        return row


def get_conversation(conversation_id: str, owner_key: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM app.conversations
            WHERE id = %s AND owner_key = %s AND archived_at IS NULL
            """,
            (conversation_id, owner_key),
        )
        return cur.fetchone()


def get_messages(conversation_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM app.messages
            WHERE conversation_id = %s
            ORDER BY seq ASC
            """,
            (conversation_id,),
        )
        return cur.fetchall()


def insert_message(
    conversation_id: str,
    role: str,
    content: str,
    speaker: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq
            FROM app.messages
            WHERE conversation_id = %s
            """,
            (conversation_id,),
        )
        row = cur.fetchone()
        seq = int(row["next_seq"])

        cur.execute(
            """
            INSERT INTO app.messages (conversation_id, seq, role, speaker, content, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (conversation_id, seq, role, speaker, content, Jsonb(metadata or {})),
        )
        row = cur.fetchone()
        conn.commit()
        return row


def update_conversation_after_turn(
    conversation_id: str,
    last_message_preview: str,
    last_state: Dict[str, Any],
    title: Optional[str] = None,
):
    with get_conn() as conn, conn.cursor() as cur:
        if title:
            cur.execute(
                """
                UPDATE app.conversations
                SET title = %s,
                    last_message_preview = %s,
                    last_state = %s,
                    updated_at = now(),
                    last_message_at = now()
                WHERE id = %s
                """,
                (title, last_message_preview, Jsonb(last_state or {}), conversation_id),
            )
        else:
            cur.execute(
                """
                UPDATE app.conversations
                SET last_message_preview = %s,
                    last_state = %s,
                    updated_at = now(),
                    last_message_at = now()
                WHERE id = %s
                """,
                (last_message_preview, Jsonb(last_state or {}), conversation_id),
            )
        conn.commit()


def create_agent_run(
    conversation_id: str,
    run_type: str,
    message_id: Optional[str] = None,
    input_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO app.agent_runs (conversation_id, message_id, run_type, status, input_json)
            VALUES (%s, %s, %s, 'running', %s)
            RETURNING *
            """,
            (conversation_id, message_id, run_type, Jsonb(input_json or {})),
        )
        row = cur.fetchone()
        conn.commit()
        return row


def finish_agent_run(run_id: str, status: str, result_json=None, error_json=None):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE app.agent_runs
            SET status = %s,
                result_json = %s,
                error_json = %s,
                completed_at = now()
            WHERE id = %s
            """,
            (status, Jsonb(result_json or {}), Jsonb(error_json or {}), run_id),
        )
        conn.commit()