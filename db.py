import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

APP_DATABASE_URL = os.environ.get("APP_DATABASE_URL")


def _normalized_db_url() -> str:
    url = (APP_DATABASE_URL or "").strip()
    if not url:
        raise RuntimeError("APP_DATABASE_URL is not set.")

    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"

    return url


@contextmanager
def get_conn():
    url = _normalized_db_url()
    conn = connect(
        url,
        row_factory=dict_row,
        connect_timeout=15,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    try:
        yield conn
    finally:
        conn.close()


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


def insert_agent_output(
    conversation_id: str,
    output_type: str,
    title: Optional[str] = None,
    run_id: Optional[str] = None,
    storage_provider: Optional[str] = None,
    storage_key: Optional[str] = None,
    mime_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO app.agent_outputs
                (conversation_id, run_id, output_type, title, storage_provider, storage_key, mime_type, metadata)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                conversation_id,
                run_id,
                output_type,
                title,
                storage_provider,
                storage_key,
                mime_type,
                Jsonb(metadata or {}),
            ),
        )
        row = cur.fetchone()
        conn.commit()
        return row


def list_artifacts(conversation_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM app.agent_outputs
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            """,
            (conversation_id,),
        )
        return cur.fetchall()


def get_artifact(artifact_id: str, owner_key: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT ao.*, c.owner_key
            FROM app.agent_outputs ao
            JOIN app.conversations c
              ON c.id = ao.conversation_id
            WHERE ao.id = %s
              AND c.owner_key = %s
              AND c.archived_at IS NULL
            """,
            (artifact_id, owner_key),
        )
        return cur.fetchone()


def delete_artifact(artifact_id: str, owner_key: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM app.agent_outputs ao
            USING app.conversations c
            WHERE ao.conversation_id = c.id
              AND ao.id = %s
              AND c.owner_key = %s
            RETURNING ao.*
            """,
            (artifact_id, owner_key),
        )
        row = cur.fetchone()
        conn.commit()
        return row


def delete_conversation(conversation_id: str, owner_key: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM app.conversations
            WHERE id = %s AND owner_key = %s
            RETURNING *
            """,
            (conversation_id, owner_key),
        )
        row = cur.fetchone()
        conn.commit()
        return row


def list_artifact_storage_keys_for_conversation(
    conversation_id: str, owner_key: str
) -> List[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT ao.storage_key
            FROM app.agent_outputs ao
            JOIN app.conversations c
              ON c.id = ao.conversation_id
            WHERE ao.conversation_id = %s
              AND c.owner_key = %s
              AND ao.storage_provider = 'local'
              AND ao.storage_key IS NOT NULL
            """,
            (conversation_id, owner_key),
        )
        rows = cur.fetchall()
        return [r["storage_key"] for r in rows if r.get("storage_key")]


def update_conversation_title(
    conversation_id: str,
    owner_key: str,
    title: str,
) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE app.conversations
            SET title = %s,
                updated_at = now()
            WHERE id = %s
              AND owner_key = %s
              AND archived_at IS NULL
            RETURNING *
            """,
            (title, conversation_id, owner_key),
        )
        row = cur.fetchone()
        conn.commit()
        return row