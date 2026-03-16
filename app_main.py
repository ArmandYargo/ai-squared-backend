import os
import uuid
import json
import time
import hmac
import base64
import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from db import (
    list_conversations,
    create_conversation,
    get_conversation,
    get_messages,
    insert_message,
    update_conversation_after_turn,
    create_agent_run,
    finish_agent_run,
    insert_agent_output,
    list_artifacts,
    get_artifact,
    delete_artifact,
    delete_conversation,
    list_artifact_storage_keys_for_conversation,
    update_conversation_title,
    check_db,
)

from agent.graph import get_graph  # noqa: E402


app = FastAPI(title="AI-Squared Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://app.ai-squared.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict[str, Any]] = {}
GRAPH = get_graph()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COOKIE_NAME = "ai_squared_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7

MAX_TEXT_PER_ARTIFACT = 18000
MAX_TOTAL_ARTIFACT_CONTEXT = 45000


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    browser_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    speaker: str = "ASSISTANT"
    raw_state: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    password: str


class ConversationCreateRequest(BaseModel):
    browser_id: Optional[str] = None
    title: Optional[str] = None


class ConversationRenameRequest(BaseModel):
    title: str
    browser_id: Optional[str] = None


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    messages: List[Dict[str, Any]]


def _is_production() -> bool:
    return os.environ.get("APP_ENV", "production").lower() == "production"


def _cookie_domain() -> Optional[str]:
    return ".ai-squared.net" if _is_production() else None


def _session_secret() -> str:
    secret = os.environ.get("APP_SESSION_SECRET", "").strip()
    if not secret:
        raise RuntimeError("APP_SESSION_SECRET is not set.")
    return secret


def _shared_password() -> str:
    password = os.environ.get("APP_SHARED_PASSWORD", "").strip()
    if not password:
        raise RuntimeError("APP_SHARED_PASSWORD is not set.")
    return password


def _sign_value(value: str) -> str:
    secret = _session_secret().encode("utf-8")
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _create_session_token() -> str:
    payload = {
        "exp": int(time.time()) + COOKIE_MAX_AGE,
        "nonce": uuid.uuid4().hex,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode("utf-8")).decode("utf-8")
    sig = _sign_value(payload_b64)
    return f"{payload_b64}.{sig}"


def _verify_session_token(token: str) -> bool:
    try:
        if not token or "." not in token:
            return False

        payload_b64, sig = token.rsplit(".", 1)
        expected_sig = _sign_value(payload_b64)
        if not hmac.compare_digest(sig, expected_sig):
            return False

        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        payload = json.loads(payload_json)
        exp = int(payload.get("exp", 0))
        if exp < int(time.time()):
            return False

        return True
    except Exception:
        return False


def _require_auth(request: Request) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if not token or not _verify_session_token(token):
        raise HTTPException(status_code=401, detail="Not authenticated.")


def _resolve_owner_key(browser_id: Optional[str]) -> str:
    browser_id = (browser_id or "").strip()
    if browser_id:
        return browser_id
    return "shared-login-user"


def _safe_unlink(path_str: Optional[str]) -> None:
    if not path_str:
        return
    try:
        path = Path(path_str)
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        pass


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Truncated]"


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return raw


def _read_csv_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(parts).strip()
    except Exception:
        try:
            import PyPDF2  # type: ignore

            reader = PyPDF2.PdfReader(str(path))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(parts).strip()
        except Exception as e:
            return f"[PDF extraction unavailable: {type(e).__name__}: {e}]"


def _read_docx_file(path: Path) -> str:
    try:
        from docx import Document  # type: ignore

        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(parts).strip()
    except Exception as e:
        return f"[DOCX extraction unavailable: {type(e).__name__}: {e}]"


def _extract_text_from_path(path_str: str, mime_type: Optional[str] = None) -> Tuple[bool, str]:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return False, "[File not found on server]"

    suffix = path.suffix.lower()

    try:
        if suffix in {".txt", ".md"}:
            return True, _read_text_file(path)
        if suffix == ".json":
            return True, _read_json_file(path)
        if suffix == ".csv":
            return True, _read_csv_file(path)
        if suffix == ".pdf":
            return True, _read_pdf_file(path)
        if suffix == ".docx":
            return True, _read_docx_file(path)

        if mime_type:
            if mime_type.startswith("text/"):
                return True, _read_text_file(path)
            if mime_type == "application/json":
                return True, _read_json_file(path)

        return False, f"[Unsupported file type for extraction: {suffix or mime_type or 'unknown'}]"
    except Exception as e:
        return False, f"[Failed to extract text: {type(e).__name__}: {e}]"


def _build_artifact_context_for_conversation(conversation_id: str, artifact_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    rows = artifact_rows if artifact_rows is not None else list_artifacts(conversation_id)

    supported_blocks: List[str] = []
    used_artifact_ids: List[str] = []
    used_titles: List[str] = []
    skipped: List[Dict[str, str]] = []
    total_chars = 0

    for row in rows:
        if row.get("output_type") != "uploaded_file":
            continue
        if row.get("storage_provider") != "local":
            skipped.append(
                {
                    "artifact_id": str(row["id"]),
                    "title": row.get("title") or "Untitled artifact",
                    "reason": "Unsupported storage provider",
                }
            )
            continue

        storage_key = row.get("storage_key")
        if not storage_key:
            skipped.append(
                {
                    "artifact_id": str(row["id"]),
                    "title": row.get("title") or "Untitled artifact",
                    "reason": "Missing storage key",
                }
            )
            continue

        ok, extracted = _extract_text_from_path(storage_key, row.get("mime_type"))
        if not ok:
            skipped.append(
                {
                    "artifact_id": str(row["id"]),
                    "title": row.get("title") or "Untitled artifact",
                    "reason": extracted,
                }
            )
            continue

        extracted = _truncate_text(extracted, MAX_TEXT_PER_ARTIFACT)
        if not extracted.strip():
            skipped.append(
                {
                    "artifact_id": str(row["id"]),
                    "title": row.get("title") or "Untitled artifact",
                    "reason": "No extractable text found",
                }
            )
            continue

        block = (
            f"Document: {row.get('title') or 'Untitled'}\n"
            f"Artifact ID: {row['id']}\n"
            f"Mime type: {row.get('mime_type') or 'unknown'}\n\n"
            f"{extracted}"
        )

        projected = total_chars + len(block)
        if projected > MAX_TOTAL_ARTIFACT_CONTEXT:
            remaining = MAX_TOTAL_ARTIFACT_CONTEXT - total_chars
            if remaining > 500:
                block = _truncate_text(block, remaining)
                supported_blocks.append(block)
                used_artifact_ids.append(str(row["id"]))
                used_titles.append(row.get("title") or "Untitled artifact")
            break

        supported_blocks.append(block)
        used_artifact_ids.append(str(row["id"]))
        used_titles.append(row.get("title") or "Untitled artifact")
        total_chars += len(block)

    context_text = "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(supported_blocks) if supported_blocks else ""

    return {
        "artifact_context": context_text.strip(),
        "used_artifact_ids": used_artifact_ids,
        "used_artifact_titles": used_titles,
        "skipped_artifacts": skipped,
    }


def _build_conversation_artifact_inventory(conversation_id: str, artifact_rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    rows = artifact_rows if artifact_rows is not None else list_artifacts(conversation_id)
    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "id": str(r["id"]),
                "conversation_id": str(r["conversation_id"]),
                "run_id": str(r["run_id"]) if r.get("run_id") else None,
                "output_type": r.get("output_type"),
                "title": r.get("title"),
                "storage_provider": r.get("storage_provider"),
                "storage_key": r.get("storage_key"),
                "mime_type": r.get("mime_type"),
                "metadata": r.get("metadata") or {},
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            }
        )
    return items


def _artifact_exists(conversation_id: str, output_type: str, storage_key: Optional[str], artifact_rows: Optional[List[Dict[str, Any]]] = None) -> bool:
    if not storage_key:
        return False
    rows = artifact_rows if artifact_rows is not None else list_artifacts(conversation_id)
    for row in rows:
        if row.get("output_type") == output_type and row.get("storage_key") == storage_key:
            return True
    return False


def _persist_generated_ram_artifacts(
    conversation_id: str,
    run_id: str,
    out: Dict[str, Any],
    artifact_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    created: List[str] = []
    wiz = (out or {}).get("ram_wizard") or {}

    candidates = [
        {
            "path": wiz.get("ram_input_path"),
            "title": Path(wiz["ram_input_path"]).name if wiz.get("ram_input_path") else None,
            "output_type": "ram_input_workbook",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        },
        {
            "path": wiz.get("classified_path"),
            "title": Path(wiz["classified_path"]).name if wiz.get("classified_path") else None,
            "output_type": "ram_classified_workbook",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        },
        {
            "path": wiz.get("sim_metadata_path"),
            "title": Path(wiz["sim_metadata_path"]).name if wiz.get("sim_metadata_path") else None,
            "output_type": "ram_simulation_metadata",
            "mime_type": "application/json",
        },
    ]

    sim_outputs = wiz.get("sim_outputs") or {}
    for key, path_str in sim_outputs.items():
        if not path_str:
            continue
        ext = Path(path_str).suffix.lower()
        mime = "text/csv" if ext == ".csv" else "application/octet-stream"
        candidates.append(
            {
                "path": path_str,
                "title": Path(path_str).name,
                "output_type": f"ram_simulation_output:{key}",
                "mime_type": mime,
            }
        )

    sim_conditions = wiz.get("sim_conditions") or {}
    for key, path_str in sim_conditions.items():
        if not path_str:
            continue
        ext = Path(path_str).suffix.lower()
        mime = "text/csv" if ext == ".csv" else "application/octet-stream"
        candidates.append(
            {
                "path": path_str,
                "title": Path(path_str).name,
                "output_type": f"ram_simulation_condition:{key}",
                "mime_type": mime,
            }
        )

    for item in candidates:
        path_str = item.get("path")
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            continue
        if _artifact_exists(conversation_id, item["output_type"], str(path.resolve()), artifact_rows=artifact_rows):
            continue

        insert_agent_output(
            conversation_id=conversation_id,
            run_id=run_id,
            output_type=item["output_type"],
            title=item.get("title"),
            storage_provider="local",
            storage_key=str(path.resolve()),
            mime_type=item.get("mime_type"),
            metadata={"generated_by": "ram_wizard"},
        )
        created.append(item["output_type"])

    return created


@app.get("/")
def root():
    return {"ok": True, "service": "ai-squared-backend", "status": "running"}


@app.get("/api/health")
def health():
    try:
        db_status = check_db()
    except Exception as e:
        return {
            "ok": False,
            "service": "ai-squared-backend",
            "database": {"ok": False, "error": f"{type(e).__name__}: {e}"},
        }

    return {"ok": True, "service": "ai-squared-backend", "database": db_status}


@app.get("/api/me")
def me(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    authenticated = bool(token and _verify_session_token(token))
    return {"authenticated": authenticated}


@app.post("/api/login")
def login(req: LoginRequest, response: Response):
    try:
        expected = _shared_password()
        if req.password != expected:
            raise HTTPException(status_code=401, detail="Incorrect password.")

        token = _create_session_token()

        response.set_cookie(
            key=COOKIE_NAME,
            value=token,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            secure=_is_production(),
            samesite="none" if _is_production() else "lax",
            domain=_cookie_domain(),
            path="/",
        )

        return {"ok": True, "authenticated": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(
        key=COOKIE_NAME,
        domain=_cookie_domain(),
        path="/",
    )
    return {"ok": True}


@app.get("/api/conversations")
def api_list_conversations(request: Request, browser_id: Optional[str] = None):
    _require_auth(request)

    owner_key = _resolve_owner_key(browser_id)
    rows = list_conversations(owner_key)

    return {
        "items": [
            {
                "id": str(r["id"]),
                "title": r.get("title"),
                "last_message_preview": r.get("last_message_preview"),
                "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
                "last_message_at": r["last_message_at"].isoformat() if r.get("last_message_at") else None,
            }
            for r in rows
        ]
    }


@app.post("/api/conversations")
def api_create_conversation(req: ConversationCreateRequest, request: Request):
    _require_auth(request)

    owner_key = _resolve_owner_key(req.browser_id)
    row = create_conversation(owner_key, req.title)

    return {
        "conversation_id": str(row["id"]),
        "title": row.get("title"),
    }


@app.patch("/api/conversations/{conversation_id}")
def api_rename_conversation(
    conversation_id: str,
    req: ConversationRenameRequest,
    request: Request,
):
    _require_auth(request)

    owner_key = _resolve_owner_key(req.browser_id)
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required.")
    if len(title) > 200:
        raise HTTPException(status_code=400, detail="Title is too long.")

    row = update_conversation_title(conversation_id, owner_key, title)
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {
        "conversation_id": str(row["id"]),
        "title": row["title"],
    }


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def api_get_conversation(conversation_id: str, request: Request, browser_id: Optional[str] = None):
    _require_auth(request)

    owner_key = _resolve_owner_key(browser_id)
    conv = get_conversation(conversation_id, owner_key)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    msgs = get_messages(conversation_id)
    return ConversationDetailResponse(
        conversation_id=str(conv["id"]),
        title=conv.get("title"),
        messages=[
            {
                "id": str(m["id"]),
                "role": m["role"],
                "speaker": m.get("speaker"),
                "content": m["content"],
                "created_at": m["created_at"].isoformat() if m.get("created_at") else None,
            }
            for m in msgs
        ],
    )


@app.get("/api/conversations/{conversation_id}/artifacts")
def api_list_artifacts(conversation_id: str, request: Request, browser_id: Optional[str] = None):
    _require_auth(request)

    owner_key = _resolve_owner_key(browser_id)
    conv = get_conversation(conversation_id, owner_key)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    rows = list_artifacts(conversation_id)
    return {
        "items": [
            {
                "id": str(r["id"]),
                "conversation_id": str(r["conversation_id"]),
                "run_id": str(r["run_id"]) if r.get("run_id") else None,
                "output_type": r.get("output_type"),
                "title": r.get("title"),
                "storage_provider": r.get("storage_provider"),
                "storage_key": r.get("storage_key"),
                "mime_type": r.get("mime_type"),
                "metadata": r.get("metadata") or {},
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "download_url": f"/api/artifacts/{r['id']}/download",
            }
            for r in rows
        ]
    }


@app.get("/api/artifacts/{artifact_id}/download")
def api_download_artifact(artifact_id: str, request: Request, browser_id: Optional[str] = None):
    _require_auth(request)

    owner_key = _resolve_owner_key(browser_id)
    artifact = get_artifact(artifact_id, owner_key)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    storage_key = artifact.get("storage_key")
    if not storage_key:
        raise HTTPException(status_code=404, detail="Artifact file not found.")

    path = Path(storage_key)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")

    filename = artifact.get("title") or path.name
    media_type = artifact.get("mime_type") or "application/octet-stream"
    return FileResponse(path=str(path), media_type=media_type, filename=filename)


@app.delete("/api/conversations/{conversation_id}")
def api_delete_conversation(conversation_id: str, request: Request, browser_id: Optional[str] = None):
    _require_auth(request)

    owner_key = _resolve_owner_key(browser_id)
    conv = get_conversation(conversation_id, owner_key)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    storage_keys = list_artifact_storage_keys_for_conversation(conversation_id, owner_key)

    deleted = delete_conversation(conversation_id, owner_key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    if conversation_id in SESSIONS:
        del SESSIONS[conversation_id]

    for key in storage_keys:
        _safe_unlink(key)

    return {"ok": True, "conversation_id": conversation_id}


@app.delete("/api/artifacts/{artifact_id}")
def api_delete_artifact(artifact_id: str, request: Request, browser_id: Optional[str] = None):
    _require_auth(request)

    owner_key = _resolve_owner_key(browser_id)
    artifact = get_artifact(artifact_id, owner_key)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    storage_key = artifact.get("storage_key")
    storage_provider = artifact.get("storage_provider")

    deleted = delete_artifact(artifact_id, owner_key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    if storage_provider == "local":
        _safe_unlink(storage_key)

    return {"ok": True, "artifact_id": artifact_id}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _require_auth(request)

    try:
        owner_key = _resolve_owner_key(req.browser_id)

        conversation_id = req.conversation_id or req.session_id
        conv = None

        if conversation_id:
            conv = get_conversation(conversation_id, owner_key)
            if not conv:
                conv = create_conversation(owner_key)
                conversation_id = str(conv["id"])
        else:
            conv = create_conversation(owner_key)
            conversation_id = str(conv["id"])

        state = SESSIONS.get(conversation_id)
        if state is None:
            saved_state = conv.get("last_state") if conv else {}
            state = saved_state or {
                "messages": [],
                "ram_wizard": {"active": False, "step": "machine"},
                "intent": "qa",
            }

        state.setdefault("messages", [])
        state.setdefault("ram_wizard", {"active": False, "step": "machine"})
        state.setdefault("intent", "qa")

        artifact_rows = list_artifacts(conversation_id)
        artifact_context_info = _build_artifact_context_for_conversation(conversation_id, artifact_rows=artifact_rows)
        state["artifact_context"] = artifact_context_info.get("artifact_context", "")
        state["artifact_meta"] = {
            "used_artifact_ids": artifact_context_info.get("used_artifact_ids", []),
            "used_artifact_titles": artifact_context_info.get("used_artifact_titles", []),
            "skipped_artifacts": artifact_context_info.get("skipped_artifacts", []),
        }
        state["conversation_artifacts"] = _build_conversation_artifact_inventory(conversation_id, artifact_rows=artifact_rows)

        user_msg_row = insert_message(
            conversation_id=conversation_id,
            role="user",
            content=req.message,
            speaker="USER",
            metadata={
                "used_artifact_ids": artifact_context_info.get("used_artifact_ids", []),
                "used_artifact_titles": artifact_context_info.get("used_artifact_titles", []),
            },
        )

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": req.message}
        ]

        run = create_agent_run(
            conversation_id=conversation_id,
            message_id=str(user_msg_row["id"]),
            run_type="chat_turn",
            input_json={
                "message": req.message,
                "used_artifact_ids": artifact_context_info.get("used_artifact_ids", []),
                "used_artifact_titles": artifact_context_info.get("used_artifact_titles", []),
                "skipped_artifacts": artifact_context_info.get("skipped_artifacts", []),
            },
        )

        out = GRAPH.invoke(state, config={"configurable": {"thread_id": conversation_id}})  # type: ignore[arg-type]
        SESSIONS[conversation_id] = out

        created_output_types = _persist_generated_ram_artifacts(
            conversation_id=conversation_id,
            run_id=str(run["id"]),
            out=out,
            artifact_rows=artifact_rows,
        )
        if created_output_types:
            artifact_rows = list_artifacts(conversation_id)
            out["conversation_artifacts"] = _build_conversation_artifact_inventory(conversation_id, artifact_rows=artifact_rows)

        msgs = out.get("messages", [])
        assistant_msg = None
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                assistant_msg = m
                break

        if not assistant_msg:
            finish_agent_run(
                run_id=str(run["id"]),
                status="failed",
                error_json={"detail": "No assistant response generated."},
            )
            raise HTTPException(status_code=500, detail="No assistant response generated.")

        assistant_text = assistant_msg.get("content", "")
        assistant_speaker = assistant_msg.get("speaker", "ASSISTANT")

        insert_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_text,
            speaker=assistant_speaker,
            metadata={
                "used_artifact_ids": artifact_context_info.get("used_artifact_ids", []),
                "used_artifact_titles": artifact_context_info.get("used_artifact_titles", []),
                "skipped_artifacts": artifact_context_info.get("skipped_artifacts", []),
                "generated_output_types": created_output_types,
            },
        )

        title = conv.get("title") if conv else None
        if not title:
            title = req.message[:60].strip()

        update_conversation_after_turn(
            conversation_id=conversation_id,
            last_message_preview=assistant_text[:120],
            last_state=out,
            title=title,
        )

        finish_agent_run(
            run_id=str(run["id"]),
            status="completed",
            result_json={
                "reply_preview": assistant_text[:200],
                "used_artifact_ids": artifact_context_info.get("used_artifact_ids", []),
                "used_artifact_titles": artifact_context_info.get("used_artifact_titles", []),
                "generated_output_types": created_output_types,
            },
        )

        return ChatResponse(
            conversation_id=conversation_id,
            reply=assistant_text,
            speaker=assistant_speaker,
            raw_state=None,
        )

    except HTTPException as e:
        run_id = locals().get("run", {}).get("id") if isinstance(locals().get("run"), dict) else None
        if run_id:
            try:
                finish_agent_run(
                    run_id=str(run_id),
                    status="failed",
                    error_json={"detail": e.detail if hasattr(e, "detail") else str(e)},
                )
            except Exception:
                traceback.print_exc()
        raise
    except Exception as e:
        traceback.print_exc()
        run_id = locals().get("run", {}).get("id") if isinstance(locals().get("run"), dict) else None
        if run_id:
            try:
                finish_agent_run(
                    run_id=str(run_id),
                    status="failed",
                    error_json={"detail": f"{type(e).__name__}: {e}"},
                )
            except Exception:
                traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    browser_id: Optional[str] = Form(None),
):
    _require_auth(request)

    try:
        owner_key = _resolve_owner_key(browser_id)

        conv = None
        if conversation_id:
            conv = get_conversation(conversation_id, owner_key)
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found.")
        else:
            conv = create_conversation(owner_key, title=file.filename)
            conversation_id = str(conv["id"])

        suffix = Path(file.filename or "").suffix
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        out_path = UPLOAD_DIR / safe_name

        content = await file.read()
        out_path.write_bytes(content)

        artifact = insert_agent_output(
            conversation_id=conversation_id,
            run_id=None,
            output_type="uploaded_file",
            title=file.filename,
            storage_provider="local",
            storage_key=str(out_path.resolve()),
            mime_type=file.content_type,
            metadata={
                "original_filename": file.filename,
                "stored_as": safe_name,
                "size_bytes": len(content),
            },
        )

        update_conversation_after_turn(
            conversation_id=conversation_id,
            last_message_preview=f"Uploaded: {file.filename}",
            last_state=(conv.get("last_state") if conv else {}) or {},
            title=(conv.get("title") if conv and conv.get("title") else file.filename),
        )

        return {
            "ok": True,
            "conversation_id": conversation_id,
            "artifact_id": str(artifact["id"]),
            "filename": file.filename,
            "stored_as": safe_name,
            "server_path": str(out_path.resolve()),
            "mime_type": file.content_type,
            "metadata": artifact.get("metadata") or {},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/reset")
def reset_session(
    request: Request,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    _require_auth(request)

    key = conversation_id or session_id
    if key and key in SESSIONS:
        del SESSIONS[key]
    return {"ok": True, "conversation_id": key}