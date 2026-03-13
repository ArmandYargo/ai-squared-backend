import os
import uuid
import json
import time
import hmac
import base64
import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from db import (
    list_conversations,
    create_conversation,
    get_conversation,
    get_messages,
    insert_message,
    update_conversation_after_turn,
    create_agent_run,
    finish_agent_run,
)

# Load .env
load_dotenv()

# Import your graph
from agent.graph import get_graph  # noqa: E402


app = FastAPI(title="AI-Squared Backend", version="0.1.0")

# Allow frontend origins
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

# In-memory cache for active sessions
# Neon is the source of truth; this just speeds up active conversations
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Build graph once
GRAPH = get_graph()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COOKIE_NAME = "ai_squared_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # backward compatibility
    browser_id: Optional[str] = None  # will be used by frontend sidebar/history


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


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    messages: List[Dict[str, Any]]


def _is_production() -> bool:
    return os.environ.get("APP_ENV", "production").lower() == "production"


def _cookie_domain() -> Optional[str]:
    # For local dev, return None
    # For production, use the shared parent domain so app.ai-squared.net can use api.ai-squared.net session flow
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
    """
    Temporary owner-key strategy until full user accounts are added.
    Once the frontend sends a stable browser_id, each browser gets its own chat list.
    For backward compatibility right now, fall back to a shared key.
    """
    browser_id = (browser_id or "").strip()
    if browser_id:
        return browser_id
    return "shared-login-user"


@app.get("/")
def root():
    return {"ok": True, "service": "ai-squared-backend", "status": "running"}


@app.get("/api/health")
def health():
    return {"ok": True, "service": "ai-squared-backend"}


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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _require_auth(request)

    try:
        owner_key = _resolve_owner_key(req.browser_id)

        # Backward compatibility: accept old session_id from current frontend
        conversation_id = req.conversation_id or req.session_id
        conv = None

        if conversation_id:
            conv = get_conversation(conversation_id, owner_key)
            if not conv:
                # If the frontend still sends an old in-memory session id that isn't in Neon yet,
                # create a fresh conversation so chat still works instead of hard failing.
                conv = create_conversation(owner_key)
                conversation_id = str(conv["id"])
        else:
            conv = create_conversation(owner_key)
            conversation_id = str(conv["id"])

        # Load from active in-memory cache first; otherwise from Neon last_state
        state = SESSIONS.get(conversation_id)
        if state is None:
            saved_state = conv.get("last_state") if conv else {}
            state = saved_state or {
                "messages": [],
                "ram_wizard": {"active": False, "step": "machine"},
                "intent": "qa",
            }

        # Ensure base state keys exist
        state.setdefault("messages", [])
        state.setdefault("ram_wizard", {"active": False, "step": "machine"})
        state.setdefault("intent", "qa")

        # Save user message to DB
        user_msg_row = insert_message(
            conversation_id=conversation_id,
            role="user",
            content=req.message,
            speaker="USER",
        )

        # Add user message to working state
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": req.message}
        ]

        run = create_agent_run(
            conversation_id=conversation_id,
            message_id=str(user_msg_row["id"]),
            run_type="chat_turn",
            input_json={"message": req.message},
        )

        out = GRAPH.invoke(state, config={"configurable": {"thread_id": conversation_id}})
        SESSIONS[conversation_id] = out

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
            result_json={"reply_preview": assistant_text[:200]},
        )

        return ChatResponse(
            conversation_id=conversation_id,
            reply=assistant_text,
            speaker=assistant_speaker,
            raw_state=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    _require_auth(request)

    try:
        suffix = Path(file.filename).suffix
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        out_path = UPLOAD_DIR / safe_name

        content = await file.read()
        out_path.write_bytes(content)

        return {
            "ok": True,
            "filename": file.filename,
            "stored_as": safe_name,
            "server_path": str(out_path.resolve()),
        }
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