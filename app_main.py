import os
import uuid
import json
import time
import hmac
import base64
import hashlib
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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

# Simple in-memory chat session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Build graph once
GRAPH = get_graph()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COOKIE_NAME = "ai_squared_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    speaker: str = "ASSISTANT"
    raw_state: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    password: str


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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _require_auth(request)

    try:
        session_id = req.session_id or str(uuid.uuid4())

        state = SESSIONS.get(session_id)
        if state is None:
            state = {
                "messages": [],
                "ram_wizard": {"active": False, "step": "machine"},
                "intent": "qa",
            }

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": req.message}
        ]

        out = GRAPH.invoke(state, config={"configurable": {"thread_id": session_id}})
        SESSIONS[session_id] = out

        msgs = out.get("messages", [])
        assistant_msg = None
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                assistant_msg = m
                break

        if not assistant_msg:
            raise HTTPException(status_code=500, detail="No assistant response generated.")

        return ChatResponse(
            session_id=session_id,
            reply=assistant_msg.get("content", ""),
            speaker=assistant_msg.get("speaker", "ASSISTANT"),
            raw_state=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Backend processing error.")


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
    except Exception:
        raise HTTPException(status_code=500, detail="Upload failed.")


@app.post("/api/reset")
def reset_session(session_id: str, request: Request = None):
    if request is not None:
        _require_auth(request)

    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"ok": True, "session_id": session_id}