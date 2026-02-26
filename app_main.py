import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Import your graph
from agent.graph import get_graph  # noqa: E402


app = FastAPI(title="AI-Squared Backend", version="0.1.0")

# Allow local frontend + Cloudflare frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://app.ai-squared.net",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session store (good for local + early deployment)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Build graph once
GRAPH = get_graph()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    speaker: str = "ASSISTANT"
    raw_state: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health():
    return {"ok": True, "service": "ai-squared-backend"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())

        # Start session state if new
        state = SESSIONS.get(session_id)
        if state is None:
            state = {
                "messages": [],
                "ram_wizard": {"active": False, "step": "machine"},
                "intent": "qa",
            }

        # Add user message
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": req.message}
        ]

        # Invoke graph (thread_id helps LangGraph checkpointing)
        out = GRAPH.invoke(state, config={"configurable": {"thread_id": session_id}})
        SESSIONS[session_id] = out

        # Find last assistant message
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
            raw_state=None,  # set to out if you want to debug
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload Excel/docs from the web UI and store on the backend.
    Returns a server-side path that the RAM wizard can use later.
    """
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
def reset_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"ok": True, "session_id": session_id}