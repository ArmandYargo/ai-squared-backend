# agent/graph.py
import os
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:
    SqliteSaver = None

from agent.state import AgentState
from agent.ram_tool import run_ram_pipeline_compat, check_ram_readiness
from agent.ram_simulation_tool import run_ram_simulation_archived
from agent.rag import RagStore
from func_define_components import ai_propose_components_coarse, ai_apply_edit_to_components

_sim_progress_store: Optional[Dict[str, Dict[str, Any]]] = None


# -------------------------
# OpenAI helper
# -------------------------
def _llm_text(messages: List[Dict[str, str]]) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {
        "hi",
        "hello",
        "hey",
        "howzit",
        "morning",
        "good morning",
        "good afternoon",
        "good evening",
    }


# -------------------------
# RAG helpers
# -------------------------
_RAG_STORE: Optional[RagStore] = None


def _get_rag_store() -> Optional[RagStore]:
    global _RAG_STORE
    if os.environ.get("RAG_ENABLED", "1") != "1":
        return None
    if _RAG_STORE is not None:
        return _RAG_STORE
    try:
        _RAG_STORE = RagStore()
        return _RAG_STORE
    except Exception as e:
        print(f"[rag] init failed: {type(e).__name__}: {e}", flush=True)
        return None


def _rag_confidence_from_distance(distance: Optional[float]) -> float:
    """
    pgvector cosine distance is lower when more similar.
    For normalized embeddings, similarity ~= 1 - distance.
    We clamp to [0,1] and express as 0..100 confidence.
    """
    if distance is None:
        return 0.0
    sim = 1.0 - float(distance)
    sim = max(0.0, min(1.0, sim))
    return sim * 100.0


def _retrieve_rag_context(query: str, *, namespace: str = "knowledge", k: int = 6) -> Dict[str, Any]:
    store = _get_rag_store()
    if store is None:
        return {
            "enabled": False,
            "used": False,
            "confidence": 0.0,
            "hits": [],
            "context": "",
        }

    try:
        hits = store.retrieve(query, k=k, namespace=namespace)
    except Exception as e:
        print(f"[rag] retrieve failed: {type(e).__name__}: {e}", flush=True)
        return {
            "enabled": True,
            "used": False,
            "confidence": 0.0,
            "hits": [],
            "context": "",
        }

    if not hits:
        return {
            "enabled": True,
            "used": False,
            "confidence": 0.0,
            "hits": [],
            "context": "",
        }

    top_conf = max(_rag_confidence_from_distance(h.get("distance")) for h in hits)
    threshold = float(os.environ.get("RAG_CONFIDENCE_THRESHOLD", "70"))

    context_blocks: List[str] = []
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta") or {}
        source = meta.get("source") or meta.get("path") or "unknown"
        text = (h.get("text") or "").strip()
        if not text:
            continue
        context_blocks.append(f"[Source {i}] {source}\n{text}")

    return {
        "enabled": True,
        "used": top_conf >= threshold,
        "confidence": top_conf,
        "hits": hits,
        "context": "\n\n".join(context_blocks),
    }


# -------------------------
# Artifact helpers
# -------------------------
def _excel_suffixes() -> set[str]:
    return {".xlsx", ".xlsm", ".xls"}


def _is_excel_artifact(artifact: Dict[str, Any]) -> bool:
    title = (artifact.get("title") or "").lower()
    mime = (artifact.get("mime_type") or "").lower()
    suffix = Path(title).suffix.lower()

    if suffix in _excel_suffixes():
        return True

    return mime in {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.ms-excel.sheet.macroenabled.12",
    }


def _latest_excel_artifact(state: AgentState) -> Optional[Dict[str, Any]]:
    artifacts = state.get("conversation_artifacts") or []
    excel_artifacts = [a for a in artifacts if _is_excel_artifact(a)]
    if not excel_artifacts:
        return None
    return excel_artifacts[0]


def _latest_uploaded_excel_for_simulation(state: AgentState, current_source_artifact_id: Optional[str]) -> Optional[Dict[str, Any]]:
    artifacts = state.get("conversation_artifacts") or []
    excel_artifacts = [a for a in artifacts if _is_excel_artifact(a)]
    if not excel_artifacts:
        return None

    latest = excel_artifacts[0]
    latest_id = latest.get("id")
    if current_source_artifact_id and str(latest_id) == str(current_source_artifact_id):
        return None
    return latest


def _artifact_prompt_hint(state: AgentState) -> str:
    latest = _latest_excel_artifact(state)
    if not latest:
        return (
            "Please upload the CMMS Excel workbook into this conversation.\n"
            "Once uploaded, I’ll let you choose whether to proceed with it or upload a new file."
        )
    return (
        f"I found an uploaded workbook in this conversation: {latest.get('title')}\n"
        "Reply with 'proceed' to use it, or 'upload new' if you want to upload a different file."
    )


# -------------------------
# AI Router helper
# -------------------------
_ALLOWED_INTENTS = ("qa", "ram_wizard")


def _safe_json_from_text(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    return bool(
        re.match(
            r"^(what|why|how|when|where|who|which|can you|could you|do you|is it|are you|summarise|summarize|compare|review|analyse|analyze|explain)\b",
            t.lower(),
        )
    )


def _looks_like_ram_start(text: str) -> bool:
    t = (text or "").strip().lower()
    phrases = [
        "create input sheet",
        "build input sheet",
        "start ram",
        "ram wizard",
        "run ram",
        "start simulation",
        "simulate ram",
        "run simulation",
    ]
    return any(p in t for p in phrases)


def _route_intent_ai(user_text: str, wiz_active: bool) -> dict:
    system = (
        "You are an intent router for a reliability engineering assistant for mines/factories.\n"
        "Choose the single best intent for the user's message.\n\n"
        "Allowed intents:\n"
        "- qa: user is asking a question or wants explanation/advice\n"
        "- ram_wizard: user wants to create/continue the RAM input sheet workflow, provide required wizard input, "
        "or proceed with simulation steps.\n\n"
        "Rules:\n"
        "- If the user asks about an uploaded file, document, summary, comparison, or attached material, choose qa.\n"
        "- If the user wants to build an input sheet, run RAM, continue a wizard, review an input sheet, "
        "or start simulation, choose ram_wizard.\n"
        "- If the user is supplying machine/date/file/category/simulation values while wizard is active, choose ram_wizard.\n"
        "- If unclear, prefer qa.\n\n"
        "Output MUST be valid JSON with keys: intent, confidence, reason."
    )
    prompt = f"Wizard active: {wiz_active}\nUser message: {user_text}"
    resp = _llm_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )
    data = _safe_json_from_text(resp)
    if isinstance(data, dict) and data.get("intent") in _ALLOWED_INTENTS:
        return data
    return {}


# -------------------------
# Router node
# -------------------------
def node_router(state: AgentState) -> AgentState:
    msgs = state.get("messages") or []
    if not msgs:
        state["intent"] = "qa"
        return state

    user_text = msgs[-1].get("content", "")
    wiz_active = state.get("ram_wizard", {}).get("active") is True
    t = (user_text or "").strip()

    if t.startswith("?") or t.lower().startswith("/ask "):
        state["intent"] = "qa"
        msgs_list = state.get("messages") or []
        if msgs_list:
            if t.startswith("?"):
                msgs_list[-1]["content"] = t[1:].strip()
            else:
                msgs_list[-1]["content"] = t[5:].strip()
        return state

    if t.lower().startswith("/ram"):
        parts = t.split(maxsplit=1)
        cmd = parts[1].strip().lower() if len(parts) > 1 else ""
        state["ram_command"] = cmd
        state["intent"] = "ram_wizard"
        return state

    if _looks_like_ram_start(t):
        state["intent"] = "ram_wizard"
        return state

    if wiz_active:
        state["intent"] = "qa" if _looks_like_question(t) or _is_greeting(t) else "ram_wizard"
        return state

    try:
        route = _route_intent_ai(t, wiz_active=False)
    except Exception:
        route = {}

    if route:
        state["intent"] = route["intent"]
        state["route_meta"] = {
            "confidence": route.get("confidence"),
            "reason": route.get("reason"),
        }
        return state

    state["intent"] = "qa"
    return state


# -------------------------
# QA node (RAG first, then generic)
# -------------------------
def node_qa(state: AgentState) -> AgentState:
    print("[node_qa] entered", flush=True)

    msgs = state.get("messages") or []
    if not msgs:
        state.setdefault("messages", []).append(
            {"role": "assistant", "content": "How can I help?", "speaker": "LLM"}
        )
        return state

    question = msgs[-1].get("content", "").strip()
    print(f"[node_qa] question: {question}", flush=True)

    if not question:
        state.setdefault("messages", []).append(
            {"role": "assistant", "content": "Please enter a question.", "speaker": "LLM"}
        )
        return state

    artifact_context = (state.get("artifact_context") or "").strip()
    artifact_meta = state.get("artifact_meta") or {}
    used_titles = artifact_meta.get("used_artifact_titles") or []
    skipped_artifacts = artifact_meta.get("skipped_artifacts") or []

    rag = _retrieve_rag_context(question)
    rag_used = rag.get("used", False)
    rag_conf = rag.get("confidence", 0.0)
    rag_context = (rag.get("context") or "").strip()
    rag_hits = rag.get("hits") or []

    state["rag_hits"] = rag_hits

    print(f"[node_qa] rag_used={rag_used} rag_conf={rag_conf}", flush=True)
    print(f"[node_qa] used_titles={used_titles}", flush=True)
    print(f"[node_qa] skipped_artifacts={skipped_artifacts}", flush=True)
    print(f"[node_qa] artifact_context_length={len(artifact_context)}", flush=True)

    q_lower = question.lower()

    asks_about_artifacts = any(
        phrase in q_lower
        for phrase in [
            "what uploaded artifacts",
            "what artifacts",
            "what documents",
            "what files",
            "what attachments",
            "uploaded artifacts do you have access to",
        ]
    )

    if asks_about_artifacts:
        if used_titles:
            reply = "I have access to these uploaded artifacts in this conversation:\n" + "\n".join(
                f"- {title}" for title in used_titles
            )
            if skipped_artifacts:
                reply += "\n\nSome uploaded artifacts were found but were not usable for text extraction:\n"
                reply += "\n".join(
                    f"- {item.get('title') or 'Untitled'}: {item.get('reason') or 'Skipped'}"
                    for item in skipped_artifacts
                )
        elif skipped_artifacts:
            reply = "I found uploaded artifacts in this conversation, but they were not usable for text extraction:\n"
            reply += "\n".join(
                f"- {item.get('title') or 'Untitled'}: {item.get('reason') or 'Skipped'}"
                for item in skipped_artifacts
            )
        else:
            reply = "I do not currently have any usable uploaded artifact text in this conversation."

        state.setdefault("messages", []).append(
            {"role": "assistant", "content": reply, "speaker": "LLM"}
        )
        return state

    system_prompt = (
        "You are a helpful engineering, reliability, and document-analysis assistant.\n"
        "Priority order:\n"
        "1. If uploaded document context is present and relevant, use it.\n"
        "2. If RAG context is present with strong confidence, use it.\n"
        "3. Otherwise answer from general knowledge and be clear when you are doing so.\n"
        "Never fabricate cited facts from missing context."
    )

    user_parts: List[str] = []

    if artifact_context:
        user_parts.append(
            "Uploaded document context:\n"
            f"{artifact_context}"
        )

    if used_titles:
        user_parts.append(
            "Usable uploaded artifacts:\n" + "\n".join(f"- {title}" for title in used_titles)
        )

    if skipped_artifacts:
        skipped_lines = [
            f"- {item.get('title') or 'Untitled'}: {item.get('reason') or 'Skipped'}"
            for item in skipped_artifacts
        ]
        user_parts.append(
            "Some uploaded artifacts were not usable for text extraction:\n" + "\n".join(skipped_lines)
        )

    if rag_used and rag_context:
        user_parts.append(
            f"RAG context (confidence {rag_conf:.1f}%):\n{rag_context}"
        )

    user_parts.append(f"User question:\n{question}")
    combined_user_prompt = "\n\n".join(user_parts)

    answer = _llm_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_user_prompt},
        ]
    ).strip()

    state.setdefault("messages", []).append(
        {"role": "assistant", "content": answer, "speaker": "LLM"}
    )
    return state


# -------------------------
# RAM Wizard + Simulation
# -------------------------
def _wizard_reply(state: AgentState, text: str, wizard_ui: Optional[Dict[str, Any]] = None) -> None:
    msg: Dict[str, Any] = {"role": "assistant", "content": text, "speaker": "WIZARD"}
    if wizard_ui:
        msg["wizard_ui"] = wizard_ui
    state.setdefault("messages", []).append(msg)


def _parse_date_yyyy_mm_dd(s: str):
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _default_dates_from_input_sheet(xlsx_path: str):
    try:
        tl0 = pd.read_excel(xlsx_path, sheet_name="timeline_0")
        start = pd.to_datetime(tl0.loc[0, "from"]).date()
        end = pd.to_datetime(tl0.loc[0, "to"]).date()
        return start, end
    except Exception:
        return None, None


def _read_input_sheets(xlsx_path: str) -> Dict[str, Any]:
    xl = pd.ExcelFile(xlsx_path)
    sheets: Dict[str, Any] = {}
    for name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=name)
        rows = []
        for _, row in df.iterrows():
            r: Dict[str, Any] = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    r[col] = ""
                elif hasattr(val, "isoformat"):
                    r[col] = val.isoformat()
                else:
                    r[col] = val
            rows.append(r)
        sheets[name] = {
            "columns": df.columns.tolist(),
            "rows": rows,
        }
    return sheets


def _write_input_sheets(xlsx_path: str, sheets_data: Dict[str, Any]) -> None:
    from openpyxl import Workbook

    if not sheets_data:
        raise ValueError("No sheet data to write.")

    wb = Workbook()
    first_sheet = True
    for sheet_name, sheet_info in sheets_data.items():
        cols = sheet_info.get("columns", [])
        rows = sheet_info.get("rows", [])
        if first_sheet:
            ws = wb.active
            ws.title = sheet_name
            first_sheet = False
        else:
            ws = wb.create_sheet(title=sheet_name)
        ws.append(cols)
        for row_data in rows:
            cell_values = []
            for c in cols:
                val = row_data.get(c, "")
                if val != "" and val is not None:
                    try:
                        val = pd.to_numeric(val)
                    except (ValueError, TypeError):
                        pass
                cell_values.append(val)
            ws.append(cell_values)
    wb.save(xlsx_path)


def _pick_excel_file_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select CMMS Excel file",
            filetypes=[
                ("Excel files", "*.xlsx *.xlsm *.xls"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        path = (path or "").strip()
        return path if path else None
    except Exception:
        return None


def _format_readiness_summary(readiness_payload: Dict[str, Any]) -> str:
    subset_rows = readiness_payload.get("subset_rows", 0)
    readiness = readiness_payload.get("readiness") or {}
    ok = readiness.get("ok_to_simulate", None)
    metrics = readiness.get("metrics") or {}
    threshold = readiness.get("min_coverage_threshold", 0.8)
    essentials = readiness.get("essentials") or []
    message = readiness.get("message", "")

    lines: List[str] = []
    lines.append(f"- Rows matched for machine: {subset_rows}")
    lines.append(f"- OK to simulate: {'Yes' if ok else 'No'}")

    if metrics:
        lines.append(f"\nField coverage (minimum required: {threshold:.0%}):")
        for field, cov in metrics.items():
            pct = f"{cov:.0%}" if isinstance(cov, (int, float)) else str(cov)
            status = "PASS" if isinstance(cov, (int, float)) and cov >= threshold else "FAIL"
            lines.append(f"  - {field}: {pct} [{status}]")

    if not ok:
        failing = [
            f for f in essentials
            if f in metrics and isinstance(metrics[f], (int, float)) and metrics[f] < threshold
        ]
        if failing:
            lines.append(f"\nInsufficient data for: {', '.join(failing)}")
        if subset_rows == 0:
            lines.append("No work-order rows matched this machine. Check the machine name or upload a different file.")
        if message:
            lines.append(f"\n{message}")

    return "Data Readiness Check:\n" + "\n".join(lines)


def _format_numbered_categories(cats: List[str]) -> str:
    if not cats:
        return "(no categories)"
    return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(cats)])


_PRACTICE_CODE_TO_NAME = {0: "Reactive", 1: "Corrective", 2: "Preventative", 3: "Condition based"}
_PRACTICE_NAME_TO_CODE = {
    "reactive": 0, "corrective": 1,
    "preventative": 2, "preventive": 2,
    "condition based": 3, "condition-based": 3, "cbm": 3,
}


def _format_component_practices(cats: List[str], practices: Dict[str, int]) -> str:
    if not cats:
        return "(no components)"
    max_len = max(len(c) for c in cats)
    lines = []
    for i, c in enumerate(cats):
        code = practices.get(c, 0)
        name = _PRACTICE_CODE_TO_NAME.get(code, "Reactive")
        lines.append(f"{i + 1}. {c:<{max_len}}  ->  {name}")
    return "\n".join(lines)


def _build_default_practices(cats: List[str], default_code: int) -> Dict[str, int]:
    return {c: default_code for c in cats}


def node_ram_wizard(state: AgentState) -> AgentState:
    wiz = state.get("ram_wizard") or {}
    state["ram_wizard"] = wiz

    wiz.setdefault("active", False)
    wiz.setdefault("step", "machine")

    wiz.setdefault("machine", None)
    wiz.setdefault("maintenance_practice", None)
    wiz.setdefault("date_range_text", None)
    wiz.setdefault("excel_path", None)
    wiz.setdefault("source_artifact_id", None)
    wiz.setdefault("source_artifact_title", None)

    wiz.setdefault("categories", None)
    wiz.setdefault("categories_last_ai", None)
    wiz.setdefault("component_practices", None)
    wiz.setdefault("readiness_payload", None)

    wiz.setdefault("ram_input_path", None)
    wiz.setdefault("classified_path", None)
    wiz.setdefault("ram_input_ready_for_review", False)

    wiz.setdefault("sim_start", None)
    wiz.setdefault("sim_end", None)
    wiz.setdefault("simulations", None)
    wiz.setdefault("sim_archive_dir", None)
    wiz.setdefault("sim_metadata_path", None)
    wiz.setdefault("sim_outputs", None)
    wiz.setdefault("sim_conditions", None)

    cmd = (state.get("ram_command") or "").strip().lower()
    if cmd:
        state["ram_command"] = ""
        if cmd == "status":
            _wizard_reply(
                state,
                "RAM wizard status:\n"
                f"- active: {wiz.get('active')}\n"
                f"- step: {wiz.get('step')}\n"
                f"- machine: {wiz.get('machine')}\n"
                f"- date_range_text: {wiz.get('date_range_text')}\n"
                f"- excel_path: {wiz.get('excel_path')}\n"
                f"- source_artifact_title: {wiz.get('source_artifact_title')}\n"
                f"- ram_input_path: {wiz.get('ram_input_path')}\n"
                f"- classified_path: {wiz.get('classified_path')}\n"
                f"- sim_start: {wiz.get('sim_start')}\n"
                f"- sim_end: {wiz.get('sim_end')}\n"
                f"- simulations: {wiz.get('simulations')}\n"
                f"- sim_archive_dir: {wiz.get('sim_archive_dir')}\n"
                f"- categories: {wiz.get('categories')}\n"
            )
            return state

        if cmd in {"reset", "cancel"}:
            state["ram_wizard"] = {"active": False, "step": "machine"}
            _wizard_reply(state, "RAM wizard cancelled/reset. You can start again by typing: create input sheet")
            return state

        _wizard_reply(state, "Unknown /ram command. Use: /ram status | /ram reset | /ram cancel")
        return state

    if not wiz.get("active"):
        wiz["active"] = True
        wiz["step"] = "machine"
        latest = _latest_excel_artifact(state)
        extra = f"\nI can already see an uploaded workbook: {latest.get('title')}" if latest else ""
        _wizard_reply(
            state,
            "RAM Input Sheet Wizard started.\n"
            "What machine are we working on?"
            f"{extra}"
        )
        return state

    _msgs = state.get("messages") or []
    user_text = (_msgs[-1].get("content", "") if _msgs else "")
    user_text_stripped = user_text.strip()
    user_lower = user_text_stripped.lower()

    if wiz["step"] == "machine":
        if not user_text_stripped:
            _wizard_reply(state, "Please enter the machine name/type.")
            return state
        wiz["machine"] = user_text_stripped
        wiz["step"] = "maintenance_practice"
        _wizard_reply(
            state,
            "What maintenance practice is currently used on this machine?\n"
            "0. Reactive\n"
            "1. Corrective\n"
            "2. Preventative\n"
            "3. Condition based\n\n"
            "Type the number or name of the practice."
        )
        return state

    _MAINT_PRACTICE_MAP = {
        "0": 0, "reactive": 0,
        "1": 1, "corrective": 1,
        "2": 2, "preventative": 2, "preventive": 2,
        "3": 3, "condition based": 3, "condition-based": 3, "cbm": 3,
    }

    if wiz["step"] == "maintenance_practice":
        if not user_text_stripped:
            _wizard_reply(state, "Please select a maintenance practice (0-3).")
            return state
        key = user_lower.strip()
        if key not in _MAINT_PRACTICE_MAP:
            _wizard_reply(
                state,
                "Unrecognised option. Please type one of:\n"
                "0. Reactive\n"
                "1. Corrective\n"
                "2. Preventative\n"
                "3. Condition based"
            )
            return state
        wiz["maintenance_practice"] = _MAINT_PRACTICE_MAP[key]
        label = {0: "Reactive", 1: "Corrective", 2: "Preventative", 3: "Condition based"}[wiz["maintenance_practice"]]
        wiz["step"] = "date"
        _wizard_reply(
            state,
            f"Maintenance practice set to: {label}.\n\n"
            "Optional: enter a date range (e.g., '2019-01-01 to 2021-12-31' or '2023-2024') or type 'skip'."
        )
        return state

    if wiz["step"] == "date":
        if not user_text_stripped:
            _wizard_reply(state, "Enter a date range or type 'skip'.")
            return state
        wiz["date_range_text"] = None if user_lower in {"skip", "none", "no"} else user_text_stripped
        wiz["step"] = "file"
        _wizard_reply(state, _artifact_prompt_hint(state))
        return state

    if wiz["step"] == "file":
        latest = _latest_excel_artifact(state)

        if user_lower in {"proceed", "use uploaded", "continue"}:
            if not latest:
                _wizard_reply(state, "I couldn't find an uploaded Excel workbook in this conversation.")
                return state
            wiz["excel_path"] = latest.get("storage_key")
            wiz["source_artifact_id"] = latest.get("id")
            wiz["source_artifact_title"] = latest.get("title")
        elif user_lower in {"upload new", "new file"}:
            _wizard_reply(state, "Okay — upload the new CMMS workbook, then type 'continue'.")
            return state
        elif user_lower in {"pick file", "pick", "browse", "choose file", "select file"}:
            picked = _pick_excel_file_dialog()
            if not picked:
                _wizard_reply(state, f"No file selected (cancelled).\n\n{_artifact_prompt_hint(state)}")
                return state
            wiz["excel_path"] = picked
            wiz["source_artifact_id"] = None
            wiz["source_artifact_title"] = Path(picked).name
        elif latest and user_lower == "continue":
            wiz["excel_path"] = latest.get("storage_key")
            wiz["source_artifact_id"] = latest.get("id")
            wiz["source_artifact_title"] = latest.get("title")
        elif user_text_stripped:
            wiz["excel_path"] = user_text_stripped
            wiz["source_artifact_id"] = None
            wiz["source_artifact_title"] = Path(user_text_stripped).name
        else:
            _wizard_reply(state, _artifact_prompt_hint(state))
            return state

        _wizard_reply(
            state,
            f"Selected file:\n{wiz['source_artifact_title'] or wiz['excel_path']}\n\nRunning data readiness check..."
        )
        try:
            payload = check_ram_readiness(
                excel_path=wiz.get("excel_path") or "",
                machine=wiz.get("machine") or "",
            )
        except Exception as e:
            _wizard_reply(state, f"Readiness check failed: {type(e).__name__}: {e}\n\n{_artifact_prompt_hint(state)}")
            wiz["step"] = "file"
            return state

        wiz["readiness_payload"] = payload
        readiness_ok = (payload.get("readiness") or {}).get("ok_to_simulate", False)
        if readiness_ok:
            prompt_suffix = (
                "\n\nProceed with this file?\n"
                "- yes (continue)\n"
                "- upload new (go back and upload another workbook)\n"
                "- cancel"
            )
        else:
            prompt_suffix = (
                "\n\nThe data does not fully meet the requirements. You can still proceed, "
                "but results may be unreliable.\n"
                "- yes (proceed anyway)\n"
                "- upload new (go back and upload another workbook)\n"
                "- cancel"
            )
        _wizard_reply(state, _format_readiness_summary(payload) + prompt_suffix)
        wiz["step"] = "readiness_confirm"
        return state

    if wiz["step"] == "readiness_confirm":
        t = user_lower

        if t in {"upload new", "new file"}:
            wiz["step"] = "file"
            _wizard_reply(state, "Okay — upload the new CMMS workbook, then type 'continue'.")
            return state

        if t in {"cancel", "no", "n"}:
            state["ram_wizard"] = {"active": False, "step": "machine"}
            _wizard_reply(state, "Cancelled. Type 'create input sheet' to start again.")
            return state

        if t not in {"yes", "y"}:
            _wizard_reply(state, "Please respond with: yes | upload new | cancel")
            return state

        wiz["step"] = "categories_edit"
        if not wiz.get("categories"):
            model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
            cats = ai_propose_components_coarse(wiz.get("machine") or "", model=model)
            wiz["categories"] = cats
            wiz["categories_last_ai"] = list(cats) if isinstance(cats, list) else cats

        _wizard_reply(
            state,
            "Coarse component categories:\n"
            f"{_format_numbered_categories(wiz['categories'])}\n\n"
            "Type 'accept' to confirm these categories.\n"
            "Or type an edit instruction (e.g., 'remove idler', 'add gearbox', 'rename idler to idler_set').\n"
            "Tip: type 'reset' to re-generate from AI."
        )
        return state

    def _go_to_maintenance_review():
        cats = wiz.get("categories") or []
        default_code = wiz.get("maintenance_practice", 0) or 0
        practices = _build_default_practices(cats, default_code)
        wiz["component_practices"] = practices
        wiz["step"] = "maintenance_review"
        default_name = _PRACTICE_CODE_TO_NAME.get(default_code, "Reactive")
        _wizard_reply(
            state,
            f"Categories accepted. Default maintenance practice: {default_name}",
            wizard_ui={
                "type": "maintenance_table",
                "categories": cats,
                "practices": practices,
                "legend": {str(k): v for k, v in _PRACTICE_CODE_TO_NAME.items()},
            },
        )
        _wizard_reply(state, "Happy with these maintenance practices? (y/n)")

    _ACCEPT_KEYWORDS = {"", "accept", "ok", "okay", "proceed", "continue", "done",
                        "y", "yes", "confirm", "looks good", "lgtm"}

    if wiz["step"] == "categories_edit":
        if user_lower in _ACCEPT_KEYWORDS:
            _go_to_maintenance_review()
            return state

        if user_lower == "reset":
            model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
            cats = ai_propose_components_coarse(wiz.get("machine") or "", model=model)
            wiz["categories"] = cats
            wiz["categories_last_ai"] = list(cats) if isinstance(cats, list) else cats
            _wizard_reply(
                state,
                "Categories reset from AI:\n"
                f"{_format_numbered_categories(wiz['categories'])}\n\n"
                "Type 'accept' to confirm, or type an edit instruction."
            )
            return state

        old_cats = list(wiz.get("categories") or [])
        model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
        try:
            wiz["categories"] = ai_apply_edit_to_components(
                wiz.get("categories") or [], user_text_stripped, model=model
            )
        except Exception as e:
            _wizard_reply(state, f"Edit failed: {type(e).__name__}: {e}\nTry a simpler edit (e.g., 'remove idler').")
            return state

        _wizard_reply(
            state,
            "Updated categories:\n"
            f"{_format_numbered_categories(wiz['categories'])}\n\n"
            "Happy with these? Type 'accept' to confirm, or type another edit."
        )
        wiz["step"] = "categories_confirm_or_edit"
        return state

    if wiz["step"] == "categories_confirm_or_edit":
        t = user_lower

        if t in _ACCEPT_KEYWORDS:
            _go_to_maintenance_review()
            return state

        if t in {"n", "no"}:
            wiz["step"] = "categories_edit"
            _wizard_reply(
                state,
                "Okay — type an edit instruction, or 'accept' to confirm:\n"
                f"{_format_numbered_categories(wiz['categories'])}"
            )
            return state

        if t == "reset":
            model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
            cats = ai_propose_components_coarse(wiz.get("machine") or "", model=model)
            wiz["categories"] = cats
            wiz["categories_last_ai"] = list(cats) if isinstance(cats, list) else cats
            _wizard_reply(
                state,
                "Categories reset from AI:\n"
                f"{_format_numbered_categories(wiz['categories'])}\n\n"
                "Happy with these? Type 'accept' to confirm, or type another edit."
            )
            return state

        model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
        try:
            wiz["categories"] = ai_apply_edit_to_components(
                wiz.get("categories") or [], user_text_stripped, model=model
            )
        except Exception as e:
            _wizard_reply(state, f"Edit failed: {type(e).__name__}: {e}\nTry a simpler edit (e.g., 'remove idler').")
            return state

        _wizard_reply(
            state,
            "Updated categories:\n"
            f"{_format_numbered_categories(wiz['categories'])}\n\n"
            "Happy with these? Type 'accept' to confirm, or type another edit."
        )
        return state

    if wiz["step"] == "maintenance_review":
        t = user_lower
        if t in {"y", "yes"}:
            wiz["step"] = "confirm_create"
            _wizard_reply(state, "Great.\n\nReady to create the RAM input sheet. Type 'yes' to proceed or 'no' to cancel.")
            return state
        if t in {"n", "no"}:
            wiz["step"] = "maintenance_edit"
            cats = wiz.get("categories") or []
            practices = wiz.get("component_practices") or {}
            _wizard_reply(
                state,
                "Edit maintenance practices using the table below, then click Save.",
                wizard_ui={
                    "type": "maintenance_table",
                    "categories": cats,
                    "practices": practices,
                    "legend": {str(k): v for k, v in _PRACTICE_CODE_TO_NAME.items()},
                    "editable": True,
                },
            )
            return state
        _wizard_reply(state, "Please type 'y' (accept) or 'n' (edit practices).")
        return state

    if wiz["step"] == "maintenance_edit":
        cats = wiz.get("categories") or []
        practices = dict(wiz.get("component_practices") or {})
        raw = user_text_stripped

        # Parse "all: Practice" or "all: 2"
        if raw.lower().startswith("all:"):
            val = raw[4:].strip().lower()
            if val.isdigit() and int(val) in _PRACTICE_CODE_TO_NAME:
                code = int(val)
            else:
                code = _PRACTICE_NAME_TO_CODE.get(val)
            if code is None:
                _wizard_reply(
                    state,
                    f"Unknown practice '{raw[4:].strip()}'. "
                    "Use: 0 (Reactive), 1 (Corrective), 2 (Preventative), 3 (Condition based)."
                )
                return state
            for c in cats:
                practices[c] = code
        else:
            # Parse "1:0, 3:2" or "1: Preventative, 3: Condition based"
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            errors = []
            for part in parts:
                if ":" not in part:
                    errors.append(f"'{part}' -- expected format 'number:practice'")
                    continue
                num_str, pval = part.split(":", 1)
                num_str = num_str.strip()
                pval = pval.strip().lower()
                if not num_str.isdigit():
                    errors.append(f"'{num_str}' is not a valid component number")
                    continue
                idx = int(num_str) - 1
                if idx < 0 or idx >= len(cats):
                    errors.append(f"Component {num_str} is out of range (1-{len(cats)})")
                    continue
                if pval.isdigit() and int(pval) in _PRACTICE_CODE_TO_NAME:
                    code = int(pval)
                else:
                    code = _PRACTICE_NAME_TO_CODE.get(pval)
                if code is None:
                    errors.append(f"Unknown practice '{pval}'. Use: 0-3 or name (Reactive, Corrective, Preventative, Condition based)")
                    continue
                practices[cats[idx]] = code
            if errors:
                _wizard_reply(state, "Could not parse some edits:\n" + "\n".join(f"  - {e}" for e in errors))
                return state

        wiz["component_practices"] = practices
        wiz["step"] = "maintenance_review"
        _wizard_reply(
            state,
            "Updated maintenance practices.",
            wizard_ui={
                "type": "maintenance_table",
                "categories": cats,
                "practices": practices,
                "legend": {str(k): v for k, v in _PRACTICE_CODE_TO_NAME.items()},
            },
        )
        _wizard_reply(state, "Happy with these? (y/n)")
        return state

    if wiz["step"] == "confirm_create":
        ans = user_lower
        if ans in {"no", "n", "cancel"}:
            state["ram_wizard"] = {"active": False, "step": "machine"}
            _wizard_reply(state, "Cancelled. Type 'create input sheet' to start again.")
            return state
        if ans not in {"yes", "y"}:
            _wizard_reply(state, "Please type 'yes' to proceed or 'no' to cancel.")
            return state

        try:
            result = run_ram_pipeline_compat(
                input_xlsx_path=wiz.get("excel_path"),
                machine_type=wiz.get("machine"),
                date_range_text=wiz.get("date_range_text"),
                preferred_categories=wiz.get("categories"),
                maintenance_practice=wiz.get("maintenance_practice"),
                component_practices=wiz.get("component_practices"),
            )
        except Exception as e:
            _wizard_reply(state, f"Input-sheet pipeline failed: {type(e).__name__}: {e}")
            return state

        outputs = result.get("outputs") or {}
        ram_input_path = (
            outputs.get("latest_path")
            or outputs.get("input_path")
            or result.get("ram_input_path")
            or result.get("output_path")
            or result.get("latest_copy_path")
        )

        wiz["ram_input_path"] = ram_input_path
        wiz["classified_path"] = outputs.get("classified_path")
        wiz["readiness_payload"] = outputs.get("readiness")
        wiz["ram_input_ready_for_review"] = True

        sheet_data = {}
        try:
            sheet_data = _read_input_sheets(ram_input_path)
        except Exception:
            pass

        _wizard_reply(
            state,
            "RAM input sheet created.\n"
            f"- source workbook: {wiz.get('source_artifact_title') or wiz.get('excel_path')}\n"
            f"- RAM input path: {ram_input_path}\n\n"
            "Reply with:\n"
            "- simulate (continue with this generated input sheet)\n"
            "- edit (review and edit the input sheet in chat)\n"
            "- upload (use a previously saved input sheet instead)",
            wizard_ui={
                "type": "input_sheet_editor",
                "sheets": sheet_data,
                "editable": False,
            } if sheet_data else None,
        )
        wiz["step"] = "input_review"
        return state

    if wiz["step"] == "input_review":
        if user_lower in {"simulate", "run sim", "run simulation"}:
            d0, d1 = _default_dates_from_input_sheet(wiz.get("ram_input_path") or "") if wiz.get("ram_input_path") else (None, None)
            wiz["_sim_default_start"] = str(d0) if d0 else None
            wiz["_sim_default_end"] = str(d1) if d1 else None
            if d0:
                _wizard_reply(state, f"Enter simulation START date (YYYY-MM-DD). Default: {d0} (type 'default')")
            else:
                _wizard_reply(state, "Enter simulation START date (YYYY-MM-DD).")
            wiz["step"] = "sim_start"
            return state

        if user_lower in {"view & edit", "view and edit", "edit", "view", "review"}:
            xlsx = wiz.get("ram_input_path") or ""
            sheet_data = {}
            try:
                sheet_data = _read_input_sheets(xlsx)
            except Exception as e:
                _wizard_reply(state, f"Could not read input sheet for editing: {e}")
                return state
            wiz["step"] = "input_sheet_edit"
            _wizard_reply(
                state,
                "Edit the input sheet below. Click Save when done.",
                wizard_ui={
                    "type": "input_sheet_editor",
                    "sheets": sheet_data,
                    "editable": True,
                },
            )
            return state

        if user_lower in {"upload", "upload file", "upload sheet"}:
            wiz["step"] = "awaiting_upload_input"
            _wizard_reply(
                state,
                "Upload your RAM input sheet (.xlsx) using the file upload button.\n"
                "It will replace the current generated sheet."
            )
            return state

        _wizard_reply(state, "Reply with 'simulate', 'edit', or 'upload'.")
        return state

    if wiz["step"] == "input_sheet_edit":
        if user_text_stripped.startswith("__SHEET_SAVE__:"):
            json_str = user_text_stripped[len("__SHEET_SAVE__:"):]
            try:
                payload = json.loads(json_str)
                sheets_data = payload.get("sheets") or payload
                xlsx = wiz.get("ram_input_path") or ""
                _write_input_sheets(xlsx, sheets_data)
            except Exception as e:
                wiz["step"] = "input_review"
                _wizard_reply(
                    state,
                    f"Failed to save edits: {type(e).__name__}: {e}\n\n"
                    "You can:\n"
                    "- edit (try editing again)\n"
                    "- upload (upload a saved input sheet instead)\n"
                    "- simulate (proceed with the original sheet)"
                )
                return state

            wiz["step"] = "input_review"
            _wizard_reply(
                state,
                "Input sheet updated successfully.\n\n"
                "Reply with:\n"
                "- simulate (continue with this input sheet)\n"
                "- edit (make more changes)\n"
                "- upload (use a different input sheet)"
            )
            return state

        _wizard_reply(state, "Use the table editor above to make changes, then click Save.")
        return state

    if wiz["step"] == "awaiting_upload_input":
        artifacts = state.get("conversation_artifacts") or []
        xlsx_artifacts = [
            a for a in artifacts
            if (a.get("mime_type") or "").startswith("application/vnd.openxmlformats")
            or (a.get("title") or "").lower().endswith(".xlsx")
        ]
        latest = xlsx_artifacts[-1] if xlsx_artifacts else None
        if latest and latest.get("storage_key"):
            uploaded_path = latest["storage_key"]
            wiz["ram_input_path"] = uploaded_path
            sheet_data = {}
            try:
                sheet_data = _read_input_sheets(uploaded_path)
            except Exception:
                pass
            wiz["step"] = "input_review"
            _wizard_reply(
                state,
                f"Using uploaded input sheet: {latest.get('title', uploaded_path)}\n\n"
                "Reply with:\n"
                "- simulate (continue with this input sheet)\n"
                "- edit (review and edit the input sheet in chat)",
                wizard_ui={
                    "type": "input_sheet_editor",
                    "sheets": sheet_data,
                    "editable": False,
                } if sheet_data else None,
            )
            return state

        _wizard_reply(
            state,
            "No uploaded input sheet found. Please upload an .xlsx file using the file upload button,\n"
            "or type 'edit' to go back to editing, or 'simulate' to use the current sheet."
        )
        if user_lower in {"edit", "simulate"}:
            wiz["step"] = "input_review"
        return state

    if wiz["step"] == "sim_confirm":
        ans = user_lower
        if ans in {"no", "n"}:
            wiz["active"] = False
            wiz["step"] = "done"
            _wizard_reply(state, "Okay — input sheet is ready. You can run simulation later. (Use /ram status to see paths.)")
            return state
        if ans not in {"yes", "y"}:
            _wizard_reply(state, "Please answer 'yes' or 'no'.")
            return state

        d0, d1 = _default_dates_from_input_sheet(wiz.get("ram_input_path") or "") if wiz.get("ram_input_path") else (None, None)
        wiz["_sim_default_start"] = str(d0) if d0 else None
        wiz["_sim_default_end"] = str(d1) if d1 else None

        if d0:
            _wizard_reply(state, f"Enter simulation START date (YYYY-MM-DD). Default: {d0} (type 'default')")
        else:
            _wizard_reply(state, "Enter simulation START date (YYYY-MM-DD).")
        wiz["step"] = "sim_start"
        return state

    if wiz["step"] == "sim_start":
        t = user_text_stripped
        if t.lower() == "default" and wiz.get("_sim_default_start"):
            t = wiz["_sim_default_start"]
        try:
            _parse_date_yyyy_mm_dd(t)
        except Exception:
            _wizard_reply(state, "Please enter a valid START date in format YYYY-MM-DD (or type 'default' if shown).")
            return state
        wiz["sim_start"] = t

        if wiz.get("_sim_default_end"):
            _wizard_reply(state, f"Enter simulation END date (YYYY-MM-DD). Default: {wiz['_sim_default_end']} (type 'default')")
        else:
            _wizard_reply(state, "Enter simulation END date (YYYY-MM-DD).")
        wiz["step"] = "sim_end"
        return state

    if wiz["step"] == "sim_end":
        t = user_text_stripped
        if t.lower() == "default" and wiz.get("_sim_default_end"):
            t = wiz["_sim_default_end"]
        try:
            _parse_date_yyyy_mm_dd(t)
        except Exception:
            _wizard_reply(state, "Please enter a valid END date in format YYYY-MM-DD (or type 'default' if shown).")
            return state
        wiz["sim_end"] = t

        _wizard_reply(state, "How many Monte Carlo simulations? Default: 2 (type a number)")
        wiz["step"] = "sim_sims"
        return state

    if wiz["step"] == "sim_sims":
        t = user_text_stripped or "2"
        try:
            sims = int(t)
            if sims <= 0:
                raise ValueError
        except Exception:
            _wizard_reply(state, "Please enter a positive integer (e.g., 2).")
            return state
        wiz["simulations"] = sims
        wiz["step"] = "sim_run"
        _wizard_reply(state, "Running simulation + archiving results...")

    if wiz["step"] == "sim_run":
        import time as _sim_time
        _sim_wall_start = _sim_time.perf_counter()
        try:
            start = _parse_date_yyyy_mm_dd(wiz.get("sim_start") or "")
            end = _parse_date_yyyy_mm_dd(wiz.get("sim_end") or "")
            sims = int(wiz.get("simulations") or 2)

            conv_id = state.get("_conversation_id") or ""
            def _progress_cb(info: dict) -> None:
                if _sim_progress_store is not None and conv_id:
                    _sim_progress_store[conv_id] = info

            archive = run_ram_simulation_archived(
                input_xlsx=wiz.get("ram_input_path") or "",
                start_date=start,
                end_date=end,
                simulations=sims,
                agg=os.environ.get("RAM_AGG", "50th_perc"),
                opp_dt_ind=int(os.environ.get("RAM_OPP_DT", "0")),
                spare_ind=int(os.environ.get("RAM_SPARES", "0")),
                out_root=os.environ.get("RAM_RUNS_DIR", "ram_runs"),
                machine_label=wiz.get("machine"),
                progress_callback=_progress_cb,
            )
        except Exception as e:
            _wizard_reply(
                state,
                "Simulation failed.\n"
                f"Error: {type(e).__name__}: {e}"
            )
            return state

        _sim_wall_elapsed = _sim_time.perf_counter() - _sim_wall_start

        wiz["sim_archive_dir"] = archive.run_dir
        wiz["sim_metadata_path"] = archive.metadata_path
        wiz["sim_outputs"] = archive.outputs
        wiz["sim_conditions"] = archive.conditions
        wiz["active"] = False
        wiz["step"] = "done"

        period_days = (end - start).days
        period_str = f"{period_days} days (~{period_days // 365} years)"

        available_plots = [
            {"id": "availability", "label": "Availability", "description": "System availability % over time"},
            {"id": "failures", "label": "Failures", "description": "Average yearly failures by component"},
            {"id": "costs_by_component", "label": "Costs by Component", "description": "Average yearly maintenance cost by component"},
            {"id": "costs_over_time", "label": "Costs over Time", "description": "Monthly maintenance cost over time"},
            {"id": "downtime_by_component", "label": "Downtime by Component", "description": "Average yearly downtime by component"},
            {"id": "downtime_over_time", "label": "Downtime over Time", "description": "Monthly total downtime over time"},
        ]

        _wizard_reply(
            state,
            "Simulation complete.\n"
            f"- simulations: {sims}\n"
            f"- period: {start} to {end} ({period_str})\n"
            f"- total time: {_sim_wall_elapsed:.1f}s\n"
            f"- saved outputs: {archive.summary.get('saved_outputs_count')}\n"
            f"- saved condition tables: {archive.summary.get('saved_conditions_count')}\n\n"
            "Select a plot below to visualise the results.",
            wizard_ui={
                "type": "sim_plots_menu",
                "plots": available_plots,
            },
        )
        return state

    _wizard_reply(state, "Wizard complete. Type 'create input sheet' to start a new run, or ask a question with '?'.")
    return state


# -------------------------
# Graph build
# -------------------------
def get_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", node_router)
    builder.add_node("qa", node_qa)
    builder.add_node("ram_wizard", node_ram_wizard)

    builder.set_entry_point("router")

    def _route(state: AgentState) -> str:
        return state.get("intent", "qa")

    builder.add_conditional_edges("router", _route, {"qa": "qa", "ram_wizard": "ram_wizard"})
    builder.add_edge("qa", END)
    builder.add_edge("ram_wizard", END)

    use_sql = os.environ.get("USE_SQL_CHECKPOINTER", "1") == "1"
    if use_sql and SqliteSaver is not None:
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    else:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)