# agent/graph.py
import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:
    SqliteSaver = None

from agent.state import AgentState
from agent.rag import RagStore
from agent.ram_tool import run_ram_pipeline_compat, check_ram_readiness

# category helpers live in ram_module funcs (already in sys.path via ram_tool.py)
from func_define_components import ai_propose_components_coarse, ai_apply_edit_to_components

from agent.ram_simulation_tool import run_ram_simulation_archived


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
        messages=messages,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"hi", "hello", "hey", "howzit", "morning", "good morning", "good afternoon", "good evening"}


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
            r"^(what|why|how|when|where|who|which|can you|could you|do you|is it|are you)\b",
            t.lower(),
        )
    )


def _route_intent_ai(user_text: str, wiz_active: bool) -> dict:
    system = (
        "You are an intent router for a reliability engineering assistant for mines/factories.\n"
        "Choose the single best intent for the user's message.\n\n"
        "Allowed intents:\n"
        "- qa: user is asking a question or wants explanation/advice\n"
        "- ram_wizard: user wants to create/continue the RAM input sheet workflow, provide required wizard input, "
        "or proceed with simulation steps.\n\n"
        "Rules:\n"
        "- If the user asks a question about regulations/laws/definitions/concepts, choose qa.\n"
        "- If the user is supplying a requested field (machine name, date range, file path) while wizard is active, choose ram_wizard.\n"
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
# RAG store singleton (lazy)
# -------------------------
_RAG_STORE: Optional[RagStore] = None


def get_rag_store() -> RagStore:
    global _RAG_STORE
    if _RAG_STORE is None:
        _RAG_STORE = RagStore()
    return _RAG_STORE


def _rag_search(store: RagStore, query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Compatibility layer: your RagStore API may not be named 'search'.

    We try common method names in order and normalize to:
      [{"text": ..., "distance": ..., ...}, ...]
    """
    if hasattr(store, "search") and callable(getattr(store, "search")):
        return store.search(query, k=k)  # type: ignore

    for name in ("query", "retrieve", "similarity_search", "search_chunks", "search_similar"):
        if hasattr(store, name) and callable(getattr(store, name)):
            fn = getattr(store, name)
            try:
                res = fn(query, k=k)
            except TypeError:
                res = fn(query, top_k=k)
            return res

    raise AttributeError(
        "RagStore has no supported retrieval method. "
        "Expected one of: search/query/retrieve/similarity_search/search_chunks/search_similar"
    )


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

    # Explicit QA shortcuts
    if t.startswith("?") or t.lower().startswith("/ask "):
        state["intent"] = "qa"
        if t.startswith("?"):
            state["messages"][-1]["content"] = t[1:].strip()
        else:
            state["messages"][-1]["content"] = t[5:].strip()
        return state

    # RAM commands
    if t.lower().startswith("/ram"):
        parts = t.split(maxsplit=1)
        cmd = parts[1].strip().lower() if len(parts) > 1 else ""
        state["ram_command"] = cmd
        state["intent"] = "ram_wizard"
        return state

    # Wizard active: keep flow, allow questions
    if wiz_active:
        state["intent"] = "qa" if _looks_like_question(t) or _is_greeting(t) else "ram_wizard"
        return state

    # Wizard inactive: ask the LLM to decide
    try:
        route = _route_intent_ai(t, wiz_active=False)
    except Exception:
        route = {}

    if route:
        state["intent"] = route["intent"]
        state["route_meta"] = {"confidence": route.get("confidence"), "reason": route.get("reason")}
        return state

    state["intent"] = "qa"
    return state


# -------------------------
# QA node
# -------------------------
RAG_MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.4"))


def node_qa(state: AgentState) -> AgentState:
    msgs = state.get("messages") or []
    if not msgs:
        state.setdefault("messages", []).append({"role": "assistant", "content": "How can I help?", "speaker": "LLM"})
        return state

    question = msgs[-1].get("content", "").strip()
    if not question:
        state.setdefault("messages", []).append(
            {"role": "assistant", "content": "Please enter a question.", "speaker": "LLM"}
        )
        return state

    store = get_rag_store()
    hits = _rag_search(store, question, k=6) or []

    # Optional confidence filter if the store returns distances
    if hits:
        d0 = hits[0].get("distance", None)
        if d0 is not None:
            try:
                if float(d0) > RAG_MAX_DISTANCE:
                    hits = []
            except Exception:
                pass

    if hits:
        context = "\n\n".join([f"[{i}] {h.get('text','')}" for i, h in enumerate(hits, start=1)])
        system = (
            "You answer ONLY using the provided knowledge base context. "
            "If the answer is not in the context, respond exactly with:\n"
            "Not found in the knowledge base."
        )
        answer = _llm_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"},
            ]
        ).strip()

        if answer == "Not found in the knowledge base.":
            answer2 = _llm_text(
                [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
            ).strip()
            state.setdefault("messages", []).append({"role": "assistant", "content": answer2, "speaker": "LLM"})
        else:
            state.setdefault("messages", []).append({"role": "assistant", "content": answer, "speaker": "RAG"})
        return state

    # fallback
    answer = _llm_text(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
    ).strip()
    state.setdefault("messages", []).append({"role": "assistant", "content": answer, "speaker": "LLM"})
    return state


# -------------------------
# RAM Wizard + Simulation
# -------------------------
def _wizard_reply(state: AgentState, text: str) -> None:
    state.setdefault("messages", []).append({"role": "assistant", "content": text, "speaker": "WIZARD"})


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


def _pick_excel_file_dialog() -> Optional[str]:
    """
    Try to open a native file picker dialog (Windows/macOS/Linux).
    Returns a selected file path, or None if cancelled/unavailable.
    """
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
    subset_rows = readiness_payload.get("subset_rows")
    readiness = readiness_payload.get("readiness") or {}
    ok = readiness.get("ok_to_simulate", None)

    interesting_keys = [
        "ok_to_simulate",
        "coverage",
        "mapping_coverage",
        "reason",
        "notes",
        "missing_required_fields",
        "issues",
        "warnings",
    ]
    lines = []
    lines.append(f"- Rows matched for machine: {subset_rows}")
    lines.append(f"- OK to simulate?: {ok}")

    for k in interesting_keys:
        if k in readiness and k not in {"ok_to_simulate"}:
            v = readiness.get(k)
            if v not in (None, "", [], {}, False):
                lines.append(f"- {k}: {v}")

    return "Data Readiness Check:\n" + "\n".join(lines)


def _format_numbered_categories(cats: List[str]) -> str:
    if not cats:
        return "(no categories)"
    return "\n".join([f"{i+1}. {c}" for i, c in enumerate(cats)])


def node_ram_wizard(state: AgentState) -> AgentState:
    wiz = state.get("ram_wizard") or {}
    state["ram_wizard"] = wiz

    wiz.setdefault("active", False)
    wiz.setdefault("step", "machine")

    wiz.setdefault("machine", None)
    wiz.setdefault("date_range_text", None)
    wiz.setdefault("excel_path", None)

    # categories state
    wiz.setdefault("categories", None)           # current working list
    wiz.setdefault("categories_last_ai", None)   # last AI-proposed list (optional)

    # readiness gate state
    wiz.setdefault("readiness_payload", None)

    wiz.setdefault("ram_input_path", None)
    wiz.setdefault("sim_start", None)
    wiz.setdefault("sim_end", None)
    wiz.setdefault("simulations", None)
    wiz.setdefault("sim_archive_dir", None)

    # ---------- RAM commands ----------
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
                f"- ram_input_path: {wiz.get('ram_input_path')}\n"
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

    # ---------- start wizard ----------
    if not wiz.get("active"):
        wiz["active"] = True
        wiz["step"] = "machine"
        _wizard_reply(state, "RAM Input Sheet Wizard started.\nWhat machine are we working on? (e.g., 'Conveyor CV-101')")
        return state

    user_text = (state.get("messages")[-1].get("content", "") if state.get("messages") else "")
    # NOTE: DO NOT strip here globally; we need to detect blank ENTER at categories step.
    user_text_stripped = user_text.strip()

    # ---------- machine ----------
    if wiz["step"] == "machine":
        if not user_text_stripped:
            _wizard_reply(state, "Please enter the machine name/type.")
            return state
        wiz["machine"] = user_text_stripped
        wiz["step"] = "date"
        _wizard_reply(state, "Optional: enter a date range (e.g., '2019-01-01 to 2021-12-31' or '2023-2024') or type 'skip'.")
        return state

    # ---------- date ----------
    if wiz["step"] == "date":
        if not user_text_stripped:
            _wizard_reply(state, "Enter a date range or type 'skip'.")
            return state
        wiz["date_range_text"] = None if user_text_stripped.lower() in {"skip", "none", "no"} else user_text_stripped
        wiz["step"] = "file"
        _wizard_reply(state, "Please provide the path to the CMMS Excel file, or type 'pick file'.")
        return state

    # ---------- file ----------
    if wiz["step"] == "file":
        if not user_text_stripped:
            _wizard_reply(state, "Please provide an Excel file path, or type 'pick file'.")
            return state

        if user_text_stripped.lower() in {"pick file", "pick", "browse", "choose file", "select file"}:
            picked = _pick_excel_file_dialog()
            if not picked:
                _wizard_reply(
                    state,
                    "No file selected (cancelled). Please provide a path, or type 'pick file' again."
                )
                return state
            wiz["excel_path"] = picked
        else:
            wiz["excel_path"] = user_text_stripped

        # readiness check immediately after file selection
        _wizard_reply(state, f"Selected file:\n{wiz['excel_path']}\n\nRunning data readiness check...")
        try:
            payload = check_ram_readiness(
                excel_path=wiz["excel_path"],
                machine=wiz["machine"],
            )
        except Exception as e:
            _wizard_reply(state, f"Readiness check failed: {type(e).__name__}: {e}\n\nType 'pick file' to select another file or paste a new path.")
            wiz["step"] = "file"
            return state

        wiz["readiness_payload"] = payload
        _wizard_reply(
            state,
            _format_readiness_summary(payload)
            + "\n\nProceed with this file?\n"
              "- yes (continue)\n"
              "- pick file (choose another file)\n"
              "- cancel (stop wizard)"
        )
        wiz["step"] = "readiness_confirm"
        return state

    # ---------- readiness confirm ----------
    if wiz["step"] == "readiness_confirm":
        t = user_text_stripped.lower()

        if t in {"pick file", "pick", "pick new file", "new file", "choose file", "browse"}:
            wiz["excel_path"] = None
            wiz["readiness_payload"] = None
            wiz["step"] = "file"
            _wizard_reply(state, "Okay — select a new CMMS Excel file (type 'pick file' or paste the path).")
            return state

        if t in {"cancel", "no", "n"}:
            state["ram_wizard"] = {"active": False, "step": "machine"}
            _wizard_reply(state, "Cancelled. Type 'create input sheet' to start again.")
            return state

        if t not in {"yes", "y"}:
            _wizard_reply(state, "Please respond with: yes | pick file | cancel")
            return state

        # move to category proposal/edit loop
        wiz["step"] = "categories_edit"
        if not wiz.get("categories"):
            model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
            cats = ai_propose_components_coarse(wiz.get("machine"), model=model)
            wiz["categories"] = cats
            wiz["categories_last_ai"] = list(cats) if isinstance(cats, list) else cats

        _wizard_reply(
            state,
            "Coarse component categories:\n"
            f"{_format_numbered_categories(wiz['categories'])}\n\n"
            "Press ENTER to accept these categories.\n"
            "Or type an edit instruction (e.g., 'remove idler', 'add gearbox', 'rename idler to idler_set').\n"
            "Tip: type 'reset' to re-generate from AI."
        )
        return state

    # ---------- categories edit: ENTER accepts, otherwise edit ----------
    if wiz["step"] == "categories_edit":
        # Rule (1): ENTER means accept
        if user_text == "":
            wiz["step"] = "confirm_create"
            _wizard_reply(state, "Categories accepted.\n\nReady to create the RAM input sheet. Type 'yes' to proceed or 'no' to cancel.")
            return state

        # Allow reset as a convenience
        if user_text_stripped.lower() == "reset":
            model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
            cats = ai_propose_components_coarse(wiz.get("machine"), model=model)
            wiz["categories"] = cats
            wiz["categories_last_ai"] = list(cats) if isinstance(cats, list) else cats
            _wizard_reply(
                state,
                "Categories reset from AI:\n"
                f"{_format_numbered_categories(wiz['categories'])}\n\n"
                "Press ENTER to accept, or type another edit."
            )
            return state

        # Rule (2): anything else is an edit instruction (no "edit:" required)
        model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
        try:
            wiz["categories"] = ai_apply_edit_to_components(wiz.get("categories") or [], user_text_stripped, model=model)
        except Exception as e:
            _wizard_reply(state, f"Edit failed: {type(e).__name__}: {e}\nTry a simpler edit (e.g., 'remove idler').")
            return state

        # Show updated numbered list + prompt continue
        _wizard_reply(
            state,
            "Updated categories:\n"
            f"{_format_numbered_categories(wiz['categories'])}\n\n"
            "Continue with these categories? (Y/N)"
        )
        wiz["step"] = "categories_confirm_or_edit"
        return state

    # ---------- categories confirm: Y continues, N prompts for edit; other = treat as edit ----------
    if wiz["step"] == "categories_confirm_or_edit":
        t = user_text_stripped.lower()

        if t in {"y", "yes"}:
            wiz["step"] = "confirm_create"
            _wizard_reply(state, "Great.\n\nReady to create the RAM input sheet. Type 'yes' to proceed or 'no' to cancel.")
            return state

        if t in {"n", "no"}:
            wiz["step"] = "categories_edit"
            _wizard_reply(
                state,
                "Okay — enter another edit, or press ENTER to accept:\n"
                f"{_format_numbered_categories(wiz['categories'])}"
            )
            return state

        # Rule (4): anything else -> treat as another edit instruction on current list
        if t == "reset":
            model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
            cats = ai_propose_components_coarse(wiz.get("machine"), model=model)
            wiz["categories"] = cats
            wiz["categories_last_ai"] = list(cats) if isinstance(cats, list) else cats
            _wizard_reply(
                state,
                "Categories reset from AI:\n"
                f"{_format_numbered_categories(wiz['categories'])}\n\n"
                "Continue with these categories? (Y/N)"
            )
            # stay in confirm_or_edit so Y/N works after reset
            return state

        model = os.environ.get("RAM_CAT_MODEL", "gpt-5.2")
        try:
            wiz["categories"] = ai_apply_edit_to_components(wiz.get("categories") or [], user_text_stripped, model=model)
        except Exception as e:
            _wizard_reply(state, f"Edit failed: {type(e).__name__}: {e}\nTry a simpler edit (e.g., 'remove idler').")
            return state

        _wizard_reply(
            state,
            "Updated categories:\n"
            f"{_format_numbered_categories(wiz['categories'])}\n\n"
            "Continue with these categories? (Y/N)"
        )
        return state

    # ---------- confirm create ----------
    if wiz["step"] == "confirm_create":
        ans = user_text_stripped.lower()
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

        readiness = outputs.get("readiness") or {}
        ok_to_simulate = outputs.get("ok_to_simulate")

        _wizard_reply(
            state,
            "RAM input sheet created.\n"
            f"- ok_to_simulate: {ok_to_simulate}\n"
            f"- readiness: {readiness}\n"
            f"- RAM input path: {ram_input_path}\n"
        )

        wiz["step"] = "sim_confirm"
        _wizard_reply(state, "Do you want to proceed to simulate the RAM model now? (yes/no)")
        return state

    # ---------- sim confirm ----------
    if wiz["step"] == "sim_confirm":
        ans = user_text_stripped.lower()
        if ans in {"no", "n"}:
            wiz["active"] = False
            wiz["step"] = "done"
            _wizard_reply(state, "Okay — input sheet is ready. You can run simulation later. (Use /ram status to see paths.)")
            return state
        if ans not in {"yes", "y"}:
            _wizard_reply(state, "Please answer 'yes' or 'no'.")
            return state

        d0, d1 = _default_dates_from_input_sheet(wiz.get("ram_input_path")) if wiz.get("ram_input_path") else (None, None)
        wiz["_sim_default_start"] = str(d0) if d0 else None
        wiz["_sim_default_end"] = str(d1) if d1 else None

        if d0:
            _wizard_reply(state, f"Enter simulation START date (YYYY-MM-DD). Default: {d0} (type 'default')")
        else:
            _wizard_reply(state, "Enter simulation START date (YYYY-MM-DD).")
        wiz["step"] = "sim_start"
        return state

    # ---------- sim start ----------
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

    # ---------- sim end ----------
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

        _wizard_reply(state, "How many Monte Carlo simulations? Default: 200 (type a number)")
        wiz["step"] = "sim_sims"
        return state

    # ---------- sim sims ----------
    if wiz["step"] == "sim_sims":
        t = user_text_stripped or "200"
        try:
            sims = int(t)
            if sims <= 0:
                raise ValueError
        except Exception:
            _wizard_reply(state, "Please enter a positive integer (e.g., 200).")
            return state
        wiz["simulations"] = sims
        wiz["step"] = "sim_run"
        _wizard_reply(state, "Running simulation + archiving results...")
        # fall through

    # ---------- sim run ----------
    if wiz["step"] == "sim_run":
        try:
            start = _parse_date_yyyy_mm_dd(wiz.get("sim_start"))
            end = _parse_date_yyyy_mm_dd(wiz.get("sim_end"))
            sims = int(wiz.get("simulations") or 200)

            archive = run_ram_simulation_archived(
                input_xlsx=wiz.get("ram_input_path"),
                start_date=start,
                end_date=end,
                simulations=sims,
                agg=os.environ.get("RAM_AGG", "50th_perc"),
                opp_dt_ind=int(os.environ.get("RAM_OPP_DT", "0")),
                spare_ind=int(os.environ.get("RAM_SPARES", "0")),
                out_root=os.environ.get("RAM_RUNS_DIR", "ram_runs"),
                machine_label=wiz.get("machine"),
            )
        except Exception as e:
            _wizard_reply(
                state,
                "Simulation failed.\n"
                f"Error: {type(e).__name__}: {e}"
            )
            return state

        wiz["sim_archive_dir"] = archive.run_dir
        wiz["active"] = False
        wiz["step"] = "done"
        _wizard_reply(
            state,
            "Simulation complete ✅\n"
            f"- archived run dir: {archive.run_dir}\n"
            f"- metadata: {archive.metadata_path}\n"
            f"- saved outputs: {archive.summary.get('saved_outputs_count')}\n"
            f"- saved condition tables: {archive.summary.get('saved_conditions_count')}\n"
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
