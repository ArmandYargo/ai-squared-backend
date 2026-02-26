# agent/state.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    intent: str

    # RAM run outputs (paths, metrics, etc.)
    ram: Dict[str, Any]

    # RAG (optional)
    rag_hits: List[Dict[str, Any]]

    # Wizard state for interactive RAM flow
    ram_wizard: Dict[str, Any]