from __future__ import annotations
from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    intent: str

    # General artifact / document context
    artifact_context: str
    artifact_meta: Dict[str, Any]
    conversation_artifacts: List[Dict[str, Any]]

    # RAM run outputs / metadata
    ram: Dict[str, Any]

    # DEM run outputs / metadata
    dem: Dict[str, Any]

    # RAG (optional)
    rag_hits: List[Dict[str, Any]]

    # Wizard state for interactive RAM flow
    ram_wizard: Dict[str, Any]
    ram_command: str
    route_meta: Dict[str, Any]

    # Runtime: conversation ID for progress tracking (set by app_main before invoke)
    _conversation_id: str