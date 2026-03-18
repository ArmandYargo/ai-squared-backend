from __future__ import annotations

from typing import Any, Dict, List

from dem_module.prompts import (
    DEM_GEOMETRY_PROMPT,
    DEM_INTRO_PROMPT,
    DEM_RESULTS_NOT_READY_PROMPT,
    DEM_RUN_NOT_READY_PROMPT,
    build_project_summary_prompt,
)
from dem_module.router import route_dem_request
from dem_module.state import DEMAgentState
from dem_module.services.project_service import (
    apply_extracted_entities,
    build_default_dem_project,
    build_dem_project_snapshot,
    build_dem_setup_wizard,
    ensure_default_scenarios,
)


def _ensure_dem_state(state: DEMAgentState) -> Dict[str, Any]:
    dem = state.get("dem")
    if not isinstance(dem, dict):
        dem = build_default_dem_project()
    else:
        base = build_default_dem_project()
        base.update(dem)
        dem = base

    state["dem"] = dem
    return dem


def _append_assistant_message(
    state: DEMAgentState,
    text: str,
    speaker: str = "DEM_AGENT",
    wizard_ui: Dict[str, Any] | None = None,
) -> None:
    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": text,
        "speaker": speaker,
    }
    if wizard_ui:
        msg["wizard_ui"] = wizard_ui

    state.setdefault("messages", []).append(msg)

    dem = _ensure_dem_state(state)
    dem["last_assistant_message"] = text


def _latest_user_text(state: DEMAgentState) -> str:
    msgs: List[Dict[str, Any]] = state.get("messages") or []
    if not msgs:
        return ""
    return (msgs[-1].get("content") or "").strip()


def handle_dem_turn(state: DEMAgentState) -> DEMAgentState:
    dem = _ensure_dem_state(state)

    user_text = _latest_user_text(state)
    dem["last_user_message"] = user_text

    route = route_dem_request(user_text)
    dem["step"] = route["next_step"]

    entities = route["entities"]
    dem = apply_extracted_entities(dem, entities)
    dem = ensure_default_scenarios(dem)

    intent = route["intent"]

    if intent == "dem_run":
        dem["run_requested"] = True
        reply = DEM_RUN_NOT_READY_PROMPT
        _append_assistant_message(state, reply)
        return state

    if intent == "dem_geometry":
        reply = DEM_GEOMETRY_PROMPT
        _append_assistant_message(state, reply)
        return state

    if intent == "dem_results":
        reply = DEM_RESULTS_NOT_READY_PROMPT
        _append_assistant_message(state, reply)
        return state

    if intent in {"dem_setup", "dem_general"}:
        if not dem.get("project_name"):
            dem["project_name"] = "DEM Project"

        dem["user_goal"] = user_text or dem.get("user_goal") or "Create a DEM model"

        summary = build_project_summary_prompt(
            project_name=dem.get("project_name"),
            geometry_hint=entities.get("geometry_hint"),
            material_name=dem.get("material_name"),
            user_goal=dem.get("user_goal"),
        )

        reply = DEM_INTRO_PROMPT + "\n\n" + summary
        _append_assistant_message(
            state,
            reply,
            wizard_ui=build_dem_setup_wizard(),
        )
        return state

    _append_assistant_message(
        state,
        "DEM module is active, but I need a bit more information to continue. "
        "Please describe the conveyor transfer, chute, hopper, or geometry you want to model."
    )
    return state