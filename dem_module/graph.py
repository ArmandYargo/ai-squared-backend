from __future__ import annotations

from typing import Any, Dict, List

from dem_module.intents import classify_dem_intent, extract_dem_entities
from dem_module.state import DEMAgentState


def _ensure_dem_state(state: DEMAgentState) -> Dict[str, Any]:
    dem = state.get("dem")
    if not isinstance(dem, dict):
        dem = {}

    dem.setdefault("active", True)
    dem.setdefault("step", "project_setup")
    dem.setdefault("project_name", None)
    dem.setdefault("user_goal", None)
    dem.setdefault("geometry_source", None)
    dem.setdefault("geometry_files", [])
    dem.setdefault("geometry_units", None)
    dem.setdefault("geometry_validated", False)
    dem.setdefault("geometry_notes", "")
    dem.setdefault("material_name", None)
    dem.setdefault("material_properties", {})
    dem.setdefault("scenarios", [])
    dem.setdefault("run_requested", False)
    dem.setdefault("run_status", "not_started")
    dem.setdefault("progress", {})
    dem.setdefault("results_summary", "")
    dem.setdefault("recommendations", [])
    dem.setdefault("artifacts", [])
    dem.setdefault("next_action", "awaiting_user_input")
    dem.setdefault("last_user_message", "")
    dem.setdefault("last_assistant_message", "")

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


def _default_setup_wizard() -> Dict[str, Any]:
    """
    This matches your frontend pattern where assistant messages can carry wizard_ui.
    You can extend this later.
    """
    return {
        "type": "dem_project_setup",
        "editable": False,
        "fields": [
            {"key": "project_name", "label": "Project name", "type": "text"},
            {"key": "application_type", "label": "Application type", "type": "text"},
            {"key": "geometry_source", "label": "Geometry source", "type": "text"},
            {"key": "material_name", "label": "Bulk material", "type": "text"},
            {"key": "target_question", "label": "Main engineering question", "type": "textarea"},
        ],
    }


def handle_dem_turn(state: DEMAgentState) -> DEMAgentState:
    """
    Minimal DEM handler.

    Goal for this first version:
    - If routed here, the user gets a DEM-specific response
    - State is initialised safely
    - We start collecting project information
    """
    dem = _ensure_dem_state(state)

    user_text = _latest_user_text(state)
    dem["last_user_message"] = user_text
    dem_intent = classify_dem_intent(user_text)
    entities = extract_dem_entities(user_text)

    if not dem.get("project_name") and entities.get("project_name"):
        dem["project_name"] = entities["project_name"]

    if not dem.get("material_name") and entities.get("material_name"):
        dem["material_name"] = entities["material_name"]

    if dem_intent == "dem_run":
        dem["step"] = "run_request_review"
        dem["run_requested"] = True
        reply = (
            "DEM module is active, but the simulation runner is not connected yet.\n\n"
            "Before we run anything, I need a valid setup with:\n"
            "- geometry source or CAD file\n"
            "- bulk material definition\n"
            "- scenario definition\n"
            "- solver settings for the first test case\n\n"
            "Next step: tell me what geometry you want to model, or upload the CAD/mesh file."
        )
        _append_assistant_message(state, reply)
        return state

    if dem_intent == "dem_geometry":
        dem["step"] = "geometry_definition"
        reply = (
            "Geometry setup noted.\n\n"
            "For the first DEM version, we should define one of these:\n"
            "- conveyor transfer chute\n"
            "- conveyor transition\n"
            "- hopper / bin discharge\n"
            "- simple impact zone\n\n"
            "Please tell me:\n"
            "1. what geometry type this is,\n"
            "2. whether you have CAD already,\n"
            "3. what file format you have available."
        )
        _append_assistant_message(state, reply)
        return state

    if dem_intent == "dem_results":
        dem["step"] = "results_review"
        reply = (
            "There are no DEM results yet because the simulation pipeline has not been run.\n\n"
            "Once the runner is connected, this module will summarise:\n"
            "- particle flow behaviour\n"
            "- impact and wear hot spots\n"
            "- retention / blockage risk areas\n"
            "- transfer performance observations\n"
            "- engineering recommendations"
        )
        _append_assistant_message(state, reply)
        return state

    if dem_intent in {"dem_setup", "dem_general"}:
        dem["step"] = "project_setup"
        if not dem.get("project_name"):
            dem["project_name"] = "DEM Project"

        dem["user_goal"] = user_text or dem.get("user_goal") or "Create a DEM model"

        reply = (
            "DEM module is active.\n\n"
            "I can help you build a DEM workflow for PyChrono-based bulk material modelling, "
            "including geometry definition, material setup, scenario creation, simulation planning, "
            "visualisation, and engineering recommendations.\n\n"
            "To start properly, I need:\n"
            "- the transfer or chute geometry\n"
            "- the bulk material\n"
            "- the operating scenario you want to investigate\n"
            "- the main engineering question you want answered\n\n"
            "You can either describe the system in plain English or upload geometry/CAD files."
        )
        _append_assistant_message(state, reply, wizard_ui=_default_setup_wizard())
        return state

    _append_assistant_message(
        state,
        "DEM module is active, but I need a bit more information to continue. "
        "Please describe the conveyor transfer, chute, hopper, or geometry you want to model."
    )
    return state