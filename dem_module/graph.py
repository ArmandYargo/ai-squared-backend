from __future__ import annotations

import json
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


def _is_dem_setup_submission(user_text: str) -> bool:
    return user_text.startswith("__DEM_SETUP__:")


def _parse_dem_setup_submission(user_text: str) -> Dict[str, Any]:
    raw = user_text[len("__DEM_SETUP__:") :].strip()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("DEM setup payload must be a JSON object.")
    return payload


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _apply_dem_setup_submission(dem: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    values = payload.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("DEM setup payload is missing a valid 'values' object.")

    project_name = _clean_str(values.get("project_name"))
    application_type = _clean_str(values.get("application_type"))
    geometry_source = _clean_str(values.get("geometry_source"))
    material_name = _clean_str(values.get("material_name"))
    target_question = _clean_str(values.get("target_question"))

    if project_name:
        dem["project_name"] = project_name

    if application_type:
        dem["application_type"] = application_type

    if geometry_source:
        dem["geometry_source"] = geometry_source

    if material_name:
        dem["material_name"] = material_name

    if target_question:
        dem["user_goal"] = target_question

    dem["step"] = "setup_saved"
    dem["next_action"] = "awaiting_geometry_and_scenario_detail"

    return dem


def _build_setup_saved_reply(dem: Dict[str, Any]) -> str:
    return (
        "DEM setup saved.\n\n"
        f"Project name: {dem.get('project_name') or 'Not set'}\n"
        f"Application type: {dem.get('application_type') or 'Not set'}\n"
        f"Geometry source: {dem.get('geometry_source') or 'Not set'}\n"
        f"Bulk material: {dem.get('material_name') or 'Not set'}\n"
        f"Main engineering question: {dem.get('user_goal') or 'Not set'}\n\n"
        "Next, I need the first simulation case definition.\n"
        "Please tell me:\n"
        "- what geometry type this is (transfer chute, conveyor transition, hopper, etc.)\n"
        "- what operating case you want to simulate first\n"
        "- whether you already have CAD/mesh files ready to upload\n"
        "- the geometry units if known"
    )


def handle_dem_turn(state: DEMAgentState) -> DEMAgentState:
    dem = _ensure_dem_state(state)

    user_text = _latest_user_text(state)
    dem["last_user_message"] = user_text

    if _is_dem_setup_submission(user_text):
        try:
            payload = _parse_dem_setup_submission(user_text)
            dem = _apply_dem_setup_submission(dem, payload)
            dem = ensure_default_scenarios(dem)
            reply = _build_setup_saved_reply(dem)
            _append_assistant_message(state, reply)
            return state
        except Exception as exc:
            _append_assistant_message(
                state,
                f"I could not save the DEM setup form because the payload was invalid. Error: {exc}"
            )
            return state

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