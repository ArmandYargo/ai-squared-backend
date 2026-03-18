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
from dem_module.services.config_service import build_dem_config, summarise_dem_config
from dem_module.services.project_service import (
    apply_extracted_entities,
    build_default_dem_project,
    build_dem_geometry_wizard,
    build_dem_scenario_wizard,
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


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_float(value: Any) -> float | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_dem_setup_submission(user_text: str) -> bool:
    return user_text.startswith("__DEM_SETUP__:")


def _is_dem_scenario_submission(user_text: str) -> bool:
    return user_text.startswith("__DEM_SCENARIO__:")


def _is_dem_geometry_submission(user_text: str) -> bool:
    return user_text.startswith("__DEM_GEOMETRY__:")


def _parse_json_submission(prefix: str, user_text: str) -> Dict[str, Any]:
    raw = user_text[len(prefix):].strip()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Submission payload must be a JSON object.")
    return payload


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
    dem["next_action"] = "awaiting_scenario_definition"
    return dem


def _apply_dem_scenario_submission(dem: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    values = payload.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("DEM scenario payload is missing a valid 'values' object.")

    scenario = {
        "name": _clean_str(values.get("scenario_name")) or "Scenario 1",
        "description": _clean_str(values.get("operating_case")) or "User-defined operating case",
        "enabled": True,
        "geometry_type": _clean_str(values.get("geometry_type")),
        "operating_case": _clean_str(values.get("operating_case")),
        "mass_flow_tph": _clean_float(values.get("mass_flow_tph")),
        "belt_speed_mps": _clean_float(values.get("belt_speed_mps")),
        "geometry_units": _clean_str(values.get("geometry_units")),
        "notes": _clean_str(values.get("notes")) or "",
    }

    existing = dem.get("scenarios") or []
    dem["scenarios"] = [scenario] + existing

    if scenario.get("geometry_units"):
        dem["geometry_units"] = scenario["geometry_units"]

    if scenario.get("geometry_type"):
        dem["geometry_notes"] = f"Geometry type: {scenario['geometry_type']}"

    dem["step"] = "scenario_saved"
    dem["next_action"] = "awaiting_geometry_definition"
    return dem


def _apply_dem_geometry_submission(dem: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    values = payload.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("DEM geometry payload is missing a valid 'values' object.")

    geometry_definition = {
        "geometry_mode": _clean_str(values.get("geometry_mode")),
        "transfer_length_mm": _clean_float(values.get("transfer_length_mm")),
        "transfer_width_mm": _clean_float(values.get("transfer_width_mm")),
        "drop_height_mm": _clean_float(values.get("drop_height_mm")),
        "wall_angle_deg": _clean_float(values.get("wall_angle_deg")),
        "outlet_width_mm": _clean_float(values.get("outlet_width_mm")),
        "notes": _clean_str(values.get("notes")) or "",
    }

    dem["geometry_definition"] = geometry_definition
    dem["step"] = "geometry_saved"
    dem["next_action"] = "ready_for_config_build"

    if geometry_definition.get("notes"):
        dem["geometry_notes"] = geometry_definition["notes"]

    dem["config"] = build_dem_config(dem)

    return dem


def _build_setup_saved_reply(dem: Dict[str, Any]) -> str:
    return (
        "DEM setup saved.\n\n"
        f"Project name: {dem.get('project_name') or 'Not set'}\n"
        f"Application type: {dem.get('application_type') or 'Not set'}\n"
        f"Geometry source: {dem.get('geometry_source') or 'Not set'}\n"
        f"Bulk material: {dem.get('material_name') or 'Not set'}\n"
        f"Main engineering question: {dem.get('user_goal') or 'Not set'}\n\n"
        "Next, define the first simulation scenario."
    )


def _build_scenario_saved_reply(dem: Dict[str, Any]) -> str:
    scenario = (dem.get("scenarios") or [{}])[0]
    return (
        "DEM scenario saved.\n\n"
        f"Scenario name: {scenario.get('name') or 'Not set'}\n"
        f"Geometry type: {scenario.get('geometry_type') or 'Not set'}\n"
        f"Operating case: {scenario.get('operating_case') or 'Not set'}\n"
        f"Mass flow: {scenario.get('mass_flow_tph') if scenario.get('mass_flow_tph') is not None else 'Not set'}\n"
        f"Belt speed: {scenario.get('belt_speed_mps') if scenario.get('belt_speed_mps') is not None else 'Not set'}\n"
        f"Geometry units: {scenario.get('geometry_units') or 'Not set'}\n\n"
        "Next, define the transfer geometry."
    )


def _build_geometry_saved_reply(dem: Dict[str, Any]) -> str:
    base = (
        "DEM geometry saved.\n\n"
        "The DEM intake workflow now has:\n"
        "- project setup\n"
        "- first scenario\n"
        "- geometry definition\n\n"
    )
    config_summary = summarise_dem_config(dem.get("config") or {})
    return base + config_summary


def handle_dem_turn(state: DEMAgentState) -> DEMAgentState:
    dem = _ensure_dem_state(state)

    user_text = _latest_user_text(state)
    dem["last_user_message"] = user_text

    if _is_dem_setup_submission(user_text):
        try:
            payload = _parse_json_submission("__DEM_SETUP__:", user_text)
            dem = _apply_dem_setup_submission(dem, payload)
            dem = ensure_default_scenarios(dem)
            _append_assistant_message(
                state,
                _build_setup_saved_reply(dem),
                wizard_ui=build_dem_scenario_wizard(),
            )
            return state
        except Exception as exc:
            _append_assistant_message(
                state,
                f"I could not save the DEM setup form because the payload was invalid. Error: {exc}"
            )
            return state

    if _is_dem_scenario_submission(user_text):
        try:
            payload = _parse_json_submission("__DEM_SCENARIO__:", user_text)
            dem = _apply_dem_scenario_submission(dem, payload)
            _append_assistant_message(
                state,
                _build_scenario_saved_reply(dem),
                wizard_ui=build_dem_geometry_wizard(),
            )
            return state
        except Exception as exc:
            _append_assistant_message(
                state,
                f"I could not save the DEM scenario form because the payload was invalid. Error: {exc}"
            )
            return state

    if _is_dem_geometry_submission(user_text):
        try:
            payload = _parse_json_submission("__DEM_GEOMETRY__:", user_text)
            dem = _apply_dem_geometry_submission(dem, payload)
            _append_assistant_message(state, _build_geometry_saved_reply(dem))
            return state
        except Exception as exc:
            _append_assistant_message(
                state,
                f"I could not save the DEM geometry form because the payload was invalid. Error: {exc}"
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
        _append_assistant_message(state, DEM_RUN_NOT_READY_PROMPT)
        return state

    if intent == "dem_geometry":
        _append_assistant_message(state, DEM_GEOMETRY_PROMPT, wizard_ui=build_dem_geometry_wizard())
        return state

    if intent == "dem_results":
        _append_assistant_message(state, DEM_RESULTS_NOT_READY_PROMPT)
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

        _append_assistant_message(
            state,
            DEM_INTRO_PROMPT + "\n\n" + summary,
            wizard_ui=build_dem_setup_wizard(),
        )
        return state

    _append_assistant_message(
        state,
        "DEM module is active, but I need a bit more information to continue. "
        "Please describe the conveyor transfer, chute, hopper, or geometry you want to model."
    )
    return state