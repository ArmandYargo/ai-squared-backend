from __future__ import annotations

from typing import Any, Dict


def build_default_dem_project() -> Dict[str, Any]:
    return {
        "active": True,
        "step": "project_setup",
        "project_id": None,
        "project_name": None,
        "user_goal": None,
        "application_type": None,
        "geometry_source": None,
        "geometry_files": [],
        "geometry_units": None,
        "geometry_validated": False,
        "geometry_notes": "",
        "material_name": None,
        "material_properties": {},
        "scenarios": [],
        "run_requested": False,
        "run_status": "not_started",
        "progress": {},
        "results_summary": "",
        "recommendations": [],
        "artifacts": [],
        "next_action": "awaiting_user_input",
        "last_user_message": "",
        "last_assistant_message": "",
    }


def apply_extracted_entities(dem: Dict[str, Any], entities: Dict[str, Any]) -> Dict[str, Any]:
    if not dem.get("project_name") and entities.get("project_name"):
        dem["project_name"] = entities["project_name"]

    if not dem.get("material_name") and entities.get("material_name"):
        dem["material_name"] = entities["material_name"]

    if entities.get("geometry_hint") and not dem.get("geometry_notes"):
        dem["geometry_notes"] = f"Initial geometry hint: {entities['geometry_hint']}"

    return dem


def ensure_default_scenarios(dem: Dict[str, Any]) -> Dict[str, Any]:
    if dem.get("scenarios"):
        return dem

    dem["scenarios"] = [
        {
            "name": "Nominal Operation",
            "description": "Baseline operating condition for first DEM setup.",
            "enabled": True,
            "notes": "Default starter scenario",
        },
        {
            "name": "Surge Case",
            "description": "Higher loading or upset feed condition.",
            "enabled": False,
            "notes": "Enable later if required",
        },
    ]
    return dem


def build_dem_setup_wizard() -> Dict[str, Any]:
    return {
        "type": "dem_project_setup",
        "editable": True,
        "fields": [
            {"key": "project_name", "label": "Project name", "type": "text"},
            {"key": "application_type", "label": "Application type", "type": "text"},
            {"key": "geometry_source", "label": "Geometry source", "type": "text"},
            {"key": "material_name", "label": "Bulk material", "type": "text"},
            {"key": "target_question", "label": "Main engineering question", "type": "textarea"},
        ],
    }


def build_dem_scenario_wizard() -> Dict[str, Any]:
    return {
        "type": "dem_scenario_setup",
        "editable": True,
        "fields": [
            {"key": "scenario_name", "label": "Scenario name", "type": "text"},
            {"key": "geometry_type", "label": "Geometry type", "type": "text"},
            {"key": "operating_case", "label": "Operating case", "type": "text"},
            {"key": "mass_flow_tph", "label": "Mass flow (tph)", "type": "text"},
            {"key": "belt_speed_mps", "label": "Belt speed (m/s)", "type": "text"},
            {"key": "geometry_units", "label": "Geometry units", "type": "text"},
            {"key": "notes", "label": "Scenario notes", "type": "textarea"},
        ],
    }