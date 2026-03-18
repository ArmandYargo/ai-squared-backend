from __future__ import annotations

from typing import Any, Dict, List


def build_dem_config(dem: Dict[str, Any]) -> Dict[str, Any]:
    scenario = (dem.get("scenarios") or [{}])[0]
    geometry = dem.get("geometry_definition") or {}

    material_name = dem.get("material_name") or "unspecified_material"
    project_name = dem.get("project_name") or "DEM Project"
    application_type = dem.get("application_type") or "general_dem"
    user_goal = dem.get("user_goal") or "Create DEM model"

    geometry_units = (
        geometry.get("geometry_units")
        or dem.get("geometry_units")
        or "mm"
    )

    geometry_mode = geometry.get("geometry_mode") or dem.get("geometry_source") or "parametric"

    config: Dict[str, Any] = {
        "project": {
            "name": project_name,
            "application_type": application_type,
            "user_goal": user_goal,
        },
        "material": {
            "name": material_name,
            "properties": dem.get("material_properties") or {},
        },
        "geometry": {
            "mode": geometry_mode,
            "units": geometry_units,
            "source": dem.get("geometry_source"),
            "files": dem.get("geometry_files") or [],
            "definition": {
                "transfer_length_mm": geometry.get("transfer_length_mm"),
                "transfer_width_mm": geometry.get("transfer_width_mm"),
                "drop_height_mm": geometry.get("drop_height_mm"),
                "wall_angle_deg": geometry.get("wall_angle_deg"),
                "outlet_width_mm": geometry.get("outlet_width_mm"),
                "notes": geometry.get("notes") or dem.get("geometry_notes") or "",
            },
        },
        "scenario": {
            "name": scenario.get("name") or "Scenario 1",
            "geometry_type": scenario.get("geometry_type"),
            "operating_case": scenario.get("operating_case"),
            "mass_flow_tph": scenario.get("mass_flow_tph"),
            "belt_speed_mps": scenario.get("belt_speed_mps"),
            "notes": scenario.get("notes") or "",
        },
        "solver": {
            "engine": "pychrono",
            "gravity_mps2": 9.81,
            "time_step_s": 1e-3,
            "duration_s": 5.0,
            "broadphase": "auto",
            "collision_model": "spheres_initial",
        },
        "particles": {
            "shape_model": "sphere",
            "size_distribution": [],
            "density_kgm3": None,
            "count_estimate": None,
        },
        "contacts": {
            "normal_restitution": None,
            "sliding_friction": None,
            "rolling_friction": None,
            "wall_friction": None,
        },
        "outputs": {
            "save_run_log": True,
            "save_summary": True,
            "save_config_json": True,
            "save_preview_data": True,
        },
    }

    return config


def summarise_dem_config(config: Dict[str, Any]) -> str:
    project = config.get("project") or {}
    material = config.get("material") or {}
    geometry = config.get("geometry") or {}
    scenario = config.get("scenario") or {}
    solver = config.get("solver") or {}

    definition = geometry.get("definition") or {}

    return (
        "DEM configuration object created.\n\n"
        f"Project: {project.get('name') or 'Not set'}\n"
        f"Application type: {project.get('application_type') or 'Not set'}\n"
        f"Material: {material.get('name') or 'Not set'}\n"
        f"Geometry mode: {geometry.get('mode') or 'Not set'}\n"
        f"Geometry units: {geometry.get('units') or 'Not set'}\n"
        f"Scenario: {scenario.get('name') or 'Not set'}\n"
        f"Operating case: {scenario.get('operating_case') or 'Not set'}\n"
        f"Mass flow (tph): {scenario.get('mass_flow_tph') if scenario.get('mass_flow_tph') is not None else 'Not set'}\n"
        f"Belt speed (m/s): {scenario.get('belt_speed_mps') if scenario.get('belt_speed_mps') is not None else 'Not set'}\n"
        f"Transfer length (mm): {definition.get('transfer_length_mm') if definition.get('transfer_length_mm') is not None else 'Not set'}\n"
        f"Transfer width (mm): {definition.get('transfer_width_mm') if definition.get('transfer_width_mm') is not None else 'Not set'}\n"
        f"Drop height (mm): {definition.get('drop_height_mm') if definition.get('drop_height_mm') is not None else 'Not set'}\n"
        f"Solver engine: {solver.get('engine') or 'Not set'}\n\n"
        "Next step: connect this config to either a CAD importer or a first PyChrono simulation stub."
    )