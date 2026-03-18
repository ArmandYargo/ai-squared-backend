from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional


class DEMScenario(TypedDict, total=False):
    name: str
    description: str
    enabled: bool
    mass_flow_tph: Optional[float]
    belt_speed_mps: Optional[float]
    notes: str


class DEMArtifact(TypedDict, total=False):
    title: str
    output_type: str
    path: str
    mime_type: str
    available: bool
    meta: Dict[str, Any]


class DEMProjectState(TypedDict, total=False):
    # Basic project info
    active: bool
    step: str
    project_id: str
    project_name: str
    user_goal: str

    # Geometry / CAD
    geometry_source: str
    geometry_files: List[str]
    geometry_units: str
    geometry_validated: bool
    geometry_notes: str

    # Material definition
    material_name: str
    material_properties: Dict[str, Any]

    # Scenario setup
    scenarios: List[DEMScenario]

    # Run state
    run_requested: bool
    run_status: str
    progress: Dict[str, Any]

    # Results
    results_summary: str
    recommendations: List[str]

    # Artifacts
    artifacts: List[DEMArtifact]

    # UI / assistant control
    next_action: str
    last_user_message: str
    last_assistant_message: str


class DEMAgentState(TypedDict, total=False):
    """
    This is the DEM module's local working state.
    It can be embedded later inside your shared app state as:
        state["dem"] = {...}
    """
    messages: List[Dict[str, Any]]
    intent: str
    dem: DEMProjectState
    route_meta: Dict[str, Any]
    _conversation_id: str