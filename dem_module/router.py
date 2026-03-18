from __future__ import annotations

from typing import Any, Dict

from dem_module.intents import classify_dem_intent, extract_dem_entities


def route_dem_request(user_text: str) -> Dict[str, Any]:
    """
    Lightweight DEM-side routing.
    This stays inside the DEM module and decides what internal step
    the DEM assistant should move to.
    """
    intent = classify_dem_intent(user_text)
    entities = extract_dem_entities(user_text)

    if intent == "dem_geometry":
        next_step = "geometry_definition"
    elif intent == "dem_run":
        next_step = "run_request_review"
    elif intent == "dem_results":
        next_step = "results_review"
    elif intent == "dem_setup":
        next_step = "project_setup"
    else:
        next_step = "project_setup"

    return {
        "intent": intent,
        "entities": entities,
        "next_step": next_step,
    }