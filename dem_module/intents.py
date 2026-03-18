from __future__ import annotations

import re
from typing import Dict, Any


DEM_COMMAND_PREFIXES = ("/dem",)

DEM_PATTERNS = [
    r"\bdem\b",
    r"\bdiscrete\s+element\b",
    r"\bpychrono\b",
    r"\bparticle\s+flow\b",
    r"\bbulk\s+solids?\b",
    r"\bchute\b",
    r"\btransfer\s+chute\b",
    r"\btransfer\s+point\b",
    r"\bconveyor\s+transition\b",
    r"\bmaterial\s+flow\b",
    r"\bimpact\s+zone\b",
    r"\bcarryback\b",
    r"\bblockage\b",
]

DEM_SETUP_PATTERNS = [
    r"\bset\s*up\b.*\bdem\b",
    r"\bcreate\b.*\bdem\b",
    r"\bbuild\b.*\bdem\b",
    r"\bstart\b.*\bdem\b",
    r"\bmodel\b.*\btransfer\b",
    r"\bmodel\b.*\bchute\b",
    r"\bvisuali[sz]e\b.*\bdem\b",
]

DEM_RUN_PATTERNS = [
    r"\brun\b.*\bdem\b",
    r"\bsimulate\b.*\bdem\b",
    r"\bstart\b.*\bsimulation\b",
    r"\bexecute\b.*\bdem\b",
]

DEM_RESULTS_PATTERNS = [
    r"\bresults?\b",
    r"\bshow\b.*\bresults?\b",
    r"\bsummary\b",
    r"\brecommendations?\b",
    r"\bfindings\b",
]

DEM_GEOMETRY_PATTERNS = [
    r"\bcad\b",
    r"\bgeometry\b",
    r"\bstep\s+file\b",
    r"\bstl\b",
    r"\bobj\b",
    r"\bimport\b.*\bgeometry\b",
]


def is_dem_related(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    if any(t.startswith(prefix) for prefix in DEM_COMMAND_PREFIXES):
        return True

    for pat in DEM_PATTERNS + DEM_SETUP_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return True

    return False


def classify_dem_intent(text: str) -> str:
    """
    Returns one of:
    - dem_setup
    - dem_geometry
    - dem_run
    - dem_results
    - dem_general
    """
    t = (text or "").strip().lower()
    if not t:
        return "dem_general"

    if any(t.startswith(prefix) for prefix in DEM_COMMAND_PREFIXES):
        return "dem_setup"

    for pat in DEM_RUN_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return "dem_run"

    for pat in DEM_GEOMETRY_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return "dem_geometry"

    for pat in DEM_RESULTS_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return "dem_results"

    for pat in DEM_SETUP_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return "dem_setup"

    if is_dem_related(t):
        return "dem_general"

    return "dem_general"


def extract_dem_entities(text: str) -> Dict[str, Any]:
    """
    Lightweight extraction only.
    Safe and simple for now.
    """
    t = (text or "").strip()

    out: Dict[str, Any] = {
        "project_name": None,
        "material_name": None,
        "geometry_hint": None,
    }

    project_match = re.search(
        r"(?:project|called|named)\s+([A-Za-z0-9_\- ][A-Za-z0-9_\- ]{1,60})",
        t,
        flags=re.I,
    )
    if project_match:
        out["project_name"] = project_match.group(1).strip(" .,")

    material_match = re.search(
        r"\b(iron ore|coal|copper ore|bauxite|gravel|sand|ore|rock)\b",
        t,
        flags=re.I,
    )
    if material_match:
        out["material_name"] = material_match.group(1).strip()

    geometry_match = re.search(
        r"\b(chute|transfer point|conveyor transition|hopper|bin|belt)\b",
        t,
        flags=re.I,
    )
    if geometry_match:
        out["geometry_hint"] = geometry_match.group(1).strip()

    return out