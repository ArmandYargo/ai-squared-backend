from __future__ import annotations


DEM_INTRO_PROMPT = """\
DEM module is active.

I can help you build a DEM workflow for PyChrono-based bulk material modelling, including:
- geometry definition
- bulk material setup
- scenario creation
- simulation planning
- visualisation
- engineering recommendations

To begin, I need:
- the transfer or chute geometry
- the bulk material
- the operating scenario you want to investigate
- the main engineering question you want answered

You can either describe the system in plain English or upload geometry/CAD files.
"""


DEM_GEOMETRY_PROMPT = """\
Geometry setup noted.

For the first DEM version, we should define one of these:
- conveyor transfer chute
- conveyor transition
- hopper / bin discharge
- simple impact zone

Please tell me:
1. what geometry type this is,
2. whether you already have CAD,
3. what file format you have available,
4. what units the geometry uses.
"""


DEM_RUN_NOT_READY_PROMPT = """\
DEM module is active, but the simulation runner is not connected yet.

Before we run anything, I need a valid setup with:
- geometry source or CAD file
- bulk material definition
- scenario definition
- solver settings for the first test case

Next step: tell me what geometry you want to model, or upload the CAD/mesh file.
"""


DEM_RESULTS_NOT_READY_PROMPT = """\
There are no DEM results yet because the simulation pipeline has not been run.

Once the runner is connected, this module will summarise:
- particle flow behaviour
- impact and wear hot spots
- retention / blockage risk areas
- transfer performance observations
- engineering recommendations
"""


def build_project_summary_prompt(
    project_name: str | None,
    geometry_hint: str | None,
    material_name: str | None,
    user_goal: str | None,
) -> str:
    return (
        "DEM project draft created.\n\n"
        f"Project name: {project_name or 'Not set'}\n"
        f"Geometry type: {geometry_hint or 'Not set'}\n"
        f"Bulk material: {material_name or 'Not set'}\n"
        f"Engineering goal: {user_goal or 'Not set'}\n\n"
        "Next, I need you to confirm or provide:\n"
        "- geometry source (CAD, mesh, or simple parametric model)\n"
        "- material properties or at least a material name\n"
        "- the operating case you want to simulate first"
    )