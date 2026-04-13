"""
Shared PyChrono import, materials, and CAD mesh loading (Chrono 9/10 compatible).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from src.dem_ai_assistant.simulation.input_builder import SimulationInput


def import_chrono() -> Any:
    try:
        import pychrono.core as chrono  # type: ignore[import-untyped]
    except ImportError:
        import pychrono as chrono  # type: ignore[import-untyped]

    if not hasattr(chrono, "ChSystemNSC"):
        raise RuntimeError(
            "Invalid PyChrono module (missing ChSystemNSC). Install: "
            "conda install projectchrono::pychrono -c conda-forge"
        )
    return chrono


def make_contact_material(chrono: Any, sim: SimulationInput) -> Any:
    if hasattr(chrono, "ChContactMaterialNSC"):
        mat = chrono.ChContactMaterialNSC()
    else:
        mat = chrono.ChMaterialSurfaceNSC()
    mu = max(
        0.05,
        min(0.95, 0.5 * (sim.friction_particle_wall + sim.friction_particle_particle)),
    )
    if hasattr(mat, "SetFriction"):
        mat.SetFriction(mu)
    if hasattr(mat, "SetRestitution"):
        mat.SetRestitution(max(0.0, min(0.95, sim.restitution)))
    cohesion_norm = min(sim.cohesion_pa / 4000.0, 1.0)
    if hasattr(mat, "SetCompliance"):
        mat.SetCompliance(1e-6 + 5e-5 * cohesion_norm)
    elif hasattr(mat, "SetComplianceNormal"):
        mat.SetComplianceNormal(1e-6 + 5e-5 * cohesion_norm)
    return mat


def configure_collision_system(chrono: Any, system: Any) -> None:
    if hasattr(chrono, "ChCollisionSystem") and hasattr(
        chrono.ChCollisionSystem, "Type_BULLET"
    ):
        try:
            system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        except Exception:
            pass
    if hasattr(chrono, "ChCollisionModel"):
        try:
            chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.002)
            chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.002)
        except Exception:
            pass


def vec3(chrono: Any, x: float, y: float, z: float) -> Any:
    if hasattr(chrono, "ChVector3d"):
        return chrono.ChVector3d(x, y, z)
    return chrono.ChVectorD(x, y, z)


def _scale_mesh_uniform(chrono: Any, mesh: Any, scale: float) -> None:
    if abs(scale - 1.0) < 1e-9:
        return
    origin = vec3(chrono, 0.0, 0.0, 0.0)
    if hasattr(chrono, "ChMatrix33d"):
        m33 = chrono.ChMatrix33d()
        try:
            if hasattr(m33, "FillDiag"):
                m33.FillDiag(vec3(chrono, scale, scale, scale))
                mesh.Transform(origin, m33)
            elif hasattr(m33, "SetElement"):
                for i in range(3):
                    for j in range(3):
                        m33.SetElement(i, j, scale if i == j else 0.0)
                mesh.Transform(origin, m33)
        except Exception:
            pass


def add_fixed_mesh_body(
    chrono: Any,
    system: Any,
    mesh_path: Path,
    material: Any,
    density: float,
    scale: float,
    position: Tuple[float, float, float],
) -> Any:
    """
    Load a triangle mesh as a fixed rigid boundary (visual + collision).
    Prefer watertight .obj; .stl supported when LoadSTLMesh exists.
    """
    path = mesh_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = chrono.ChTriangleMeshConnected()
    suffix = path.suffix.lower()
    loaded = False
    if suffix == ".obj":
        loaded = bool(mesh.LoadWavefrontMesh(str(path)))
    elif suffix == ".stl":
        if hasattr(mesh, "LoadSTLMesh"):
            loaded = bool(mesh.LoadSTLMesh(str(path)))
        elif hasattr(mesh, "LoadSTLFile"):
            loaded = bool(mesh.LoadSTLFile(str(path)))

    if not loaded:
        raise RuntimeError(
            f"Could not load mesh '{path}'. Export CAD to watertight .obj or binary/ASCII .stl "
            "(metres in the file; use cad_mesh_scale in YAML if you export in mm)."
        )

    _scale_mesh_uniform(chrono, mesh, max(1e-9, float(scale)))

    body = chrono.ChBodyAuxRef()
    body.SetMass(float(density))
    body.SetInertiaXX(vec3(chrono, 1.0, 1.0, 1.0))
    if hasattr(body, "SetFixed"):
        body.SetFixed(True)
    elif hasattr(body, "SetBodyFixed"):
        body.SetBodyFixed(True)
    px, py, pz = position
    body.SetPos(vec3(chrono, px, py, pz))

    if hasattr(chrono, "ChVisualShapeTriangleMesh"):
        vshape = chrono.ChVisualShapeTriangleMesh()
        vshape.SetMesh(mesh)
        body.AddVisualShape(vshape)

    if hasattr(chrono, "ChCollisionShapeTriangleMesh"):
        cshape = chrono.ChCollisionShapeTriangleMesh(material, mesh, False, False)
        body.AddCollisionShape(cshape)
    if hasattr(body, "EnableCollision"):
        body.EnableCollision(True)

    system.Add(body)
    return body
