from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.dem_ai_assistant.core.models import GeometryConfig, MaterialConfig, ScenarioConfig


class SimulationInput(BaseModel):
    material_name: str
    scenario_name: str
    transfer_point_name: str
    belt_speed_m_s: float
    throughput_tph: float
    transition_length_mm: float
    trough_angle_deg: float
    hood_angle_deg: float
    spoon_angle_deg: float
    skirt_spacing_mm: float
    lateral_offset_mm: float
    cohesion_pa: float
    friction_particle_particle: float
    friction_particle_wall: float
    restitution: float
    rolling_resistance: float
    duration_s: float
    timestep_s: float
    restart_enabled: bool
    # Absolute or project-relative path to collision mesh (from geometry cad_file).
    cad_mesh_path: Optional[str] = None
    cad_mesh_scale: float = 1.0


def _resolve_mesh_path(cad_file: str, project_root: Optional[Path]) -> str:
    p = Path(cad_file)
    if p.is_absolute():
        return str(p.resolve())
    if project_root is not None:
        return str((project_root / p).resolve())
    return str((Path.cwd() / p).resolve())


def build_simulation_input(
    material_config: MaterialConfig,
    scenario_config: ScenarioConfig,
    geometry_config: GeometryConfig,
    project_root: Optional[Path] = None,
) -> SimulationInput:
    cad_mesh_path: Optional[str] = None
    if geometry_config.cad_file:
        cad_mesh_path = _resolve_mesh_path(geometry_config.cad_file, project_root)

    return SimulationInput(
        material_name=material_config.name,
        scenario_name=scenario_config.scenario_name,
        transfer_point_name=geometry_config.transfer_point_name,
        belt_speed_m_s=geometry_config.belt.speed_m_s,
        throughput_tph=scenario_config.feed_profile.throughput_tph,
        transition_length_mm=geometry_config.belt.transition_length_mm,
        trough_angle_deg=geometry_config.belt.trough_angle_deg,
        hood_angle_deg=geometry_config.chute.hood_angle_deg,
        spoon_angle_deg=geometry_config.chute.spoon_angle_deg,
        skirt_spacing_mm=geometry_config.chute.skirt_spacing_mm,
        lateral_offset_mm=geometry_config.chute.lateral_offset_mm,
        cohesion_pa=material_config.cohesion_pa,
        friction_particle_particle=material_config.friction_particle_particle,
        friction_particle_wall=material_config.friction_particle_wall,
        restitution=material_config.restitution,
        rolling_resistance=material_config.rolling_resistance,
        duration_s=scenario_config.duration_s,
        timestep_s=scenario_config.timestep_s,
        restart_enabled=scenario_config.restart_profile.enabled,
        cad_mesh_path=cad_mesh_path,
        cad_mesh_scale=geometry_config.cad_mesh_scale,
    )