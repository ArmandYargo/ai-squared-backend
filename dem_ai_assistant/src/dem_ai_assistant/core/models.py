from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class PSDBand(BaseModel):
    size_min_mm: float
    size_max_mm: float
    mass_fraction: float


class MaterialConfig(BaseModel):
    name: str
    bulk_density_kg_m3: float
    friction_particle_particle: float
    friction_particle_wall: float
    restitution: float
    rolling_resistance: float
    cohesion_pa: float
    psd_bands: List[PSDBand]


class BeltConfig(BaseModel):
    width_mm: float
    speed_m_s: float
    trough_angle_deg: float
    transition_length_mm: float


class ChuteConfig(BaseModel):
    hood_angle_deg: float
    spoon_angle_deg: float
    skirt_spacing_mm: float
    lateral_offset_mm: float = 0.0


class EgressZoneConfig(BaseModel):
    left_limit_mm: float
    right_limit_mm: float
    top_limit_mm: float


class GeometryConfig(BaseModel):
    transfer_point_name: str
    cad_file: Optional[str] = None
    belt: BeltConfig
    chute: ChuteConfig
    egress_zone: EgressZoneConfig


class FeedProfile(BaseModel):
    mode: str
    throughput_tph: float
    surge_multiplier: float = 1.0


class RestartProfile(BaseModel):
    enabled: bool = False
    stop_time_s: float = 0.0
    restart_ramp_s: float = 0.0


class ScenarioConfig(BaseModel):
    scenario_name: str
    duration_s: float
    timestep_s: float
    feed_profile: FeedProfile
    restart_profile: RestartProfile


class SimulationResult(BaseModel):
    run_id: str
    output_dir: Path
    status: str
    scenario_name: str
    notes: str = ""
    raw_metrics: dict = Field(default_factory=dict)


class KPIResult(BaseModel):
    egress_events: int
    max_impact_energy_j: float
    mean_edge_margin_mm: float
    min_edge_margin_mm: float
    throughput_proxy_tph: float
    stability_score: float


class EngineeringFinding(BaseModel):
    severity: str
    title: str
    detail: str