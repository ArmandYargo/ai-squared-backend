from pydantic import BaseModel, Field


class ParticleStats(BaseModel):
    total_particles: int
    coarse_particles: int
    fines_fraction: float


class FlowStats(BaseModel):
    throughput_tph: float
    estimated_spill_events: int
    estimated_recirculation_index: float


class EdgeStats(BaseModel):
    mean_edge_margin_mm: float
    min_edge_margin_mm: float
    left_edge_bias: float
    right_edge_bias: float


class ImpactStats(BaseModel):
    max_impact_energy_j: float
    p95_impact_energy_j: float


class SimulationRawOutput(BaseModel):
    particle_stats: ParticleStats
    flow_stats: FlowStats
    edge_stats: EdgeStats
    impact_stats: ImpactStats
    stability_score: float
    notes: str = ""
    metadata: dict = Field(default_factory=dict)