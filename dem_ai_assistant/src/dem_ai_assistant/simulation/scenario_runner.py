from pathlib import Path
from uuid import uuid4

from src.dem_ai_assistant.core.models import (
    GeometryConfig,
    MaterialConfig,
    ScenarioConfig,
    SimulationResult,
)


def run_simulation_stub(
    material_config: MaterialConfig,
    scenario_config: ScenarioConfig,
    geometry_config: GeometryConfig,
    output_dir: Path,
) -> SimulationResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{scenario_config.scenario_name}_{uuid4().hex[:8]}"

    raw_metrics = {
        "material_name": material_config.name,
        "transfer_point_name": geometry_config.transfer_point_name,
        "belt_speed_m_s": geometry_config.belt.speed_m_s,
        "throughput_tph": scenario_config.feed_profile.throughput_tph,
        "estimated_egress_events": 7,
        "estimated_max_impact_energy_j": 42.5,
        "estimated_mean_edge_margin_mm": 85.0,
        "estimated_min_edge_margin_mm": 18.0,
        "stability_score": 0.61,
    }

    return SimulationResult(
        run_id=run_id,
        output_dir=output_dir,
        status="success",
        scenario_name=scenario_config.scenario_name,
        notes="Stub simulation only. No real DEM executed yet.",
        raw_metrics=raw_metrics,
    )