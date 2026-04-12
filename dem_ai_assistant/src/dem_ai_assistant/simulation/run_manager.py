import json
from pathlib import Path
from uuid import uuid4

from src.dem_ai_assistant.core.models import GeometryConfig, MaterialConfig, ScenarioConfig
from src.dem_ai_assistant.simulation.input_builder import SimulationInput, build_simulation_input
from src.dem_ai_assistant.simulation.output_schema import (
    EdgeStats,
    FlowStats,
    ImpactStats,
    ParticleStats,
    SimulationRawOutput,
)


def create_run_folder(base_runs_dir: Path, scenario_name: str) -> Path:
    run_id = f"{scenario_name}_{uuid4().hex[:8]}"
    run_dir = base_runs_dir / run_id
    (run_dir / "inputs_snapshot").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "processed").mkdir(parents=True, exist_ok=True)
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_inputs_snapshot(
    run_dir: Path,
    material_config: MaterialConfig,
    scenario_config: ScenarioConfig,
    geometry_config: GeometryConfig,
    sim_input: SimulationInput,
) -> None:
    snapshots = {
        "material_config.json": material_config.model_dump(),
        "scenario_config.json": scenario_config.model_dump(),
        "geometry_config.json": geometry_config.model_dump(),
        "simulation_input.json": sim_input.model_dump(),
    }

    for filename, payload in snapshots.items():
        path = run_dir / "inputs_snapshot" / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_placeholder_solver(sim_input: SimulationInput) -> SimulationRawOutput:
    edge_penalty = max(0.0, 100 - sim_input.skirt_spacing_mm) / 100
    cohesion_penalty = min(sim_input.cohesion_pa / 3000.0, 1.0)
    throughput_penalty = min(sim_input.throughput_tph / 5000.0, 1.0)

    estimated_spill_events = int(3 + 6 * edge_penalty + 4 * cohesion_penalty + 3 * throughput_penalty)
    min_edge_margin = max(5.0, 40.0 - 15.0 * throughput_penalty - 10.0 * cohesion_penalty)
    mean_edge_margin = max(min_edge_margin + 20.0, 90.0 - 10.0 * throughput_penalty)
    stability_score = max(0.2, 0.9 - 0.2 * cohesion_penalty - 0.15 * throughput_penalty - 0.1 * edge_penalty)

    return SimulationRawOutput(
        particle_stats=ParticleStats(
            total_particles=25000,
            coarse_particles=3500,
            fines_fraction=0.25,
        ),
        flow_stats=FlowStats(
            throughput_tph=sim_input.throughput_tph,
            estimated_spill_events=estimated_spill_events,
            estimated_recirculation_index=0.18 + 0.2 * cohesion_penalty,
        ),
        edge_stats=EdgeStats(
            mean_edge_margin_mm=mean_edge_margin,
            min_edge_margin_mm=min_edge_margin,
            left_edge_bias=0.08,
            right_edge_bias=0.11,
        ),
        impact_stats=ImpactStats(
            max_impact_energy_j=30.0 + 25.0 * throughput_penalty,
            p95_impact_energy_j=20.0 + 10.0 * throughput_penalty,
        ),
        stability_score=stability_score,
        notes="Placeholder solver output only. No DEM engine yet.",
        metadata={"solver": "placeholder"},
    )


def save_raw_output(run_dir: Path, raw_output: SimulationRawOutput) -> None:
    path = run_dir / "raw" / "simulation_output.json"
    path.write_text(
        json.dumps(raw_output.model_dump(), indent=2),
        encoding="utf-8",
    )


def execute_case(
    base_runs_dir: Path,
    material_config: MaterialConfig,
    scenario_config: ScenarioConfig,
    geometry_config: GeometryConfig,
) -> tuple[Path, SimulationRawOutput]:
    run_dir = create_run_folder(base_runs_dir, scenario_config.scenario_name)
    sim_input = build_simulation_input(material_config, scenario_config, geometry_config)
    save_inputs_snapshot(run_dir, material_config, scenario_config, geometry_config, sim_input)
    raw_output = run_placeholder_solver(sim_input)
    save_raw_output(run_dir, raw_output)
    return run_dir, raw_output