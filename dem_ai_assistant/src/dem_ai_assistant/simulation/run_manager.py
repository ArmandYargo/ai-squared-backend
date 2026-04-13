import json
from pathlib import Path
from uuid import uuid4

from src.dem_ai_assistant.core.models import GeometryConfig, MaterialConfig, ScenarioConfig
from src.dem_ai_assistant.simulation.input_builder import SimulationInput, build_simulation_input
from src.dem_ai_assistant.simulation.output_schema import SimulationRawOutput
from src.dem_ai_assistant.simulation.solver_adapter import get_solver_adapter


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
    solver_name: str = "placeholder",
) -> tuple[Path, SimulationRawOutput]:
    run_dir = create_run_folder(base_runs_dir, scenario_config.scenario_name)
    project_root = base_runs_dir.parent
    sim_input = build_simulation_input(
        material_config,
        scenario_config,
        geometry_config,
        project_root=project_root,
    )
    save_inputs_snapshot(run_dir, material_config, scenario_config, geometry_config, sim_input)
    solver = get_solver_adapter(solver_name)
    raw_output = solver.run(sim_input)
    save_raw_output(run_dir, raw_output)
    return run_dir, raw_output