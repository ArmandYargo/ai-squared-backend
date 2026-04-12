from pathlib import Path

from src.dem_ai_assistant.core.config_loader import load_yaml_config
from src.dem_ai_assistant.simulation.scenario_runner import run_simulation_stub
from src.dem_ai_assistant.postprocessing.kpi_calculator import calculate_kpis
from src.dem_ai_assistant.engineering.rules_engine import evaluate_run
from src.dem_ai_assistant.reporting.summary_writer import build_summary


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    material_cfg = load_yaml_config(base_dir / "configs" / "materials" / "iron_ore_wet_fines.yaml")
    scenario_cfg = load_yaml_config(base_dir / "configs" / "scenarios" / "steady_state.yaml")
    geometry_cfg = load_yaml_config(base_dir / "configs" / "geometry" / "transfer_point_template.yaml")

    sim_result = run_simulation_stub(
        material_config=material_cfg,
        scenario_config=scenario_cfg,
        geometry_config=geometry_cfg,
        output_dir=base_dir / "runs" / "baseline_case_001",
    )

    kpis = calculate_kpis(sim_result)
    findings = evaluate_run(kpis)
    summary = build_summary(sim_result, kpis, findings)

    print(summary)


if __name__ == "__main__":
    main()