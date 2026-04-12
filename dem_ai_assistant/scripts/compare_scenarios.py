from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dem_ai_assistant.core.config_loader import load_yaml_config
from src.dem_ai_assistant.postprocessing.kpi_calculator import calculate_kpis
from src.dem_ai_assistant.simulation.run_manager import execute_case


def run_one_case(scenario_file):
    material_cfg = load_yaml_config(
        PROJECT_ROOT / "configs" / "materials" / "iron_ore_wet_fines.yaml"
    )

    scenario_cfg = load_yaml_config(
        PROJECT_ROOT / "configs" / "scenarios" / scenario_file
    )

    geometry_cfg = load_yaml_config(
        PROJECT_ROOT / "configs" / "geometry" / "transfer_point_template.yaml"
    )

    run_dir, raw_output = execute_case(
        base_runs_dir=PROJECT_ROOT / "runs",
        material_config=material_cfg,
        scenario_config=scenario_cfg,
        geometry_config=geometry_cfg,
    )

    kpis = calculate_kpis(raw_output)

    return scenario_cfg.scenario_name, kpis


def main():
    scenarios = [
        "steady_state.yaml",
        "surge.yaml",
        "stop_restart.yaml",
    ]

    print("\nSCENARIO COMPARISON")
    print("=" * 70)

    for scenario_file in scenarios:
        name, kpis = run_one_case(scenario_file)

        print(f"\n{name.upper()}")
        print("-" * 30)
        print(f"Egress Events: {kpis.egress_events}")
        print(f"Max Impact Energy: {kpis.max_impact_energy_j:.2f} J")
        print(f"Min Edge Margin: {kpis.min_edge_margin_mm:.2f} mm")
        print(f"Stability Score: {kpis.stability_score:.2f}")


if __name__ == "__main__":
    main()