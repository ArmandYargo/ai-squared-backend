from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dem_ai_assistant.core.config_loader import load_yaml_config
from src.dem_ai_assistant.postprocessing.kpi_calculator import calculate_kpis
from src.dem_ai_assistant.engineering.rules_engine import evaluate_run
from src.dem_ai_assistant.reporting.summary_writer import build_summary
from src.dem_ai_assistant.simulation.run_manager import execute_case


def main() -> None:
    material_cfg = load_yaml_config(
        PROJECT_ROOT / "configs" / "materials" / "iron_ore_wet_fines.yaml"
    )
    scenario_cfg = load_yaml_config(
        PROJECT_ROOT / "configs" / "scenarios" / "steady_state.yaml"
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
    findings = evaluate_run(kpis)

    class SimResultShim:
        def __init__(self, run_dir, scenario_name, notes):
            self.run_id = run_dir.name
            self.output_dir = run_dir
            self.status = "success"
            self.scenario_name = scenario_name
            self.notes = notes

    sim_result = SimResultShim(run_dir, scenario_cfg.scenario_name, raw_output.notes)
    summary = build_summary(sim_result, kpis, findings)

    print(summary)

    report_path = run_dir / "report" / "summary.txt"
    report_path.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()