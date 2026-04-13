from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dem_ai_assistant.core.config_loader import load_yaml_config
from src.dem_ai_assistant.core.models import GeometryConfig, MaterialConfig, ScenarioConfig
from src.dem_ai_assistant.postprocessing.comparison import (
    ComparisonCase,
    build_comparison_result,
)
from src.dem_ai_assistant.postprocessing.kpi_calculator import calculate_kpis
from src.dem_ai_assistant.reporting.comparison_writer import (
    build_comparison_summary,
    save_comparison_outputs,
)
from src.dem_ai_assistant.simulation.run_manager import execute_case


def _load_material(path: Path) -> MaterialConfig:
    config = load_yaml_config(path)
    if not isinstance(config, MaterialConfig):
        raise TypeError(f"Expected MaterialConfig at {path}")
    return config


def _load_scenario(path: Path) -> ScenarioConfig:
    config = load_yaml_config(path)
    if not isinstance(config, ScenarioConfig):
        raise TypeError(f"Expected ScenarioConfig at {path}")
    return config


def _load_geometry(path: Path) -> GeometryConfig:
    config = load_yaml_config(path)
    if not isinstance(config, GeometryConfig):
        raise TypeError(f"Expected GeometryConfig at {path}")
    return config


def main() -> None:
    scenario_cfg = _load_scenario(PROJECT_ROOT / "configs" / "scenarios" / "surge.yaml")
    material_cfg = _load_material(
        PROJECT_ROOT / "configs" / "materials" / "iron_ore_wet_fines.yaml"
    )

    design_option_files = [
        "option_01_baseline.yaml",
        "option_02_tighter_skirt.yaml",
        "option_03_longer_transition.yaml",
    ]

    comparison_cases: list[ComparisonCase] = []
    for option_file in design_option_files:
        geometry_cfg = _load_geometry(
            PROJECT_ROOT / "configs" / "design_options" / option_file
        )
        run_dir, raw_output = execute_case(
            base_runs_dir=PROJECT_ROOT / "runs",
            material_config=material_cfg,
            scenario_config=scenario_cfg,
            geometry_config=geometry_cfg,
        )
        comparison_cases.append(
            ComparisonCase(
                case_name=geometry_cfg.transfer_point_name,
                run_dir=run_dir,
                kpis=calculate_kpis(raw_output),
            )
        )

    baseline_case = comparison_cases[0].case_name
    comparison_result = build_comparison_result(
        comparison_name="design_option_comparison",
        cases=comparison_cases,
        baseline_case=baseline_case,
    )

    comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = (
        PROJECT_ROOT / "runs" / "comparisons" / f"design_option_comparison_{comparison_id}"
    )
    json_path, txt_path = save_comparison_outputs(
        comparison_dir=comparison_dir,
        comparison=comparison_result,
    )

    top_ranked = comparison_result.ranking[0]
    print(build_comparison_summary(comparison_result))
    print(
        f"Recommended option (simple KPI score): {top_ranked.case_name} "
        f"(score={top_ranked.score:.2f})"
    )
    print(f"Saved JSON report: {json_path}")
    print(f"Saved text report: {txt_path}")


if __name__ == "__main__":
    main()
