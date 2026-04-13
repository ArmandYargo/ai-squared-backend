import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dem_ai_assistant.core.config_loader import load_yaml_config
from src.dem_ai_assistant.core.models import GeometryConfig, MaterialConfig, ScenarioConfig
from src.dem_ai_assistant.core.study_loader import load_design_study_config
from src.dem_ai_assistant.postprocessing.comparison import (
    ComparisonCase,
    ComparisonResult,
    build_comparison_result,
)
from src.dem_ai_assistant.postprocessing.kpi_calculator import calculate_kpis
from src.dem_ai_assistant.postprocessing.study_recommendation import (
    StudyRecommendation,
    build_study_recommendation,
)
from src.dem_ai_assistant.reporting.comparison_writer import (
    build_comparison_summary,
    save_comparison_outputs,
)
from src.dem_ai_assistant.reporting.study_writer import build_study_summary, save_study_outputs
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


def _run_scenario_comparison(
    scenario_cfg: ScenarioConfig,
    material_cfg: MaterialConfig,
    design_option_files: list[str],
    baseline_option_file: str,
    solver_name: str,
) -> ComparisonResult:
    comparison_cases: list[ComparisonCase] = []
    baseline_case_name = ""

    for option_file in design_option_files:
        geometry_cfg = _load_geometry(PROJECT_ROOT / "configs" / "design_options" / option_file)
        run_dir, raw_output = execute_case(
            base_runs_dir=PROJECT_ROOT / "runs",
            material_config=material_cfg,
            scenario_config=scenario_cfg,
            geometry_config=geometry_cfg,
            solver_name=solver_name,
        )

        comparison_cases.append(
            ComparisonCase(
                case_name=geometry_cfg.transfer_point_name,
                run_dir=run_dir,
                kpis=calculate_kpis(raw_output),
            )
        )

        if option_file == baseline_option_file:
            baseline_case_name = geometry_cfg.transfer_point_name

    if not baseline_case_name:
        raise ValueError(
            "Baseline option file is not included in design_option_files."
        )

    return build_comparison_result(
        comparison_name=scenario_cfg.scenario_name,
        cases=comparison_cases,
        baseline_case=baseline_case_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded design study from a study YAML.")
    parser.add_argument(
        "--study",
        type=str,
        default="configs/studies/transfer_point_study_v1.yaml",
        help="Path to study YAML relative to project root, or absolute path.",
    )
    args = parser.parse_args()

    study_path = Path(args.study)
    if not study_path.is_absolute():
        study_path = PROJECT_ROOT / study_path

    study_cfg = load_design_study_config(study_path)
    material_cfg = _load_material(
        PROJECT_ROOT / "configs" / "materials" / study_cfg.material_file
    )

    study_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = PROJECT_ROOT / "runs" / "studies" / f"{study_cfg.study_name}_{study_id}"

    scenario_comparisons: list[ComparisonResult] = []
    for scenario_file in study_cfg.scenario_files:
        scenario_cfg = _load_scenario(PROJECT_ROOT / "configs" / "scenarios" / scenario_file)
        comparison = _run_scenario_comparison(
            scenario_cfg=scenario_cfg,
            material_cfg=material_cfg,
            design_option_files=study_cfg.design_option_files,
            baseline_option_file=study_cfg.baseline_option_file,
            solver_name=study_cfg.solver_name,
        )
        scenario_comparisons.append(comparison)

        scenario_dir = study_dir / "scenario_comparisons" / scenario_cfg.scenario_name
        save_comparison_outputs(scenario_dir, comparison)
        print(build_comparison_summary(comparison))

    recommendation: StudyRecommendation = build_study_recommendation(
        study_name=study_cfg.study_name,
        scenario_comparisons=scenario_comparisons,
    )
    json_path, txt_path = save_study_outputs(
        study_dir, recommendation, scenario_comparisons=scenario_comparisons
    )

    print(build_study_summary(recommendation, scenario_comparisons))
    print(f"Saved study JSON: {json_path}")
    print(f"Saved study text summary: {txt_path}")


if __name__ == "__main__":
    main()
