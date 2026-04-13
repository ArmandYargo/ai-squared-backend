import json
from pathlib import Path

from src.dem_ai_assistant.engineering.recommendation_engine import (
    build_study_recommendation_lines,
)
from src.dem_ai_assistant.postprocessing.comparison import ComparisonResult
from src.dem_ai_assistant.postprocessing.study_recommendation import StudyRecommendation


def build_study_summary(
    recommendation: StudyRecommendation,
    scenario_comparisons: list[ComparisonResult] | None = None,
) -> str:
    lines = [
        "=" * 70,
        f"DEM AI ASSISTANT - STUDY SUMMARY: {recommendation.study_name}",
        "=" * 70,
        f"Scenarios assessed: {', '.join(recommendation.scenario_names)}",
        f"Recommended option: {recommendation.recommended_option}",
        "",
        "OPTION PERFORMANCE",
        "-" * 70,
    ]

    for item in recommendation.option_scores:
        lines.append(f"{item.case_name}")
        lines.append(f"  Wins (rank #1 count): {item.wins}")
        lines.append(f"  Average rank: {item.average_rank:.2f}")
        lines.append(f"  Average score: {item.average_score:.2f}")
        lines.append("")

    if scenario_comparisons:
        lines.append("STUDY-LEVEL RECOMMENDATIONS (RULE-BASED)")
        lines.append("-" * 70)
        for bullet in build_study_recommendation_lines(
            recommendation, scenario_comparisons
        ):
            lines.append(bullet)
        lines.append("")

    return "\n".join(lines)


def save_study_outputs(
    study_dir: Path,
    recommendation: StudyRecommendation,
    scenario_comparisons: list[ComparisonResult] | None = None,
) -> tuple[Path, Path]:
    study_dir.mkdir(parents=True, exist_ok=True)

    json_path = study_dir / "study_recommendation.json"
    txt_path = study_dir / "study_summary.txt"

    json_path.write_text(
        json.dumps(recommendation.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(
        build_study_summary(recommendation, scenario_comparisons),
        encoding="utf-8",
    )

    return json_path, txt_path
