from pydantic import BaseModel

from src.dem_ai_assistant.postprocessing.comparison import ComparisonResult


class OptionStudyScore(BaseModel):
    case_name: str
    wins: int
    average_rank: float
    average_score: float


class StudyRecommendation(BaseModel):
    study_name: str
    scenario_names: list[str]
    option_scores: list[OptionStudyScore]
    recommended_option: str


def build_study_recommendation(
    study_name: str,
    scenario_comparisons: list[ComparisonResult],
) -> StudyRecommendation:
    if not scenario_comparisons:
        raise ValueError("At least one scenario comparison is required.")

    score_map: dict[str, dict[str, float]] = {}
    scenario_names: list[str] = []

    for comparison in scenario_comparisons:
        scenario_names.append(comparison.comparison_name)
        for ranked in comparison.ranking:
            if ranked.case_name not in score_map:
                score_map[ranked.case_name] = {
                    "wins": 0.0,
                    "rank_total": 0.0,
                    "score_total": 0.0,
                    "count": 0.0,
                }
            row = score_map[ranked.case_name]
            row["rank_total"] += float(ranked.rank)
            row["score_total"] += ranked.score
            row["count"] += 1.0
            if ranked.rank == 1:
                row["wins"] += 1.0

    option_scores: list[OptionStudyScore] = []
    for case_name, row in score_map.items():
        option_scores.append(
            OptionStudyScore(
                case_name=case_name,
                wins=int(row["wins"]),
                average_rank=row["rank_total"] / row["count"],
                average_score=row["score_total"] / row["count"],
            )
        )

    option_scores.sort(
        key=lambda item: (-item.wins, item.average_rank, -item.average_score)
    )
    recommended_option = option_scores[0].case_name

    return StudyRecommendation(
        study_name=study_name,
        scenario_names=scenario_names,
        option_scores=option_scores,
        recommended_option=recommended_option,
    )
