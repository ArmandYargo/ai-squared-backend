from pathlib import Path

from pydantic import BaseModel

from src.dem_ai_assistant.core.models import KPIResult


class ComparisonCase(BaseModel):
    case_name: str
    run_dir: Path
    kpis: KPIResult


class ComparisonDelta(BaseModel):
    case_name: str
    delta_egress_events: int
    delta_max_impact_energy_j: float
    delta_mean_edge_margin_mm: float
    delta_min_edge_margin_mm: float
    delta_throughput_proxy_tph: float
    delta_stability_score: float


class RankedCase(BaseModel):
    case_name: str
    score: float
    rank: int


class ComparisonResult(BaseModel):
    comparison_name: str
    baseline_case: str
    cases: list[ComparisonCase]
    deltas_from_baseline: list[ComparisonDelta]
    ranking: list[RankedCase]


def _calculate_rank_score(kpis: KPIResult) -> float:
    """
    Simple weighted score for quick proof-of-concept ranking.
    Higher is better.
    """
    return (
        100.0 * kpis.stability_score
        + 1.2 * kpis.min_edge_margin_mm
        + 0.3 * kpis.mean_edge_margin_mm
        - 8.0 * kpis.egress_events
        - 0.25 * kpis.max_impact_energy_j
    )


def build_comparison_result(
    comparison_name: str,
    cases: list[ComparisonCase],
    baseline_case: str,
) -> ComparisonResult:
    if not cases:
        raise ValueError("At least one case is required for comparison.")

    baseline = next((case for case in cases if case.case_name == baseline_case), None)
    if baseline is None:
        raise ValueError(f"Baseline case '{baseline_case}' not found in cases.")

    deltas: list[ComparisonDelta] = []
    for case in cases:
        case_kpis = case.kpis
        base_kpis = baseline.kpis
        deltas.append(
            ComparisonDelta(
                case_name=case.case_name,
                delta_egress_events=case_kpis.egress_events - base_kpis.egress_events,
                delta_max_impact_energy_j=case_kpis.max_impact_energy_j
                - base_kpis.max_impact_energy_j,
                delta_mean_edge_margin_mm=case_kpis.mean_edge_margin_mm
                - base_kpis.mean_edge_margin_mm,
                delta_min_edge_margin_mm=case_kpis.min_edge_margin_mm
                - base_kpis.min_edge_margin_mm,
                delta_throughput_proxy_tph=case_kpis.throughput_proxy_tph
                - base_kpis.throughput_proxy_tph,
                delta_stability_score=case_kpis.stability_score - base_kpis.stability_score,
            )
        )

    scored_cases = [
        (case.case_name, _calculate_rank_score(case.kpis))
        for case in cases
    ]
    scored_cases.sort(key=lambda item: item[1], reverse=True)

    ranking: list[RankedCase] = []
    for index, (case_name, score) in enumerate(scored_cases, start=1):
        ranking.append(RankedCase(case_name=case_name, score=score, rank=index))

    return ComparisonResult(
        comparison_name=comparison_name,
        baseline_case=baseline_case,
        cases=cases,
        deltas_from_baseline=deltas,
        ranking=ranking,
    )
