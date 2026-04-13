"""
Rule-based design hints from KPIs and deltas (not an LLM).
Placeholder-solver aware: language stays general for later PyChrono results.
"""

from src.dem_ai_assistant.core.models import EngineeringFinding, KPIResult
from src.dem_ai_assistant.postprocessing.comparison import ComparisonResult
from src.dem_ai_assistant.postprocessing.study_recommendation import StudyRecommendation


def build_single_run_recommendation_lines(
    kpis: KPIResult,
    findings: list[EngineeringFinding] | None = None,
) -> list[str]:
    """
    Action-oriented hints for one run, aligned with the same KPI bands as the rules engine.
    """
    lines: list[str] = []

    if findings:
        highs = [f for f in findings if f.severity.lower() == "high"]
        if highs:
            lines.append(
                "Address high-severity findings first (above), then iterate geometry or "
                "operating limits in small steps and re-run."
            )

    if kpis.egress_events > 5:
        lines.append(
            "Egress/spill proxy is elevated: review skirt clearance and sealing, surge "
            "handling, and chute wear plates; confirm against worst-case throughput."
        )
    if kpis.min_edge_margin_mm < 25:
        lines.append(
            "Minimum edge margin is tight: check chute centreline and lateral offset, "
            "idler alignment, and lengthen trough-to-flat transition if migration persists."
        )
    if kpis.stability_score < 0.70:
        lines.append(
            "Stability score is below the comfort band: improve burden shaping before "
            "the transfer (hood/spoon, spoon height, load centre) and verify belt support "
            "through the transition."
        )
    if kpis.max_impact_energy_j > 48:
        lines.append(
            "Impact energy is relatively high: soften impact (deflector, rock box, "
            "material bed) or adjust drop height/angle if geometry allows."
        )

    if not lines:
        lines.append(
            "No extra rule-based actions beyond findings: keep this run as a baseline "
            "and compare under surge or stop/restart when those scenarios matter."
        )

    lines.append(
        "Re-run after bounded edits only (parametric YAML changes), then compare KPIs "
        "to this run folder."
    )
    return lines


def build_comparison_recommendation_lines(comparison: ComparisonResult) -> list[str]:
    """Plain-language bullets comparing each case to baseline."""
    lines: list[str] = []
    baseline = comparison.baseline_case

    for delta in comparison.deltas_from_baseline:
        if delta.case_name == baseline:
            continue

        parts: list[str] = []
        if delta.delta_egress_events > 0:
            parts.append(
                "more spill/egress tendency than baseline: review skirt clearance, "
                "throughput peaks, and chute sealing."
            )
        if delta.delta_min_edge_margin_mm < -1.5:
            parts.append(
                "tighter minimum edge margin than baseline: check chute centreline, "
                "lateral offset, and transition length for edge migration."
            )
        if delta.delta_stability_score < -0.02:
            parts.append(
                "lower stability score than baseline: review trough-to-flat transition "
                "and burden shaping before the transfer."
            )
        if delta.delta_max_impact_energy_j > 1.5:
            parts.append(
                "higher impact energy than baseline: consider hood/spoon angles, "
                "deflector geometry, or impact bed height."
            )
        if delta.delta_max_impact_energy_j < -1.5:
            parts.append(
                "lower impact than baseline: confirm this is acceptable for wear "
                "and centre loading, not masking poor guidance."
            )

        if parts:
            lines.append(f"{delta.case_name} vs baseline:")
            for p in parts:
                lines.append(f"  - {p}")
        else:
            lines.append(
                f"{delta.case_name}: similar risk profile to baseline on tracked KPIs; "
                "no strong rule-based flag from deltas alone."
            )

    if not lines:
        lines.append("No delta-driven recommendations (only baseline or no material change).")

    # Rank context
    top = comparison.ranking[0].case_name if comparison.ranking else ""
    if top and top != baseline:
        lines.append(f"Highest-ranked option in this scenario: {top}.")
    elif top:
        lines.append("Baseline ranks highest for this scenario under the simple score.")

    return lines


def build_study_recommendation_lines(
    recommendation: StudyRecommendation,
    scenario_comparisons: list[ComparisonResult],
) -> list[str]:
    """Cross-scenario study-level hints."""
    lines: list[str] = []
    rec = recommendation.recommended_option
    lines.append(
        f"Preferred option across scenarios (simple score): {rec}. "
        "Confirm against your worst credible operating case (e.g. surge, stop/restart)."
    )

    if len(recommendation.option_scores) >= 2:
        first = recommendation.option_scores[0]
        second = recommendation.option_scores[1]
        if first.wins == second.wins and first.average_rank == second.average_rank:
            lines.append(
                f"{first.case_name} and {second.case_name} are close on wins/rank; "
                "consider a focused field check or a narrower KPI set before committing."
            )
        elif second.average_rank <= 2.0 and (first.average_score - second.average_score) < 3.0:
            lines.append(
                f"{second.case_name} is a plausible alternative; "
                "compare egress and min edge margin under surge if throughput varies."
            )

    # Worst-case scenario name for recommended option egress
    worst_name = ""
    worst_egress = -1
    for comp in scenario_comparisons:
        for case in comp.cases:
            if case.case_name != rec:
                continue
            if case.kpis.egress_events > worst_egress:
                worst_egress = case.kpis.egress_events
                worst_name = comp.comparison_name
    if worst_name:
        lines.append(
            f"For {rec}, highest egress among assessed scenarios is in '{worst_name}' "
            f"({worst_egress} events in this PoC). Prioritise mitigations for that regime."
        )

    lines.append(
        "Next engineering step: keep changes parametric (skirt spacing, transition length, "
        "chute angles) and re-run this study after each bounded edit."
    )
    return lines
