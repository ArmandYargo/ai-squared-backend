from src.dem_ai_assistant.core.models import EngineeringFinding, KPIResult, SimulationResult
from src.dem_ai_assistant.engineering.recommendation_engine import (
    build_single_run_recommendation_lines,
)


def build_summary(
    sim_result: SimulationResult,
    kpis: KPIResult,
    findings: list[EngineeringFinding],
    include_recommendations: bool = True,
) -> str:
    lines = [
        "=" * 70,
        "DEM AI ASSISTANT - RUN SUMMARY",
        "=" * 70,
        f"Run ID: {sim_result.run_id}",
        f"Scenario: {sim_result.scenario_name}",
        f"Status: {sim_result.status}",
        f"Notes: {sim_result.notes}",
        "",
        "KPIs",
        "-" * 70,
        f"Egress events: {kpis.egress_events}",
        f"Max impact energy (J): {kpis.max_impact_energy_j:.2f}",
        f"Mean edge margin (mm): {kpis.mean_edge_margin_mm:.2f}",
        f"Min edge margin (mm): {kpis.min_edge_margin_mm:.2f}",
        f"Throughput proxy (tph): {kpis.throughput_proxy_tph:.2f}",
        f"Stability score: {kpis.stability_score:.2f}",
        "",
        "Findings",
        "-" * 70,
    ]

    for idx, finding in enumerate(findings, start=1):
        lines.append(f"{idx}. [{finding.severity.upper()}] {finding.title}")
        lines.append(f"   {finding.detail}")

    if include_recommendations:
        lines.append("")
        lines.append("Recommendations (rule-based)")
        lines.append("-" * 70)
        for bullet in build_single_run_recommendation_lines(kpis, findings):
            lines.append(f"- {bullet}")

    lines.append("")
    return "\n".join(lines)