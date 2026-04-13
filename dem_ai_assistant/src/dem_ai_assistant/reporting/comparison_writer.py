import json
from pathlib import Path

from src.dem_ai_assistant.engineering.recommendation_engine import (
    build_comparison_recommendation_lines,
)
from src.dem_ai_assistant.postprocessing.comparison import ComparisonResult


def build_comparison_summary(comparison: ComparisonResult) -> str:
    lines = [
        "=" * 70,
        f"DEM AI ASSISTANT - {comparison.comparison_name.upper()}",
        "=" * 70,
        f"Baseline case: {comparison.baseline_case}",
        "",
        "CASE KPIs",
        "-" * 70,
    ]

    for case in comparison.cases:
        lines.append(f"{case.case_name}")
        lines.append(f"  Run folder: {case.run_dir}")
        lines.append(f"  Egress events: {case.kpis.egress_events}")
        lines.append(f"  Max impact energy (J): {case.kpis.max_impact_energy_j:.2f}")
        lines.append(f"  Mean edge margin (mm): {case.kpis.mean_edge_margin_mm:.2f}")
        lines.append(f"  Min edge margin (mm): {case.kpis.min_edge_margin_mm:.2f}")
        lines.append(f"  Throughput proxy (tph): {case.kpis.throughput_proxy_tph:.2f}")
        lines.append(f"  Stability score: {case.kpis.stability_score:.2f}")
        lines.append("")

    lines.append("DELTA FROM BASELINE")
    lines.append("-" * 70)
    for delta in comparison.deltas_from_baseline:
        lines.append(f"{delta.case_name}")
        lines.append(f"  dEgress events: {delta.delta_egress_events:+d}")
        lines.append(f"  dMax impact energy (J): {delta.delta_max_impact_energy_j:+.2f}")
        lines.append(f"  dMean edge margin (mm): {delta.delta_mean_edge_margin_mm:+.2f}")
        lines.append(f"  dMin edge margin (mm): {delta.delta_min_edge_margin_mm:+.2f}")
        lines.append(f"  dThroughput proxy (tph): {delta.delta_throughput_proxy_tph:+.2f}")
        lines.append(f"  dStability score: {delta.delta_stability_score:+.3f}")
        lines.append("")

    lines.append("RANKING (HIGHER SCORE IS BETTER)")
    lines.append("-" * 70)
    for ranked_case in comparison.ranking:
        lines.append(
            f"{ranked_case.rank}. {ranked_case.case_name} | score={ranked_case.score:.2f}"
        )
    lines.append("")

    lines.append("ENGINEERING RECOMMENDATIONS (RULE-BASED)")
    lines.append("-" * 70)
    for bullet in build_comparison_recommendation_lines(comparison):
        lines.append(bullet)
    lines.append("")

    return "\n".join(lines)


def save_comparison_outputs(comparison_dir: Path, comparison: ComparisonResult) -> tuple[Path, Path]:
    comparison_dir.mkdir(parents=True, exist_ok=True)

    json_path = comparison_dir / "comparison.json"
    txt_path = comparison_dir / "comparison_summary.txt"

    json_path.write_text(
        json.dumps(comparison.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(build_comparison_summary(comparison), encoding="utf-8")

    return json_path, txt_path
