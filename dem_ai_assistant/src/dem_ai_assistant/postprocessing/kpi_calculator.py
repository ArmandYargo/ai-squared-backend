from src.dem_ai_assistant.core.models import KPIResult
from src.dem_ai_assistant.simulation.output_schema import SimulationRawOutput


def calculate_kpis(raw_output: SimulationRawOutput) -> KPIResult:
    return KPIResult(
        egress_events=raw_output.flow_stats.estimated_spill_events,
        max_impact_energy_j=raw_output.impact_stats.max_impact_energy_j,
        mean_edge_margin_mm=raw_output.edge_stats.mean_edge_margin_mm,
        min_edge_margin_mm=raw_output.edge_stats.min_edge_margin_mm,
        throughput_proxy_tph=raw_output.flow_stats.throughput_tph,
        stability_score=raw_output.stability_score,
    )