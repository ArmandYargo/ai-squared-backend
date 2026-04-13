from src.dem_ai_assistant.simulation.input_builder import SimulationInput
from src.dem_ai_assistant.simulation.output_schema import (
    EdgeStats,
    FlowStats,
    ImpactStats,
    ParticleStats,
    SimulationRawOutput,
)


def run_placeholder_solver(sim_input: SimulationInput) -> SimulationRawOutput:
    """Heuristic placeholder: reacts to throughput, cohesion, skirt, offset, transition, chute angles."""
    edge_penalty = max(0.0, 100 - sim_input.skirt_spacing_mm) / 100
    cohesion_penalty = min(sim_input.cohesion_pa / 3000.0, 1.0)
    throughput_penalty = min(sim_input.throughput_tph / 5000.0, 1.0)
    offset_penalty = min(abs(sim_input.lateral_offset_mm) / 150.0, 0.45)
    # Longer transition (vs 5 m reference) slightly improves burden guidance (placeholder only).
    transition_boost = min(max((sim_input.transition_length_mm - 5000.0) / 10000.0, 0.0), 0.15)
    chute_steepness = min(
        max((sim_input.hood_angle_deg + sim_input.spoon_angle_deg - 75.0) / 120.0, 0.0),
        0.35,
    )

    base_spill = 3 + 6 * edge_penalty + 4 * cohesion_penalty + 3 * throughput_penalty
    spill_offset = 2.5 * offset_penalty
    spill_restart = 1.5 if sim_input.restart_enabled else 0.0
    spill_transition = -2.0 * transition_boost
    estimated_spill_events = int(max(0, base_spill + spill_offset + spill_restart + spill_transition))

    min_edge_margin = max(
        5.0,
        40.0
        - 15.0 * throughput_penalty
        - 10.0 * cohesion_penalty
        - 14.0 * offset_penalty
        + 10.0 * transition_boost,
    )
    mean_edge_margin = max(
        min_edge_margin + 18.0,
        88.0 - 10.0 * throughput_penalty - 6.0 * offset_penalty + 8.0 * transition_boost,
    )
    stability_score = max(
        0.2,
        0.9
        - 0.2 * cohesion_penalty
        - 0.15 * throughput_penalty
        - 0.1 * edge_penalty
        - 0.12 * offset_penalty
        + 0.1 * transition_boost
        - 0.04 * chute_steepness,
    )

    max_impact = 30.0 + 25.0 * throughput_penalty + 12.0 * chute_steepness
    p95_impact = 20.0 + 10.0 * throughput_penalty + 8.0 * chute_steepness

    lo = sim_input.lateral_offset_mm
    left_bias = 0.08 + max(0.0, -lo) / 500.0
    right_bias = 0.11 + max(0.0, lo) / 500.0

    return SimulationRawOutput(
        particle_stats=ParticleStats(
            total_particles=25000,
            coarse_particles=3500,
            fines_fraction=0.25,
        ),
        flow_stats=FlowStats(
            throughput_tph=sim_input.throughput_tph,
            estimated_spill_events=estimated_spill_events,
            estimated_recirculation_index=0.18 + 0.2 * cohesion_penalty,
        ),
        edge_stats=EdgeStats(
            mean_edge_margin_mm=mean_edge_margin,
            min_edge_margin_mm=min_edge_margin,
            left_edge_bias=left_bias,
            right_edge_bias=right_bias,
        ),
        impact_stats=ImpactStats(
            max_impact_energy_j=max_impact,
            p95_impact_energy_j=p95_impact,
        ),
        stability_score=stability_score,
        notes="Placeholder solver output only. No DEM engine yet.",
        metadata={
            "solver": "placeholder",
            "edge_penalty": edge_penalty,
            "offset_penalty": offset_penalty,
            "transition_boost": transition_boost,
            "chute_steepness": chute_steepness,
        },
    )
