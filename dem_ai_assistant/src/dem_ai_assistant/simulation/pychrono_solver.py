from src.dem_ai_assistant.simulation.input_builder import SimulationInput
from src.dem_ai_assistant.simulation.output_schema import SimulationRawOutput
from src.dem_ai_assistant.simulation.pychrono_transfer_slice import (
    run_pychrono_transfer_slice,
)


def run_pychrono_solver(sim_input: SimulationInput) -> SimulationRawOutput:
    """
    CPU rigid-sphere NSC transfer slice. Not full GPU DEM or CAD mesh import yet.
    """
    return run_pychrono_transfer_slice(sim_input)
