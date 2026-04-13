from typing import Protocol

from src.dem_ai_assistant.simulation.input_builder import SimulationInput
from src.dem_ai_assistant.simulation.output_schema import SimulationRawOutput
from src.dem_ai_assistant.simulation.placeholder_solver import run_placeholder_solver
from src.dem_ai_assistant.simulation.pychrono_solver import run_pychrono_solver


class SolverAdapter(Protocol):
    def run(self, sim_input: SimulationInput) -> SimulationRawOutput:
        ...


class PlaceholderSolverAdapter:
    def run(self, sim_input: SimulationInput) -> SimulationRawOutput:
        return run_placeholder_solver(sim_input)


class PyChronoSolverAdapter:
    def run(self, sim_input: SimulationInput) -> SimulationRawOutput:
        return run_pychrono_solver(sim_input)


def get_solver_adapter(solver_name: str) -> SolverAdapter:
    normalized = solver_name.strip().lower()
    if normalized == "placeholder":
        return PlaceholderSolverAdapter()
    if normalized == "pychrono":
        return PyChronoSolverAdapter()
    raise ValueError(
        f"Unsupported solver '{solver_name}'. "
        "Valid options: 'placeholder', 'pychrono'."
    )
