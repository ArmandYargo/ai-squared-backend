"""Smoke test PyChrono with CAD mesh from configs/geometry/transfer_point_with_mesh_example.yaml."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dem_ai_assistant.core.config_loader import load_yaml_config
from src.dem_ai_assistant.core.models import GeometryConfig, MaterialConfig, ScenarioConfig
from src.dem_ai_assistant.simulation.input_builder import build_simulation_input
from src.dem_ai_assistant.simulation.pychrono_transfer_slice import run_pychrono_transfer_slice


def main() -> None:
    m = load_yaml_config(PROJECT_ROOT / "configs" / "materials" / "iron_ore_wet_fines.yaml")
    s = load_yaml_config(PROJECT_ROOT / "configs" / "scenarios" / "pychrono_smoke.yaml")
    g = load_yaml_config(
        PROJECT_ROOT / "configs" / "geometry" / "transfer_point_with_mesh_example.yaml"
    )
    if not isinstance(m, MaterialConfig) or not isinstance(s, ScenarioConfig):
        raise TypeError("Expected material and scenario configs")
    if not isinstance(g, GeometryConfig):
        raise TypeError("Expected geometry config")

    sim_input = build_simulation_input(m, s, g, project_root=PROJECT_ROOT)
    out = run_pychrono_transfer_slice(sim_input)
    print(out.notes)
    print("metadata:", out.metadata)


if __name__ == "__main__":
    main()
