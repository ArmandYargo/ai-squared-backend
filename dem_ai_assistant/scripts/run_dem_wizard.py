"""
Interactive DEM wizard for Spyder / terminal (Anaconda env with PyChrono).

Run from project root with working directory = project root:
    python scripts/run_dem_wizard.py

In Spyder: open this file, set run configuration working directory to the repo root, then Run.

Workflow merges your checklist with the recommended CAD order:
  geometry (CAD + optional simplified collision mesh) → particles → material →
  gravity → solver → PyChrono run → Irrlicht (optional) → report files under runs/wizard_runs/.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dem_ai_assistant.simulation.pychrono_wizard_runner import run_wizard_dem
from src.dem_ai_assistant.wizard.wizard_collect import collect_wizard_interactive


def main() -> None:
    cfg = collect_wizard_interactive()
    run_dir, rep = run_wizard_dem(cfg, PROJECT_ROOT)
    print("\nDone.")
    print(f"Outputs: {run_dir}")
    print(f"  wizard_config.json  wizard_report.json  wizard_summary.txt")


if __name__ == "__main__":
    main()
