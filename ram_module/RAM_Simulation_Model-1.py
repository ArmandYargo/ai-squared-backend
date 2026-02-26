# ram_module/RAM_Simulation_Model.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict
import types


def _legacy_path() -> Path:
    p = Path(__file__).resolve().parent / "RAM_Simulation_Model_3.py"
    if not p.exists():
        raise FileNotFoundError(f"Expected legacy model at: {p}")
    return p


def _load_legacy_module() -> types.ModuleType:
    code_path = _legacy_path()
    code = code_path.read_text(encoding="utf-8", errors="ignore")
    mod = types.ModuleType("ram_legacy_runtime")
    mod.__dict__["__name__"] = "ram_legacy_runtime"
    mod.__file__ = str(code_path)
    exec(compile(code, str(code_path), "exec"), mod.__dict__)
    return mod


def run_ram_simulation(
    *,
    input_xlsx: str,
    start_date: date,
    end_date: date,
    simulations: int = 200,
    agg: str = "50th_perc",
    opp_dt_ind: int = 0,
    spare_ind: int = 0,
) -> Dict[str, Any]:
    """
    Agent-facing entrypoint.
    Accepts agent kwargs, forwards to legacy *_param kwargs.
    """
    # ✅ compute inside function (runtime), not at definition/import time
    input_xlsx_abs = str(Path(input_xlsx).expanduser().resolve())

    mod = _load_legacy_module()
    legacy_fn = getattr(mod, "run_ram_simulation")

    return legacy_fn(
        input_xlsx_param=input_xlsx_abs,
        start_date_param=start_date,
        end_date_param=end_date,
        simulations_param=int(simulations),
        agg_param=str(agg),
        opp_dt_ind_param=int(opp_dt_ind),
        spare_ind_param=int(spare_ind),
    )
