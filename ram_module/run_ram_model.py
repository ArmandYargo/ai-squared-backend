"""
run_ram_model.py

Convenience CLI/script for Spyder to:
- run the RAM simulation using the generated input workbook
- persist results into an archive folder for trending over time

Usage (CLI):
  python run_ram_model.py --input ram_input_sheet.xlsx --start 2011-01-01 --end 2017-12-31 --sims 200 --out ram_runs

Spyder:
  from run_ram_model import run_and_save
  manifest = run_and_save("ram_input_sheet.xlsx", "2011-01-01", "2017-12-31", sims=200)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Use the fixed/refactored model file by default.
# Rename to RAM_Simulation_Model.py in your project once you're happy.
from RAM_Simulation_Model_FIXED import run_ram_simulation  # type: ignore

from ram_results import save_ram_results


def _parse_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return pd.to_datetime(str(d)).date()


def run_and_save(
    input_xlsx: str | Path,
    start: str | date,
    end: str | date,
    *,
    sims: int = 200,
    agg: str = "50th_perc",
    opp_dt_ind: int = 0,
    spare_ind: int = 0,
    out_root: str | Path = "ram_runs",
    run_id: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs the model and archives results. Returns the archive manifest.
    """
    input_xlsx = Path(input_xlsx)
    start_d = _parse_date(start)
    end_d = _parse_date(end)

    results = run_ram_simulation(
        input_xlsx=str(input_xlsx),
        start_date=start_d,
        end_date=end_d,
        simulations=int(sims),
        agg=str(agg),
        opp_dt_ind=int(opp_dt_ind),
        spare_ind=int(spare_ind),
    )

    manifest = save_ram_results(
        results,
        input_xlsx=input_xlsx,
        out_root=out_root,
        run_id=run_id,
        extra_meta=extra_meta,
    )
    return manifest


def _main():
    import argparse

    p = argparse.ArgumentParser(description="Run RAM simulation + archive outputs")
    p.add_argument("--input", required=True, help="Path to RAM input workbook (.xlsx)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--sims", type=int, default=200, help="Number of Monte Carlo simulations")
    p.add_argument("--agg", default="50th_perc", help="Condition aggregation method")
    p.add_argument("--opp_dt_ind", type=int, default=0, help="Opportunistic downtime flag (0/1)")
    p.add_argument("--spare_ind", type=int, default=0, help="Spare systems flag (0/1)")
    p.add_argument("--out", default="ram_runs", help="Archive root folder")
    p.add_argument("--run_id", default=None, help="Optional run_id folder name")

    args = p.parse_args()

    manifest = run_and_save(
        args.input,
        args.start,
        args.end,
        sims=args.sims,
        agg=args.agg,
        opp_dt_ind=args.opp_dt_ind,
        spare_ind=args.spare_ind,
        out_root=args.out,
        run_id=args.run_id,
    )
    print("Archived:", manifest["run_dir"])
    print("Metadata:", manifest["metadata_path"])


if __name__ == "__main__":
    _main()
