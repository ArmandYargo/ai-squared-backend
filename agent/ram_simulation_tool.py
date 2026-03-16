# agent/ram_simulation_tool.py
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Mapping
from ram_module.RAM_Simulation_Model import run_ram_simulation


import pandas as pd
import numpy as np


class _SafeJSONEncoder(json.JSONEncoder):
    """Handles date, datetime, numpy, pandas, and other non-standard types during archiving."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat(timespec="seconds")
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, pd.DataFrame):
            return f"<DataFrame {o.shape[0]}x{o.shape[1]}>"
        if isinstance(o, pd.Series):
            return o.tolist()
        if isinstance(o, (pd.Timedelta, np.timedelta64)):
            return str(o)
        return super().default(o)


@dataclass
class RamSimArchiveResult:
    run_dir: str
    metadata_path: str
    outputs: Dict[str, str]
    conditions: Dict[str, str]
    summary: Dict[str, Any]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_df(df: pd.DataFrame, path_base: Path) -> str:
    try:
        df.to_parquet(str(path_base.with_suffix(".parquet")), index=False)
        return str(path_base.with_suffix(".parquet"))
    except Exception:
        df.to_csv(str(path_base.with_suffix(".csv")), index=False)
        return str(path_base.with_suffix(".csv"))


def _normalize_results(obj: Any) -> Dict[str, Any]:
    """
    Accepts either:
      - RAMResults dataclass (preferred)
      - dict with keys: outputs/conditions/parameters (new format)
      - dict with legacy keys (yearly_component, condition_data, parameters, ...)

    Returns normalized:
      {"outputs": {...}, "conditions": {...}, "parameters": {...}}
    """
    # RAMResults-style object (duck-typing)
    if hasattr(obj, "outputs") and hasattr(obj, "conditions") and hasattr(obj, "parameters"):
        outputs = getattr(obj, "outputs")
        conditions = getattr(obj, "conditions")
        params = getattr(obj, "parameters")
        return {"outputs": outputs or {}, "conditions": conditions or {}, "parameters": params or {}}

    if isinstance(obj, Mapping):
        d = dict(obj)

        # Already normalized
        if "outputs" in d and "conditions" in d:
            return {
                "outputs": d.get("outputs") or {},
                "conditions": d.get("conditions") or {},
                "parameters": d.get("parameters") or {},
            }

        # Legacy shape -> normalize
        outputs: Dict[str, Any] = {}
        # typical legacy keys from RAM model
        for k in (
            "yearly_component",
            "monthly_component",
            "yearly_subcomponent",
            "monthly_subcomponent",
            "yearly_simulations",
            "monthly_simulations",
        ):
            if k in d:
                outputs[k] = d.get(k)

        # condition_data -> conditions
        conditions: Dict[str, Any] = {}
        cond = d.get("condition_data")
        if isinstance(cond, dict):
            # keep this nested structure; archiver already supports nested dicts
            conditions["condition_data"] = cond

        params = d.get("parameters") or {}
        return {"outputs": outputs, "conditions": conditions, "parameters": params}

    # Unknown
    return {"outputs": {}, "conditions": {}, "parameters": {}}


MIN_TF_ETA_DAYS = 7
MIN_TF_BETA = 0.1
MAX_EXPECTED_EVENTS_PER_SIM = 50_000

def _estimate_failures(tf_eta: float, tf_beta: float, period_days: int) -> float:
    """Rough estimate of expected failure count for a Weibull component over a period."""
    import math
    if tf_eta <= 0 or tf_beta <= 0 or period_days <= 0:
        return 0.0
    mean_tbf = tf_eta * math.gamma(1 + 1.0 / tf_beta)
    if mean_tbf <= 0:
        return float("inf")
    return period_days / mean_tbf


def _validate_comp_att(comp_att: pd.DataFrame, input_xlsx: str,
                       period_days: int = 0) -> None:
    """Pre-flight validation of the comp_att sheet to catch bad data before simulation."""
    required_cols = {"component", "subcomponent", "tf_eta", "tf_beta", "pf_n", "weight",
                     "mttr_fail", "mttr_repair", "mttr_replace", "time_tbl", "use_tbl"}
    missing_cols = required_cols - set(comp_att.columns)
    if missing_cols:
        raise ValueError(
            f"comp_att sheet in '{input_xlsx}' is missing required columns: {sorted(missing_cols)}"
        )

    if comp_att.empty:
        raise ValueError(f"comp_att sheet in '{input_xlsx}' has no rows (no components defined).")

    total_estimated_events = 0.0

    for idx, row in comp_att.iterrows():
        label = f"comp_att row {idx} ({row.get('component', '?')}/{row.get('subcomponent', '?')})"

        for col in ("tf_eta", "tf_beta", "pf_n"):
            val = row.get(col)
            if val is None or (isinstance(val, float) and (pd.isna(val) or val <= 0)):
                raise ValueError(f"{label}: '{col}' must be a positive number, got {val!r}")

        tf_eta = float(row.get("tf_eta", 0))
        tf_beta = float(row.get("tf_beta", 0))

        if tf_eta < MIN_TF_ETA_DAYS:
            raise ValueError(
                f"{label}: 'tf_eta' is {tf_eta:.2f} days (< {MIN_TF_ETA_DAYS} days). "
                "This would cause an extreme number of failure events. "
                "Check your input data -- tf_eta should be in days."
            )

        if tf_beta < MIN_TF_BETA:
            raise ValueError(
                f"{label}: 'tf_beta' is {tf_beta:.4f} (< {MIN_TF_BETA}). "
                "This is an unrealistically low Weibull shape parameter and will "
                "produce unpredictable failure patterns."
            )

        if period_days > 0:
            est = _estimate_failures(tf_eta, tf_beta, period_days)
            total_estimated_events += est
            if est > 10_000:
                raise ValueError(
                    f"{label}: estimated ~{est:,.0f} failures over {period_days} days "
                    f"(tf_eta={tf_eta:.1f}, tf_beta={tf_beta:.2f}). "
                    "This will make the simulation extremely slow. "
                    "Consider increasing tf_eta or shortening the simulation period."
                )

        for col in ("mttr_fail", "mttr_repair", "mttr_replace"):
            val = row.get(col)
            if val is not None and isinstance(val, (int, float)) and not pd.isna(val):
                if val < 0:
                    raise ValueError(f"{label}: '{col}' cannot be negative, got {val}")
                if val > 3650:
                    raise ValueError(
                        f"{label}: '{col}' is {val} days (~{val / 365:.1f} years). "
                        "This looks unreasonably large and will cause date overflow errors."
                    )

        weight = row.get("weight")
        if weight is None or (isinstance(weight, float) and pd.isna(weight)):
            raise ValueError(f"{label}: 'weight' is missing or NaN")

        time_tbl = row.get("time_tbl")
        if time_tbl is not None and not pd.isna(time_tbl):
            if int(time_tbl) not in (0, 1, 2, 3):
                raise ValueError(
                    f"{label}: 'time_tbl' must be 0 (Reactive), 1 (Corrective), "
                    f"2 (Preventative), or 3 (Condition based), got {time_tbl}"
                )

    if period_days > 0 and total_estimated_events > MAX_EXPECTED_EVENTS_PER_SIM:
        raise ValueError(
            f"Total estimated failure events per simulation: ~{total_estimated_events:,.0f} "
            f"(across {len(comp_att)} components over {period_days} days). "
            f"Maximum allowed: {MAX_EXPECTED_EVENTS_PER_SIM:,}. "
            "The simulation would be extremely slow. "
            "Consider fewer components, higher tf_eta values, or a shorter period."
        )


def run_ram_simulation_archived(
    input_xlsx: str,
    start_date: date,
    end_date: date,
    simulations: int = 200,
    agg: str = "50th_perc",
    opp_dt_ind: int = 0,
    spare_ind: int = 0,
    out_root: str = "ram_runs",
    machine_label: Optional[str] = None,
) -> RamSimArchiveResult:
    """
    Runs ram_module.RAM_Simulation_Model.run_ram_simulation(...) and archives results.
    """
    if not input_xlsx or not Path(input_xlsx).exists():
        raise FileNotFoundError(f"Input sheet not found: {input_xlsx}")

    if start_date >= end_date:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date}).")

    if simulations < 1:
        raise ValueError(f"Simulations must be >= 1, got {simulations}.")

    MAX_PERIOD_DAYS = 365 * 150  # ~150 years
    period_days = (end_date - start_date).days
    if period_days > MAX_PERIOD_DAYS:
        raise ValueError(
            f"Simulation period is {period_days} days (~{period_days // 365} years). "
            f"Maximum allowed is {MAX_PERIOD_DAYS} days (~150 years). "
            "Use a shorter date range."
        )

    required_sheets = {"comp_att", "timeline_0", "usage_0"}
    try:
        xl = pd.ExcelFile(input_xlsx)
        missing = required_sheets - set(xl.sheet_names)
        if missing:
            raise ValueError(
                f"Input sheet is missing required tabs: {sorted(missing)}. "
                f"Found: {xl.sheet_names}"
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Cannot read input sheet '{input_xlsx}': {type(e).__name__}: {e}")

    comp_att = pd.read_excel(input_xlsx, sheet_name="comp_att")
    period_days = (end_date - start_date).days
    _validate_comp_att(comp_att, input_xlsx, period_days=period_days)

    try:
        from ram_module import RAM_Simulation_Model as ram_model
    except Exception as e:
        raise RuntimeError(
            "Failed to import ram_module.RAM_Simulation_Model.\n"
            "Check that:\n"
            "1) ram_module is in your project\n"
            "2) your model doesn't execute at import time\n"
            "3) simpy is installed\n\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    if not hasattr(ram_model, "run_ram_simulation"):
        raise RuntimeError("RAM_Simulation_Model.py does not expose run_ram_simulation(...).")

    try:
        raw_results = ram_model.run_ram_simulation(
            input_xlsx_param=input_xlsx,
            start_date_param=start_date,
            end_date_param=end_date,
            simulations_param=int(simulations),
            agg_param=agg,
            opp_dt_ind_param=int(opp_dt_ind),
            spare_ind_param=int(spare_ind),
        )
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" in msg:
            raise RuntimeError(
                f"Parameter mismatch calling run_ram_simulation: {e}\n"
                "Check that ram_simulation_tool.py keyword args match "
                "RAM_Simulation_Model.run_ram_simulation() signature."
            ) from e
        raise
    except ValueError as e:
        msg = str(e)
        if "no longer supported" in msg or "Invalid frequency" in msg:
            raise RuntimeError(
                f"Pandas frequency deprecation in RAM_Simulation_Model: {e}\n"
                "Replace deprecated freq aliases (e.g. 'M' -> 'ME', 'Y' -> 'YE', "
                "'H' -> 'h', 'Q' -> 'QE') in the simulation model."
            ) from e
        raise

    results = _normalize_results(raw_results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_machine = (machine_label or "machine").strip().replace(" ", "_").replace("/", "_")
    run_dir = Path(out_root) / f"{ts}_{safe_machine}"
    _ensure_dir(run_dir)

    out_dir = run_dir / "outputs"
    cond_dir = run_dir / "conditions"
    _ensure_dir(out_dir)
    _ensure_dir(cond_dir)

    saved_outputs: Dict[str, str] = {}
    outputs = results.get("outputs") or {}
    for key, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            saved_outputs[key] = _save_df(df, out_dir / key)

    saved_conditions: Dict[str, str] = {}
    conditions = results.get("conditions") or {}
    for key, val in conditions.items():
        if isinstance(val, pd.DataFrame):
            saved_conditions[key] = _save_df(val, cond_dir / key)
        elif isinstance(val, dict):
            subdir = cond_dir / key
            _ensure_dir(subdir)
            for k2, v2 in val.items():
                if isinstance(v2, pd.DataFrame):
                    saved_conditions[f"{key}.{k2}"] = _save_df(v2, subdir / str(k2))

    def _sanitize_params(d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert non-JSON-safe values so metadata can be serialized."""
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, pd.DataFrame):
                out[k] = f"<DataFrame {v.shape[0]}x{v.shape[1]}>"
            elif isinstance(v, pd.Series):
                out[k] = v.tolist()
            elif isinstance(v, (datetime, date)):
                out[k] = v.isoformat()
            elif isinstance(v, pd.Timestamp):
                out[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, dict):
                out[k] = _sanitize_params(v)
            elif isinstance(v, list):
                out[k] = [
                    _sanitize_params(item) if isinstance(item, dict)
                    else str(item) if isinstance(item, (pd.DataFrame, pd.Series))
                    else item
                    for item in v
                ]
            else:
                out[k] = v
        return out

    raw_params = results.get("parameters") or {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "simulations": simulations,
        "aggregation": agg,
        "opportunistic_downtime": opp_dt_ind,
        "spare_systems": spare_ind,
    }

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_xlsx": str(Path(input_xlsx).resolve()),
        "input_sha256": _sha256_file(input_xlsx) if Path(input_xlsx).exists() else None,
        "machine_label": machine_label,
        "parameters": _sanitize_params(raw_params) if isinstance(raw_params, dict) else raw_params,
        "saved_outputs": saved_outputs,
        "saved_conditions": saved_conditions,
    }

    meta_path = run_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, cls=_SafeJSONEncoder), encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "saved_outputs_count": len(saved_outputs),
        "saved_conditions_count": len(saved_conditions),
    }

    return RamSimArchiveResult(
        run_dir=str(run_dir),
        metadata_path=str(meta_path),
        outputs=saved_outputs,
        conditions=saved_conditions,
        summary=summary,
    )
