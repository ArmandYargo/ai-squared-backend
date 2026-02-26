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

    raw_results = ram_model.run_ram_simulation(
        input_xlsx=input_xlsx,
        start_date=start_date,
        end_date=end_date,
        simulations=int(simulations),
        agg=agg,
        opp_dt_ind=int(opp_dt_ind),
        spare_ind=int(spare_ind),
    )

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

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_xlsx": str(Path(input_xlsx).resolve()),
        "input_sha256": _sha256_file(input_xlsx) if Path(input_xlsx).exists() else None,
        "machine_label": machine_label,
        "parameters": results.get("parameters") or {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "simulations": simulations,
            "aggregation": agg,
            "opportunistic_downtime": opp_dt_ind,
            "spare_systems": spare_ind,
        },
        "saved_outputs": saved_outputs,
        "saved_conditions": saved_conditions,
    }

    meta_path = run_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
