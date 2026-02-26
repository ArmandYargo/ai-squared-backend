"""
ram_results.py

Utilities to persist RAM simulation outputs in an archive-friendly way.

Design goals:
- deterministic folder structure per run
- machine-readable metadata.json for later trend dashboards
- table outputs saved as parquet when available, else csv
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_write_parquet(df: pd.DataFrame, out_path: Path) -> bool:
    try:
        df.to_parquet(out_path, index=False)
        return True
    except Exception:
        return False


def _write_table(df: pd.DataFrame, out_dir: Path, name: str) -> Tuple[str, str]:
    """
    Returns (format, filepath).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"{name}.parquet"
    if _try_write_parquet(df, pq):
        return ("parquet", str(pq))
    csv = out_dir / f"{name}.csv"
    df.to_csv(csv, index=False)
    return ("csv", str(csv))


@dataclass
class RamArchive:
    run_dir: Path
    tables_dir: Path
    conditions_dir: Path
    meta_path: Path


def make_run_dir(root: str | Path, run_id: Optional[str] = None) -> RamArchive:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = root / run_id
    tables_dir = run_dir / "tables"
    conditions_dir = run_dir / "conditions"
    meta_path = run_dir / "metadata.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    return RamArchive(run_dir=run_dir, tables_dir=tables_dir, conditions_dir=conditions_dir, meta_path=meta_path)


def save_ram_results(
    results: Dict[str, Any],
    *,
    input_xlsx: str | Path,
    out_root: str | Path = "ram_runs",
    run_id: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist RAM results to disk and return a manifest:

    {
      "run_dir": "...",
      "metadata_path": "...",
      "tables": { "yearly_component": {"format": "...", "path": "..."}, ... },
      "conditions": {...}
    }
    """
    input_xlsx = Path(input_xlsx)
    archive = make_run_dir(out_root, run_id=run_id)

    tables_manifest: Dict[str, Dict[str, str]] = {}
    cond_manifest: Dict[str, Any] = {}

    # ---- core tables
    for key in [
        "yearly_component",
        "monthly_component",
        "yearly_subcomponent",
        "monthly_subcomponent",
        "yearly_simulations",
        "monthly_simulations",
    ]:
        df = results.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            fmt, p = _write_table(df, archive.tables_dir, key)
            tables_manifest[key] = {"format": fmt, "path": p}
        elif isinstance(df, pd.DataFrame):
            fmt, p = _write_table(df, archive.tables_dir, key)
            tables_manifest[key] = {"format": fmt, "path": p}

    # ---- condition tables (nested)
    cond = results.get("condition_data", {}) or {}
    if isinstance(cond, dict):
        ac = cond.get("all_components")
        if isinstance(ac, pd.DataFrame):
            fmt, p = _write_table(ac, archive.conditions_dir, "all_components")
            cond_manifest["all_components"] = {"format": fmt, "path": p}

        # percentiles
        percs = cond.get("percentiles", {}) or {}
        if isinstance(percs, dict):
            pct_dir = archive.conditions_dir / "percentiles"
            for name, df in percs.items():
                if isinstance(df, pd.DataFrame):
                    fmt, p = _write_table(df, pct_dir, f"cond_{name}")
                    cond_manifest.setdefault("percentiles", {})[name] = {"format": fmt, "path": p}

        # component_conditions can be massive; store each as separate table
        comp_conds = cond.get("component_conditions", {}) or {}
        if isinstance(comp_conds, dict) and comp_conds:
            cc_dir = archive.conditions_dir / "component_conditions"
            cc_dir.mkdir(parents=True, exist_ok=True)
            cond_manifest["component_conditions"] = {}
            for name, df in comp_conds.items():
                if isinstance(df, pd.DataFrame):
                    safe = name.replace("/", "_")
                    fmt, p = _write_table(df, cc_dir, safe)
                    cond_manifest["component_conditions"][name] = {"format": fmt, "path": p}

        # simulations (if list/df)
        sims = cond.get("simulations")
        if isinstance(sims, pd.DataFrame):
            fmt, p = _write_table(sims, archive.conditions_dir, "all_comp_sims")
            cond_manifest["simulations"] = {"format": fmt, "path": p}

    # ---- metadata
    params = results.get("parameters", {}) or {}
    meta = {
        "created_at_utc": _utc_now_iso(),
        "input_xlsx": str(input_xlsx),
        "input_sha256": _sha256_file(input_xlsx) if input_xlsx.exists() else None,
        "parameters": _jsonify(params),
        "outputs": {
            "tables": tables_manifest,
            "conditions": cond_manifest,
        },
    }
    if extra_meta:
        meta["extra"] = _jsonify(extra_meta)

    archive.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "run_dir": str(archive.run_dir),
        "metadata_path": str(archive.meta_path),
        "tables": tables_manifest,
        "conditions": cond_manifest,
    }


def _jsonify(obj: Any) -> Any:
    """
    Convert objects (date, Timestamp, DataFrame, numpy types) to JSON-safe values.
    """
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return {"_type": "DataFrame", "rows": int(len(obj)), "cols": int(len(obj.columns))}
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if np is not None:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    return str(obj)
