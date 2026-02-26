# agent/ram_tool.py
from __future__ import annotations

import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date

import pandas as pd

# Ensure the folder containing func_*.py is importable.
RAM_MODULE_DIR = Path(__file__).resolve().parents[1] / "ram_module"
if str(RAM_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(RAM_MODULE_DIR))

from func_ingest_data import ingest_cmms_workbook, assess_ram_readiness_for_machine
from func_define_components import ai_propose_components_coarse, ai_apply_edit_to_components
from func_classify_data import step4_with_query
from func_analyse_data import analyse_and_append_to_excel  # IMPORTANT: signature is (classified_df, excel_path, ...)
from func_date_filter import extract_dates_and_update_output  # used only for stamping, best-effort

from func_inputsheet import (
    build_comp_att_from_summary,
    build_timelines,
    build_usage0,
    export_input_workbook,
)


def _safe_slug(s: str) -> str:
    return "".join([c.lower() if c.isalnum() else "_" for c in (s or "").strip()]).strip("_")


def _log(msg: str):
    print(msg, flush=True)


def check_ram_readiness(
    excel_path: str,
    machine: str,
    *,
    model: str = "gpt-5.2",
    min_coverage: float = 0.50,
    match_scope: str = "both",
) -> Dict[str, Any]:
    """
    Run only the early pipeline steps needed to decide whether the selected file
    is "ready" for building a RAM input sheet.
    """
    df_master, report = ingest_cmms_workbook(excel_path, model=model, min_coverage=min_coverage)

    subset, readiness, filt_meta = assess_ram_readiness_for_machine(
        df_master,
        report["mapping"],
        machine,
        model=model,
        min_coverage=min_coverage,
        match_scope=match_scope,
    )

    return {
        "excel_path": excel_path,
        "machine": machine,
        "subset_rows": int(len(subset)),
        "mapping": report.get("mapping", {}),
        "readiness": readiness,
        "filt_meta": filt_meta,
    }


# -----------------------------
# Date filtering helpers
# -----------------------------
def _pick_date_column(df: pd.DataFrame, mapping: Dict[str, Any]) -> Optional[str]:
    # Prefer mapping hints if present
    candidates: List[str] = []
    for k in ("start_date", "date", "breakdown_date", "notification_date", "workorder_date"):
        v = mapping.get(k)
        if isinstance(v, str) and v in df.columns:
            candidates.append(v)

    # fallback: any column with "date" in name
    if not candidates:
        for c in df.columns:
            if "date" in str(c).lower():
                candidates.append(c)

    # choose the most parseable
    best = None
    best_n = 0
    for c in candidates:
        try:
            s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            n = int(s.notna().sum())
            if n > best_n:
                best_n = n
                best = c
        except Exception:
            continue
    return best


def _parse_yearish_range(text: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Minimal parser supporting:
      - "2023-2024" / "2023 to 2024"
      - "2024"
      - "2024-01 to 2024-03"
    """
    t = (text or "").strip()
    if not t:
        return None

    import re

    m = re.match(r"^\s*(\d{4})\s*(?:-|/|to)\s*(\d{4})\s*$", t, re.IGNORECASE)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        start = pd.Timestamp(datetime(y1, 1, 1))
        end = pd.Timestamp(datetime(y2, 12, 31, 23, 59, 59))
        return start, end

    m = re.match(r"^\s*(\d{4})\s*$", t)
    if m:
        y = int(m.group(1))
        start = pd.Timestamp(datetime(y, 1, 1))
        end = pd.Timestamp(datetime(y, 12, 31, 23, 59, 59))
        return start, end

    m = re.match(r"^\s*(\d{4})-(\d{2})\s*(?:-|/|to)\s*(\d{4})-(\d{2})\s*$", t, re.IGNORECASE)
    if m:
        y1, mo1, y2, mo2 = map(int, m.groups())
        start = pd.Timestamp(datetime(y1, mo1, 1))
        end = (pd.Timestamp(datetime(y2, mo2, 1)) + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
        return start, end

    return None


def _apply_date_filter(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    user_date_range_text: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    text = (user_date_range_text or "").strip()
    if not text:
        return df, {"skipped": True, "reason": "no date filter text provided"}

    date_col = _pick_date_column(df, mapping)
    if not date_col:
        return df, {"skipped": True, "reason": "no usable date column found"}

    parsed = _parse_yearish_range(text)
    if not parsed:
        return df, {"skipped": True, "reason": f"could not parse date range: {text}", "date_col": date_col}

    start, end = parsed
    s = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    mask = s.notna() & (s >= start) & (s <= end)
    out = df.loc[mask].copy()

    meta = {
        "date_col": date_col,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "n_before": int(len(df)),
        "n_after": int(len(out)),
        "skipped": False,
    }
    return out, meta


def _days_in_range_from_meta(date_meta: Dict[str, Any]) -> Optional[int]:
    if not date_meta or date_meta.get("skipped"):
        return None
    try:
        start = pd.to_datetime(date_meta["start_date"])
        end = pd.to_datetime(date_meta["end_date"])
        # inclusive-ish
        return int((end - start).days) + 1
    except Exception:
        return None


def _start_date_for_timelines(date_meta: Dict[str, Any]) -> date:
    if not date_meta or date_meta.get("skipped"):
        return datetime.today().date()
    try:
        return pd.to_datetime(date_meta["start_date"]).date()
    except Exception:
        return datetime.today().date()


# -----------------------------
# Main pipeline
# -----------------------------
def run_ram_pipeline(
    *,
    machine: str,
    date_range_text: str | None = None,
    excel_path: str | None = None,
    preferred_categories: Optional[List[str]] = None,
    category_edit_text: Optional[str] = None,
    model: str = "gpt-5.2",
    min_coverage: float = 0.50,
    match_scope: str = "both",
    outputs_dir: str | Path = "outputs",
    latest_name: str = "ram_input_sheet.xlsx",
) -> Dict[str, Any]:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []

    def log(msg: str):
        log_lines.append(msg)
        _log(msg)

    if not excel_path:
        return {"ok": False, "message": "No excel_path provided.", "outputs": {}, "log": log_lines}

    excel_path = str(excel_path)
    excel_name = Path(excel_path).name

    machine_slug = _safe_slug(machine)
    file_slug = _safe_slug(Path(excel_path).stem)

    classified_path = outputs_dir / f"{file_slug}_{machine_slug}_classified.xlsx"
    input_path = outputs_dir / f"{file_slug}_{machine_slug}_input.xlsx"
    latest_path = outputs_dir / latest_name

    log("=== RAM/Inputsheet pipeline started ===")
    log(f"✅ File selected: {excel_name}")

    # Step 2/7: ingest workbook
    log("Step 2/7: Ingest workbook + decide column mapping…")
    df_master, report = ingest_cmms_workbook(excel_path, model=model, min_coverage=min_coverage)
    mapping = report["mapping"]
    log("✅ Ingest complete.")

    # Step 3/7: readiness check
    log("Step 3/7: Assess readiness for selected machine scope…")
    df_subset, readiness, filt_meta = assess_ram_readiness_for_machine(
        df_master,
        mapping,
        machine,
        model=model,
        min_coverage=min_coverage,
        match_scope=match_scope,
    )
    ok_to_simulate = bool(readiness.get("ok_to_simulate", False))
    log(f"✅ Readiness check complete. OK to simulate?: {ok_to_simulate}")

    # Step 4/7: apply date filter
    log(f"Step 4/7: Apply date filter: {date_range_text or '(none)'}")
    df_filtered, date_meta = _apply_date_filter(df_subset, mapping, date_range_text)
    if date_meta.get("skipped"):
        log(f"ℹ️ Date filter skipped: {date_meta.get('reason')}")
    log(f"✅ Rows after filter: {len(df_filtered)}")

    # Step 5/7: categories
    log("Step 5/7: Prepare coarse breakdown categories…")
    if preferred_categories is None:
        preferred_categories = ai_propose_components_coarse(machine, model=model)
        if category_edit_text:
            preferred_categories = ai_apply_edit_to_components(preferred_categories, category_edit_text, model=model)
    log(f"✅ Categories set ({len(preferred_categories)}): {preferred_categories}")

    # Step 6/7: classify
    log("Step 6/7: Run Step 4 classification (this can take a bit)…")

    # IMPORTANT: your step4_with_query does NOT accept mapping=...
    classified_df, mapping2, end_row, stats, meta = step4_with_query(
        df_filtered,
        machine,
        model=model,
        preferred_categories=preferred_categories,
    )
    log(f"✅ Step 4 complete. Rows classified: {len(classified_df)}")

    # Write classified data to workbook
    log("Writing classified workbook…")
    with pd.ExcelWriter(classified_path, engine="openpyxl") as writer:
        classified_df.to_excel(writer, sheet_name="classified", index=False)
    log(f"✅ Classified workbook written: {classified_path}")

    # Build summary + append into the SAME workbook (correct signature)
    log("Building breakdown summary…")
    days_in_range = _days_in_range_from_meta(date_meta)
    summary_df = analyse_and_append_to_excel(
        classified_df,
        classified_path,
        category_col="breakdown_category",
        duration_col="breakdown_duration_hours",
        sheet_name="category_summary",
        days_in_range=days_in_range,
        model=model,
    )
    log("✅ Summary complete.")

    # Best-effort: stamp the date range using your existing helper (if it works)
    try:
        _ = extract_dates_and_update_output(
            df_master=df_master,
            mapping=mapping,
            machine_query=machine,
            user_date_range_text=(date_range_text or ""),
            out_excel_path=str(classified_path),
            model=model,
            match_scope=match_scope,
        )
        log("✅ Date range stamped into workbook.")
    except Exception:
        # Don’t fail the pipeline if stamping fails
        pass

    # Step 7/7: Build RAM input workbook (using func_inputsheet correctly)
    log("Step 7/7: Export RAM input workbook…")
    comp_att = build_comp_att_from_summary(summary_df, machine_type=machine, model=model)
    timelines = build_timelines(_start_date_for_timelines(date_meta))
    usage0 = build_usage0()

    export_input_workbook(
        input_path,
        comp_att=comp_att,
        timelines=timelines,
        usage0=usage0,
        autosize=True,
        overwrite=True,
    )

    # Latest copy
    shutil.copyfile(input_path, latest_path)

    log(f"✅ Input workbook written: {input_path}")
    log(f"✅ Latest copy written: {latest_path}")
    log("=== RAM/Inputsheet pipeline finished ===")

    return {
        "ok": True,
        "message": "RAM/Inputsheet pipeline completed.",
        "outputs": {
            "classified_path": str(classified_path),
            "input_path": str(input_path),
            "latest_path": str(latest_path),
            "ok_to_simulate": ok_to_simulate,
            "readiness": readiness,
            "date_meta": date_meta,
        },
        "log": log_lines,
    }


# -------------------------
# Compatibility wrapper
# -------------------------
def run_ram_pipeline_compat(**kwargs):
    """
    Compatibility wrapper so the graph can call a stable function name.

    Your wizard currently calls:
      - input_xlsx_path=...
      - machine_type=...

    But run_ram_pipeline expects:
      - excel_path=...
      - machine=...

    This wrapper normalizes common aliases into the canonical names and drops unknown keys.
    """
    if "run_ram_pipeline" not in globals():
        raise ImportError("run_ram_pipeline not found in agent.ram_tool.py")

    mapped = dict(kwargs)

    # ---- Excel file path aliases -> excel_path ----
    if "excel_path" not in mapped or mapped.get("excel_path") in (None, ""):
        for k in ("input_xlsx_path", "input_xlsx", "workbook_path", "path", "file_path"):
            if k in mapped and mapped.get(k):
                mapped["excel_path"] = mapped.get(k)
                break
    # remove alias keys so they don't get passed to run_ram_pipeline
    for k in ("input_xlsx_path", "input_xlsx", "workbook_path", "path", "file_path"):
        mapped.pop(k, None)

    # ---- Machine aliases -> machine ----
    if "machine" not in mapped or mapped.get("machine") in (None, ""):
        for k in ("machine_type", "asset", "equipment"):
            if k in mapped and mapped.get(k):
                mapped["machine"] = mapped.get(k)
                break
    for k in ("machine_type", "asset", "equipment"):
        mapped.pop(k, None)

    # ---- Date range aliases -> date_range_text ----
    if "date_range_text" not in mapped and "date_range" in mapped:
        mapped["date_range_text"] = mapped.pop("date_range")
    if "date_range_text" not in mapped and "date_filter_text" in mapped:
        mapped["date_range_text"] = mapped.pop("date_filter_text")

    # ---- Categories aliases -> preferred_categories ----
    if "preferred_categories" not in mapped and "categories" in mapped:
        mapped["preferred_categories"] = mapped.pop("categories")
    if "preferred_categories" not in mapped and "preferred_category_list" in mapped:
        mapped["preferred_categories"] = mapped.pop("preferred_category_list")

    # ---- Output dir aliases -> outputs_dir ----
    if "outputs_dir" not in mapped and "output_dir" in mapped:
        mapped["outputs_dir"] = mapped.pop("output_dir")
    if "outputs_dir" not in mapped and "out_dir" in mapped:
        mapped["outputs_dir"] = mapped.pop("out_dir")

    # ---- Drop unknown keys (run_ram_pipeline has a strict signature) ----
    allowed = {
        "machine",
        "date_range_text",
        "excel_path",
        "preferred_categories",
        "category_edit_text",
        "model",
        "min_coverage",
        "match_scope",
        "outputs_dir",
        "latest_name",
    }
    mapped = {k: v for k, v in mapped.items() if k in allowed}

    return run_ram_pipeline(**mapped)
