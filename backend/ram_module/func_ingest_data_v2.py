# func_ingest_data.py
from __future__ import annotations

import json
import math
import re
import difflib
from typing import Dict, List, Tuple, Optional

import pandas as pd

# Reuse helpers from your Step 4 module:
# - _snake : header normalizer
# - ai_pick_two_columns : pick FLOC & Description
# - ai_pick_duration_column : pick downtime (hours)
# - _llm_json : JSON-call helper
# - ai_build_machine_filter : derive include/exclude regex from machine query + FLOC samples
from func_classify_data import (
    _snake,
    ai_pick_two_columns,
    ai_pick_duration_column,
    _llm_json,
    ai_build_machine_filter,
)


# ----------------------------
# 1) Read workbook (Excel/CSV)
# ----------------------------
def read_all_sheets(path: str) -> Dict[str, pd.DataFrame]:
    """
    Reads .xlsx (all sheets) or .csv into a dict of DataFrames.
    Keys are sheet names ('Sheet1' for CSV).
    """
    p = str(path)
    if p.lower().endswith(".csv"):
        df = pd.read_csv(p, encoding_errors="ignore")
        return {"Sheet1": df}
    elif p.lower().endswith((".xlsx", ".xlsm", ".xls")):
        xls = pd.ExcelFile(p)
        return {s: xls.parse(s) for s in xls.sheet_names}
    else:
        raise ValueError("Unsupported file type. Please provide .xlsx or .csv")


# ------------------------------------------
# 2) Heuristics to score sheets / find roles
# ------------------------------------------
_FUNCLOC_HINTS = [
    "functional location", "func. loc.", "func loc", "floc", "equipment", "asset",
    "location", "tag", "equipment id", "equipment_no", "functional_loc",
]
_DESC_HINTS = [
    "description", "fault description", "work description", "breakdown", "remarks",
    "comments", "text", "problem", "failure description", "symptoms",
]
_DURATION_HINTS = [
    "breakdown duration", "downtime", "duration", "time lost", "time loss",
    "repair time", "fix time", "outage duration", "time to repair", "mttr",
    "stoppage time", "down time", "delay time", "hours", "hrs",
]
_DATETIME_HINTS = [
    "start", "end", "date", "time", "malfunct", "reported", "created", "completed",
]

def _name_match_score(col: str, hints: List[str]) -> float:
    c = _snake(col)
    best = 0.0
    for h in hints:
        r = difflib.SequenceMatcher(a=c, b=_snake(h)).ratio()
        if r > best:
            best = r
    return best


# -------------------------------------------
# 3) Discover relationships (key candidates)
# -------------------------------------------
def _normalize_values(series: pd.Series, max_unique: int = 3000) -> pd.Series:
    """String-normalize for set overlap comparisons (lowercase, strip)."""
    s = series.dropna().astype(str).str.strip().str.lower()
    if s.nunique(dropna=True) > max_unique:
        # sample down to keep it light
        s = s.sample(n=max_unique, random_state=0)
    return s

def _jaccard_on_columns(a: pd.Series, b: pd.Series) -> float:
    sa = set(_normalize_values(a))
    sb = set(_normalize_values(b))
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / max(1, len(union))


# ------------------------------------------
# 4) Score sheets & propose joins
# ------------------------------------------
def _sheet_score(df: pd.DataFrame) -> float:
    """Score how likely a sheet is the base 'event' table."""
    if df.empty:
        return 0.0
    cols = list(df.columns)
    score = 0.0
    # presence/strength of key roles
    best_floc = max((_name_match_score(c, _FUNCLOC_HINTS) for c in cols), default=0.0)
    best_desc = max((_name_match_score(c, _DESC_HINTS) for c in cols), default=0.0)
    best_dur  = max((_name_match_score(c, _DURATION_HINTS) for c in cols), default=0.0)
    best_dt   = max((_name_match_score(c, _DATETIME_HINTS) for c in cols), default=0.0)
    score += 3.0 * best_floc + 2.0 * best_desc + 1.5 * best_dur + 1.0 * best_dt
    # reward row count (log scale)
    score += 0.5 * math.log10(max(len(df), 1))
    return score

def propose_joins(
    sheets: Dict[str, pd.DataFrame],
    *,
    min_overlap: float = 0.35,
) -> List[Tuple[str, str, str, str, float]]:
    """
    For each pair of sheets, propose a join (left_col, right_col) with highest Jaccard overlap.
    Returns tuples: (left_sheet, right_sheet, left_col, right_col, overlap)
    Only includes proposals with overlap >= min_overlap.
    """
    names = list(sheets.keys())
    proposals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            L, R = names[i], names[j]
            dfL, dfR = sheets[L], sheets[R]
            best = (None, None, 0.0)
            for cL in dfL.columns:
                for cR in dfR.columns:
                    # quick name prefilter to avoid O(C^2) heavy compares
                    if difflib.SequenceMatcher(a=_snake(cL), b=_snake(cR)).ratio() < 0.45:
                        continue
                    ov = _jaccard_on_columns(dfL[cL], dfR[cR])
                    if ov > best[2]:
                        best = (cL, cR, ov)
            if best[0] and best[1] and best[2] >= min_overlap:
                proposals.append((L, R, best[0], best[1], best[2]))
    proposals.sort(key=lambda t: t[4], reverse=True)
    return proposals


# ----------------------------------------------------
# 5) Build unified event table by joining useful sheets
# ----------------------------------------------------
def _useful_columns(df: pd.DataFrame) -> List[str]:
    """Columns that might add value for RAM: duration & datetime-ish."""
    cols = []
    for c in df.columns:
        if _name_match_score(c, _DURATION_HINTS) >= 0.5:
            cols.append(c)
        elif _name_match_score(c, _DATETIME_HINTS) >= 0.55:
            cols.append(c)
    return list(dict.fromkeys(cols))  # preserve order, dedupe

def build_unified_event_table(sheets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """
    Picks a base sheet, then left-joins other sheets that add useful columns
    using strongest join proposals (one per right sheet).
    Returns (unified_df, relation_report)
    """
    if not sheets:
        raise ValueError("No sheets found.")
    if len(sheets) == 1:
        name = next(iter(sheets))
        return sheets[name].copy(), {"base_sheet": name, "joins": []}

    # Choose base by score
    scores = {name: _sheet_score(df) for name, df in sheets.items()}
    base_name = max(scores, key=scores.get)
    base = sheets[base_name].copy()

    # Propose joins
    joins = propose_joins(sheets)

    used_right = set()
    relation_report = {"base_sheet": base_name, "joins": []}

    for L, R, cL, cR, ov in joins:
        if L != base_name and R != base_name:
            continue
        right_name = R if L == base_name else L
        if right_name in used_right:
            continue

        df_right = sheets[right_name]
        # Only bring columns that look useful
        add_cols = _useful_columns(df_right)
        if not add_cols:
            continue

        # Pick correct left/right column orientation
        left_col, right_col = (cL, cR) if L == base_name else (cR, cL)

        before_cols = set(base.columns)
        try:
            base = base.merge(
                df_right[[right_col] + add_cols],
                left_on=left_col,
                right_on=right_col,
                how="left",
                suffixes=("", f"__{right_name}"),
            )
            used_right.add(right_name)
            new_cols = [c for c in base.columns if c not in before_cols]
            relation_report["joins"].append(
                {
                    "right_sheet": right_name,
                    "left_key": left_col,
                    "right_key": right_col,
                    "overlap": round(ov, 3),
                    "added_columns": new_cols,
                }
            )
        except Exception as e:
            relation_report["joins"].append(
                {
                    "right_sheet": right_name,
                    "left_key": left_col,
                    "right_key": right_col,
                    "overlap": round(ov, 3),
                    "error": str(e),
                }
            )
            continue

    return base, relation_report


# ----------------------------------------------------
# 6) Canonical mapping via GPT-5.2 helpers (from Step 4)
# ----------------------------------------------------
def canonical_map_and_attach_duration(
    df_unified: pd.DataFrame,
    *,
    model: str = "gpt-5.2",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Uses ai_pick_two_columns + ai_pick_duration_column to:
      - map Functional Location & Description
      - attach best downtime-in-hours as 'breakdown_duration_hours' (if found)
    Returns (df_ready, mapping_report)
    """
    mapping = ai_pick_two_columns(df_unified, model=model)
    floc_col = mapping.get("functional_location")
    desc_col = mapping.get("description")

    if floc_col is None or desc_col is None:
        raise ValueError(f"Could not identify required columns. Mapping={mapping}")

    ready = df_unified.copy()
    dur_col = ai_pick_duration_column(ready, model=model)
    used_col = None
    non_null = 0
    note = ""

    if dur_col and dur_col in ready.columns:
        ready["breakdown_duration_hours"] = pd.to_numeric(ready[dur_col], errors="coerce")
        used_col = dur_col
        non_null = int(ready["breakdown_duration_hours"].notna().sum())
        note = "duration attached from selected column"

    mapping["duration"] = used_col
    mapping["duration_non_null"] = non_null
    mapping["duration_note"] = note or ("no plausible duration column found" if used_col is None else "")

    return ready, {"mapping": mapping}


# ------------------------------------------
# 7) RAM readiness (coverage & pass/fail)
# ------------------------------------------
def assess_ram_readiness(
    df_ready: pd.DataFrame,
    *,
    floc_col: Optional[str],
    desc_col: Optional[str],
    duration_col: Optional[str] = "breakdown_duration_hours",
    min_coverage: float = 0.50,
) -> Dict:
    """
    Computes coverage for essential RAM inputs on the given dataframe and returns pass/fail.
    """
    def coverage(series: pd.Series) -> float:
        if series is None or series is False:
            return 0.0
        return float(series.notna().mean()) if len(series) else 0.0

    metrics = {}

    if floc_col and floc_col in df_ready.columns:
        metrics["functional_location"] = round(coverage(df_ready[floc_col]), 3)
    else:
        metrics["functional_location"] = 0.0

    if desc_col and desc_col in df_ready.columns:
        metrics["description"] = round(coverage(df_ready[desc_col]), 3)
    else:
        metrics["description"] = 0.0

    if duration_col and duration_col in df_ready.columns:
        s = pd.to_numeric(df_ready[duration_col], errors="coerce")
        metrics["duration_hours"] = round(float(s.notna().mean()), 3)
    else:
        metrics["duration_hours"] = 0.0

    essentials = ["functional_location", "description", "duration_hours"]
    ok = all(metrics[k] >= min_coverage for k in essentials)

    details = {
        "min_coverage_threshold": min_coverage,
        "essentials": essentials,
        "metrics": metrics,
        "ok_to_simulate": ok,
        "message": (
            "Data meets minimum coverage to proceed with RAM classification/simulation."
            if ok else
            "Coverage is below the minimum threshold. Please provide missing fields or more complete data before simulating."
        )
    }
    return details


# ---------------------------------------------------
# 8) One-call orchestrator
# ---------------------------------------------------
def ingest_cmms_workbook(
    path: str,
    *,
    model: str = "gpt-5.2",
    min_coverage: float = 0.50,
) -> Tuple[pd.DataFrame, Dict]:
    """
    High-level ingest:
      - Read all sheets
      - Build unified event table by joining useful sheets
      - Map canonical columns and attach duration (hours)
      - Assess RAM readiness (overall, before machine filter)
    Returns (df_ready, report)
    """
    sheets = read_all_sheets(path)

    sheets_summary = {
        name: {"rows": int(len(df)), "cols": int(len(df.columns)), "columns": list(map(str, df.columns))}
        for name, df in sheets.items()
    }

    unified, relations = build_unified_event_table(sheets)
    ready, mapping_report = canonical_map_and_attach_duration(unified, model=model)
    mapping = mapping_report["mapping"]

    readiness = assess_ram_readiness(
        ready,
        floc_col=mapping.get("functional_location"),
        desc_col=mapping.get("description"),
        duration_col="breakdown_duration_hours",
        min_coverage=min_coverage,
    )

    report = {
        "sheets_summary": sheets_summary,
        "relations": relations,
        "mapping": mapping,
        "readiness": readiness,
    }
    return ready, report


# ---------------------------------------------------
# 9) AI feedback for missing/low coverage + guidance
# ---------------------------------------------------
def generate_readiness_feedback(
    report: Dict,
    *,
    model: str = "gpt-5.2",
) -> Dict:
    """
    Produces natural-language guidance using GPT-5.2:
      - missing_fields: essentials with 0.0 coverage
      - low_coverage: essentials present but < threshold
      - messages: {'missing': str|None, 'low': str|None, 'overall': str}
    """
    readiness = report.get("readiness", {})
    mapping = report.get("mapping", {})
    threshold = float(readiness.get("min_coverage_threshold", 0.50))
    metrics = readiness.get("metrics", {})
    essentials = readiness.get("essentials", ["functional_location", "description", "duration_hours"])
    ok = bool(readiness.get("ok_to_simulate", False))

    cov_floc = float(metrics.get("functional_location", 0.0))
    cov_desc = float(metrics.get("description", 0.0))
    cov_dur  = float(metrics.get("duration_hours", 0.0))

    missing_fields: List[str] = []
    if cov_floc == 0.0:
        missing_fields.append("functional_location")
    if cov_desc == 0.0:
        missing_fields.append("description")
    if cov_dur == 0.0:
        missing_fields.append("duration_hours")

    low_cov: List[Dict] = []
    for field, cov in [("functional_location", cov_floc), ("description", cov_desc), ("duration_hours", cov_dur)]:
        if 0.0 < cov < threshold:
            low_cov.append({"field": field, "coverage_pct": round(cov * 100.0, 1)})

    messages = {"overall": readiness.get("message", ""), "missing": None, "low": None}

    ctx = {
        "threshold_pct": int(threshold * 100),
        "metrics": metrics,
        "mapping": mapping,
        "missing_fields": missing_fields,
        "low_coverage": low_cov,
        "essentials": essentials,
    }

    if missing_fields:
        schema_missing = {
            "name": "missing_feedback",
            "strict": True,
            "schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        }
        prompt_missing = {
            "role": "user",
            "content": (
                "You are a reliability/maintenance/RAM expert. "
                "Explain to a plant engineer, in concise bullet points, why each of these missing inputs is essential "
                "for a credible RAM simulation, what the model cannot estimate without them, and what to provide to fix it. "
                "Return JSON with key 'message' (string). "
                f"Context:\n{json.dumps(ctx, ensure_ascii=False)}"
            ),
        }
        try:
            obj = _llm_json(
                model=model,
                messages=[{"role": "system", "content": "Write concise, practical plant engineering guidance."}, prompt_missing],
                json_schema=schema_missing,
                temperature=0.2,
                max_tokens=800,
            )
            messages["missing"] = obj.get("message")
        except Exception:
            bullets = []
            if "functional_location" in missing_fields:
                bullets.append("- Functional Location/tag missing: cannot tie events to assets; failure rates become meaningless.")
            if "description" in missing_fields:
                bullets.append("- Description missing: cannot classify breakdown cause/mode; category analytics and mitigation suffer.")
            if "duration_hours" in missing_fields:
                bullets.append("- Duration hours missing: cannot compute downtime/MTTR; availability impact becomes inaccurate.")
            messages["missing"] = "\n".join(bullets)

    if low_cov:
        schema_low = {
            "name": "low_feedback",
            "strict": True,
            "schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        }
        prompt_low = {
            "role": "user",
            "content": (
                "You are a reliability/maintenance/RAM expert. "
                "Explain the risks of running a RAM model with low completeness in essentials, field by field. "
                "Note the likely bias/variance and what to do next to improve each field. "
                "Return JSON with key 'message' (string). "
                f"Context:\n{json.dumps(ctx, ensure_ascii=False)}"
            ),
        }
        try:
            obj2 = _llm_json(
                model=model,
                messages=[{"role": "system", "content": "Write concise, practical plant engineering guidance."}, prompt_low],
                json_schema=schema_low,
                temperature=0.2,
                max_tokens=800,
            )
            messages["low"] = obj2.get("message")
        except Exception:
            bullets = [f"- {x['field']}: {x['coverage_pct']}% filled → high uncertainty; enrich before simulation."]
            messages["low"] = "\n".join(bullets)

    return {
        "ok_to_simulate": ok,
        "threshold": threshold,
        "metrics": metrics,
        "missing_fields": missing_fields,
        "low_coverage": low_cov,
        "messages": messages,
    }


# ---------------------------------------------------
# 10) Machine-specific readiness on filtered subset
# ---------------------------------------------------
def assess_ram_readiness_for_machine(
    df_ready: pd.DataFrame,
    mapping: Dict,
    machine_query: str,
    *,
    model: str = "gpt-5.2",
    min_coverage: float = 0.50,
    match_scope: str = "floc",   # 'floc' | 'desc' | 'both'
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Builds regex for the given machine (via FLOC samples), filters df_ready by
    FLOC/Description per match_scope, and computes readiness on the subset.
    Returns (subset_df, readiness_dict, filter_meta)
    """
    floc_col = mapping.get("functional_location")
    desc_col = mapping.get("description")
    if not floc_col or not desc_col or floc_col not in df_ready.columns or desc_col not in df_ready.columns:
        raise ValueError(f"Invalid mapping for readiness: {mapping}")

    # 1) Build include/exclude regex via AI (based on FLOC samples)
    inc_re, exc_re, meta = ai_build_machine_filter(df_ready[floc_col], machine_query, model=model)
    
def _to_noncapturing(pat: str) -> str:
    # Convert ONLY plain capturing groups "(" into "(?:"
    # - not preceded by backslash (so we don't touch literal "\(")
    # - not followed by "?" (so we don't touch "(?= ...", "(?! ...", "(?: ...", "(?P<...>", etc.)
    return re.sub(r"(?<!\\)\((?!\?)", "(?:", pat)

# Apply conversion safely ONLY if it compiles; otherwise keep original.
for _name in ("inc_re", "exc_re"):
    _pat = locals().get(_name)
    if isinstance(_pat, str) and _pat:
        try:
            re.compile(_pat)  # ensure original is valid
        except re.error:
            # If AI generated an invalid regex, let the original error surface later,
            # but DON'T try to rewrite it (rewriting can make diagnostics worse).
            continue

        _cand = _to_noncapturing(_pat)
        if _cand != _pat:
            try:
                re.compile(_cand)  # only accept if rewritten is still valid
                if _name == "inc_re":
                    inc_re = _cand
                else:
                    exc_re = _cand
            except re.error:
                # keep original if rewrite broke it
                pass

    # 2) Build masks over both columns
    floc_s = df_ready[floc_col].astype(str)
    desc_s = df_ready[desc_col].astype(str)

    m_inc_floc = floc_s.str.contains(inc_re, na=False)
    m_inc_desc = desc_s.str.contains(inc_re, na=False)
    if exc_re:
        m_exc_floc = floc_s.str.contains(exc_re, na=False)
        m_exc_desc = desc_s.str.contains(exc_re, na=False)
    else:
        m_exc_floc = pd.Series(False, index=df_ready.index)
        m_exc_desc = pd.Series(False, index=df_ready.index)

    # 3) Choose scope
    if match_scope == "floc":
        selected_mask = m_inc_floc & ~m_exc_floc
    elif match_scope == "desc":
        selected_mask = m_inc_desc & ~m_exc_desc
    elif match_scope == "both":
        selected_mask = ((m_inc_floc | m_inc_desc) & ~(m_exc_floc | m_exc_desc))
    else:
        raise ValueError("match_scope must be one of: 'floc', 'desc', 'both'")

    subset = df_ready[selected_mask].copy()
    stats = {
        "n_total": int(len(df_ready)),
        "n_selected": int(len(subset)),
        "matches_floc": int(m_inc_floc.sum()),
        "matches_desc": int(m_inc_desc.sum()),
        "scope_used": match_scope,
    }

    # 4) If nothing matched in 'floc', auto-fallback to 'both'
    if stats["n_selected"] == 0 and match_scope == "floc":
        union_mask = ((m_inc_floc | m_inc_desc) & ~(m_exc_floc | m_exc_desc))
        subset = df_ready[union_mask].copy()
        stats["scope_used"] = "both(fallback)"
        stats["n_selected"] = int(len(subset))

    # 5) Compute readiness on the subset
    readiness = assess_ram_readiness(
        subset,
        floc_col=floc_col,
        desc_col=desc_col,
        duration_col="breakdown_duration_hours",
        min_coverage=min_coverage,
    )

    return subset, readiness, {"regex_meta": meta, "stats": stats, "include_regex": inc_re, "exclude_regex": exc_re}

