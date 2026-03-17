# ram_module/func_classify_data.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


# =============================================================================
# Backwards-compatible helpers expected by func_ingest_data.py
# =============================================================================
def _snake(s: Any) -> str:
    """
    Header normalizer used across the RAM module.
    Kept for backward compatibility (func_ingest_data imports it).
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _chunks(rows: List[dict], n: int) -> Iterable[List[dict]]:
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def _normalize_cat(s: Any) -> str:
    return _snake(s)


# =============================================================================
# OpenAI helpers (Responses API preferred, chat.completions fallback)
# =============================================================================
def _llm_json(
    *,
    model: str,
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """
    Unified JSON-call helper.
    Compatible with gpt-5.x models (Responses API + Chat fallback).
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from e

    client = OpenAI()

    # ---- Preferred path: Responses API (gpt-5.x compatible) ----
    try:
        resp = client.responses.create(
            model=model,
            input=messages,
            text={"format": {"type": "json_schema", "json_schema": json_schema}},
            temperature=temperature,
            max_output_tokens=max_tokens,   # ✅ correct for Responses API
        )
        txt = getattr(resp, "output_text", None)
        if txt:
            return json.loads(txt)
    except Exception:
        pass

    # ---- Fallback: Chat Completions (use max_completion_tokens) ----
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,  # ✅ correct for gpt-5.x
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content or "{}"
    return json.loads(txt)



# =============================================================================
# Column picking / machine filter
# =============================================================================
_DOMAIN_LINE = "You are a expert reliability engineer, expert section engineer and RAM modelling expert."


def ai_pick_two_columns(df: pd.DataFrame, *, model: str) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    prompt = {
        "role": "user",
        "content": (
            f"{_DOMAIN_LINE}\n"
            "You are helping map columns in an Excel export.\n"
            "Return JSON with keys: functional_location, description.\n"
            f"Available columns: {cols}\n"
            "Pick the best matches.\n"
        ),
    }
    schema = {
        "name": "column_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "functional_location": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
            },
            "required": ["functional_location", "description"],
            "additionalProperties": False,
        },
    }
    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": "Return only JSON."}, prompt],
        json_schema=schema,
        temperature=0.0,
        max_tokens=400,
    )
    return {
        "functional_location": obj.get("functional_location"),
        "description": obj.get("description"),
    }


def ai_pick_duration_column(df: pd.DataFrame, *, model: str) -> Optional[str]:
    """
    Backwards-compatible helper expected by func_ingest_data.py.
    Attempts to pick a downtime duration column (in hours) from df columns.
    """
    cols = list(df.columns)
    prompt = {
        "role": "user",
        "content": (
            f"{_DOMAIN_LINE}\n"
            "Pick the downtime duration column for RAM analysis.\n"
            "Return JSON with key: duration_hours_column (nullable).\n"
            "If there is no downtime duration column, return null.\n"
            f"Available columns: {cols}\n"
            "Hints: duration, downtime, down time, hours, repair time, outage duration.\n"
        ),
    }
    schema = {
        "name": "duration_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "duration_hours_column": {"type": ["string", "null"]},
            },
            "required": ["duration_hours_column"],
            "additionalProperties": False,
        },
    }
    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": "Return only JSON."}, prompt],
        json_schema=schema,
        temperature=0.0,
        max_tokens=300,
    )
    col = obj.get("duration_hours_column")
    if isinstance(col, str) and col in df.columns:
        return col
    return None


def ai_build_machine_filter(series: pd.Series, user_machine_query: str, *, model: str) -> Tuple[str, Optional[str], dict]:
    sample = series.dropna().astype(str).head(50).tolist()
    prompt = {
        "role": "user",
        "content": (
            f"{_DOMAIN_LINE}\n"
            "We want a regex include/exclude filter for functional locations.\n"
            f"User machine query: {user_machine_query}\n"
            f"Sample functional locations: {sample}\n"
            "Return JSON: include_regex, exclude_regex (nullable), rationale.\n"
            "Keep patterns conservative but useful.\n"
        ),
    }
    schema_floc = {
        "name": "machine_filter",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "include_regex": {"type": "string"},
                "exclude_regex": {"type": ["string", "null"]},
                "rationale": {"type": "string"},
            },
            "required": ["include_regex", "exclude_regex", "rationale"],
            "additionalProperties": False,
        },
    }
    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": "Return only JSON."}, prompt],
        json_schema=schema_floc,
        temperature=0.0,
        max_tokens=600,
    )
    return obj["include_regex"], obj.get("exclude_regex"), {"rationale": obj.get("rationale", "")}


# =============================================================================
# Failure mechanism inference guardrail
# =============================================================================
MECHANISM_ENUM = ["Random", "Wear", "Fatigue", "Corrosion"]

_WEAR_KW = re.compile(
    r"\b(wear|worn|abrasion|abrad|fray|frayed|scuff|scuffed|rub|rubbing|"
    r"seiz|seized|stuck|jam|jammed|overheat|hot\b|vibration|"
    r"tear\b|splic|lacing|lagging|slip|slipping)\b",
    re.I,
)
_FATIGUE_KW = re.compile(
    r"\b(fatigue|crack|cracked|fracture|fractured|snapped|shear|sheared|"
    r"broken\s*shaft|weld\s*crack|repeat(ed)?\s*crack)\b",
    re.I,
)
_CORROSION_KW = re.compile(
    r"\b(corrosion|corrod|rust|rusted|oxid|pitt|pitting|scale|"
    r"leak\b.*(corros|rust)|hole\b.*(rust|corros))\b",
    re.I,
)


def infer_failure_mechanism(text: str) -> str:
    if not text:
        return "Random"
    if _CORROSION_KW.search(text):
        return "Corrosion"
    if _FATIGUE_KW.search(text):
        return "Fatigue"
    if _WEAR_KW.search(text):
        return "Wear"
    return "Random"


# =============================================================================
# Deterministic fallback rules (only used when AI low confidence)
# =============================================================================
@dataclass
class RuleResult:
    category: str
    confidence: float
    rationale: str
    mechanism_hint: Optional[str] = None


CONVEYOR_RULES: List[Tuple[str, re.Pattern, str, Optional[str]]] = [
    ("belt_damage", re.compile(r"\b(splice|lacing|fastener|fray|frayed|tear|belt\s*tear|belt\s*damage|abrasion)\b", re.I), "belt", "Wear"),
    ("belt_tracking", re.compile(r"\b(skew|belt\s*track|tracking|drift|misalign)\b", re.I), "belt", "Wear"),
    ("idler_seized", re.compile(r"\b(idler|roller).*(seiz|seized|jam|jammed|stuck)\b|\b(seiz|seized).*(idler|roller)\b", re.I), "idler", "Wear"),
    ("pulley_lagging", re.compile(r"\b(pulley|lagging)\b", re.I), "pulley", "Wear"),
    ("chute_blockage", re.compile(r"\b(chute|blockage|blocked|plugged|hang[-\s]?up|hung\s*up)\b", re.I), "chute", "Wear"),
    ("drive_trip", re.compile(r"\b(underspeed|overload|trip|vfd|drive\s*trip|control\s*trip)\b", re.I), "electrical_control", "Random"),
]


def _deterministic_fallback(description: str, machine_hint: str, allowed_set: set) -> Optional[RuleResult]:
    d = (description or "").strip()
    if not d:
        return None

    mh = (machine_hint or "").lower()
    use_conveyor = ("conveyor" in mh) or ("cnv" in mh)

    if use_conveyor:
        for rule_name, pat, cat, mech in CONVEYOR_RULES:
            if pat.search(d):
                if allowed_set and (cat not in allowed_set) and (cat != "other"):
                    continue
                return RuleResult(
                    category=cat,
                    confidence=0.85,
                    rationale=f"REGEX fallback: {rule_name}",
                    mechanism_hint=mech,
                )
    return None


# =============================================================================
# AI-first classifier (two-pass)
# =============================================================================
def _llm_classify_batch(
    rows: List[dict],
    *,
    model: str,
    machine_hint: str,
    allowed_categories: List[str],
    temperature: float = 0.0,
    mode: str = "fast",
) -> List[dict]:
    allowed = [c for c in (allowed_categories or []) if c]
    allowed_norm = sorted(set([_normalize_cat(c) for c in allowed if _normalize_cat(c)]))
    if "other" not in allowed_norm:
        allowed_norm.append("other")

    schema = {
        "name": "classification_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "breakdown_category": {"type": "string"},
                            "failure_modes": {"type": "string", "enum": MECHANISM_ENUM},
                            "confidence": {"type": "number"},
                            "evidence": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["id", "breakdown_category", "failure_modes", "confidence", "evidence", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }

    guidance = (
        f"{_DOMAIN_LINE}\n"
        "You classify maintenance event text for industrial/mining equipment.\n"
        "Pick breakdown_category ONLY from the allowed list (normalized).\n"
        "If none fit, use 'other'.\n"
        "failure_modes MUST be one of: Random, Wear, Fatigue, Corrosion.\n"
        "Return an evidence snippet copied from the description (short phrase).\n"
        "Confidence is 0.0–1.0; use <0.7 when uncertain or description is vague.\n"
    )
    if mode == "refine":
        guidance += (
            "This is a refinement pass. Be conservative: if evidence is weak, lower confidence.\n"
            "Ensure category aligns with the evidence phrase.\n"
        )

    prompt = {
        "role": "user",
        "content": (
            f"{guidance}\n"
            f"Machine/equipment scope hint: {machine_hint}\n"
            f"Allowed breakdown categories (normalized): {allowed_norm}\n"
            f"Rows:\n{rows}\n"
            "Return ONLY JSON.\n"
        ),
    }

    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": "Return only JSON."}, prompt],
        json_schema=schema,
        temperature=temperature,
        max_tokens=1400 if mode == "refine" else 1200,
    )
    return obj.get("items", [])


def _validate_item(
    it: dict,
    *,
    allowed_set: set,
    description: str,
) -> Tuple[str, str, float, str]:
    cat = _normalize_cat(it.get("breakdown_category", "")) or "other"
    mech = it.get("failure_modes", "Random")
    evidence = str(it.get("evidence", "") or "").strip()
    rationale = str(it.get("rationale", "") or "").strip()

    try:
        conf = float(it.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    if allowed_set and (cat not in allowed_set) and cat != "other":
        cat = "other"
        conf = min(conf, 0.55)
        rationale = (rationale + " | forced to other (not in allowed category set)").strip(" |")

    if mech not in MECHANISM_ENUM:
        mech = infer_failure_mechanism(description)
        conf = min(conf, 0.65)
        rationale = (rationale + " | mechanism corrected by guardrail").strip(" |")

    if not evidence:
        conf = min(conf, 0.60)
        rationale = (rationale + " | no evidence phrase provided").strip(" |")

    # We keep evidence inside rationale so the sheet stays compact.
    if evidence:
        rationale = (rationale + f" | evidence: {evidence[:80]}").strip(" |")

    return cat, mech, conf, rationale


# =============================================================================
# Two-signal classification: FLOC code (primary) + description regex (secondary)
# =============================================================================

def _build_floc_code_map(
    floc_series: pd.Series,
    categories: List[str],
    machine_hint: str,
    *,
    model: str = "gpt-5.2",
) -> Dict[str, str]:
    """
    Extract unique FLOC type-codes and ask the LLM to map each code to a
    category.  FLOC codes like 'BET-001', 'ELE-PNL-001', 'CHU-001' are the
    primary signal for classification.
    Returns {code_prefix: category}, e.g. {'BET': 'belt', 'ELE': 'electrical'}.
    """
    raw_flocs = floc_series.dropna().astype(str).unique().tolist()

    codes: List[str] = []
    for f in raw_flocs:
        parts = f.split("-")
        for i in range(len(parts) - 1, 0, -1):
            candidate = "-".join(parts[i:])
            if any(c.isalpha() for c in candidate):
                codes.append(candidate)
                break

    unique_codes = sorted(set(codes))[:80]
    if not unique_codes:
        return {}

    cats_norm = [_normalize_cat(c) for c in categories if _normalize_cat(c)]

    schema = {
        "name": "floc_code_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "category": {"type": "string"},
                        },
                        "required": ["code", "category"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["mappings"],
            "additionalProperties": False,
        },
    }

    prompt = {
        "role": "user",
        "content": (
            f"{_DOMAIN_LINE}\n"
            "I have equipment FLOC (Functional Location) type-codes from SAP and a set of "
            "coarse breakdown categories.  Map each code to the BEST matching category.\n\n"
            "Common SAP FLOC naming conventions:\n"
            "- BET = belt, PUL = pulley, IDL = idler, DRV/DRI = drive\n"
            "- ELE/ELC = electrical, PNL = panel, SWH = switch, MOT = motor\n"
            "- CHU = chute, SCA = scraper, STR/STL = structure\n"
            "- INE/INS = instrumentation, TSM = transmitter, DTE = detector, MET = meter\n"
            "- HYD = hydraulic, VLV = valve, GBX = gearbox\n"
            "- PWR/POW = power_supply\n\n"
            "If a code does not fit any category, map it to 'other'.\n\n"
            f"Machine: {machine_hint}\n"
            f"Categories: {cats_norm}\n"
            f"FLOC type-codes: {unique_codes}\n\n"
            "Return ONLY JSON."
        ),
    }

    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": "Return only JSON."}, prompt],
        json_schema=schema,
        temperature=0.0,
        max_tokens=1500,
    )

    code_map: Dict[str, str] = {}
    allowed = set(cats_norm) | {"other"}
    for item in obj.get("mappings", []):
        code = (item.get("code") or "").strip()
        cat = _normalize_cat(item.get("category", ""))
        if code and cat and cat in allowed:
            code_map[code.upper()] = cat
    return code_map


def ai_build_category_regexes(
    descriptions: List[str],
    categories: List[str],
    machine_hint: str,
    *,
    model: str = "gpt-5.2",
) -> Dict[str, "re.Pattern[str]"]:
    """
    Ask the LLM once to produce a regex for each category based on sample
    descriptions.  Returns {category: compiled_regex}.  Used as a secondary
    classifier when the FLOC code doesn't resolve a category.
    """
    unique_descs = list(dict.fromkeys(d.strip() for d in descriptions if d.strip()))
    sample = unique_descs[:150]

    cats_norm = [_normalize_cat(c) for c in categories if _normalize_cat(c)]
    if not cats_norm:
        return {}

    schema = {
        "name": "category_regexes",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "regex": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["category", "regex", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["patterns"],
            "additionalProperties": False,
        },
    }

    prompt = {
        "role": "user",
        "content": (
            f"{_DOMAIN_LINE}\n"
            "I have maintenance work-order DESCRIPTION text (free-text, often written by "
            "semi-literate operators — expect abbreviations, misspellings, and shorthand).  "
            "For EACH breakdown category, write a Python case-insensitive regex that matches "
            "descriptions belonging to that category.\n\n"
            "Rules:\n"
            "- Study the sample descriptions carefully.\n"
            "- Be BROAD: use alternation (|) generously for synonyms, abbreviations, "
            "common misspellings, partial words.\n"
            "- Keywords to look for: part names (belt, roller, idler, pulley, chute, motor, "
            "scraper, gearbox), actions (tear, broken, worn, skew, alignment, sparks, noise), "
            "sensor/electrical terms (underspeed, blockchute, alarm, fault, trip, interlock, "
            "sima code, pullwire, retain).\n"
            "- Do NOT use look-aheads/look-behinds. Use \\b for word boundaries where helpful.\n"
            "- Do NOT include an 'other' category.\n\n"
            f"Machine scope: {machine_hint}\n"
            f"Categories (snake_case): {cats_norm}\n\n"
            f"Sample descriptions ({len(sample)} of {len(unique_descs)} unique):\n"
            + "\n".join(f"  - {d}" for d in sample)
            + "\n\nReturn ONLY JSON."
        ),
    }

    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": "Return only JSON."}, prompt],
        json_schema=schema,
        temperature=0.0,
        max_tokens=3500,
    )

    patterns: Dict[str, "re.Pattern[str]"] = {}
    for item in obj.get("patterns", []):
        cat = _normalize_cat(item.get("category", ""))
        raw = (item.get("regex") or "").strip()
        if not cat or not raw or cat == "other":
            continue
        try:
            patterns[cat] = re.compile(raw, re.IGNORECASE)
        except re.error:
            continue
    return patterns


def classify_two_signal(
    work: pd.DataFrame,
    floc_code_map: Dict[str, str],
    desc_patterns: Dict[str, "re.Pattern[str]"],
    floc_col: str = "functional_location",
    desc_col: str = "description",
) -> Dict[str, int]:
    """
    Classify every row using two signals:
      1. FLOC type-code (primary, high confidence)
      2. Description regex (secondary, lower confidence)
    Mutates work in-place.  Returns stats dict.
    """
    stats = {"floc_matched": 0, "desc_matched": 0, "other": 0}
    desc_ordered = list(desc_patterns.items())

    for i in work.index:
        floc_val = str(work.at[i, floc_col]).upper()
        desc_val = str(work.at[i, desc_col])

        matched_cat = None
        source = ""

        for code, cat in floc_code_map.items():
            if cat == "other":
                continue
            if code.upper() in floc_val:
                matched_cat = cat
                source = f"FLOC code contains '{code}'"
                stats["floc_matched"] += 1
                break

        if not matched_cat:
            for cat, pat in desc_ordered:
                if pat.search(desc_val):
                    matched_cat = cat
                    source = f"description regex: /{pat.pattern}/"
                    stats["desc_matched"] += 1
                    break

        if matched_cat:
            work.at[i, "breakdown_category"] = matched_cat
            work.at[i, "ai_confidence"] = 0.95 if "FLOC" in source else 0.80
            work.at[i, "ai_rationale"] = source
        else:
            work.at[i, "breakdown_category"] = "other"
            work.at[i, "ai_confidence"] = 0.0
            work.at[i, "ai_rationale"] = "no FLOC code or description match"
            stats["other"] += 1

        work.at[i, "failure_modes"] = infer_failure_mechanism(desc_val)

    return stats


# =============================================================================
# Main step4 processing (NO failure_mode column)
# =============================================================================
def step4_process(
    df: pd.DataFrame,
    *,
    blank_run: int = 5,
    model: str = "gpt-5.2",
    machine_include: Optional[Union[str, re.Pattern]] = None,
    machine_exclude: Optional[Union[str, re.Pattern]] = None,
    require_match: bool = True,
    machine_hint: Optional[str] = None,
    preferred_categories: Optional[List[str]] = None,
    ai_confidence_threshold: float = 0.70,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], int, Dict[str, int]]:
    mapping = ai_pick_two_columns(df, model=model)
    floc_col = mapping.get("functional_location")
    desc_col = mapping.get("description")
    if floc_col is None or desc_col is None:
        raise ValueError(f"Could not identify required columns. Mapping={mapping}")

    floc = df[floc_col].astype(str).fillna("")
    desc = df[desc_col].astype(str).fillna("")

    # Filter
    if machine_include is not None:
        inc_pat = re.compile(machine_include, re.I) if isinstance(machine_include, str) else machine_include
        mask_inc = floc.str.contains(inc_pat, na=False)
    else:
        mask_inc = pd.Series([True] * len(df), index=df.index)

    if machine_exclude:
        exc_pat = re.compile(machine_exclude, re.I) if isinstance(machine_exclude, str) else machine_exclude
        mask_exc = floc.str.contains(exc_pat, na=False)
    else:
        mask_exc = pd.Series([False] * len(df), index=df.index)

    mask = mask_inc & (~mask_exc)
    if require_match and not bool(mask.any()):
        raise ValueError("No rows matched machine filter (require_match=True).")

    filtered = df.loc[mask].copy()
    end_idx = int(filtered.index.max()) if len(filtered) else 0

    # Allowed categories
    preferred = [_normalize_cat(p) for p in (preferred_categories or [])]
    preferred = [p for p in preferred if p]
    allowed_set = set(preferred)

    # Output
    work = pd.DataFrame(index=filtered.index)
    work["functional_location"] = filtered[floc_col].astype(str)
    work["description"] = filtered[desc_col].astype(str)
    work["breakdown_category"] = "other"
    work["failure_modes"] = "Random"
    work["ai_confidence"] = 0.0
    work["ai_rationale"] = ""

    # duration best-effort (ingest also handles this elsewhere)
    duration_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "duration" in cl or ("down" in cl and "hour" in cl) or ("time" in cl and "hour" in cl):
            duration_col = c
            break
    used_col = None
    if duration_col is not None:
        work["breakdown_duration_hours"] = pd.to_numeric(filtered[duration_col], errors="coerce")
        used_col = duration_col
    else:
        work["breakdown_duration_hours"] = pd.NA

    mapping_out: Dict[str, Optional[str]] = {"functional_location": floc_col, "description": desc_col, "duration": used_col}

    # Signal 1: map FLOC type-codes to categories (1 LLM call)
    floc_code_map = _build_floc_code_map(
        work["functional_location"],
        preferred or [],
        machine_hint or "",
        model=model,
    )

    # Signal 2: build description regexes (1 LLM call)
    all_descs = work["description"].tolist()
    desc_patterns = ai_build_category_regexes(
        all_descs,
        preferred or [],
        machine_hint or "",
        model=model,
    )

    # Apply both signals: FLOC first, description fallback
    stats = classify_two_signal(
        work, floc_code_map, desc_patterns,
        floc_col="functional_location", desc_col="description",
    )

    # Final safety: mechanism enum always valid
    for i in work.index:
        mech = str(work.at[i, "failure_modes"] or "")
        if mech not in MECHANISM_ENUM:
            work.at[i, "failure_modes"] = infer_failure_mechanism(str(work.at[i, "description"]))

    return work, mapping_out, end_idx, stats


def step4_with_query(
    df: pd.DataFrame,
    user_machine_query: str,
    *,
    blank_run: int = 5,
    model: str = "gpt-5.2",
    require_match: bool = True,
    preferred_categories: Optional[List[str]] = None,
    ai_confidence_threshold: float = 0.70,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], int, Dict[str, int], dict]:
    mapping = ai_pick_two_columns(df, model=model)
    floc_col = mapping.get("functional_location")
    if floc_col is None:
        raise ValueError(f"Could not identify functional location column. Mapping={mapping}")

    inc_re, exc_re, meta = ai_build_machine_filter(df[floc_col], user_machine_query, model=model)

    classified_df, mapping2, end_row, stats = step4_process(
        df,
        blank_run=blank_run,
        model=model,
        machine_include=inc_re,
        machine_exclude=exc_re,
        require_match=require_match,
        machine_hint=user_machine_query,
        preferred_categories=preferred_categories,
        ai_confidence_threshold=ai_confidence_threshold,
    )
    return classified_df, mapping2, end_row, stats, meta
