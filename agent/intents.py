# intents.py
from __future__ import annotations

import re
from typing import Dict, Any


# High-precision signals only (avoid accidental routing)
RAM_COMMAND_PREFIXES = ("/ram",)

# If user explicitly expresses they want to build/run the RAM input sheet workflow,
# we route to run_ram. This is intentionally *small* and *specific*.
RUN_RAM_PATTERNS = [
    r"\bbuild\b.*\binput\s*sheet\b",
    r"\bcreate\b.*\binput\s*sheet\b",
    r"\bgenerate\b.*\binput\s*sheet\b",
    r"\brun\b.*\binput\s*sheet\b",
    r"\bbuild\b.*\bram\b",
    r"\bcreate\b.*\bram\b",
    r"\bgenerate\b.*\bram\b",
    r"\brun\b.*\bram\b",
    r"\bram\b.*\binput\s*sheet\b",
    r"\bsimulate\b.*\bram\b",
]


def heuristic_intent(text: str) -> str:
    """
    Minimal, safety-first intent routing.

    - Only routes to run_ram when user is explicitly asking to run/build/generate a RAM input workflow
      or they use explicit /ram commands.
    - Everything else routes to ask (QA), where your graph should do RAG-first then LLM fallback.

    This avoids brittle keyword lists and reduces accidental action routing.
    """
    t = (text or "").strip().lower()
    if not t:
        return "ask"

    # Explicit wizard commands should always route to run_ram / wizard handler
    if any(t.startswith(p) for p in RAM_COMMAND_PREFIXES):
        return "run_ram"

    # Strong “run RAM workflow” signals only
    for pat in RUN_RAM_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return "run_ram"

    # Default: QA (RAG-first happens downstream)
    return "ask"


def extract_machine_and_range(text: str) -> Dict[str, Any]:
    """
    Very lightweight extraction:
    - machine: first 'wordy' token after 'for' or 'machine'
    - date_range: common patterns like 2023-2024 or 2023 to 2024

    (This is safe to keep; it doesn't affect routing.)
    """
    t = (text or "").strip()
    out: Dict[str, Any] = {"machine": None, "date_range": None}

    m = re.search(r"(?:for|machine)\s+([A-Za-z][A-Za-z0-9_\-\s]{1,40})", t, flags=re.I)
    if m:
        out["machine"] = m.group(1).strip().strip(".,")

    d = re.search(r"\b(20\d{2})\s*[-to]+\s*(20\d{2})\b", t, flags=re.I)
    if d:
        out["date_range"] = f"{d.group(1)}-{d.group(2)}"
    else:
        d2 = re.search(r"\b(20\d{2})-(20\d{2})\b", t)
        if d2:
            out["date_range"] = f"{d2.group(1)}-{d2.group(2)}"

    return out
