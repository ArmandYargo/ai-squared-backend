# func_define_components.py
from __future__ import annotations
from typing import List, Optional
import json

# Reuse the LLM JSON helper from the classifier:
from func_classify_data import _llm_json

MODEL_DEFAULT = "gpt-5.2"

def ai_propose_components_coarse(
    machine_hint: str,
    *,
    top_k: int = 10,
    model: str = MODEL_DEFAULT,
) -> List[str]:
    """
    Propose ~top_k HIGH-LEVEL breakdown categories (components OR broad fault themes)
    for the given machine. Use short, coarse, snake_case (e.g., idler, alignment, belt, chute).
    """
    schema = {
        "name": "coarse_component_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": max(10, top_k),
                    "items": {"type": "string"},
                },
            },
            "required": ["components"],
            "additionalProperties": False,
        },
    }

    sys = (
        "You are a world-class reliability/process/maintenance engineer. "
        "List HIGH-LEVEL downtime categories for the machine: components or broad themes "
        "(not specific failure modes). Examples of coarse categories: idler, alignment, belt, chute, "
        "pulley, scraper, drive, gearbox, motor, structure, take_up, electrical_control, sensor, "
        "lubrication, spillage, guarding, power_supply. Use snake_case, very short terms."
    )
    usr = (
        f"Machine type: {machine_hint}\n"
        f"Return ONLY JSON with key 'components' (array of ~8–{top_k} strings). "
        f"Make them COARSE (high-level) for broader matching. No sentences."
    )

    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        json_schema=schema,
        temperature=0.0,
        max_tokens=600,
    )
    comps = obj.get("components", [])
    # tidy + de-dup
    out, seen = [], set()
    for c in comps:
        s = str(c).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        s = "_".join([t for t in s.split("_") if t])  # collapse repeats
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def ai_apply_edit_to_components(
    current: List[str],
    user_message: str,
    *,
    model: str = MODEL_DEFAULT,
) -> List[str]:
    """Apply natural-language edits (add/remove/rename) to the coarse list."""
    schema = {
        "name": "component_edit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "final": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 40},
            },
            "required": ["final"],
            "additionalProperties": False,
        },
    }

    sys = (
        "You are helping a user edit a COARSE list of downtime categories (snake_case). "
        "Interpret the user's message to add/remove/rename/expand, but keep items HIGH-LEVEL "
        "(components/themes) rather than specific failure modes.\n\n"
        "IMPORTANT: When the user asks to EXPAND or BREAK DOWN a category:\n"
        "- REMOVE the original parent category\n"
        "- ADD its detailed subcategories\n"
        "Example: 'expand motor' → remove 'motor', add 'motor_bearings', 'motor_windings', 'motor_housing'\n\n"
        "Support multiple languages (English, Afrikaans, etc.) - treat words like 'expand', 'verbreed', "
        "'break down', 'uitbrei' as meaning the same thing."
    )
    usr = (
        f"Current list: {json.dumps(current, ensure_ascii=False)}\n"
        f"User message: {user_message}\n"
        "Return ONLY JSON with 'final' (array of strings). Keep snake_case; no duplicates.\n"
        "Remember: when expanding a category, REPLACE it with its subcategories (don't keep both)."
    )

    obj = _llm_json(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        json_schema=schema,
        temperature=0.0,
        max_tokens=600,
    )
    final = obj.get("final", [])
    # tidy + de-dup
    out, seen = [], set()
    for c in final:
        s = str(c).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        s = "_".join([t for t in s.split("_") if t])
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if not out:
        raise ValueError("Edit produced an empty component list.")
    return out

def define_breakdown_components_interactive(
    machine_hint: str,
    *,
    model: str = MODEL_DEFAULT,
) -> List[str]:
    """Interactive loop: propose COARSE categories → user edits → confirm.
    NEW: If the user presses ENTER on the edit prompt (blank), accept the current list immediately.
    """
    components = ai_propose_components_coarse(machine_hint, top_k=10, model=model)

    def show_list(lst: List[str]):
        print("\nProposed COARSE breakdown categories (you can edit):")
        for i, c in enumerate(lst, 1):
            print(f"  {i:>2}. {c}")

    show_list(components)

    while True:
        msg = input(
            "\nType natural-language edits (e.g., 'remove scraper, add alignment & drive'),\n"
            "or press ENTER to accept the list: "
        ).strip()

        # If blank, immediately accept current list
        if msg == "":
            print("Accepted proposed categories.")
            break

        try:
            new_components = ai_apply_edit_to_components(components, msg, model=model)
            show_list(new_components)
            ok2 = input("Apply these changes and use this list? (Y/n): ").strip().lower()
            if ok2 in ("", "y", "yes"):
                components = new_components
                break
            else:
                # Keep editing; retain the last shown list so user can build iteratively
                components = new_components
                continue
        except Exception as e:
            print(f"Edit error: {e}")
            show_list(components)

    return components
