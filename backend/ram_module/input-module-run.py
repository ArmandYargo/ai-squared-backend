# input-module-run.py
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Optional
import shutil
import pandas as pd

from func_select_data import select_site_data
from func_ingest_data import (
    ingest_cmms_workbook,
    generate_readiness_feedback,
    assess_ram_readiness_for_machine,
)
from func_define_components import define_breakdown_components_interactive
from func_classify_data import step4_with_query, _llm_json
from func_analyse_data import (
    analyse_and_append_to_excel,
    summarise_breakdowns,
    append_summary_sheet_to_excel,
)
from func_date_filter import (
    _ai_pick_start_date_column,
    _ai_parse_date_range,
    extract_dates_and_update_output,
)
from func_weibull_fit import (
    fit_weibull_by_category,
    apply_weibull_to_summary,
)
from func_inputsheet import (
    build_comp_att_from_summary,
    build_timelines,
    build_usage0,
    export_input_workbook,
)

# ---- Config ----
MODEL = "gpt-5.2"
MIN_COVERAGE = 0.50
BLANK_RUN = 5
AGGRESSIVENESS = "assertive"  # 'conservative' | 'balanced' | 'assertive'
DAYFIRST = True               # interpret ambiguous dates as DD/MM/YYYY

# Silence pandas per-element parse warnings from pd.to_datetime
warnings.filterwarnings(
    "ignore",
    message=r"Could not infer format, so each element will be parsed individually, falling back to `dateutil`.*",
    category=UserWarning,
)

# Common equipment tokens (fallback heuristic)
_MACHINE_TOKENS = {
    "conveyor","pump","crusher","screen","motor","gearbox","gear","gearpump","fan","blower","compressor",
    "turbine","generator","kiln","boiler","mill","agitator","mixer","thickener","filter","press","centrifuge",
    "separator","chiller","hvac","vfd","feeder","valve","pipeline","pipe","sump","hoist","winch",
    "stacker","reclaimer","bucket","elevator","belt","idler","pulley","drive","take_up","scraper",
    "chute","clarifier","flotation","dryer","cooler","jaw","cone","impact"
}

def _looks_like_machine_term_heuristic(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t or len(t) < 3:
        return False
    if any(tok in t for tok in _MACHINE_TOKENS):
        return True
    if re.search(r"(pump|convey|motor|gear|screen|crusher|mill|fan|blower|compress|drive|feeder|valve|kiln|boiler|turbine|generator)", t):
        return True
    return False

def _ai_is_machine_term(term: str, *, model: str = MODEL) -> Optional[bool]:
    """
    Use GPT-5.2 to decide if `term` is an industrial machine/equipment type.
    Returns True/False or None if uncertain/failed.
    """
    t = (term or "").strip()
    if not t:
        return False

    schema = {
        "name": "machine_check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_machine": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
            "required": ["is_machine", "confidence"],
            "additionalProperties": False,
        },
    }
    sys = (
        "Classify whether a user-provided term is an INDUSTRIAL MACHINE or EQUIPMENT TYPE "
        "(e.g., conveyor, pump, jaw crusher, screen, motor, gearbox, compressor, kiln, generator, VFD). "
        "If ambiguous (fruit, verb, animal, etc.), prefer False."
    )
    usr = f"Term: {t}\nReturn JSON only."

    try:
        obj = _llm_json(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            json_schema=schema,
            temperature=0.0,
            max_tokens=150,
        )
        is_machine = bool(obj.get("is_machine", False))
        conf = float(obj.get("confidence", 0.0))
        if conf >= 0.65:
            return is_machine
        return None
    except Exception:
        return None

def is_machine(term: str) -> bool:
    """AI-first; fall back to heuristic when AI is uncertain/unavailable."""
    ai = _ai_is_machine_term(term)
    if ai is True:
        return True
    if ai is False:
        return False
    return _looks_like_machine_term_heuristic(term)

def _kind_reprompt() -> None:
    print("\nThat doesn’t look like an equipment/machine type.")
    print("Please enter something like: conveyor, pump, crusher, screen, motor, gearbox.\n")

def _ensure_summary_structure(df: pd.DataFrame) -> None:
    required = [
        "Breakdown Category", "Primary Failure Mode", "pf_n",
        "Beta", "Eta", "MTTR_fail", "MTTR_replace", "MTTR_repair",
        "Downtime Count",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in 'category_summary': {missing}")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set in the environment.")
        print("Set it in your Conda env, restart Spyder, and rerun.")
        return

    # -------------------
    # 1) Ask machine type (AI-first validation)
    # -------------------
    while True:
        machine = input("Enter machine type (e.g., conveyor, pump, crusher): ").strip()
        if not is_machine(machine):
            _kind_reprompt()
            continue
        break

    # -------------------
    # 2) Ask date range (optional)
    # -------------------
    date_text = input(
        "\nOptional date filter (inclusive).\n"
        "Examples: 'from 2025/03/12 to 2025/06/02'  |  'November to Jan'  |  '2024-01 to 2024-03'\n"
        "Press ENTER to skip: "
    ).strip()

    # -------------------
    # 3) Ask to upload file
    # -------------------
    while True:
        path = select_site_data()
        if not path:
            print("No file selected. Exiting.")
            return
        print(f"\nSelected: {path}")

        # 4) Ingest workbook
        df_ready, report = ingest_cmms_workbook(path, model=MODEL, min_coverage=MIN_COVERAGE)

        print("\n=== Ingest report ===")
        print("Sheets found:", list(report["sheets_summary"].keys()))
        print("Base sheet:", report["relations"]["base_sheet"])
        joins = report["relations"]["joins"]
        if joins:
            print("Joins:")
            for j in joins:
                right = j.get("right_sheet")
                lk = j.get("left_key")
                rk = j.get("right_key")
                ov = j.get("overlap")
                added = j.get("added_columns", [])
                err = j.get("error")
                if err:
                    print(f"  - {right}: {lk} ↔ {rk} (overlap {ov}) -> ERROR {err}")
                else:
                    print(f"  - {right}: {lk} ↔ {rk} (overlap {ov}) -> +{len(added)} cols")
        else:
            print("No useful joins added (single-sheet or no strong overlaps).")

        print("\nColumn mapping decided by AI:")
        print(report["mapping"])

        df_master = df_ready.copy(deep=True)
        floc_col = report["mapping"].get("functional_location")
        desc_col = report["mapping"].get("description")

        # 5) Validate that the chosen machine exists in THIS file; re-prompt as needed
        while True:
            try:
                subset, machine_ready, filt_meta = assess_ram_readiness_for_machine(
                    df_master,
                    mapping=report["mapping"],
                    machine_query=machine,
                    model=MODEL,
                    min_coverage=MIN_COVERAGE,
                    match_scope="both",
                )
            except Exception:
                # Suppress detailed traces and previews; show a short prompt only
                ans = input(
                    "\nIt seems the machine you entered could not be found. "
                    "Would you like to re-enter the machine (r) / upload a new file (u) / quit (q): "
                ).strip().lower()
                if ans in ("u", "upload"):
                    break
                if ans in ("q", "quit"):
                    return
                while True:
                    machine = input("\nEnter machine type (e.g., conveyor, pump, crusher): ").strip()
                    if is_machine(machine):
                        break
                    _kind_reprompt()
                continue

            stats = filt_meta.get("stats", {})
            if stats.get("n_selected", 0) == 0:
                # Suppress previews; show short prompt only
                ans = input(
                    "\nIt seems the machine you entered could not be found. "
                    "Would you like to re-enter the machine (r) / upload a new file (u) / quit (q): "
                ).strip().lower()
                if ans in ("u", "upload"):
                    break
                if ans in ("q", "quit"):
                    return
                while True:
                    machine = input("\nEnter machine type (e.g., conveyor, pump, crusher): ").strip()
                    if is_machine(machine):
                        break
                    _kind_reprompt()
                continue  # re-try on same file

            # Found! Proceed with normal flow on this file & machine
            print("\n--- Machine-specific readiness (on filtered rows) ---")
            print("Selection stats:", stats, "| include_regex:", filt_meta.get("include_regex"))
            print("Metrics:", machine_ready["metrics"])
            print("OK to simulate?:", machine_ready["ok_to_simulate"])
            print(machine_ready["message"])

            # === Apply optional date range prior to classification ===
            prefilter_mask = None
            parsed_range = None
            date_col_used = None
            start_d = end_d = None
            days_inclusive: Optional[int] = None

            if date_text:
                # Build machine mask for df_master
                inc_re = filt_meta.get("include_regex")
                exc_re = filt_meta.get("exclude_regex")
                floc_s = df_master[floc_col].astype(str)
                desc_s = df_master[desc_col].astype(str)
                m_inc_floc = floc_s.str.contains(inc_re, na=False)
                m_inc_desc = desc_s.str.contains(inc_re, na=False)
                if exc_re:
                    m_exc_floc = floc_s.str.contains(exc_re, na=False)
                    m_exc_desc = desc_s.str.contains(exc_re, na=False)
                else:
                    m_exc_floc = pd.Series(False, index=df_master.index)
                    m_exc_desc = pd.Series(False, index=df_master.index)
                machine_mask = (m_inc_floc | m_inc_desc) & ~(m_exc_floc | m_exc_desc)

                machine_subset = df_master[machine_mask].copy()
                date_col_used = _ai_pick_start_date_column(machine_subset, model=MODEL, dayfirst=DAYFIRST)

                try:
                    start_d, end_d, _meta = _ai_parse_date_range(date_text, model=MODEL)
                    parsed_range = (start_d, end_d)
                    days_inclusive = (end_d - start_d).days + 1
                except Exception as e:
                    print(f"Warning: couldn't interpret date range ({e}). Continuing without date filter.")
                    parsed_range = None

                if parsed_range and date_col_used:
                    parsed_all = pd.to_datetime(df_master[date_col_used], errors="coerce", dayfirst=DAYFIRST).dt.date
                    prefilter_mask = machine_mask & (parsed_all >= start_d) & (parsed_all <= end_d)
                    n_pref = int(prefilter_mask.sum())
                    print(f"\nDate filter recognized: {start_d.isoformat()} → {end_d.isoformat()} using '{date_col_used}'.")
                    print(f"Rows in date range before classification: {n_pref}")
                elif parsed_range and not date_col_used:
                    print("\nNo plausible start-date column found for this machine subset. Proceeding without date filter.")

            # Optional coarse categories
            preferred_categories = None
            try:
                use_coarse = input(
                    "\nHave the AI propose COARSE breakdown categories and let you edit them? (Y/n): "
                ).strip().lower()
                if use_coarse in ("", "y", "yes"):
                    preferred_categories = define_breakdown_components_interactive(machine, model=MODEL)
                    print("\nUsing preferred coarse categories:")
                    print(preferred_categories)
            except Exception as e:
                print(f"(Skipping coarse-category definition due to error: {e})")

            # 6) Step 4 classification (use date prefilter if present)
            print("\nRunning Step 4 (classification)…")
            df_for_step4 = df_master[prefilter_mask].copy() if isinstance(prefilter_mask, pd.Series) else df_master.copy()

            classified_df, mapping2, end_row, stats2, meta = step4_with_query(
                df_for_step4,
                user_machine_query=machine,
                model=MODEL,
                blank_run=BLANK_RUN,
                require_match=True,
                site_terms=None,
                aggressiveness=AGGRESSIVENESS,
                preferred_categories=preferred_categories,
            )

            # Output-only rename: failure_modes -> primary_failure_modes
            if "failure_modes" in classified_df.columns:
                classified_df = classified_df.rename(columns={"failure_modes": "primary_failure_modes"})

            print("\n--- Step 4 results ---")
            print("Column mapping (Step 4):", mapping2)
            print("End-of-data row:", end_row)
            print("Selection stats:", stats2)
            print("Regex builder meta:", meta)

            # 7) Save 'classified' sheet
            in_path = Path(path)
            out_path = in_path.with_name(f"{in_path.stem}_{machine.lower()}_classified.xlsx")
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                classified_df.to_excel(writer, sheet_name="classified", index=False)

            # 8) Build & write category_summary (with MTTR/MTBF/Beta/Eta and Primary Failure Mode/pf_n)
            try:
                summary_df = analyse_and_append_to_excel(
                    classified_df,
                    str(out_path),
                    sheet_name="category_summary",
                    days_in_range=days_inclusive,  # may be None if user skipped
                    model=MODEL,
                )
            except Exception as e:
                print(f"Warning: failed to append summary sheet: {e}")
                summary_df = summarise_breakdowns(classified_df)

            # 8b) Optional: override Beta/Eta via Weibull fits if we can
            if parsed_range and date_col_used:
                try:
                    date_series_for_classified = pd.to_datetime(
                        df_master.loc[classified_df.index, date_col_used],
                        errors="coerce",
                        dayfirst=DAYFIRST
                    )
                    weibull_df = fit_weibull_by_category(
                        classified_df,
                        category_col="breakdown_category",
                        date_series=date_series_for_classified,
                        within_start=pd.Timestamp(start_d),
                        within_end=pd.Timestamp(end_d),
                    )
                    if not weibull_df.empty:
                        summary_weib = apply_weibull_to_summary(
                            summary_df,
                            weibull_df,
                            beta_col_out="Beta",
                            eta_col_out="Eta",
                        )
                        append_summary_sheet_to_excel(summary_weib, str(out_path), sheet_name="category_summary")
                        summary_df = summary_weib
                        print("Applied Weibull fits to Beta/Eta where sufficient intervals existed.")
                except Exception as e:
                    print(f"Warning: Weibull fitting step skipped due to error: {e}")

            # 9) Stamp the date-range stats into summary (if a range was provided)
            if parsed_range:
                try:
                    _ = extract_dates_and_update_output(
                        df_master=df_master,
                        mapping=report["mapping"],
                        machine_query=machine,
                        user_date_range_text=date_text,
                        out_excel_path=str(out_path),
                        model=MODEL,
                        match_scope="both",
                        dayfirst=DAYFIRST,
                        summary_sheet="category_summary",
                    )
                except Exception as e:
                    print(f"Warning: failed to stamp date-range info: {e}")

            print(f"\nClassified workbook written to: {out_path}")

            # ---- Ask whether to create input sheet now ----
            def _build_and_optionally_edit_input_sheet(current_summary: pd.DataFrame, start_date_fallback: Optional[pd.Timestamp] = None):
                # Determine timeline start date
                if parsed_range and start_d is not None:
                    start_date_for_input = start_d
                else:
                    start_date_for_input = (start_date_fallback.date() if isinstance(start_date_fallback, pd.Timestamp)
                                            else pd.Timestamp("today").date())

                # Build input components/timelines/usage (base/exported template)
                comp_att_df = build_comp_att_from_summary(
                    current_summary,
                    machine_type=machine,
                    model=MODEL,
                )
                timelines = build_timelines(start_date_for_input)
                usage0 = build_usage0()

                input_out_path = in_path.with_name(f"{in_path.stem}_{machine.lower()}_input.xlsx")
                export_input_workbook(
                    input_out_path,
                    comp_att=comp_att_df,
                    timelines=timelines,
                    usage0=usage0,
                    autosize=True,
                    overwrite=True,
                )
                print(f"\nInput workbook written to: {input_out_path}")

                dest = input_out_path.with_name("ram_input_sheet.xlsx")

                # Offer to edit & re-upload input workbook
                try:
                    ans_edit = input(
                        "\nWould you like to edit the exported input sheet and upload the edited file? (y/N): "
                    ).strip().lower()
                    if ans_edit in ("y", "yes"):
                        edited_path = select_site_data(
                            allowed_exts={".xlsx", ".xls"},
                            title="Select your edited RAM input workbook",
                        )
                        if edited_path:
                            try:
                                shutil.copy2(edited_path, dest)
                                print(f"Edited input workbook saved as: {dest}")
                            except Exception as e:
                                print(f"Error saving edited input as ram_input_sheet.xlsx: {e}")
                        else:
                            # No file chosen; fall back to saving the exported template as ram_input_sheet.xlsx
                            try:
                                shutil.copy2(input_out_path, dest)
                                print(f"No file selected. Exported input saved as: {dest}")
                            except Exception as e:
                                print(f"Error saving exported input as ram_input_sheet.xlsx: {e}")
                    else:
                        # User chose not to edit: save the exported input as ram_input_sheet.xlsx
                        try:
                            shutil.copy2(input_out_path, dest)
                            print(f"Exported input saved as: {dest}")
                        except Exception as e:
                            print(f"Error saving exported input as ram_input_sheet.xlsx: {e}")
                except Exception as e:
                    print(f"Warning: edit/upload step skipped due to error: {e}")
                    # Best-effort: still save the exported input as ram_input_sheet.xlsx
                    try:
                        shutil.copy2(input_out_path, dest)
                        print(f"(Fallback) Exported input saved as: {dest}")
                    except Exception as e2:
                        print(f"(Fallback) Error saving exported input as ram_input_sheet.xlsx: {e2}")

                # Intentionally no in-memory df update.

            create_now = input("\nCreate RAM input sheet now from the current output? (Y/n): ").strip().lower()
            if create_now in ("", "y", "yes"):
                _build_and_optionally_edit_input_sheet(summary_df, start_date_fallback=None)
                return  # done
            else:
                while True:
                    choice = input(
                        "\nWhat would you like to do next?\n"
                        "  [h] I'm happy with the output file (finish without creating input sheet)\n"
                        "  [e] Edit the output file and re-upload (then create input sheet from the uploaded file)\n"
                        "  [q] Quit now\n"
                        "Choose (h/e/q): "
                    ).strip().lower()

                    if choice == "h":
                        print("Finishing without creating an input sheet.")
                        return
                    elif choice == "q":
                        print("Quitting.")
                        return
                    elif choice == "e":
                        edited_out = select_site_data(
                            allowed_exts={".xlsx", ".xls"},
                            title="Select your edited classified workbook (same structure)",
                        )
                        if not edited_out:
                            print("No file selected.")
                            continue
                        try:
                            edited_summary = pd.read_excel(edited_out, sheet_name="category_summary")
                            _ensure_summary_structure(edited_summary)
                        except Exception as e:
                            print(
                                f"Edited file not in expected format: {e}\n"
                                "Please upload the classified workbook with an unchanged 'category_summary' sheet."
                            )
                            continue

                        summary_df = edited_summary  # overwrite in-memory
                        # Try to infer a start-date fallback from any datetime column in 'classified' sheet
                        start_fallback = None
                        try:
                            edited_class = pd.read_excel(edited_out, sheet_name="classified")
                            dt_cols = edited_class.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"])
                            if not dt_cols.empty:
                                start_fallback = pd.to_datetime(dt_cols.min().min())
                        except Exception:
                            pass

                        _build_and_optionally_edit_input_sheet(summary_df, start_date_fallback=start_fallback)
                        return  # done
                    else:
                        print("Please choose 'h', 'e', or 'q'.")
                        continue

        # If we got here, inner machine loop broke to upload a new file
        continue


if __name__ == "__main__":
    main()

