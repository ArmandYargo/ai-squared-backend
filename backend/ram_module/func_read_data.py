from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Optional
import pandas as pd


def read_site_data(
    path: Union[str, Path],
    *,
    excel_sheet: Optional[str] = None,
    csv_dayfirst: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Read a CSV or Excel file with minimal assumptions.
    - CSV -> single DataFrame
    - Excel -> dict of DataFrames (unless excel_sheet is specified)

    Parameters
    ----------
    path : str | Path
        File path selected in Step 2.
    excel_sheet : str | None
        If set and file is Excel, read only this sheet (by name).
    csv_dayfirst : bool
        If True, later date parsing (if you do it) should prefer DD/MM/YYYY.

    Returns
    -------
    DataFrame | dict[str, DataFrame]
        DataFrame for CSV; dict of sheet_name -> DataFrame for Excel.

    Notes
    -----
    - This function does NOT scrub/clean. That’s Step 4.
    - For CSV delimiter autodetect we use engine='python' with sep=None.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()

    if ext == ".csv":
        # Minimal, robust CSV read
        df = pd.read_csv(
            p,
            sep=None,             # autodetect delimiter
            engine="python",      # needed for sep=None
            encoding="utf-8-sig", # tolerate BOM
            low_memory=False,     # avoid mixed dtypes fragmentation
        )
        # We just return raw df; any parsing/scrub will be Step 4.
        return df

    if ext in {".xlsx", ".xls", ".xlsm"}:
        if excel_sheet is not None:
            # Read only one sheet by name
            return pd.read_excel(p, sheet_name=excel_sheet)
        # Read all sheets
        sheets: Dict[str, pd.DataFrame] = pd.read_excel(p, sheet_name=None)
        return sheets

    raise ValueError(f"Unsupported file type: {ext} (use .csv, .xlsx, .xls, .xlsm)")


def summarize_loaded(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
    """
    Convenience: print shapes and column names so you can see what was loaded.
    """
    if isinstance(data, dict):
        print(f"Excel workbook with {len(data)} sheet(s):")
        for name, df in data.items():
            print(f" - {name}: {df.shape[0]} rows x {df.shape[1]} cols")
            print(f"   columns: {list(df.columns)}")
    else:
        print(f"CSV: {data.shape[0]} rows x {data.shape[1]} cols")
        print(f"columns: {list(data.columns)}")
