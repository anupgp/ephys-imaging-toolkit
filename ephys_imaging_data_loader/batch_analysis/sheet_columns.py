"""
sheet_columns.py
================

Add facts from the experiment sheet (the experiment sheet (CSV)) to a batch results table,
so the results can be filtered by brain region, age, temperature, ... without loading
and cross-checking the sheet each time.

    add_sheet_columns(results, sheet)

For every results row the sheet rows of the FILES it was computed from (the results'
``files`` column, e.g. "26326005.abf" or "26322014.abf;26322015.abf") are looked up and
their values summarised:

    text columns     (region, area, side, animal, strain, sex, patchclamp, ...)
                     the distinct values joined with ';' (normally a single value)
    number columns   (age, temp, accessres, zoom, ...)   the mean over the files

Each added column is prefixed with ``sheet_`` (sheet_region, sheet_area, ...), so it is
clear where it comes from and it never collides with a results column. Rows whose files
are not in the sheet get empty values. The values are those of the sheet at the time of the
call; re-run it after editing the sheet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# sheet column -> how it is summarised over the files of one results row
SHEET_TEXT_COLUMNS = ["region", "area", "side", "animal", "strain", "sex", "dob", "patchclamp", "chan_resp"]
SHEET_NUMBER_COLUMNS = ["age", "temp", "accessres", "zoom", "stimpower", "stimdur", "uncaging_targets"]
PREFIX = "sheet_"


def _file_stems(sheet: pd.DataFrame) -> pd.Series:
    """'26326005' per sheet row: the catalog's ephys_file_name, else prefix with 'xxx' -> 3-digit ephysid."""
    if "ephys_file_name" in sheet:
        names = sheet["ephys_file_name"].astype("string")
    else:
        ephysid = pd.to_numeric(sheet["ephysid"], errors="coerce")
        names = pd.Series([str(p).replace("xxx", f"{int(i):03d}") if pd.notna(i) else pd.NA
                           for p, i in zip(sheet["ephysfile_prefix"], ephysid)], index=sheet.index, dtype="string")
    return names.str.replace(r"\.abf$", "", regex=True, case=False).str.lower()


def read_sheet(path: str | Path) -> pd.DataFrame:
    """The sheet as text (blank separator rows dropped)."""
    sheet = pd.read_csv(path, dtype=str)
    return sheet[sheet["ephysid"].notna()].reset_index(drop=True)


def add_sheet_columns(results: pd.DataFrame, sheet: pd.DataFrame | str | Path,
                      text_columns=None, number_columns=None) -> pd.DataFrame:
    """Copy of ``results`` with sheet_<column> added (see module docstring)."""
    if not isinstance(sheet, pd.DataFrame):
        sheet = read_sheet(sheet)
    text_columns = [c for c in (text_columns or SHEET_TEXT_COLUMNS) if c in sheet]
    number_columns = [c for c in (number_columns or SHEET_NUMBER_COLUMNS) if c in sheet]
    by_file = sheet.assign(_stem=_file_stems(sheet)).drop_duplicates("_stem").set_index("_stem")
    numbers = by_file[number_columns].apply(pd.to_numeric, errors="coerce")

    added = {f"{PREFIX}{c}": [] for c in text_columns + number_columns}
    for files in results["files"].astype("string").fillna(""):
        stems = [Path(f.strip()).stem.lower() for f in files.split(";") if f.strip()]
        stems = [s for s in stems if s in by_file.index]
        for column in text_columns:
            values = sorted({str(v).strip() for v in by_file.loc[stems, column] if pd.notna(v) and str(v).strip()})
            added[f"{PREFIX}{column}"].append(";".join(values) if values else pd.NA)
        for column in number_columns:
            values = numbers.loc[stems, column].dropna()
            added[f"{PREFIX}{column}"].append(float(values.mean()) if len(values) else np.nan)
    result = results.drop(columns=[c for c in added if c in results.columns])
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(added)], axis=1)
