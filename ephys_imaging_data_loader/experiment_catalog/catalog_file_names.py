"""
catalog_file_names.py
=====================

Turns the file columns of the experiment sheet into real file names, and
parses the "badsweeps" column.

How ephys file names are written in the sheet
---------------------------------------------
``ephysfile_prefix`` + ``ephysid``, where the run of "x" characters in the
prefix is replaced by the id, zero-padded to the same width:

    prefix            id    ->  file name
    ----------------  ----      -----------------
    26322xxx          7         26322007.abf
    2027_02_18_xxxx   3         2027_02_18_0003.abf
    24d05050          50        24d05050.abf        (prefix is already a full name)
    (empty)           8         24413008.abf        (INFERRED, see below)

When the prefix is empty, the name is inferred from the date using the
classic Clampex default naming  YY + month code + DD + 3-digit number, with
month code 1-9 for Jan-Sep and o/n/d for Oct/Nov/Dec (e.g. 2025-10-14 ->
"25o14").
Newer Clampex versions name files "YYYY_MM_DD_NNNN" instead, so inferred names
are flagged (column ``ephys_file_name_source`` = "inferred from date") and a
missing file then gives a clear "not found" message.
"""

from __future__ import annotations

import math
import re

# Clampex month codes: January = "1" ... September = "9", then o, n, d.
CLAMPEX_MONTH_CODES = "123456789ond"


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() in ("", "nan", "<NA>")


def build_ephys_file_name(prefix, ephys_id, expdate: str | None) -> tuple[str | None, str]:
    """
    Return (file name without extension, how it was obtained).

    "how" is one of: "template", "full name", "inferred from date", "missing".
    """
    if _is_missing(ephys_id) and _is_missing(prefix):
        return None, "missing"

    if not _is_missing(prefix):
        prefix = str(prefix).strip()
        # Replace the LAST run of x's (e.g. "26322xxx" or "2027_02_18_xxxx").
        runs = list(re.finditer(r"x+", prefix, flags=re.IGNORECASE))
        if runs:
            if _is_missing(ephys_id):
                return None, "missing"
            run = runs[-1]
            number = str(int(float(ephys_id)))
            if len(number) > len(run.group()):
                return None, "missing"  # id does not fit the template
            return prefix[: run.start()] + number.zfill(len(run.group())) + prefix[run.end():], "template"
        return prefix, "full name"

    # No prefix: infer the classic Clampex name from the date.
    if expdate and len(str(expdate)) == 8 and str(expdate).isdigit():
        year, month, day = str(expdate)[2:4], int(str(expdate)[4:6]), str(expdate)[6:8]
        if 1 <= month <= 12:
            return f"{year}{CLAMPEX_MONTH_CODES[month - 1]}{day}{int(float(ephys_id)):03d}", "inferred from date"
    return None, "missing"


def fill_file_template(prefix, file_id) -> str | None:
    """
    Replace the run of "x" in ``prefix`` with ``file_id``, zero-padded to the
    run's width: ("Imagexxxx", 5) -> "Image0005", ("26322xxx", 7) -> "26322007".
    None if the prefix has no x-run, the id is missing, or the id does not fit.
    """
    if _is_missing(prefix) or _is_missing(file_id):
        return None
    runs = list(re.finditer(r"x+", str(prefix).strip(), flags=re.IGNORECASE))
    if not runs:
        return None
    run = runs[-1]
    number = str(int(float(file_id)))
    if len(number) > len(run.group()):
        return None
    prefix = str(prefix).strip()
    return prefix[: run.start()] + number.zfill(len(run.group())) + prefix[run.end():]


def build_ophys_file_names(prefix, ophys_ids) -> list[str]:
    """
    Imaging file names of a row: ``ophysfile_prefix`` (e.g. "Imagexxxx") with
    each id of ``ophysid`` (one id, or several separated by ";", e.g. "12;13"),
    plus the ".oib" extension: ["Image0012.oib", "Image0013.oib"].
    """
    if _is_missing(ophys_ids):
        return []
    names = []
    for part in re.split(r"[;,\s]+", str(ophys_ids).strip()):
        if part:
            name = fill_file_template(prefix, part)
            if name:
                names.append(name if name.lower().endswith((".oib", ".oif")) else name + ".oib")
    return names


def parse_bad_sweeps(text) -> list[int]:
    """
    "1;2;3;4" -> [1, 2, 3, 4]. Commas are accepted too; empty -> [].

    The numbers are returned exactly as written in the sheet; converting them
    to 0-based sweep indices is done by the catalog (see ``badsweeps_numbering``).
    """
    if _is_missing(text):
        return []
    return [int(float(part)) for part in re.split(r"[;,\s]+", str(text).strip()) if part]
