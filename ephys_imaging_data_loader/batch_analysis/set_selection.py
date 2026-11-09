"""
set_selection.py
================

Which "sets" a batch run analyses. A set = expdate + neuronid + dendriteid +
group, e.g. 20260322 / neuron 1 / dendrite 1 / post-apv; each set gets its own
results folder.

``resolve_sets(catalog, sets)`` accepts:

* ``None``                                  every set in the sheet (optionally only
                                            some experiment names: ``expnames=[...]``)
* one set as a tuple                        ``(20260322, 1, 1)`` or ``(20260322, 1, 1, "post-apv")``
* one set as text                           ``"20260322 1 1"`` / ``"20260322,1,1,post-apv"``
* a list of the above                       ``[(20260322, 1, 1), "20260326 2 1 ctrl"]``
* the path of a text file                   one set per line, written as above;
                                            empty lines are ignored, and # starts a comment
                                            anywhere on a line (the rest of the line is ignored)

Without a group, every group of that dendrite in the sheet is analysed (one set each).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SetKey:
    expdate: str
    neuronid: int
    dendriteid: int
    group: str | None = None       # None = every group of the dendrite (expanded by resolve_sets)

    @property
    def name(self) -> str:
        """Folder / file name: '20260322_1_1_post-apv' (characters unsafe in file names -> '_')."""
        group = re.sub(r"[^\w.+-]+", "_", str(self.group)) if self.group is not None else "allgroups"
        return f"{self.expdate}_{self.neuronid}_{self.dendriteid}_{group}"

    def __str__(self) -> str:
        return f"{self.expdate} neuron {self.neuronid} dendrite {self.dendriteid}" + (
            f" group {self.group}" if self.group is not None else "")


def parse_set_text(text: str) -> SetKey:
    """'20260322 1 1', '20260322,1,1,post-apv', '20260322; 1; 1' -> SetKey."""
    fields = [f for f in re.split(r"[,;\s]+", text.strip()) if f]
    if len(fields) not in (3, 4):
        raise ValueError(f"A set needs expdate, neuronid, dendriteid [, group]; got {text!r}.")
    return SetKey(fields[0].replace(".0", ""), int(float(fields[1])), int(float(fields[2])),
                  fields[3] if len(fields) == 4 else None)


def _as_key(item) -> SetKey:
    if isinstance(item, SetKey):
        return item
    if isinstance(item, str):
        return parse_set_text(item)
    if isinstance(item, (tuple, list)) and len(item) in (3, 4):
        return parse_set_text(",".join(str(v) for v in item))
    raise ValueError(f"Cannot read a set from {item!r}; use (expdate, neuronid, dendriteid[, group]).")


def read_set_file(path: str | Path) -> list[SetKey]:
    """Sets from a text file (see module docstring)."""
    keys = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            keys.append(parse_set_text(line))
    return keys


def all_sets(catalog, expnames: list[str] | None = None) -> list[SetKey]:
    """Every expdate / neuronid / dendriteid / group in the sheet that has an ephys file."""
    table = catalog.select(expname=expnames)
    table = table.dropna(subset=["expdate", "neuronid", "dendriteid"])
    keys = {SetKey(str(r.expdate), int(r.neuronid), int(r.dendriteid), None if pd.isna(r.group) else str(r.group))
            for r in table[["expdate", "neuronid", "dendriteid", "group"]].itertuples()}
    return sorted(keys, key=lambda k: (k.expdate, k.neuronid, k.dendriteid, str(k.group)))


def resolve_sets(catalog, sets=None, expnames: list[str] | None = None) -> list[SetKey]:
    """The list of sets to analyse (see module docstring), each with its group."""
    if sets is None:
        return all_sets(catalog, expnames)
    if isinstance(sets, Path) or (isinstance(sets, str) and Path(sets).is_file()):
        keys = read_set_file(sets)
    elif isinstance(sets, (str, SetKey)) or (isinstance(sets, tuple) and not isinstance(sets[0], (tuple, list, str, SetKey))):
        keys = [_as_key(sets)]
    else:
        keys = []
        for item in sets:
            keys += read_set_file(item) if isinstance(item, (str, Path)) and Path(item).is_file() else [_as_key(item)]
    expanded = []
    for key in keys:
        if key.group is not None:
            expanded.append(key)
            continue
        groups = catalog.distinct("group", expdate=key.expdate, neuronid=key.neuronid, dendriteid=key.dendriteid,
                                  expname=expnames)
        if not groups:
            raise LookupError(f"No rows in the sheet for {key}.")
        expanded += [SetKey(key.expdate, key.neuronid, key.dendriteid, str(g)) for g in groups]
    return list(dict.fromkeys(expanded))
