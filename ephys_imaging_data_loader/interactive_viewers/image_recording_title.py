"""
image_recording_title.py
========================

Builds the one-line title shown above the image by ``image.show()``,
in the same '|'-separated style as the ephys viewer:

    20260322 | e/o ids: 5/3 | nds: 120 | 40 d Female | ca3 | basal | LP: 30%, 2ms / 5.0ms | 32.0 °C

* expdate              experiment date
* e/o ids: <e>/<o>     ephysid of the recording(s) that used this image / this
                       image's number (Image0010 -> 10); z-stacks: "o id: <o>"
* nds: <n><d><s>       neuron, dendrite and spine id pooled (112 = n1 d1 s2;
                       "1.1.12" if an id has two digits); "nd:" for z-stacks,
                       "n:" for images not listed in the sheet
* "<age> d <Sex>"      age in days on the experiment day, and sex
* ca1/l5, apical/basal area and side, as written in the sheet
* LP: <p>%, <t>ms / <T>ms  uncaging laser power, illumination time per target, and total time
                       of the laser sequence incl. travel between targets: from the image file's own
                       laser metadata (authoritative); the sheet's value only
                       if the file does not have it; omitted for images without
                       uncaging targets (e.g. z-stacks)
* <temp> °C            recording temperature
If several sheet rows use the image, differing values are joined with "/".

Finding the sheet rows of an image (first match wins):
1. rows whose ``ophysfile`` lists the image (as for the uncaging-target labels);
2. rows whose ``zstack`` lists the image number (formats "Image00034", "34",
   "16;17;18", "62;63.64;65"); a z-stack belongs to a cell/dendrite, not to one
   recording, so no eid and no sid are shown;
3. the image is not listed, but its folder gives date and neuron
   (<expdate>/c<neuron>/): the cell's other rows provide age, sex, area ...
Items that are not available are simply left out.
"""

from __future__ import annotations

import re

import pandas as pd

from ..recording_data_containers.image_recording import ImageRecording

# At most this many ephys file names are listed before "…".
MAX_EPHYS_FILES_IN_TITLE = 3


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip() not in ("", "nan", "<NA>")


def _number(value) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(v) for v in values if _present(v)))


def _image_number(file_name: str) -> int | None:
    match = re.search(r"(\d+)", str(file_name))
    return int(match.group(1)) if match else None


def _zstack_rows(catalog, recording: ImageRecording) -> pd.DataFrame:
    """
    Rows of the same date (and neuron) whose z-stack columns list this image
    number. Works with the old single "zstack" column and the newer
    "zstack_basal", "zstack_soma", "zstack_apical", "zstack_dend" columns.
    """
    from ..experiment_catalog.uncaging_target_labels import _image_identity

    expdate, neuron = _image_identity(recording)
    number = _image_number(recording.file_name)
    table = catalog.table
    zstack_columns = [c for c in table.columns if c.startswith("zstack")]
    if expdate is None or number is None or not zstack_columns:
        return table.iloc[0:0]
    candidates = table[table["expdate"] == expdate]
    if neuron is not None:
        candidates = candidates[candidates["neuronid"] == neuron]

    def lists_image(row) -> bool:
        return any(
            number in [int(n) for n in re.findall(r"\d+", str(row[column]))]
            for column in zstack_columns
            if pd.notna(row[column])
        )

    return candidates[candidates.apply(lists_image, axis=1)] if len(candidates) else candidates


def _cell_rows(catalog, recording: ImageRecording) -> pd.DataFrame:
    """All rows of the image's cell (date + neuron from its folder)."""
    from ..experiment_catalog.uncaging_target_labels import _image_identity

    expdate, neuron = _image_identity(recording)
    if expdate is None or neuron is None:
        return catalog.table.iloc[0:0]
    table = catalog.table
    return table[(table["expdate"] == expdate) & (table["neuronid"] == neuron)]


def image_title_parts(recording: ImageRecording, catalog=None) -> tuple[str, str]:
    """
    (title, details) for an image.

    title   : the '|'-separated line described in the module docstring
    details : extra information for the caption (experiment name, group, how
              many sheet rows use this image), or why sheet details are missing
    """
    from ..experiment_catalog.uncaging_target_labels import _rows_for_image

    if catalog is None:
        from ..experiment_catalog.ephys_experiment_catalog import get_default_catalog

        catalog = get_default_catalog()
    rows = _rows_for_image(catalog, recording) if catalog is not None else pd.DataFrame()
    how_found = "ophysfile"
    if rows.empty and catalog is not None:
        rows, how_found = _zstack_rows(catalog, recording), "zstack"
    if rows.empty and catalog is not None:
        rows, how_found = _cell_rows(catalog, recording), "cell"

    parts: list[str] = []
    details: list[str] = []
    column = lambda name: rows[name] if (not rows.empty and name in rows.columns) else pd.Series(dtype=object)
    joined = lambda values: "/".join(_unique(values))

    # expdate
    dates = _unique(column("expdate"))
    if not dates and recording.acquisition_datetime:
        dates = [recording.acquisition_datetime.strftime("%Y%m%d")]
    if dates:
        parts.append("/".join(dates))

    # Ephys / imaging ids in one item: "e/o ids: 13/10" (ephysid / this image's number,
    # Image0010 -> 10). Several recordings using one image: "e/o ids: 13,31/10".
    # Z-stacks have no recording: "o id: 34".
    eids = ",".join(_unique(_number(v) for v in column("ephysid").dropna())) if how_found == "ophysfile" else ""
    image_number = _image_number(recording.file_name)
    if eids and image_number is not None:
        parts.append(f"e/o ids: {eids}/{image_number}")
    elif image_number is not None:
        parts.append(f"o id: {image_number}")
    elif eids:
        parts.append(f"e id: {eids}")

    # Neuron / dendrite / spine ids pooled into one item, e.g. "nds: 112" for
    # neuron 1, dendrite 1, spine 2. If any id has two digits the ids are joined
    # with dots ("nds: 1.1.12"), otherwise "1112" would be ambiguous.
    # Z-stacks have no spine ("nd: 11"); unlisted images only a neuron ("n: 1").
    from ..experiment_catalog.uncaging_target_labels import _image_identity

    def pooled(ids: list) -> str:
        texts = [_number(i) for i in ids]
        return "".join(texts) if all(len(t) == 1 for t in texts) else ".".join(texts)

    if how_found == "ophysfile":
        label, keys = "nds", ("neuronid", "dendriteid", "spineid")
    elif how_found == "zstack":
        label, keys = "nd", ("neuronid", "dendriteid")
    else:
        label, keys = "n", ("neuronid",)
    combos = _unique(
        pooled([getattr(r, k) for k in keys])
        for r in (rows.itertuples() if not rows.empty else [])
        if all(_present(getattr(r, k)) for k in keys)
    )
    if not combos and _image_identity(recording)[1] is not None:
        label, combos = "n", [str(_image_identity(recording)[1])]
    if combos:
        parts.append(f"{label}: {'/'.join(combos)}")

    # "<age> d <Sex>", e.g. "40 d Female"
    ages = joined(f"{int(a)}" for a in column("age_days").dropna())
    sexes = joined(str(v).capitalize() for v in column("sex").dropna())
    if ages or sexes:
        parts.append(" ".join(x for x in (f"{ages} d" if ages else "", sexes) if x))

    # area (ca1 / l5), side (apical / basal), as written in the sheet
    for name in ("area", "side"):
        value = joined(column(name).dropna())
        if value:
            parts.append(value)

    # LP = uncaging laser power: image metadata first (authoritative), then the sheet
    power = recording.metadata.get("uncaging_laser_power_percent")
    power_text = _number(power) if power is not None else joined(_number(v) for v in column("stim_power_percent").dropna())
    if power is None and power_text:
        details.append("laser power from the experiment sheet")
    # "LP: 60%, 0.2ms / 5.04ms" = power, illumination time per target, and total
    # time of the whole laser sequence incl. travel between targets
    # (= n_targets x (illumination + move); equal to the per-target time for a
    # single target). File metadata first, the sheet's stimdur / stimdur_total otherwise.
    def ms(value) -> str:
        return f"{round(float(value), 2):g}"

    duration = recording.metadata.get("uncaging_duration_ms")
    total = recording.metadata.get("uncaging_total_time_ms")
    duration_text = ms(duration) if duration is not None else joined(ms(v) for v in column("stimdur").dropna())
    total_text = ms(total) if total is not None else joined(ms(v) for v in column("stimdur_total").dropna())
    if duration_text and not total_text and len(recording.uncaging_points) == 1:
        total_text = duration_text
    if power_text and not recording.uncaging_points.empty:
        timing = f", {duration_text}ms / {total_text}ms" if duration_text and total_text else (
            f", {duration_text}ms" if duration_text else "")
        parts.append(f"LP: {power_text}%{timing}")

    # temperature
    temperature = joined(_number(v) for v in column("temp").dropna())
    if temperature:
        parts.append(f"{temperature} °C")

    if how_found == "ophysfile" and not rows.empty:
        details.append(" | ".join(_unique(column("expname")) + _unique(column("group"))))
        if len(rows) > 1:
            details.append(f"{len(rows)} rows of the experiment sheet use this image")
    elif how_found == "zstack":
        details.append("z-stack listed in a zstack column of the experiment sheet")
    elif how_found == "cell" and not rows.empty:
        details.append("image not listed in the experiment sheet; values from the cell's other rows")
    else:
        details.append("image not found in the experiment sheet")

    return " | ".join(parts), " | ".join(d for d in details if d)
