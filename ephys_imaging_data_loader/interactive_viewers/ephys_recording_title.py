"""
ephys_recording_title.py
========================

Builds the one-line title shown by ``recording.show()``:

    26322005.abf | example_protocol.pro | 3/3 sweeps | 2 pulses | IPI: 100 ms | CA3 | basal | female | stim power 30 | P40 | 32.0 °C

* "3/3 sweeps"     sweeps shown / sweeps in the file
* "2 pulses"       number of STIMULUS pulses in the first shown sweep
* "IPI: 200 ms"    inter-pulse interval (mean time between stimulus onsets)
* CA3 ... 32.0 °C  from the experiment sheet (area, side, sex, stim power,
                   age as postnatal day from date of birth, temperature)

Which pulses are "the stimulus"?
--------------------------------
A protocol often drives several digital outputs (Olympus rig: DO0 = a 200 ms
gate during the test step, DO1 = uncaging). Only the stimulus output is
reported. It is identified with ``identify_stimulus``: a digital output is
preferred when a RECORDED channel (e.g. the light-pulse channel IN7) shows
pulses at exactly its onsets, and the sheet's ISI is used when known. The
choice is stated in the details line under the title, e.g.
"stimulus: digital output 1, confirmed by IN7".

Where do the sheet details come from?
-------------------------------------
From the catalog row attached by ``ExperimentCatalog.load_ephys``, or else by
looking the file name up in the sheet configured in the settings. If the
file is not in the sheet, those items are simply left out.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..experiment_catalog.ephys_experiment_catalog import get_default_catalog
from ..experiment_catalog.stimulus_onset_detection import StimulusDetectionError, identify_stimulus
from ..recording_data_containers.ephys_recording import EphysRecording

logger = logging.getLogger(__name__)


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip() not in ("", "nan", "<NA>")


def _number_text(value) -> str:
    """44.0 -> '44', 25.5 -> '25.5'."""
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def format_ipi(interval_ms: float) -> str:
    """Round to whole ms from 10 ms up (200.1 -> '200 ms'), else one decimal."""
    return f"{round(interval_ms):d} ms" if interval_ms >= 10 else f"{interval_ms:.1f} ms"


def sheet_row_for(recording: EphysRecording) -> dict | None:
    """The experiment-sheet row of a recording, if it can be found."""
    row = recording.header_metadata.get("catalog_row")
    if row:
        return row
    catalog = get_default_catalog()
    if catalog is None:
        return None
    rows = catalog.rows_for_file(recording.file_name)
    return rows.iloc[0].to_dict() if not rows.empty else None


def describe_stimulus(recording: EphysRecording, sweep: int, sheet_row: dict | None):
    """The identified stimulus of one sweep, or None if there is none."""
    expected_isi = float(sheet_row["isi"]) if sheet_row and _present(sheet_row.get("isi")) else None
    response = None
    if sheet_row:
        response = next((sheet_row.get(k) for k in ("response_channel", "chan_resp_verified", "chan_resp") if _present(sheet_row.get(k))), None)
    try:
        return identify_stimulus(recording, sweep, expected_isi, response)
    except StimulusDetectionError:
        if expected_isi is None:
            return None
        # The file does not match the sheet's ISI (e.g. a sheet error): describe the
        # file as it is, without using the sheet's ISI.
        try:
            return identify_stimulus(recording, sweep, None, response)
        except StimulusDetectionError:
            return None


def ephys_title_parts(recording: EphysRecording, sweeps: list[int] | None = None) -> tuple[str, str]:
    """
    (title, details) for a recording.

    title   : the '|'-separated line described in the module docstring
    details : second line for the caption: how the stimulus was identified,
              experiment name / date / neuron / dendrite / spine, and any
              disagreement between the file and the sheet
    """
    sweeps = recording.good_sweeps if sweeps is None else list(sweeps)
    sheet_row = sheet_row_for(recording)

    parts = [recording.file_name]
    if recording.protocol_name:
        parts.append(recording.protocol_name)
    parts.append(f"{len(sweeps)}/{recording.n_sweeps} sweeps")

    details = []
    stimulus = describe_stimulus(recording, sweeps[0], sheet_row) if sweeps else None
    if stimulus is None:
        parts.append("no stimulus pulses found")
    else:
        parts.append(f"{stimulus.n_pulses} pulse{'s' if stimulus.n_pulses != 1 else ''}")
        if stimulus.mean_interval_ms is not None:
            parts.append(f"IPI: {format_ipi(stimulus.mean_interval_ms)}")
        details.append(f"stimulus: {stimulus.source}")

    if sheet_row:
        if _present(sheet_row.get("area")):
            parts.append(str(sheet_row["area"]).upper())
        for key in ("side", "sex"):
            if _present(sheet_row.get(key)):
                parts.append(str(sheet_row[key]))
        # Laser-metadata power if known (authoritative), else the typed-in value.
        power = sheet_row.get("stim_power_percent", sheet_row.get("stimpower"))
        if _present(power):
            parts.append(f"stim power {_number_text(power)}")
        if _present(sheet_row.get("age_days")):
            parts.append(f"P{int(sheet_row['age_days'])}")
        if _present(sheet_row.get("temp")):
            parts.append(f"{_number_text(sheet_row['temp'])} °C")

        identity = [
            f"{label} {sheet_row[key] if key == 'expdate' else _number_text(sheet_row[key])}"
            for key, label in (("expdate", "date"), ("neuronid", "neuron"), ("dendriteid", "dendrite"), ("spineid", "spine"))
            if _present(sheet_row.get(key))
        ]
        details.append(" | ".join(identity + ([str(sheet_row["expname"])] if _present(sheet_row.get("expname")) else [])))
        sheet_isi = float(sheet_row["isi"]) if _present(sheet_row.get("isi")) else None
        if stimulus and stimulus.mean_interval_ms and sheet_isi and abs(stimulus.mean_interval_ms - sheet_isi) > max(2.0, 0.05 * sheet_isi):
            details.append(f"⚠ sheet ISI {sheet_isi:g} ms ≠ file IPI {format_ipi(stimulus.mean_interval_ms)}")
    else:
        details.append("not found in the experiment sheet")

    return " | ".join(parts), " | ".join(details)
