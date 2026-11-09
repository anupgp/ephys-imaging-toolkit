"""
traces_by_isi.py
================

Collects, for ONE spine (expdate + neuron + dendrite + spine), the response
traces of every recording, cut around the stimulus and sorted by
inter-stimulus interval (ISI). The result (``TracesByIsi``) is plotted by
``interactive_viewers/traces_by_isi_viewer.py`` (interactive and static).

Why a separate "collect" step?
------------------------------
Loading and aligning sweeps is the slow part (files may come from Box). It is
done once here; the viewer can then re-draw instantly when you change which
ISIs, groups or overlays are shown.

What happens for every catalog row (= one .abf file)
----------------------------------------------------
1. The recording is loaded with the sheet's bad sweeps removed.
2. The response channel is the one given by the caller, else the sheet's
   ``chan_resp_verified`` column, or ``chan_resp`` if that is missing
   (matched ignoring case: "ImLeft" == "ImLEFT").
3. In EVERY good sweep, the stimulus onsets are found (see
   ``stimulus_onset_detection.identify_stimulus``: the digital output matching
   the sheet ISI, cross-checked against recorded trigger / light-pulse
   channels) and the sweep is cut from
   ``time_window_ms[0]`` before to ``time_window_ms[1]`` after the first stimulus.
   By default the window is 50 ms before the first pulse to ISI + 150 ms after it.
4. Time is expressed in ms relative to the first stimulus (0 = first pulse).
5. The ISI decoded from the file is compared with the sheet's ISI; a
   disagreement is reported as a warning (the sheet value is used for grouping).

Problems with single files (not found, all sweeps bad, no stimulus detected)
never stop the whole collection: the file is skipped and a warning explains why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..recording_data_containers.ephys_recording import EphysRecording
from .stimulus_onset_detection import (
    StimulusDetectionError,
    _interval_matches,
    find_stimulus_onsets,
    identify_stimulus,
    trigger_like_channels,
)

logger = logging.getLogger(__name__)

# Sheet columns shown in the header of the traces-by-ISI view, in this order.
METADATA_COLUMNS = [
    "expdate", "neuronid", "dendriteid", "spineid", "region", "area", "side", "patchclamp",
    "signal", "stim_power_percent", "stimdur", "clustersize", "temp", "accessres", "animal", "sex", "expname",
]
METADATA_LABELS = {
    "expdate": "date", "neuronid": "neuron", "dendriteid": "dendrite", "spineid": "spine",
    "patchclamp": "clamp", "stim_power_percent": "stim power", "stimpower": "stim power", "stimdur": "stim dur", "clustersize": "cluster size",
    "temp": "temp (°C)", "accessres": "access R (MΩ)", "expname": "experiment",
}


@dataclass
class IsiTraceSet:
    """The aligned sweeps of ONE recording file."""

    isi_ms: float                 # ISI from the sheet (used for grouping)
    group: str                    # e.g. "pre-apv", "post-apv", "ctrl"
    file_name: str
    catalog_row: int              # row label in the catalog table
    channel: str
    unit: str
    sweeps: list[int]             # 0-based sweep indices used
    time_ms: np.ndarray           # (n_samples,) relative to the first stimulus
    traces: np.ndarray            # (n_sweeps, n_samples)
    stimulus_onsets_ms: np.ndarray  # onsets relative to the first stimulus (from the first used sweep)
    decoded_interval_ms: float | None
    stimulus_source: str


@dataclass
class TracesByIsi:
    """Everything needed to draw the traces-by-ISI figure."""

    trace_sets: list[IsiTraceSet]
    metadata: dict[str, str]      # sheet column -> value(s) shown in the header
    warnings: list[str] = field(default_factory=list)

    # ---- simple queries ----------------------------------------------------------
    @property
    def isis(self) -> list[float]:
        """All ISIs present, ascending."""
        return sorted({s.isi_ms for s in self.trace_sets})

    @property
    def groups(self) -> list[str]:
        """All groups present, in order of first appearance."""
        return list(dict.fromkeys(s.group for s in self.trace_sets))

    def sets_for(self, isi_ms: float, group: str | None = None) -> list[IsiTraceSet]:
        return [s for s in self.trace_sets if s.isi_ms == isi_ms and (group is None or s.group == group)]

    def sweep_count(self, isi_ms: float, group: str | None = None) -> int:
        """Number of sweeps averaged for one ISI (and group)."""
        return sum(len(s.sweeps) for s in self.sets_for(isi_ms, group))

    def mean_trace(self, isi_ms: float, group: str | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Average over ALL sweeps (of all files) for one ISI and group.

        Returns (time_ms, mean trace, number of sweeps). Files are put on the
        time grid of the first file (interpolation only matters if sampling
        rates differ between files).
        """
        sets = self.sets_for(isi_ms, group)
        if not sets:
            raise KeyError(f"No traces for ISI {isi_ms} ms, group {group}.")
        time_ms = sets[0].time_ms
        stacked = [
            s.traces if np.array_equal(s.time_ms, time_ms)
            else np.array([np.interp(time_ms, s.time_ms, trace) for trace in s.traces])
            for s in sets
        ]
        all_sweeps = np.vstack(stacked)
        return time_ms, all_sweeps.mean(axis=0), len(all_sweeps)

    # ---- text for figure headers ---------------------------------------------------
    @staticmethod
    def format_isi(isi_ms: float) -> str:
        return f"{isi_ms:g} ms"

    def isi_summary_text(self, isis: list[float] | None = None, groups: list[str] | None = None) -> str:
        """
        "pre-apv: 30 ms (3), 40 ms (3), ... | post-apv: 30 ms (3), ..."

        The number in brackets is the number of sweeps averaged for that ISI.
        ISIs are sorted numerically (30 before 100).
        """
        isis = self.isis if isis is None else sorted(isis)
        groups = self.groups if groups is None else groups
        parts = []
        for group in groups:
            entries = [
                f"{self.format_isi(isi)} ({self.sweep_count(isi, group)})"
                for isi in isis
                if self.sweep_count(isi, group) > 0
            ]
            if entries:
                parts.append((f"{group}: " if group else "") + ", ".join(entries))
        return " | ".join(parts) if parts else "no traces"

    def metadata_text(self) -> str:
        """'date 20260322 | neuron 1 | dendrite 1 | spine 0 | area ca1 | ...'."""
        return " | ".join(
            f"{METADATA_LABELS.get(column, column)} {value}" for column, value in self.metadata.items() if value
        )


def _unique_values_text(rows: pd.DataFrame, column: str) -> str:
    """All distinct values of a column, e.g. '41' or '41/45' (empty if missing)."""
    if column not in rows.columns:
        return ""

    def as_text(value) -> str:
        # 44.0 -> "44", 25.5 -> "25.5", "ca1" -> "ca1"
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)

    values = [as_text(v) for v in rows[column].dropna().unique() if str(v).strip() not in ("", "nan", "<NA>")]
    return "/".join(dict.fromkeys(values))  # unique, original order


def _check_sheet_stimulus_channel(
    recording: EphysRecording,
    sheet_channel: str | None,
    response_channel: str | None,
    isi_ms: float,
    sweep: int,
    warnings: list[str],
) -> str | None:
    """
    Use the sheet's ``chan_stim`` only if it really carries trigger pulses.

    If it is missing, flat or not a trigger, the OTHER recorded channels are
    scanned (the response channel excluded) and a channel whose pulses are
    the sheet ISI apart is used instead, with a warning naming both channels.
    Returns the channel to use, or None to fall back to the digital outputs.
    """
    triggers = trigger_like_channels(recording, sweep, exclude=[response_channel] if response_channel else [])
    trigger_names = [t.confirmed_by[0] for t in triggers]
    if sheet_channel and sheet_channel.replace(" ", "").lower() in [n.replace(" ", "").lower() for n in trigger_names]:
        return sheet_channel

    def fits_isi(onsets) -> bool:
        if isi_ms == 0:
            return onsets.n_pulses == 1
        return onsets.mean_interval_ms is not None and _interval_matches(onsets.mean_interval_ms, isi_ms)

    matching = [t.confirmed_by[0] for t in triggers if fits_isi(t)]
    problem = f"sheet chan_stim '{sheet_channel}' shows no trigger pulses" if sheet_channel else "no chan_stim in the sheet"
    if matching:
        warnings.append(f"{recording.file_name}: {problem}; using recorded channel {matching[0]} instead.")
        return matching[0]
    warnings.append(
        f"{recording.file_name}: {problem}, and no other channel has pulses {isi_ms:g} ms apart "
        f"(trigger-like channels: {trigger_names or 'none'}); using the digital outputs instead."
    )
    return None


def default_time_window_ms(isi_ms: float) -> tuple[float, float]:
    """50 ms before the first pulse until 150 ms after the last (second) pulse."""
    return (-50.0, float(isi_ms) + 150.0)


def collect_traces_by_isi(
    rows: pd.DataFrame,
    load_ephys: Callable[[int], EphysRecording],
    channel: str | None = None,
    digital_output_bit: int | str = "auto",
    stimulus_channel: str | None = None,
    stimulus_threshold: float | None = None,
    time_window_ms: tuple[float, float] | None = None,
) -> TracesByIsi:
    """
    Load and align the traces of the given catalog rows (see module docstring).

    Parameters
    ----------
    rows : catalog rows (one per .abf file), e.g. from ``ExperimentCatalog.select``
    load_ephys : function row label -> EphysRecording with bad sweeps applied
    channel : response channel; None = the sheet's ``chan_resp`` per row
    digital_output_bit, stimulus_channel, stimulus_threshold :
        how to find the stimulus (see ``find_stimulus_onsets``). Use
        stimulus_channel="catalog" to take the sheet's ``chan_stim`` column.
    time_window_ms : (before, after) relative to the first stimulus, e.g.
        (-50, 300); None = ``default_time_window_ms(isi)`` for each ISI
    """
    trace_sets: list[IsiTraceSet] = []
    warnings: list[str] = []

    for row_label, row in rows.iterrows():
        isi_ms = float(row["isi"]) if pd.notna(row.get("isi")) else float("nan")
        group = str(row.get("group")) if pd.notna(row.get("group")) else ""
        file_label = row.get("ephys_file_name") or f"row {row_label}"
        if np.isnan(isi_ms):
            warnings.append(f"{file_label}: no ISI in the sheet — skipped.")
            continue

        # ---- load ------------------------------------------------------------------
        try:
            recording = load_ephys(row_label)
        except Exception as error:  # not found, unreadable, missing file name ...
            warnings.append(f"{file_label}: could not be loaded ({error}) — skipped.")
            continue
        warnings.extend(recording.header_metadata.get("catalog_warnings", []))

        # response channel: argument, else the sheet's (file-verified) response channel
        response_channel = channel or row.get("response_channel", row.get("chan_resp"))
        try:
            channel_index = recording.channel_index(response_channel)
        except (KeyError, TypeError):
            warnings.append(
                f"{recording.file_name}: response channel '{response_channel}' not in "
                f"{recording.channel_names} — skipped."
            )
            continue

        good_sweeps = recording.good_sweeps
        if not good_sweeps:
            warnings.append(f"{recording.file_name}: all sweeps are marked bad — skipped.")
            continue

        use_channel = stimulus_channel
        if stimulus_channel == "catalog":
            sheet_stim = row.get("stimulus_channel", row.get("chan_stim"))
            use_channel = sheet_stim if pd.notna(sheet_stim) else None
            use_channel = _check_sheet_stimulus_channel(
                recording, use_channel, response_channel, isi_ms, good_sweeps[0], warnings
            )

        window = time_window_ms or default_time_window_ms(isi_ms)
        rate_hz = recording.sampling_rate_hz
        first_sample_offset = int(round(window[0] / 1000 * rate_hz))
        n_window_samples = int(round((window[1] - window[0]) / 1000 * rate_hz))

        # ---- cut every good sweep around its own first stimulus ---------------------------
        cut_traces, used_sweeps = [], []
        first_onsets = None
        detection_errors: list[str] = []
        for sweep in good_sweeps:
            try:
                if use_channel is None and digital_output_bit == "auto":
                    # Default: digital output matching the sheet ISI, cross-checked
                    # against recorded trigger / light-pulse channels.
                    onsets = identify_stimulus(recording, sweep, isi_ms, recording.channel_names[channel_index])
                else:
                    onsets = find_stimulus_onsets(
                        recording, sweep,
                        expected_interval_ms=isi_ms,
                        digital_output_bit=digital_output_bit,
                        stimulus_channel=use_channel,
                        threshold=stimulus_threshold,
                    )
            except StimulusDetectionError as error:
                detection_errors.append(str(error))
                continue
            start = int(round(onsets.onsets_s[0] * rate_hz)) + first_sample_offset
            stop = start + n_window_samples
            if start < 0 or stop > recording.n_samples_per_sweep:
                warnings.append(
                    f"{recording.file_name}, sweep {sweep}: the time window {window} ms does not fit "
                    "inside the sweep — sweep skipped."
                )
                continue
            cut_traces.append(recording.data[sweep, channel_index, start:stop])
            used_sweeps.append(sweep)
            if first_onsets is None:
                first_onsets = onsets

        # One message per FILE (not per sweep) when stimulus detection failed.
        if detection_errors:
            if used_sweeps:
                warnings.append(
                    f"{recording.file_name}: stimulus not found in {len(detection_errors)} of "
                    f"{len(good_sweeps)} sweeps (those skipped). First problem: {detection_errors[0]}"
                )
            else:
                warnings.append(f"{detection_errors[0]} — whole file skipped.")
        if not used_sweeps:
            continue

        decoded_interval = first_onsets.mean_interval_ms
        if decoded_interval is not None and isi_ms > 0 and abs(decoded_interval - isi_ms) > max(2.0, 0.05 * isi_ms):
            warnings.append(
                f"{recording.file_name}: sheet ISI {isi_ms:g} ms but the file's stimulus interval is "
                f"{decoded_interval:.1f} ms ({first_onsets.source}); plotted under {isi_ms:g} ms."
            )

        time_ms = (np.arange(n_window_samples) + first_sample_offset) / rate_hz * 1000
        trace_sets.append(
            IsiTraceSet(
                isi_ms=isi_ms,
                group=group,
                file_name=recording.file_name,
                catalog_row=int(row_label),
                channel=recording.channel_names[channel_index],
                unit=recording.channel_units[channel_index],
                sweeps=used_sweeps,
                time_ms=time_ms,
                traces=np.array(cut_traces),
                stimulus_onsets_ms=(first_onsets.onsets_s - first_onsets.onsets_s[0]) * 1000,
                decoded_interval_ms=decoded_interval,
                stimulus_source=first_onsets.source,
            )
        )

    metadata = {column: _unique_values_text(rows, column) for column in METADATA_COLUMNS}
    # Present region/area/side as one entry, e.g. "hc ca1 apical".
    location = " ".join(v for v in (metadata.pop("region"), metadata.pop("area"), metadata.pop("side")) if v)
    metadata = {**{k: metadata[k] for k in ("expdate", "neuronid", "dendriteid", "spineid")},
                "location": location,
                **{k: v for k, v in metadata.items() if k not in ("expdate", "neuronid", "dendriteid", "spineid")}}
    for message in warnings:
        logger.warning(message)
    return TracesByIsi(trace_sets=trace_sets, metadata=metadata, warnings=warnings)
