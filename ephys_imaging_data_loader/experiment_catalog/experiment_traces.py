"""
experiment_traces.py
====================

Collects ALL response traces of one experiment sample, chosen from the
experiment sheet by

    expdate + neuronid + dendriteid (+ spineid, optional)

and lets them be AVERAGED by any sheet column(s), e.g. ``average_by="isi"``
(all sweeps with the same ISI averaged together) or ``["isi", "group"]``
(one average per ISI and group, e.g. ctrl / ctz). Without ``spineid`` every
spine of that dendrite is included.

The result (``ExperimentTraces``) is drawn by
``interactive_viewers/experiment_traces_viewer.py``; use
``ExperimentCatalog.show_experiment_traces(...)`` rather than calling this
module directly.

How the traces are obtained
---------------------------
Every sheet row (= one .abf file) is loaded with its bad sweeps removed and
every good sweep is cut around its OWN first stimulus, exactly as for the
traces-by-ISI view (``traces_by_isi.collect_traces_by_isi``):

* time 0 = first stimulus of the sweep;
* the stimulus is found on the sheet's ``chan_stim`` channel (checked to really
  carry pulses; otherwise another trigger channel / the digital outputs are
  used, with a warning);
* default window per file: 50 ms before the first pulse until 150 ms after the
  LAST pulse (last pulse = isi x (nstims - 1), nstims from the sheet, else 2).
  ``time_window_ms=(before, after)`` sets one common window instead.

Averaging traces with different lengths
---------------------------------------
Files with different ISIs have windows of different length. When they are
averaged together (e.g. ``average_by="group"``), every trace is placed on a
common time axis and padded with NaN where it has no data; the average at
each time point is the mean of the traces that HAVE data there (so the late
part of such an average comes from the long-ISI sweeps only). Traces with
different units (e.g. pA and mV) are never averaged together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..recording_data_containers.ephys_recording import EphysRecording
from .traces_by_isi import IsiTraceSet, _unique_values_text, collect_traces_by_isi

logger = logging.getLogger(__name__)

# Sheet columns summarised in the header of the view, in this order.
HEADER_COLUMNS = ["expdate", "neuronid", "dendriteid", "spineid", "expname", "area", "side", "patchclamp",
                  "stim_power_percent", "temp", "sex", "age"]


def format_value(column: str, value: Any) -> str:
    """How a sheet value is written in legends / titles: '30 ms', 'spine 3', 'ctrl', 'temp 24'."""
    if pd.isna(value):
        return f"{column} ?"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    if column == "isi":
        return f"{value} ms"
    if column in ("spineid", "dendriteid", "neuronid"):
        return f"{column.replace('id', '')} {value}"
    if column in ("group", "expname"):
        return str(value)
    return f"{column} {value}"


def _sort_key(value: Any):
    """Numbers numerically (30 before 100), then text; missing values last."""
    if pd.isna(value):
        return (2, 0.0, "")
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


@dataclass
class ExperimentTraceSet:
    """The aligned sweeps of ONE recording file together with its sheet row."""

    row: pd.Series            # the sheet row (row.name = row label in the catalog)
    traces: IsiTraceSet       # time_ms, traces (n_sweeps x n_samples), onsets, channel, unit ...

    def value(self, column: str) -> Any:
        return self.row.get(column)


@dataclass
class AveragedTraces:
    """All sweeps sharing the same values of the ``average_by`` columns, and their mean."""

    key: dict[str, Any]       # e.g. {"isi": 30.0, "group": "ctrl"}
    label: str                # e.g. "30 ms · ctrl"
    time_ms: np.ndarray       # common time axis (ms from the first stimulus)
    sweeps: np.ndarray        # (n_sweeps, n_samples), NaN where a sweep has no data
    stimulus_onsets_ms: list[float]   # all onsets of the member files (ms from the 1st pulse)
    unit: str
    channel: str
    n_files: int

    @property
    def n_sweeps(self) -> int:
        return len(self.sweeps)

    @property
    def mean(self) -> np.ndarray:
        """Mean over the sweeps that have data at each time point."""
        with np.errstate(all="ignore"):
            counts = np.sum(~np.isnan(self.sweeps), axis=0)
            total = np.nansum(self.sweeps, axis=0)
            return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


@dataclass
class ExperimentTraces:
    """Everything needed to draw the traces of one experiment sample."""

    trace_sets: list[ExperimentTraceSet]
    criteria: dict[str, Any]          # what was asked for (expdate, neuronid, ...)
    metadata: dict[str, str]          # sheet column -> value(s), for the header
    warnings: list[str] = field(default_factory=list)

    # ---- simple queries ----------------------------------------------------------
    @property
    def n_sweeps(self) -> int:
        return sum(len(s.traces.sweeps) for s in self.trace_sets)

    def values(self, column: str) -> list:
        """Distinct values of a sheet column among the loaded files, sorted (numbers numerically)."""
        return sorted({s.value(column) for s in self.trace_sets}, key=_sort_key)

    def subset(self, **wanted) -> "ExperimentTraces":
        """Only the files whose sheet values are in the given lists, e.g. subset(isi=[10, 20], group=["ctrl"])."""
        def keep(trace_set: ExperimentTraceSet) -> bool:
            for column, allowed in wanted.items():
                if allowed is None:
                    continue
                allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
                value = trace_set.value(column)
                if not any(value == a or _sort_key(value) == _sort_key(a) for a in allowed):
                    return False
            return True

        return ExperimentTraces([s for s in self.trace_sets if keep(s)], self.criteria, self.metadata, self.warnings)

    @property
    def units(self) -> list[str]:
        return list(dict.fromkeys(s.traces.unit for s in self.trace_sets))

    def baseline_subtracted(self, method: str = "mean", window_ms: float | None = None) -> "ExperimentTraces":
        """
        Copy in which every sweep has its baseline subtracted, so sweeps recorded at
        different holding currents / potentials can be overlaid and averaged.

        method "mean" (default): mean of the sweep before the first stimulus (time < 0;
                                 the last ``window_ms`` ms only, if given).
        method "line":           a straight line is fitted to the last ``window_ms`` ms
                                 before the stimulus (default 30 ms) and its value AT the
                                 stimulus (t = 0) is subtracted. A sweep that is drifting
                                 is then zeroed where the response starts, not at the
                                 average level of the window (which lags behind the drift).
                                 Only this one number is subtracted: the slope is NOT
                                 extended over the sweep (slow drift is not straight over
                                 hundreds of ms; extending it makes later values worse).
        """
        if method not in ("mean", "line"):
            raise ValueError("method must be 'mean' or 'line'.")
        if method == "line" and window_ms is None:
            window_ms = 30.0

        def subtract(trace_set: ExperimentTraceSet) -> ExperimentTraceSet:
            time = trace_set.traces.time_ms
            before = time < 0
            if window_ms is not None:
                before &= time >= -float(window_ms)
            if not before.any():
                return trace_set
            values = trace_set.traces.traces[:, before]
            if method == "line" and before.sum() >= 3:
                slope, intercept = np.polynomial.polynomial.polyfit(time[before], values.T, 1)[::-1]
                baseline = intercept[:, None]                  # line value at t = 0, one per sweep
            else:
                baseline = values.mean(axis=1, keepdims=True)
            return ExperimentTraceSet(trace_set.row, replace(trace_set.traces, traces=trace_set.traces.traces - baseline))

        return ExperimentTraces([subtract(s) for s in self.trace_sets], self.criteria, self.metadata, self.warnings)

    # ---- grouping / averaging ---------------------------------------------------------
    def grouped(self, by: str | list[str] | None) -> list[AveragedTraces]:
        """
        Split the sweeps by the values of the ``by`` column(s) and average each part.

        ``by=None`` or ``[]`` -> one part with every sweep. The unit is always part
        of the key, so e.g. voltage- and current-clamp traces are never mixed.
        Parts are sorted by their values (ISI 10, 20, ... 1000).
        """
        columns = [by] if isinstance(by, str) else list(by or [])
        parts: dict[tuple, list[ExperimentTraceSet]] = {}
        for trace_set in self.trace_sets:
            key = tuple(trace_set.value(c) for c in columns) + (trace_set.traces.unit,)
            parts.setdefault(key, []).append(trace_set)

        averaged = []
        for key in sorted(parts, key=lambda k: tuple(_sort_key(v) for v in k)):
            members = parts[key]
            # Common time axis: the finest sampling step over the widest window of the members.
            step = min(float(np.median(np.diff(m.traces.time_ms))) for m in members)
            start = min(float(m.traces.time_ms[0]) for m in members)
            stop = max(float(m.traces.time_ms[-1]) for m in members)
            time_ms = start + np.arange(int(round((stop - start) / step)) + 1) * step
            sweeps = [
                np.interp(time_ms, m.traces.time_ms, trace, left=np.nan, right=np.nan)
                for m in members for trace in m.traces.traces
            ]
            onsets = sorted({round(float(o), 3) for m in members for o in m.traces.stimulus_onsets_ms})
            label = " · ".join(format_value(c, v) for c, v in zip(columns, key)) or "all sweeps"
            units = {m.traces.unit for m in members}
            if len({k[:-1] for k in parts}) < len(parts):   # same values exist with another unit
                label += f" ({key[-1]})"
            averaged.append(AveragedTraces(
                key=dict(zip(columns, key)), label=label, time_ms=time_ms, sweeps=np.array(sweeps),
                stimulus_onsets_ms=onsets, unit="/".join(units), channel="/".join({m.traces.channel for m in members}),
                n_files=len(members),
            ))
        return averaged

    # ---- text for figure headers ---------------------------------------------------
    def header_text(self) -> str:
        """'expdate 20250125 | neuron 1 | dendrite 1 | spine 1-10 | ... | 44 files, 92 sweeps'."""
        prefix = {"neuronid": "neuron ", "dendriteid": "dendrite ", "spineid": "spine ", "stim_power_percent": "LP ",
                  "temp": "", "age": ""}
        suffix = {"stim_power_percent": " %", "temp": " °C", "age": " d"}
        parts = []
        for column in HEADER_COLUMNS:
            text = self.metadata.get(column, "")
            if text:
                parts.append(f"{prefix.get(column, '')}{text}{suffix.get(column, '')}")
        parts.append(f"{len(self.trace_sets)} files, {self.n_sweeps} sweeps")
        return " | ".join(parts)


def _last_pulse_ms(row: pd.Series) -> float:
    """Time of the last pulse after the first one: isi x (nstims - 1) (nstims from the sheet, default 2)."""
    isi = float(row["isi"]) if pd.notna(row.get("isi")) else 0.0
    n_pulses = pd.to_numeric(row.get("nstims"), errors="coerce")
    n_pulses = 2 if pd.isna(n_pulses) else int(n_pulses)
    return isi * max(n_pulses - 1, 0)


def collect_experiment_traces(
    rows: pd.DataFrame,
    load_ephys: Callable[[int], EphysRecording],
    criteria: dict[str, Any] | None = None,
    channel: str | None = None,
    stimulus_channel: str | None = "catalog",
    time_window_ms: tuple[float, float] | None = None,
    before_ms: float = 50.0,
    after_ms: float = 150.0,
) -> ExperimentTraces:
    """
    Load and align the traces of the given catalog rows (see module docstring).

    Parameters
    ----------
    rows : catalog rows (one per .abf file), e.g. from ``ExperimentCatalog.select``
    load_ephys : function row label -> EphysRecording with bad sweeps applied
    channel : response channel; None = each row's response channel from the sheet
    stimulus_channel : "catalog" = the sheet's chan_stim (default, checked against the file);
        a channel name; or None = automatic choice among the digital outputs
    time_window_ms : one (before, after) window for all files, e.g. (-50, 500);
        None = per file from ``before_ms`` before the first pulse to ``after_ms`` after the last
    """
    trace_sets: list[ExperimentTraceSet] = []
    warnings: list[str] = []
    for row_label, row in rows.iterrows():
        window = time_window_ms or (-before_ms, _last_pulse_ms(row) + after_ms)
        collected = collect_traces_by_isi(
            rows.loc[[row_label]], load_ephys, channel=channel,
            stimulus_channel=stimulus_channel, time_window_ms=window,
        )
        warnings.extend(collected.warnings)
        trace_sets.extend(ExperimentTraceSet(row=row, traces=t) for t in collected.trace_sets)

    metadata = {column: _unique_values_text(rows, column) for column in HEADER_COLUMNS}
    if "spineid" in metadata and rows["spineid"].nunique() > 3:
        spines = sorted(rows["spineid"].dropna().astype(int).unique())
        metadata["spineid"] = f"{spines[0]}-{spines[-1]} ({len(spines)} spines)"
    return ExperimentTraces(trace_sets=trace_sets, criteria=dict(criteria or {}), metadata=metadata, warnings=warnings)
