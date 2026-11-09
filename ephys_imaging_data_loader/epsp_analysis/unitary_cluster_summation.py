"""
unitary_cluster_summation.py
============================

Analysis of ONE "set" = all recordings of one expdate + neuronid + dendriteid +
group (e.g. 20260322 / neuron 1 / dendrite 1 / post-apv), for EPSPs (current
clamp, mV) and EPSCs (voltage clamp, pA) alike. Two questions are answered:

1. TEMPORAL summation (every single spine and every cluster on its own):
   the deconvolution of ``summated_epsp_deconvolution`` over all ISIs that input
   was recorded with -> amplitude, rise, decay of the 1st, 2nd ... response.

2. SPATIAL summation (clusters only): is the response to a cluster of n spines
   bigger (supralinear), equal (linear) or smaller (sublinear) than the
   ARITHMETIC SUM of the responses of its spines activated one by one?

Which recordings are single spines and which are clusters
---------------------------------------------------------
From the sheet's ``spineid`` and ``uncaging_targets`` columns:

* ``uncaging_targets`` = 1                -> single spine ``spineid`` (also 10, 11 ...)
* ``spineid`` = 0 and ``uncaging_targets`` = n > 1
                                          -> cluster of "all" n spines; its members are
                                             taken to be the single spines recorded in
                                             the same set (the first n by number when
                                             there are more; noted in the table)
* ``spineid`` with several digits, e.g. 12345 or 234
                                          -> cluster of those spines (one digit each),
                                             activated in that order
* ``spineid`` with one digit but ``uncaging_targets`` > 1
                                          -> single spine (in some experiments the column
                                             holds the number of spines on the dendrite;
                                             noted in the table)

The 1st response (used for the spatial summation)
-------------------------------------------------
The 1st stimulus of every sweep comes after rest, so the 1st response of a
recording with a long ISI is an isolated response. For every input the sweeps
of its ``summation_isi_ms`` recording(s) (default 1000 ms; if there is none,
the longest ISI >= ``threshold_isi_ms``, with a note) are averaged and the
averaged response from 10 ms before to min(ISI, ``after_ms``) after the 1st
stimulus is fitted with the EPSP model of ``epsp_waveform_model``, with ALL
shape parameters free (unlike the deconvolution's EPSP1, whose shape is the
template's):

* ``measured_peak``            peak of the fitted curve (mV or pA), searched from
                               the EPSP onset (fit latency) to onset + 30 ms
                               (PEAK_WINDOW_MS; a later maximum is more likely
                               polysynaptic); if the curve still rises at 30 ms
                               the value there is used and the notes say so
* ``measured_raw_peak``        peak of the averaged trace itself (1 ms running
                               mean, same window) — a model-free check
* ``measured_rise_20_80_ms``   20-80 % rise time of the fitted curve
* ``measured_decay_tau_ms``    decay time constant of the fit
* ``measured_max_slope``       largest dV/dt of the fitted curve (mV/ms; pA/ms
                               for EPSCs) from the EPSP onset (fit latency) to the
                               EPSP peak (of the fit)
* ``measured_raw_max_slope``   the same on the averaged trace itself (1 ms running
                               mean, slope over 1 ms) — a model-free check
* ``measured_fit_r2``          goodness of that fit

The linear sum (expected cluster response)
------------------------------------------
For a cluster of spines s1, s2, ... sn, the averaged 1st responses of the
single spines are ADDED as traces (not just their peaks, because they do not
peak at exactly the same time). The spines of a cluster are uncaged one after
the other, so with ``uncaging_offset=True`` (default) each spine's response is
shifted by the time its target is uncaged in the cluster's sequence:

* step = ``stimdur_total`` / ``n_laser_points`` (time per laser target incl. the
  mirror travel; ``stimdur`` when ``stimdur_total`` is empty);
* ``n_laser_points`` (read from the image file) counts laser POINTS, while
  ``uncaging_targets`` counts SPINES (sheet changed 15 Oct 2026; in older sheets
  ``uncaging_targets`` held the point count and is used instead). In the
  shrinking-cluster experiments all 8 points are fired and the unwanted ones
  are moved off the dendrite (cluster 1234: 4 spines, 8 points); then spine k
  is taken as target k, uncaged at (k - 1) x step. Otherwise the k-th listed
  spine is uncaged at (k - 1) x step;
* a single-spine recording that was itself made with several targets (spine k
  of a shrinking cluster) already contains its (k - 1) x step delay, which is
  subtracted — unless its image is shared with other single spines (then it is
  the cluster image, e.g. gt7spines 2021, and the single spine is taken as
  uncaged alone at t = 0).

The shifts used are in ``uncaging_shifts_ms``. The summed trace is fitted and
measured exactly like a recording:

* ``linear_sum_peak`` / ``_raw_peak`` / ``_rise_20_80_ms`` / ``_decay_tau_ms`` /
  ``_max_slope`` / ``_raw_max_slope``
* ``linear_sum_unshifted_*``   the same measures of the sum WITHOUT the shifts (every
                               spine's response added at t = 0), and ``gain_unshifted``
                               = measured_peak / linear_sum_unshifted_peak
* ``sum_of_unitary_peaks``     plain sum of the single-spine peaks (for reference)
* ``gain``                     measured_peak / linear_sum_peak
                               (1 = linear, > 1 supralinear, < 1 sublinear)
* ``gain_raw``                 the same with the raw (model-free) peaks
* ``missing_spines``           cluster members without a single-spine recording
                               (then there is no linear sum)

EPSCs
-----
Voltage-clamp responses (pA) are made POSITIVE before the analysis: the sign
of the largest deflection in the first 100 ms of the set's mean response is
the set's ``polarity`` (-1 for inward AMPA currents at negative potentials,
+1 for outward currents, e.g. NMDA at +40 mV); traces are multiplied by it.
All amplitudes in the table are therefore positive sizes; the absolute
baselines (holding current before each stimulus) are given with their real sign.
EPSCs have no h-current sag, so the model is fitted without the sag term.

Table
-----
``SetSummationAnalysis.table()``: one row per input (single spine or cluster),
ISI and STIMULUS (``stimid`` 1, 2 ... of the train; empty when the input could
not be deconvolved), always with the same columns in the same order
(``SET_TABLE_COLUMNS``), so the tables of all sets stack directly:

* identifiers: expdate ... input, kind, spineid, member_spines, n_spines,
  ``cluster_spread_um`` (clusters: mean distance of the member spines' laser
  points from their centroid, in µm, in the image plane; see
  experiment_catalog/set_spine_map_image.py) and ``cluster_spread_source``
* the recording: files, ``laser_power_percent`` (the sheet's ``stimpower``,
  uncaging laser power in %; "a;b" when pooled recordings differ), isi_ms,
  n_pulses, n_sweeps, stimid
* this stimulus' response from the deconvolution: baseline (absolute), onset_ms,
  peak, rise_20_80_ms, decay_tau_ms, sag, latency_ms, ratio_to_stim1,
  max_slope_per_ms (mV or pA = ``unit``)
* per ISI (repeated on each stimid row): paired_pulse_ratio, r_squared, rms and
  the template's values
* only on the stimid-1 row of the ISI used for it: the 1st-response and
  linear-sum columns above (so averaging a column never counts them twice)
* missing_spines, notes (what was assumed or missing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..experiment_catalog.experiment_traces import ExperimentTraces, ExperimentTraceSet
from .epsp_waveform_model import (
    PEAK_WINDOW_MS,
    EpspMeasures,
    EpspParameters,
    fit_epsp,
    max_slope_window_ms,
    measure_epsp,
)
from .summated_epsp_deconvolution import SummatedEpspAnalysis, analyse_summated_epsps

logger = logging.getLogger(__name__)

FIT_BEFORE_MS = 10.0            # the 1st-response fit starts 10 ms before the stimulus
RAW_PEAK_SMOOTH_MS = 1.0        # the raw peak: largest 1 ms running mean, onset -> onset + PEAK_WINDOW_MS
RAW_SLOPE_SPAN_MS = 1.0         # raw max slope: 1 ms running mean, slope over 1 ms
POLARITY_WINDOW_MS = 100.0      # window for the sign of voltage-clamp responses


def _number(value) -> float:
    """Sheet value -> float (NaN when empty / not a number)."""
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if pd.notna(value) else np.nan


# =============================================================================
# Inputs: single spines and clusters
# =============================================================================
@dataclass
class SpineInput:
    """One stimulated input of a set: a single spine or a cluster, with its recordings."""

    kind: str                          # "unitary" or "cluster"
    spines: tuple[int, ...] | None     # member spines in activation order; None = "all" (spineid 0)
    n_spines: int
    sheet_spineid: str                 # spineid as written in the sheet
    indices: list[int] = field(default_factory=list)   # positions of its recordings in the set's trace_sets
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """'spine 3', 'cluster 1-2-3', 'all 5 spines'."""
        if self.kind == "unitary":
            return f"spine {self.spines[0]}"
        if self.spines is None:
            return f"all {self.n_spines} spines"
        return "cluster " + "-".join(str(s) for s in self.spines)

    @property
    def file_label(self) -> str:
        """For file names: 'spine3', 'cluster123', 'cluster_all5'."""
        if self.kind == "unitary":
            return f"spine{self.spines[0]}"
        if self.spines is None:
            return f"cluster_all{self.n_spines}"
        return "cluster" + "".join(str(s) for s in self.spines)


def classify_row(row: pd.Series) -> tuple[str, tuple[int, ...] | None, int, str]:
    """(kind, member spines, number of spines, note) of a sheet row (see module docstring)."""
    spineid, targets = _number(row.get("spineid")), _number(row.get("uncaging_targets"))
    if not np.isfinite(spineid):
        return "unitary", (-1,), 1, "no spineid in the sheet"
    text = str(int(spineid))
    if targets == 1:
        return "unitary", (int(text),), 1, ""
    if text == "0":
        if np.isfinite(targets) and targets > 1:
            return "cluster", None, int(targets), ""
        return "unitary", (0,), 1, "spineid 0 without uncaging_targets: treated as one input"
    if len(text) == 1:
        note = (f"sheet uncaging_targets = {targets:g} for spineid {text}: treated as a single spine"
                if np.isfinite(targets) else "")
        return "unitary", (int(text),), 1, note
    return "cluster", tuple(int(c) for c in text), len(text), ""


def identify_inputs(data: ExperimentTraces) -> list[SpineInput]:
    """Group the recordings of a set by input; single spines first (by number), then clusters (by size)."""
    found: dict[tuple, SpineInput] = {}
    for index, trace_set in enumerate(data.trace_sets):
        kind, spines, n_spines, note = classify_row(trace_set.row)
        key = (kind, spines if spines is not None else ("all", n_spines))
        if key not in found:
            found[key] = SpineInput(kind, spines, n_spines, str(trace_set.value("spineid")))
        found[key].indices.append(index)
        if note and note not in found[key].notes:
            found[key].notes.append(note)
    return sorted(found.values(), key=lambda s: (s.kind != "unitary", s.n_spines, s.spines or ()))


# =============================================================================
# 1st response of an input, and the linear sum of single spines
# =============================================================================
@dataclass
class FirstResponse:
    """Averaged, fitted 1st response of one input (or a linear sum of inputs)."""

    isi_ms: float                 # ISI of the recording(s) used (NaN for a linear sum)
    n_sweeps: int
    time_ms: np.ndarray           # from the 1st stimulus
    trace: np.ndarray             # averaged, baseline-subtracted, sign-corrected
    parameters: EpspParameters
    measures: EpspMeasures
    r_squared: float
    raw_peak: float
    raw_max_slope: float = np.nan        # max dV/dt of the averaged trace itself (see _raw_max_slope)
    raw_max_slope_time_ms: float = np.nan
    raw_peak_time_ms: float = np.nan     # when the raw peak occurs (from the stimulus)

    @property
    def fitted(self) -> np.ndarray:
        return self.parameters.waveform(self.time_ms)


def _raw_max_slope(time_ms: np.ndarray, trace: np.ndarray, measures: EpspMeasures) -> tuple[float, float]:
    """
    (max dV/dt, its time) of the averaged trace itself, between the EPSP onset
    and the EPSP peak of the fit (``max_slope_window_ms``): the trace is
    smoothed with a 1 ms running mean and the slope taken over 1 ms
    ((v(t + 0.5 ms) - v(t - 0.5 ms)) / 1 ms), since a point-to-point derivative
    of a recorded trace is dominated by noise. A model-free check of the
    fitted max slope, like ``raw_peak`` for the peak.
    """
    start, end = max_slope_window_ms(measures)
    if not (np.isfinite(start) and np.isfinite(end)):
        return np.nan, np.nan
    step = float(np.median(np.diff(time_ms)))
    width = max(1, int(round(RAW_SLOPE_SPAN_MS / step)))
    smooth = np.convolve(np.nan_to_num(trace), np.ones(width) / width, mode="same")
    half = max(1, width // 2)
    slope = np.full(len(smooth), np.nan)
    slope[half:-half] = (smooth[2 * half:] - smooth[:-2 * half]) / (2 * half * step)
    window = (time_ms >= start) & (time_ms <= min(end, float(time_ms[-1]))) & np.isfinite(slope)
    if not window.any():
        return np.nan, np.nan
    best = int(np.argmax(np.where(window, slope, -np.inf)))
    return float(slope[best]), float(time_ms[best])


def _raw_peak(time_ms: np.ndarray, trace: np.ndarray, onset_ms: float = 0.0) -> tuple[float, float]:
    """
    (value, time) of the largest 1 ms running mean between the EPSP onset (the
    fit's latency; the stimulus when unknown) and onset + PEAK_WINDOW_MS (30 ms):
    a later maximum is more likely a polysynaptic / late component.
    """
    onset_ms = onset_ms if np.isfinite(onset_ms) else 0.0
    step = float(np.median(np.diff(time_ms)))
    width = max(1, int(round(RAW_PEAK_SMOOTH_MS / step)))
    smooth = np.convolve(np.nan_to_num(trace), np.ones(width) / width, mode="same")
    window = (time_ms >= onset_ms) & (time_ms <= onset_ms + PEAK_WINDOW_MS)
    if not window.any():
        return np.nan, np.nan
    best = int(np.argmax(np.where(window, smooth, -np.inf)))
    return float(smooth[best]), float(time_ms[best])


def fit_first_response(time_ms: np.ndarray, trace: np.ndarray, isi_ms: float, n_sweeps: int,
                       end_ms: float, with_sag: bool) -> FirstResponse:
    """Fit the EPSP model (all shape parameters free) from -10 ms to ``end_ms``."""
    window = np.isfinite(trace) & (time_ms >= -FIT_BEFORE_MS) & (time_ms < end_ms)
    t, y = time_ms[window], trace[window]
    parameters, _ = fit_epsp(t, y, with_sag)
    residual = y - parameters.waveform(t)
    r_squared = 1.0 - float(np.sum(residual ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
    measures = measure_epsp(parameters)
    raw_peak, raw_peak_time = _raw_peak(t, y, measures.latency_ms)
    return FirstResponse(isi_ms, n_sweeps, t, y, parameters, measures, r_squared, raw_peak,
                         *_raw_max_slope(t, y, measures), raw_peak_time_ms=raw_peak_time)


def _summation_recordings(data: ExperimentTraces, spine_input: SpineInput, summation_isi_ms: float,
                          threshold_isi_ms: float) -> tuple[float, list[ExperimentTraceSet], str]:
    """(ISI used, its recordings, note) for the 1st response of an input; ISI NaN when nothing fits."""
    members = [data.trace_sets[i] for i in spine_input.indices]
    isis = sorted({_number(m.value("isi")) for m in members if np.isfinite(_number(m.value("isi")))})
    chosen, note = np.nan, ""
    if any(np.isclose(isi, summation_isi_ms) for isi in isis):
        chosen = float(summation_isi_ms)
    else:
        long = [isi for isi in isis if isi >= threshold_isi_ms]
        if long:
            chosen = max(long)
            note = f"no {summation_isi_ms:g} ms recording: 1st response from the {chosen:g} ms recording"
        else:
            note = f"no recording with ISI >= {threshold_isi_ms:g} ms: no isolated 1st response"
    used = [m for m in members if np.isfinite(chosen) and np.isclose(_number(m.value("isi")), chosen)]
    return chosen, used, note


def measure_first_response(recordings: list[ExperimentTraceSet], isi_ms: float, after_ms: float,
                           with_sag: bool) -> FirstResponse:
    """Average the sweeps of ``recordings`` and fit their 1st response."""
    averaged = ExperimentTraces(recordings, {}, {}).grouped(None)[0]
    end = min(isi_ms, after_ms) if np.isfinite(isi_ms) else after_ms
    return fit_first_response(averaged.time_ms, averaged.mean, isi_ms, averaged.n_sweeps, end, with_sag)


@dataclass
class LinearSum:
    """Arithmetic sum of the 1st responses of a cluster's single spines."""

    response: FirstResponse        # the summed trace, fitted and measured like a recording
    member_traces: list[np.ndarray]   # each spine's response, shifted, on response.time_ms
    shifts_ms: list[float]         # shift applied to each member (activation time in the cluster sequence)
    sum_of_peaks: float

    @property
    def step_ms(self) -> float:
        """Largest shift between consecutive members (0 without shifts)."""
        return float(np.max(np.diff(self.shifts_ms))) if len(self.shifts_ms) > 1 else 0.0


# -----------------------------------------------------------------------------
# Activation times within a multi-target uncaging sequence
# -----------------------------------------------------------------------------
# The sheet's "uncaging_targets" is the number of laser POINTS stored in the image
# file, not the number of spines: in the "shrinking cluster" experiments (dend_int)
# all 8 points are always fired and the unwanted ones are moved off the dendrite,
# so cluster 1234 has 8 targets, spine k being target k (fired at (k - 1) x step).
# step = time per target incl. the mirror travel = stimdur_total / uncaging_targets.
def _laser_points(recording: ExperimentTraceSet) -> float:
    """
    Laser points fired: the sheet's "n_laser_points" (from the image file; added
    15 Oct 2026 when "uncaging_targets" was set to the number of SPINES), else
    "uncaging_targets" (older sheets, where it held the point count).
    """
    value = _number(recording.value("n_laser_points"))
    return value if np.isfinite(value) else _number(recording.value("uncaging_targets"))


def target_step_ms(recordings: list[ExperimentTraceSet]) -> float:
    """Time per laser target (activation + travel) from the sheet; 0 when unknown."""
    totals = [_number(r.value("stimdur_total")) for r in recordings]
    targets = [_laser_points(r) for r in recordings]
    steps = [t / n for t, n in zip(totals, targets) if np.isfinite(t) and np.isfinite(n) and n > 0]
    if steps:
        return float(np.mean(steps))
    per_target = [_number(r.value("stimdur")) for r in recordings]
    per_target = [v for v in per_target if np.isfinite(v)]
    return float(np.mean(per_target)) if per_target else 0.0


def _n_targets(recordings: list[ExperimentTraceSet]) -> float:
    """Largest number of laser points fired among the recordings."""
    values = [_laser_points(r) for r in recordings]
    values = [v for v in values if np.isfinite(v)]
    return max(values) if values else np.nan


def _image_positions(recordings: list[ExperimentTraceSet], on_spine_positions, n_members: int) -> list[int] | None:
    """
    1-based scan positions of the laser points that are ON spines, read from the
    recording's image (``on_spine_positions(recording)``, see
    experiment_catalog/uncaging_target_labels.py); None when no image resolves it
    or the number of points on spines differs from ``n_members``.
    """
    if on_spine_positions is None:
        return None
    for recording in recordings:
        try:
            positions = on_spine_positions(recording)
        except Exception:      # image missing / unreadable: fall back to the sheet
            positions = None
        if positions and len(positions) == n_members:
            return list(positions)
    return None


def cluster_activation_times_ms(cluster: SpineInput, recordings: list[ExperimentTraceSet],
                                members: tuple[int, ...], on_spine_positions=None) -> tuple[list[float], str]:
    """
    Time after the trigger at which each member spine of a cluster is uncaged, and a note.

    * the image tells which laser points are on spines (``on_spine_positions``):
      the spines (sorted) are uncaged at those points, (position - 1) x step;
    * else, more laser targets than spines (shrinking cluster, e.g. 1234 with 8
      targets): spine k is target k -> (k - 1) x step;
    * otherwise the spines are uncaged in the order listed (12345 or spineid 0): k-th -> (k - 1) x step.
    """
    step = target_step_ms(recordings)
    n_targets = _n_targets(recordings)
    positions = _image_positions(recordings, on_spine_positions, len(members))
    if positions is not None and np.isfinite(n_targets) and n_targets > len(members):
        by_spine = dict(zip(sorted(members), positions))
        note = ""
        if positions != sorted(members):
            note = (f"laser points on spines (from the image) {'-'.join(map(str, positions))} ≠ spineid "
                    f"{''.join(map(str, members))}: timing from the image; check which spines were really stimulated")
        return [(by_spine[s] - 1) * step for s in members], note
    if cluster.spines is not None and np.isfinite(n_targets) and n_targets > len(members):
        return [(s - 1) * step for s in members], (f"{n_targets:g} laser targets for {len(members)} spines: "
                                                    "spine k taken as target k")
    return [k * step for k in range(len(members))], ""


def unitary_own_delay_ms(spine: int, recordings: list[ExperimentTraceSet], shared_image: bool,
                         on_spine_positions=None) -> tuple[float, str]:
    """
    Delay already contained in a single-spine recording, and a note: a single
    spine recorded with a multi-target protocol (e.g. spine 4 of a shrinking
    cluster: 8 targets, one on a spine) is uncaged at (position - 1) x step, the
    position of its on-spine point from the image, else = spine id. Not applied
    when the row's image is shared with other single spines (then the image is
    the cluster image, e.g. gt7spines 2021, and says nothing about this recording).
    """
    if shared_image or not np.isfinite(_n_targets(recordings)) or _n_targets(recordings) <= 1:
        return 0.0, ""
    positions = _image_positions(recordings, on_spine_positions, 1)
    if positions is not None:
        note = "" if positions[0] == spine else (f"spine {spine} recorded at laser point {positions[0]} "
                                                 "(from the image)")
        return (positions[0] - 1) * target_step_ms(recordings), note
    return max(spine - 1, 0) * target_step_ms(recordings), ""


def linear_sum(members: list[FirstResponse], shifts_ms: list[float], end_ms: float, with_sag: bool) -> LinearSum:
    """Add the members' averaged 1st responses, each shifted by its ``shifts_ms``; fit the sum."""
    dt = min(float(np.median(np.diff(m.time_ms))) for m in members)
    end = min([end_ms] + [float(m.time_ms[-1]) + shift for m, shift in zip(members, shifts_ms)])
    grid = np.arange(-FIT_BEFORE_MS, end, dt)
    # before its own recording starts (10 ms before its stimulus) a response is 0
    shifted = [np.interp(grid - shift, m.time_ms, m.trace, left=0.0) for m, shift in zip(members, shifts_ms)]
    total = np.sum(shifted, axis=0)
    response = fit_first_response(grid, total, np.nan, min(m.n_sweeps for m in members), end + 1, with_sag)
    return LinearSum(response, shifted, list(shifts_ms), float(sum(m.measures.peak_mv for m in members)))


# =============================================================================
# One set
# =============================================================================
@dataclass
class SetSummationAnalysis:
    """Everything computed for one set and one recording unit (mV or pA)."""

    set_info: dict                     # expdate, neuronid, dendriteid, group
    unit: str                          # "mV" or "pA"
    polarity: int                      # traces were multiplied by this (always 1 for mV)
    data: ExperimentTraces             # the set's traces, baseline-subtracted and sign-corrected
    inputs: list[SpineInput]
    deconvolutions: dict[str, SummatedEpspAnalysis] = field(default_factory=dict)   # by input label
    first_responses: dict[str, FirstResponse] = field(default_factory=dict)
    linear_sums: dict[str, LinearSum] = field(default_factory=dict)            # spines shifted by uncaging time
    linear_sums_unshifted: dict[str, LinearSum] = field(default_factory=dict)  # all spines added at t = 0
    cluster_spread: dict = field(default_factory=dict)   # input label -> ClusterSpread (set_spine_map_image.py)
    members: dict[str, tuple[int, ...]] = field(default_factory=dict)        # cluster label -> spines used
    missing: dict[str, tuple[int, ...]] = field(default_factory=dict)        # cluster label -> spines without data
    warnings: list[str] = field(default_factory=list)

    @property
    def response_name(self) -> str:
        return "EPSC" if self.unit.lower() == "pa" else "EPSP"

    @property
    def name(self) -> str:
        """'20260322_1_1_post-apv' (folder / file names)."""
        info = self.set_info
        return f"{info['expdate']}_{info['neuronid']}_{info['dendriteid']}_{info['group']}"

    def clusters(self) -> list[SpineInput]:
        return [s for s in self.inputs if s.kind == "cluster"]

    def recordings_of(self, spine_input: SpineInput) -> list[ExperimentTraceSet]:
        return [self.data.trace_sets[i] for i in spine_input.indices]

    def table(self) -> pd.DataFrame:
        """
        One row per input, ISI and stimulus (``stimid``), always with the columns of
        ``SET_TABLE_COLUMNS`` in that order (see module docstring), so the tables of
        different sets can be stacked directly.
        """
        tables = [self._input_table(spine_input) for spine_input in self.inputs]
        tables = [t for t in tables if not t.empty]
        table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        return conform_set_table(table)

    def _input_table(self, spine_input: SpineInput) -> pd.DataFrame:
        recordings = self.recordings_of(spine_input)
        label = spine_input.label
        members = self.members.get(label, spine_input.spines or ())
        spread = self.cluster_spread.get(label)
        identity = {
            **self.set_info,
            "expname": "|".join(dict.fromkeys(str(r.value("expname")) for r in recordings)),
            "input": label, "kind": spine_input.kind, "spineid": spine_input.sheet_spineid,
            "member_spines": "-".join(str(s) for s in members), "n_spines": spine_input.n_spines,
            "cluster_spread_um": spread.spread_um if spread is not None else np.nan,
            "cluster_spread_source": spread.source if spread is not None else "",
            "unit": self.unit, "polarity": self.polarity,
        }
        analysis = self.deconvolutions.get(label)
        if analysis is not None and analysis.fits:
            rows = _long_deconvolution_table(analysis.table(), self.polarity)
        else:                                  # no deconvolution: one row per ISI, no stimid
            rows = pd.DataFrame({"isi_ms": sorted({_number(r.value("isi")) for r in recordings}), "stimid": np.nan})
        by_isi: dict[float, list] = {}
        for r in recordings:
            by_isi.setdefault(_number(r.value("isi")), []).append(r)
        rows["files"] = [";".join(dict.fromkeys(str(r.traces.file_name) for r in by_isi.get(isi, [])))
                         for isi in rows["isi_ms"]]
        rows["laser_power_percent"] = [_laser_power(by_isi.get(isi, [])) for isi in rows["isi_ms"]]
        for column, value in identity.items():
            rows[column] = value

        notes = list(spine_input.notes)
        first = self.first_responses.get(label)
        if first is not None:
            # the 1st-response / linear-sum values go on ONE row: stimulus 1 of the ISI used
            same_isi = np.isclose(rows["isi_ms"].astype(float), first.isi_ms)
            matching = np.nonzero(same_isi & ((rows["stimid"] == 1) | rows["stimid"].isna()))[0]
            if not matching.size:
                extra = {**identity, "isi_ms": first.isi_ms, "stimid": 1,
                         "laser_power_percent": _laser_power(by_isi.get(first.isi_ms, []))}
                rows = pd.concat([rows, pd.DataFrame([extra])], ignore_index=True)
                matching = [len(rows) - 1]
            values = {
                "summation_isi_ms": first.isi_ms, "summation_n_sweeps": first.n_sweeps,
                "measured_peak": first.measures.peak_mv, "measured_raw_peak": first.raw_peak,
                "measured_rise_20_80_ms": first.measures.rise_20_80_ms,
                "measured_decay_tau_ms": first.measures.decay_tau_ms,
                "measured_max_slope": first.measures.max_slope_mv_per_ms, "measured_raw_max_slope": first.raw_max_slope,
                "measured_fit_r2": first.r_squared,
            }
            summed = self.linear_sums.get(label)
            if summed is not None:
                expected = summed.response
                values.update({
                    "linear_sum_peak": expected.measures.peak_mv, "linear_sum_raw_peak": expected.raw_peak,
                    "linear_sum_rise_20_80_ms": expected.measures.rise_20_80_ms,
                    "linear_sum_decay_tau_ms": expected.measures.decay_tau_ms,
                    "linear_sum_max_slope": expected.measures.max_slope_mv_per_ms,
                    "linear_sum_raw_max_slope": expected.raw_max_slope,
                    "sum_of_unitary_peaks": summed.sum_of_peaks,
                    "uncaging_shifts_ms": ";".join(f"{s:.2f}" for s in summed.shifts_ms),
                    "gain": first.measures.peak_mv / expected.measures.peak_mv if expected.measures.peak_mv else np.nan,
                    "gain_raw": first.raw_peak / expected.raw_peak if expected.raw_peak else np.nan,
                })
            unshifted = self.linear_sums_unshifted.get(label)
            if unshifted is not None:
                expected = unshifted.response
                values.update({
                    "linear_sum_unshifted_peak": expected.measures.peak_mv,
                    "linear_sum_unshifted_raw_peak": expected.raw_peak,
                    "linear_sum_unshifted_rise_20_80_ms": expected.measures.rise_20_80_ms,
                    "linear_sum_unshifted_decay_tau_ms": expected.measures.decay_tau_ms,
                    "linear_sum_unshifted_max_slope": expected.measures.max_slope_mv_per_ms,
                    "linear_sum_unshifted_raw_max_slope": expected.raw_max_slope,
                    "gain_unshifted": (first.measures.peak_mv / expected.measures.peak_mv
                                       if expected.measures.peak_mv else np.nan),
                })
            for column, value in values.items():
                if column not in rows.columns:
                    rows[column] = "" if isinstance(value, str) else np.nan
                rows.loc[rows.index[matching[0]], column] = value
        if self.missing.get(label):
            rows["missing_spines"] = "-".join(str(s) for s in self.missing[label])
        # peaks taken at onset + 30 ms because the fitted curve was still rising there
        capped = []
        if first is not None and first.measures.peak_capped:
            capped.append(f"1st response ({first.isi_ms:g} ms)")
        for name, sums in (("linear sum", self.linear_sums), ("unshifted linear sum", self.linear_sums_unshifted)):
            if label in sums and sums[label].response.measures.peak_capped:
                capped.append(name)
        if analysis is not None:
            capped += [f"stim {k + 1} at {fit.isi_ms:g} ms" for fit in analysis.fits
                       for k, m in enumerate(fit.measures) if m.peak_capped]
        if capped:
            notes.append(f"fit still rising {PEAK_WINDOW_MS:g} ms after onset, peak taken there: " + ", ".join(capped))
        if spread is not None and spread.note and spread.note not in notes:
            notes.append(spread.note)
        powers = {p for p in rows["laser_power_percent"] if isinstance(p, str)}
        if powers:
            notes.append(f"recordings of different laser power pooled ({', '.join(sorted(powers))} %)")
        rows["notes"] = "; ".join(notes)
        return rows.sort_values(["isi_ms", "stimid"], kind="stable", na_position="first").reset_index(drop=True)


# =============================================================================
# The fixed columns of a set table
# =============================================================================
# Per response (one row per stimulus, ``stimid`` 1, 2 ...; ``resp`` = EPSP or EPSC, in ``unit``)
RESPONSE_COLUMNS = ["baseline", "onset_ms", "peak", "rise_20_80_ms", "decay_tau_ms", "sag", "latency_ms",
                    "ratio_to_stim1", "max_slope_per_ms", "local_baseline_shift"]
SET_TABLE_COLUMNS = [
    # identifiers of the set and the input
    "expdate", "neuronid", "dendriteid", "group", "set", "expname", "input", "kind", "spineid", "member_spines",
    "n_spines", "cluster_spread_um", "cluster_spread_source", "unit", "polarity",
    # the recording(s)
    "files", "laser_power_percent", "isi_ms", "n_pulses", "n_sweeps", "stimid",
    # this stimulus' response (deconvolution)
    *RESPONSE_COLUMNS,
    # the whole train at this ISI (the same on every stimid row of the ISI)
    "paired_pulse_ratio", "r_squared", "rms", "threshold_isi_ms", "template_n_events", "template_peak",
    "template_rise_20_80_ms", "template_decay_tau_ms", "template_sag",
    # isolated 1st response and linear sums (only on stimid 1 of the ISI used: summation_isi_ms)
    "summation_isi_ms", "summation_n_sweeps", "measured_peak", "measured_raw_peak", "measured_rise_20_80_ms",
    "measured_decay_tau_ms", "measured_max_slope", "measured_raw_max_slope", "measured_fit_r2",
    "linear_sum_peak", "linear_sum_raw_peak", "linear_sum_rise_20_80_ms", "linear_sum_decay_tau_ms",
    "linear_sum_max_slope", "linear_sum_raw_max_slope", "sum_of_unitary_peaks", "uncaging_shifts_ms",
    "gain", "gain_raw",
    "linear_sum_unshifted_peak", "linear_sum_unshifted_raw_peak", "linear_sum_unshifted_rise_20_80_ms",
    "linear_sum_unshifted_decay_tau_ms", "linear_sum_unshifted_max_slope", "linear_sum_unshifted_raw_max_slope",
    "gain_unshifted",
    # fit of the PPR against the ISI for this input (ppr_recovery.py; same on every row of the input)
    "ppr_fit_A", "ppr_fit_A_se", "ppr_fit_tau_ms", "ppr_fit_tau_se_ms", "ppr_fit_r_squared", "ppr_fit_rms",
    "ppr_fit_n_isis", "ppr_fit_tau_meaningful",
    "missing_spines", "notes",
]


def conform_set_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    The columns of ``SET_TABLE_COLUMNS``, in that order (missing ones empty);
    a column not in the list (should not happen) is kept at the end.
    """
    extra = [c for c in table.columns if c not in SET_TABLE_COLUMNS]
    return table.reindex(columns=SET_TABLE_COLUMNS + extra)


def _laser_power(recordings: list[ExperimentTraceSet]):
    """Uncaging laser power (%) of the sheet's ``stimpower``; 'a;b' when pooled recordings differ."""
    powers = sorted({p for p in (_number(r.value("stimpower")) for r in recordings) if np.isfinite(p)})
    if not powers:
        return np.nan
    return powers[0] if len(powers) == 1 else ";".join(f"{p:g}" for p in powers)


def _long_deconvolution_table(table: pd.DataFrame, polarity: int) -> pd.DataFrame:
    """
    Deconvolution table (one row per ISI, epsp1_ ... epspN_ columns) -> one row
    per ISI and stimulus: ``stimid`` k with that stimulus' values under the
    ``RESPONSE_COLUMNS`` names ('epsp' -> 'resp', no '_mv': EPSCs are in pA;
    ``ratio_to_epsp1`` -> ``ratio_to_stim1``). Absolute baselines get their real
    sign back for sign-corrected EPSCs. The identifiers are added by the caller.
    """
    table = table.drop(columns=[c for c in ("expdate", "neuronid", "dendriteid", "spineid", "expname", "group")
                                if c in table.columns])
    names = ["baseline_mv", "onset_ms", "peak_mv", "rise_20_80_ms", "decay_tau_ms", "sag_mv", "latency_ms",
             "ratio_to_epsp1", "max_slope_mv_per_ms", "local_baseline_step_mv"]
    per_stim = [c for c in table.columns if c.startswith("epsp") and "_" in c and c.split("_")[0][4:].isdigit()]
    shared = [c for c in table.columns if c not in per_stim]
    rows = []
    for _, row in table.iterrows():
        for k in range(1, int(row["n_pulses"]) + 1):
            entry = {c.replace("_mv", ""): row[c] for c in shared}
            entry["stimid"] = k
            entry.update({new: row.get(f"epsp{k}_{old}", np.nan) for old, new in zip(names, RESPONSE_COLUMNS)})
            entry["baseline"] = entry["baseline"] * polarity
            rows.append(entry)
    return pd.DataFrame(rows)


def _response_polarity(data: ExperimentTraces) -> int:
    """+1 / -1: sign of the largest deflection of the mean response in the first 100 ms."""
    averaged = data.baseline_subtracted().grouped(None)[0]
    window = (averaged.time_ms >= 0) & (averaged.time_ms <= POLARITY_WINDOW_MS)
    values = averaged.mean[window]
    values = values[np.isfinite(values)]
    if not values.size:
        return 1
    return 1 if values.max() >= -values.min() else -1


def _sign_corrected(data: ExperimentTraces, polarity: int) -> ExperimentTraces:
    if polarity == 1:
        return data
    flipped = [ExperimentTraceSet(s.row, replace(s.traces, traces=s.traces.traces * polarity)) for s in data.trace_sets]
    return ExperimentTraces(flipped, data.criteria, data.metadata, data.warnings)


def analyse_set(
    data: ExperimentTraces,
    threshold_isi_ms: float = 200.0,
    summation_isi_ms: float = 1000.0,
    template_events: str = "both",
    template_length_ms: float | None = None,
    after_ms: float = 300.0,
    uncaging_offset: bool = True,
    on_spine_positions=None,
    baseline_method: str = "line",
    baseline_window_ms: float = 30.0,
    local_baseline: bool = True,
) -> list[SetSummationAnalysis]:
    """
    Deconvolution of every input + spatial summation of every cluster for the
    traces of ONE set (one expdate / neuron / dendrite / group, loaded with a
    window reaching ``after_ms`` after the last stimulus). One result per unit
    (normally just one: mV or pA). One input that cannot be analysed never
    stops the others; the reasons are in ``warnings`` and the table's notes.

    ``on_spine_positions``: optional function recording -> 1-based scan positions
    of its laser points that are on spines (from the image; None = unknown),
    used for the uncaging times of shrinking clusters; ``ExperimentCatalog``
    passes one based on ``uncaging_target_labels``. Without it: spine k = point k.

    Baseline (see ``ExperimentTraces.baseline_subtracted`` and ``fit_epsp_train``):
    ``baseline_method`` "line" (default) zeroes every sweep at the value, at the 1st
    stimulus, of a straight line fitted to the last ``baseline_window_ms`` (30) ms before
    it; "mean" = the mean before the stimulus (the method before 2026-11-02, window: all
    loaded pre-stimulus time). ``local_baseline`` (default True): at ISIs >= ``threshold_isi_ms``
    the 2nd EPSP is measured from a local baseline fitted from ``baseline_window_ms`` before
    its stimulus (absorbs the slow drift since the 1st stimulus).
    """
    groups = [g for g in data.values("group") if pd.notna(g)]
    results = []
    for unit in data.units:
        raw = ExperimentTraces([s for s in data.trace_sets if s.traces.unit == unit], data.criteria,
                               data.metadata, list(data.warnings))
        is_current = unit.lower() == "pa"
        polarity = _response_polarity(raw) if is_current else 1
        raw = _sign_corrected(raw, polarity)
        corrected = raw.baseline_subtracted(baseline_method, baseline_window_ms if baseline_method == "line" else None)
        set_info = {key: data.criteria.get(key) for key in ("expdate", "neuronid", "dendriteid")}
        set_info["group"] = "|".join(str(g) for g in groups) or "nogroup"
        result = SetSummationAnalysis(set_info, unit, polarity, corrected, identify_inputs(corrected),
                                      warnings=list(data.warnings))
        if polarity == -1:
            result.warnings.append("Inward currents analysed with their sign inverted (amplitudes are sizes).")

        # --- temporal summation and 1st response of every input -----------------------------
        for spine_input in result.inputs:
            label = spine_input.label
            subset = ExperimentTraces([corrected.trace_sets[i] for i in spine_input.indices],
                                      {**data.criteria, "spineid": spine_input.sheet_spineid},
                                      {**data.metadata, "spineid": label}, [])
            raw_subset = ExperimentTraces([raw.trace_sets[i] for i in spine_input.indices], {}, {})
            try:
                analysis = analyse_summated_epsps(
                    subset, set_info["group"], threshold_isi_ms, template_events, template_length_ms, after_ms,
                    raw=raw_subset, with_sag=not is_current, unit=unit,
                    local_baseline_min_isi_ms=threshold_isi_ms if local_baseline else None,
                    local_baseline_window_ms=baseline_window_ms)
                result.deconvolutions[label] = analysis
                spine_input.notes += [w for w in analysis.warnings if "skipped" not in w]
            except Exception as error:
                spine_input.notes.append(f"deconvolution not possible ({error})")
            isi, recordings, note = _summation_recordings(corrected, spine_input, summation_isi_ms, threshold_isi_ms)
            if note:
                spine_input.notes.append(note)
            if recordings:
                try:
                    result.first_responses[label] = measure_first_response(recordings, isi, after_ms, not is_current)
                except Exception as error:
                    spine_input.notes.append(f"1st-response fit failed ({error})")

        # --- spatial summation: clusters vs the sum of their single spines -----------------------
        singles = {s.spines[0]: s.label for s in result.inputs if s.kind == "unitary"}
        single_inputs = {s.spines[0]: s for s in result.inputs if s.kind == "unitary"}
        # single spines whose image (ophysid) is also listed for another single spine
        image_owners: dict[str, set] = {}
        for spine, single in single_inputs.items():
            for recording in result.recordings_of(single):
                image = str(recording.value("ophysid"))
                if image not in ("None", "nan", "<NA>", ""):
                    image_owners.setdefault(image, set()).add(spine)
        shared_image_spines = {s for owners in image_owners.values() if len(owners) > 1 for s in owners}
        for cluster in result.clusters():
            label = cluster.label
            if cluster.spines is None:                     # spineid 0 = "all spines"
                available = sorted(k for k in singles if k > 0)
                members = tuple(available[:cluster.n_spines])
                if len(available) > cluster.n_spines:
                    cluster.notes.append(f"spineid 0: assumed spines {'-'.join(map(str, members))} "
                                         f"of the single spines {'-'.join(map(str, available))}")
                if len(members) < cluster.n_spines:
                    cluster.notes.append(f"spineid 0: only {len(members)} single spines recorded for "
                                         f"{cluster.n_spines} targets")
            else:
                members = cluster.spines
            result.members[label] = members
            missing = tuple(s for s in members if singles.get(s) not in result.first_responses)
            if len(members) < cluster.n_spines:
                missing += tuple(f"?{k}" for k in range(len(members) + 1, cluster.n_spines + 1))
            if missing:
                result.missing[label] = missing
                cluster.notes.append("no linear sum: single-spine response missing for "
                                     + "-".join(str(s) for s in missing))
                continue
            first = result.first_responses.get(label)
            if first is None:
                continue
            shifts = [0.0] * len(members)
            if uncaging_offset:
                times, note = cluster_activation_times_ms(cluster, result.recordings_of(cluster), members,
                                                          on_spine_positions)
                if note:
                    cluster.notes.append(note)
                own = []
                for s in members:
                    delay, note = unitary_own_delay_ms(s, result.recordings_of(single_inputs[s]),
                                                       s in shared_image_spines, on_spine_positions)
                    own.append(delay)
                    if note and note not in single_inputs[s].notes:
                        single_inputs[s].notes.append(note)
                shifts = [t - o for t, o in zip(times, own)]
            member_responses = [result.first_responses[singles[s]] for s in members]
            try:
                result.linear_sums[label] = linear_sum(member_responses, shifts, float(first.time_ms[-1]),
                                                       not is_current)
                # the same sum with every spine at t = 0 (as if all were uncaged at once)
                result.linear_sums_unshifted[label] = linear_sum(member_responses, [0.0] * len(members),
                                                                 float(first.time_ms[-1]), not is_current)
            except Exception as error:
                cluster.notes.append(f"linear sum failed ({error})")
        results.append(result)
    return results


def set_summation_table(analyses: list[SetSummationAnalysis]) -> pd.DataFrame:
    """The tables of several sets / units below each other."""
    tables = [analysis.table() for analysis in analyses]
    tables = [t for t in tables if not t.empty]
    return conform_set_table(pd.concat(tables, ignore_index=True) if tables else pd.DataFrame())
