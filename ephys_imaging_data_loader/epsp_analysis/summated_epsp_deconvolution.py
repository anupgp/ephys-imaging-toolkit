"""
summated_epsp_deconvolution.py
==============================

Separates the EPSPs of a multi-pulse recording (paired pulses or trains of any
length) when later EPSPs ride on earlier ones (short inter-stimulus intervals),
so that every EPSP can be measured.

Input: the traces of one experiment sample (expdate + neuron + dendrite [+ spine])
as collected by ``experiment_catalog.experiment_traces`` — current clamp (mV),
aligned to the first stimulus (t = 0). The analysis works on baseline-subtracted
traces; the raw traces are used only for the absolute baseline potentials.

Step 1 — EPSP template from ISOLATED EPSPs
------------------------------------------
A sweep is "isolated" when all its stimuli are at least ``threshold_isi_ms``
apart (or it has a single stimulus). From those sweeps:
* the 1st EPSPs (each used up to the next stimulus of its sweep), and, with
  ``template_events="both"``,
* every later EPSP, cut out AFTER subtracting the earlier EPSPs of the same
  sweep including their slow sag tails (modelled with a provisional template
  made from the 1st EPSPs alone, scaled to each EPSP by least squares) — even at
  a 200 ms interval the sag of the previous EPSP is still recovering,
are aligned to their stimulus and averaged -> the template: the typical EPSP
shape including its sag. The template is as long as the fit window
(``after_ms``); because events end at different times, each event is scaled to
a peak of 1 before averaging and the average is scaled back to the mean peak.

Step 2 — model of the template
------------------------------
The template is fitted with the double-exponential EPSP + sag model of
``epsp_waveform_model`` (amplitude, latency, rise and decay time constants,
sag weight and sag time constants; the sag constrained slower than the EPSP).

Step 3 — deconvolution of every ISI
-----------------------------------
For each ISI (and number of pulses), the AVERAGED trace y(t) is fitted as

    y(t) = EPSP1(t) + EPSP2(t - onset2) + ... + EPSPn(t - onsetn)

* EPSP1 has the template SHAPE (the 1st stimulus comes after rest in every
  sweep); only its amplitude and latency are fitted, per ISI;
* every later EPSP is free in amplitude, latency, rise and decay time
  constants; its sag scales with its amplitude using the template's sag weight
  and time constants (the sag is too slow and small to estimate per EPSP);
* onsets = the measured stimulus times in the file.

Results per ISI: the fitted curve of every EPSP, their sum (to compare with the
recorded trace), the DECONVOLVED k-th EPSP = recorded trace - all other fitted
EPSPs (model-free, shows the real waveform), and for every EPSP:
* baseline (mV, ABSOLUTE): EPSP1 = mean of the 10 ms before the 1st stimulus
  (resting potential); later EPSPs = mean of the 2 ms before their own stimulus
  (the potential they start from, including the summation of earlier EPSPs);
* peak (mV, of that EPSP alone), 20-80 % rise time, decay tau, sag, latency;
* peak ratio to EPSP1; plus the goodness of fit (R², residual RMS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..experiment_catalog.experiment_traces import ExperimentTraces
from .epsp_waveform_model import (
    EpspMeasures,
    EpspParameters,
    fit_epsp,
    fit_waveform,
    measure_epsp,
)

logger = logging.getLogger(__name__)

BASELINE_BEFORE_MS = 10.0          # baseline of a later EPSP in the template = mean of the 10 ms before its stimulus
RESTING_BASELINE_MS = 10.0         # absolute baseline of EPSP1: the 10 ms before the 1st stimulus
LATER_BASELINE_MS = 2.0            # absolute baseline of later EPSPs: the 2 ms before their stimulus


@dataclass
class EpspTemplate:
    """Average of the isolated EPSPs and its model fit."""

    time_ms: np.ndarray            # from the stimulus
    events: np.ndarray             # (n_events, n_samples) the aligned isolated EPSPs
    mean: np.ndarray               # the template waveform
    parameters: EpspParameters     # model fit of the template
    fit_rms_mv: float
    measures: EpspMeasures
    n_first: int                   # number of 1st EPSPs averaged
    n_second: int                  # number of later (2nd, 3rd ...) EPSPs averaged
    threshold_isi_ms: float

    @property
    def fitted(self) -> np.ndarray:
        return self.parameters.waveform(self.time_ms)


@dataclass
class MultiEpspFit:
    """Deconvolution of the averaged trace of one ISI into its EPSPs."""

    isi_ms: float                  # sheet ISI
    onsets_ms: list[float]         # measured stimulus times (first = 0)
    n_sweeps: int
    time_ms: np.ndarray
    trace: np.ndarray              # recorded (averaged, baseline-subtracted) trace
    parameters: list[EpspParameters]   # one per EPSP
    measures: list[EpspMeasures]       # one per EPSP
    baselines_mv: list[float]          # absolute baseline per EPSP (see module docstring)
    r_squared: float
    rms_mv: float
    # LOCAL BASELINE (``fit_epsp_train(local_baseline_min_isi_ms=...)``): per EPSP the baseline
    # shift fitted from ``local_baseline_window_ms`` before its stimulus on (0 = none; EPSP1
    # always 0). It absorbs the drift of the baseline since the 1st stimulus, so the EPSP is
    # measured from the level it starts from.
    baseline_steps_mv: list[float] = field(default_factory=list)
    local_baseline_window_ms: float = 30.0

    # ---- curves ----------------------------------------------------------------------
    @property
    def n_epsps(self) -> int:
        return len(self.parameters)

    def curve(self, index: int) -> np.ndarray:
        """Fitted EPSP number ``index`` (0 = first) on the trace's time axis."""
        return self.parameters[index].waveform(self.time_ms, self.onsets_ms[index])

    @property
    def curves(self) -> list[np.ndarray]:
        return [self.curve(k) for k in range(self.n_epsps)]

    @property
    def baseline_curve(self) -> np.ndarray:
        """The fitted local-baseline steps on the trace's time axis (zeros without them)."""
        return _baseline_steps(self.time_ms, self.onsets_ms, self.baseline_steps_mv, self.local_baseline_window_ms)

    @property
    def summed_curve(self) -> np.ndarray:
        """Sum of the fitted EPSPs + the local-baseline steps (= the whole model)."""
        return np.sum(self.curves, axis=0) + self.baseline_curve

    def deconvolved(self, index: int) -> np.ndarray:
        """Recorded trace minus all OTHER fitted EPSPs and the local-baseline steps = EPSP
        ``index`` alone, from its own baseline (model-free)."""
        return self.trace - (self.summed_curve - self.curve(index))

    def peak_ratio(self, index: int) -> float:
        first = self.measures[0].peak_mv
        return self.measures[index].peak_mv / first if first else np.nan

    # ---- paired-pulse shortcuts ---------------------------------------------------------
    @property
    def interval_ms(self) -> float:
        return self.onsets_ms[1] if self.n_epsps > 1 else np.nan

    @property
    def first_measures(self) -> EpspMeasures:
        return self.measures[0]

    @property
    def second_measures(self) -> EpspMeasures:
        return self.measures[1]

    @property
    def first_curve(self) -> np.ndarray:
        return self.curve(0)

    @property
    def second_curve(self) -> np.ndarray:
        return self.curve(1)

    @property
    def second_deconvolved(self) -> np.ndarray:
        return self.deconvolved(1)

    @property
    def paired_pulse_ratio(self) -> float:
        return self.peak_ratio(1) if self.n_epsps > 1 else np.nan


PairedEpspFit = MultiEpspFit   # earlier name (two EPSPs)


@dataclass
class SummatedEpspAnalysis:
    """Template + per-ISI deconvolution for one group (e.g. ctrl, pre-apv)."""

    group: str
    header: str                    # experiment description for figure titles
    template: EpspTemplate
    fits: list[MultiEpspFit]
    warnings: list[str] = field(default_factory=list)
    sample: dict = field(default_factory=dict)   # expdate, neuronid, dendriteid, spineid, expname
    # "mV" for EPSPs; "pA" for EPSCs (analysed with their sign made positive, see
    # epsp_analysis/unitary_cluster_summation.py). Only used for labels: the
    # numbers keep the "_mv" column names of the EPSP analysis.
    unit: str = "mV"

    @property
    def response_name(self) -> str:
        """'EPSP' (current clamp) or 'EPSC' (voltage clamp), for figure labels."""
        return "EPSC" if self.unit.lower() == "pa" else "EPSP"

    @property
    def max_epsps(self) -> int:
        return max((fit.n_epsps for fit in self.fits), default=0)

    def table(self) -> pd.DataFrame:
        """
        One row per ISI: sample identifiers, group, ISI, number of pulses, and for
        EVERY EPSP (epsp1_, epsp2_, ... up to the longest train): absolute baseline
        (mV), onset (ms), peak (mV), 20-80 % rise (ms), decay tau (ms), sag (mV),
        latency (ms), the peak ratio to EPSP1, the max slope dV/dt of the fit
        (mV/ms, onset -> peak) and the local-baseline step (mV, 0 = none); then fit quality and the
        template's values for reference. Columns of EPSPs a trace does not have are empty.
        """
        template = self.template
        names = ["baseline_mv", "onset_ms", "peak_mv", "rise_20_80_ms", "decay_tau_ms", "sag_mv",
                 "latency_ms", "ratio_to_epsp1", "max_slope_mv_per_ms", "local_baseline_step_mv"]
        rows = []
        for fit in self.fits:
            row = {**self.sample, "group": self.group, "isi_ms": fit.isi_ms, "n_pulses": fit.n_epsps,
                   "n_sweeps": fit.n_sweeps}
            for k in range(self.max_epsps):
                if k < fit.n_epsps:
                    m = fit.measures[k]
                    step = fit.baseline_steps_mv[k] if k < len(fit.baseline_steps_mv) else 0.0
                    values = [fit.baselines_mv[k], fit.onsets_ms[k], m.peak_mv, m.rise_20_80_ms, m.decay_tau_ms,
                              m.sag_mv, m.latency_ms, fit.peak_ratio(k), m.max_slope_mv_per_ms, step]
                else:
                    values = [np.nan] * len(names)
                row.update({f"epsp{k + 1}_{name}": value for name, value in zip(names, values)})
            row.update(paired_pulse_ratio=fit.paired_pulse_ratio, r_squared=fit.r_squared, rms_mv=fit.rms_mv)
            row.update(
                threshold_isi_ms=template.threshold_isi_ms, template_n_events=template.n_first + template.n_second,
                template_peak_mv=template.measures.peak_mv, template_rise_20_80_ms=template.measures.rise_20_80_ms,
                template_decay_tau_ms=template.measures.decay_tau_ms, template_sag_mv=template.measures.sag_mv,
            )
            rows.append(row)
        return pd.DataFrame(rows).round(4)


def epsp_deconvolution_table(analyses: list[SummatedEpspAnalysis]) -> pd.DataFrame:
    """The tables of all groups (and samples) below each other."""
    tables = [analysis.table() for analysis in analyses]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# =============================================================================
# Step 1 + 2: template
# =============================================================================
def _average_events(grid: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Peak-normalised average of aligned events (NaN where an event has no data),
    scaled back to the mean peak. Events without a clear EPSP (peak in the first
    50 ms below 20 % of the median peak) are left out. Returns (mean, kept mask).
    """
    early = (grid >= 0) & (grid <= 50)
    with np.errstate(all="ignore"):
        peaks = np.nanmax(events[:, early], axis=1)
        kept = np.isfinite(peaks) & (peaks > 0.2 * np.nanmedian(peaks))
        mean = np.nanmean(events[kept] / peaks[kept, None], axis=0) * peaks[kept].mean()
    return mean, kept


def _is_isolated(onsets_ms: np.ndarray, threshold_isi_ms: float) -> bool:
    """All stimuli at least ``threshold_isi_ms`` apart (a single stimulus counts as isolated)."""
    return len(onsets_ms) == 1 or float(np.min(np.diff(onsets_ms))) >= threshold_isi_ms


def build_epsp_template(
    data: ExperimentTraces,
    threshold_isi_ms: float = 200.0,
    template_events: str = "both",
    length_ms: float | None = None,
    before_ms: float = 10.0,
    with_sag: bool = True,
) -> EpspTemplate:
    """
    Average of the isolated EPSPs (see module docstring) and its model fit.

    ``template_events``: "both" (1st and later EPSPs), "first" or "second"
    (= the later EPSPs only). ``length_ms``: template length after the stimulus
    (default 300 ms); every event contributes only up to the next stimulus.
    ``with_sag=False``: model without the sag term (EPSCs).
    """
    if template_events not in ("both", "first", "second"):
        raise ValueError("template_events must be 'both', 'first' or 'second'.")
    length_ms = float(length_ms or 300.0)
    step = min(float(np.median(np.diff(s.traces.time_ms))) for s in data.trace_sets)
    grid = np.arange(-before_ms, length_ms, step)

    # isolated sweeps: (time, sweep, stimulus onsets)
    sweeps = [
        (trace_set.traces.time_ms, sweep, np.asarray(trace_set.traces.stimulus_onsets_ms, dtype=float))
        for trace_set in data.trace_sets
        if _is_isolated(np.asarray(trace_set.traces.stimulus_onsets_ms, dtype=float), threshold_isi_ms)
        for sweep in trace_set.traces.traces
    ]
    if not sweeps:
        raise ValueError(f"No isolated EPSPs: no sweeps whose stimuli are >= {threshold_isi_ms:g} ms apart. "
                         "Lower threshold_isi_ms.")

    def cut(t, values, onsets, k):
        """Event k aligned to its stimulus, only up to the next stimulus."""
        event = np.interp(grid, t - onsets[k], values, left=np.nan, right=np.nan)
        if k + 1 < len(onsets):
            event[grid >= onsets[k + 1] - onsets[k]] = np.nan
        return event

    first_events = np.array([cut(t, sweep, onsets, 0) for t, sweep, onsets in sweeps])

    later_events = []
    if template_events in ("both", "second"):
        first_mean, _ = _average_events(grid, first_events)
        valid = np.isfinite(first_mean)
        provisional, _ = fit_epsp(grid[valid], first_mean[valid], with_sag)
        for t, sweep, onsets in sweeps:
            corrected = sweep.copy()
            for k in range(len(onsets)):
                if k > 0:
                    event = cut(t, corrected, onsets, k)
                    shifted = t - onsets[k]
                    baseline = corrected[(shifted >= -BASELINE_BEFORE_MS) & (shifted < 0)]
                    if baseline.size and np.isfinite(event).sum() > 0.3 * len(grid):
                        later_events.append(event - baseline.mean())
                # remove EPSP k (scaled provisional shape, incl. its sag tail) before the next one
                shape = provisional.waveform(t, onsets[k])
                window = (t >= onsets[k]) & (t < (onsets[k + 1] if k + 1 < len(onsets) else t[-1] + 1))
                scale = float(np.dot(shape[window], corrected[window]) /
                              max(np.dot(shape[window], shape[window]), 1e-12))
                corrected = corrected - scale * shape

    chosen, kinds = [], []
    if template_events in ("both", "first"):
        chosen.append(first_events)
        kinds += ["first"] * len(first_events)
    if template_events in ("both", "second") and later_events:
        chosen.append(np.array(later_events))
        kinds += ["later"] * len(later_events)
    if not chosen:
        raise ValueError("No isolated later EPSPs long enough for the template; use template_events='first'.")
    events, kinds = np.vstack(chosen), np.array(kinds)
    mean, kept = _average_events(grid, events)
    usable = np.isfinite(mean)
    grid, events, mean = grid[usable], events[kept][:, usable], mean[usable]
    parameters, rms = fit_epsp(grid, mean, with_sag)
    return EpspTemplate(grid, events, mean, parameters, rms, measure_epsp(parameters),
                        int(np.sum(kinds[kept] == "first")), int(np.sum(kinds[kept] == "later")), threshold_isi_ms)


# =============================================================================
# Step 3: deconvolution per ISI
# =============================================================================
def _baseline_steps(time_ms, onsets_ms, steps_mv, window_ms) -> np.ndarray:
    """Sum of steps: steps_mv[k] from onsets_ms[k] - window_ms on (missing / 0 = no step)."""
    curve = np.zeros_like(np.asarray(time_ms, dtype=float))
    for onset, step in zip(onsets_ms, steps_mv or []):
        if step:
            curve = curve + step * (np.asarray(time_ms) >= onset - window_ms)
    return curve


def fit_epsp_train(
    template: EpspTemplate,
    time_ms: np.ndarray,
    trace: np.ndarray,
    onsets_ms: list[float],
    isi_ms: float,
    n_sweeps: int,
    after_ms: float = 300.0,
    baselines_mv: list[float] | None = None,
    local_baseline_min_isi_ms: float | None = None,
    local_baseline_window_ms: float = 30.0,
) -> MultiEpspFit:
    """
    Fit y = EPSP1 + EPSP2(t - onset2) + ... to one averaged trace (see module docstring).

    ``local_baseline_min_isi_ms``: when a stimulus comes at least this long after the
    previous one, the fit gets one more free value for it: a baseline STEP starting
    ``local_baseline_window_ms`` before that stimulus. The baseline of a sweep drifts slowly
    (a median 0.16 mV, often > 0.6 mV, over 1 s in these recordings); without the step that
    drift is added to - or taken from - the later EPSP. The earlier EPSPs' tails are still
    modelled under the step, so real summation is not removed. None (default) = no steps.
    Short intervals get no step: the drift over tens of ms is negligible and a step there
    could absorb a real difference of EPSP1's tail.
    """
    shape = template.parameters
    onsets = [float(o) for o in onsets_ms]
    n = len(onsets)
    window = np.isfinite(trace) & (time_ms <= onsets[-1] + after_ms)
    t, y = time_ms[window], trace[window]
    stepped = [k for k in range(2, n + 1) if local_baseline_min_isi_ms is not None
               and onsets[k - 1] - onsets[k - 2] >= local_baseline_min_isi_ms]

    def epsps_of(p) -> list[EpspParameters]:
        found = [shape.with_values(amplitude=p["a1"], latency=p["d1"])]
        for k in range(2, n + 1):
            found.append(shape.with_values(amplitude=p[f"a{k}"], latency=p[f"d{k}"], tau_rise=p[f"r{k}"],
                                           tau_decay=p[f"k{k}"]))
        return found

    def steps_of(p) -> list[float]:
        return [p.get(f"c{k}", 0.0) for k in range(1, n + 1)]

    def model(times, p):
        return (sum(e.waveform(times, onset) for e, onset in zip(epsps_of(p), onsets))
                + _baseline_steps(times, onsets, steps_of(p), local_baseline_window_ms))

    # Starting values, one EPSP after the other: EPSP1 scaled to the trace at the
    # template's peak time (if before the 2nd stimulus); every later EPSP = what
    # remains after its stimulus once the earlier guesses are subtracted.
    peak_time = template.measures.peak_time_ms
    template_peak_value = shape.waveform(np.array([peak_time]))[0]
    a1 = shape.amplitude
    if (n == 1 or peak_time < onsets[1] + shape.latency) and template_peak_value > 0:
        a1 *= max(float(np.interp(peak_time, t, y)), 0.05 * template_peak_value) / template_peak_value
    start = {"a1": a1, "d1": shape.latency}
    bounds = {"a1": (0.0, 200.0), "d1": (0.0, 20.0)}
    explained = shape.with_values(amplitude=a1).waveform(t)
    for k in range(2, n + 1):
        onset = onsets[k - 1]
        if k in stepped:          # start: the level just before the stimulus that EPSP1 ... do not explain
            before = (t >= onset - local_baseline_window_ms) & (t < onset)
            level = float(np.mean((y - explained)[before])) if before.any() else 0.0
            start[f"c{k}"] = level
            bounds[f"c{k}"] = (-50.0, 50.0)
            explained = explained + level * (t >= onset - local_baseline_window_ms)
        end = min(onset + 60.0, onsets[k] if k < n else np.inf)
        remaining = (y - explained)[(t >= onset) & (t <= end)]
        amplitude = max(float(remaining.max()) if remaining.size else 0.0, 0.05 * a1)
        start.update({f"a{k}": amplitude, f"d{k}": shape.latency, f"r{k}": shape.tau_rise, f"k{k}": shape.tau_decay})
        bounds.update({f"a{k}": (0.0, 200.0), f"d{k}": (0.0, 20.0), f"r{k}": (0.1, 30.0), f"k{k}": (1.0, 1000.0)})
        explained = explained + shape.with_values(amplitude=amplitude).waveform(t, onset)

    def varied(amplitude_scale=1.0, decay_scale=1.0):
        return {name: value * (amplitude_scale if name[0] == "a" and name != "a1" else 1.0)
                * (decay_scale if name[0] == "k" else 1.0) for name, value in start.items()}

    starts = [varied(), varied(decay_scale=0.5), varied(decay_scale=2.0), varied(amplitude_scale=0.5)]
    fitted, rms = fit_waveform(model, t, y, starts if n > 1 else starts[:1], bounds)
    epsps = epsps_of(fitted)
    residual = y - model(t, fitted)
    r_squared = 1.0 - float(np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2))
    return MultiEpspFit(
        isi_ms=isi_ms, onsets_ms=onsets, n_sweeps=n_sweeps, time_ms=t, trace=y, parameters=epsps,
        measures=[measure_epsp(e) for e in epsps],
        baselines_mv=list(baselines_mv) if baselines_mv is not None else [np.nan] * n,
        r_squared=r_squared, rms_mv=rms,
        baseline_steps_mv=steps_of(fitted), local_baseline_window_ms=local_baseline_window_ms,
    )


def fit_paired_epsps(template, time_ms, trace, interval_ms, isi_ms, n_sweeps, after_ms=300.0) -> MultiEpspFit:
    """Two-pulse shortcut of ``fit_epsp_train``."""
    return fit_epsp_train(template, time_ms, trace, [0.0, interval_ms], isi_ms, n_sweeps, after_ms)


def _absolute_baselines(raw: ExperimentTraces, onsets: list[float]) -> list[float]:
    """Absolute potential before every stimulus, from the raw (not baseline-subtracted) average."""
    averaged = raw.grouped(None)[0]
    t, v = averaged.time_ms, averaged.mean
    baselines = []
    for k, onset in enumerate(onsets):
        length = RESTING_BASELINE_MS if k == 0 else LATER_BASELINE_MS
        window = (t >= onset - length) & (t < onset)
        baselines.append(float(np.nanmean(v[window])) if window.any() else np.nan)
    return baselines


def analyse_summated_epsps(
    data: ExperimentTraces,
    group: str = "",
    threshold_isi_ms: float = 200.0,
    template_events: str = "both",
    template_length_ms: float | None = None,
    after_ms: float = 300.0,
    raw: ExperimentTraces | None = None,
    with_sag: bool = True,
    unit: str = "mV",
    local_baseline_min_isi_ms: float | None = None,
    local_baseline_window_ms: float = 30.0,
) -> SummatedEpspAnalysis:
    """
    Template + deconvolution of every ISI for traces of ONE group (all in mV,
    baseline-subtracted). ``raw``: the same traces WITHOUT baseline subtraction
    (same order), for the absolute baselines; without it they stay empty.
    ``local_baseline_min_isi_ms`` / ``local_baseline_window_ms``: local baseline of later
    EPSPs at long intervals (see ``fit_epsp_train``; None = off).
    Recordings with the same ISI but a different number of pulses are fitted separately.
    ``with_sag`` / ``unit``: False / "pA" for EPSCs (traces already made positive).
    """
    template = build_epsp_template(data, threshold_isi_ms, template_events, template_length_ms or after_ms,
                                   with_sag=with_sag)
    raw_of = {}
    if raw is not None and len(raw.trace_sets) == len(data.trace_sets):
        raw_of = {id(s): r for s, r in zip(data.trace_sets, raw.trace_sets)}
    warnings, fits = [], []
    for isi in data.values("isi"):
        by_pulses: dict[int, list] = {}
        for trace_set in data.subset(isi=[isi]).trace_sets:
            by_pulses.setdefault(len(trace_set.traces.stimulus_onsets_ms), []).append(trace_set)
        for n_pulses, members in sorted(by_pulses.items()):
            if n_pulses < 2:
                warnings.append(f"ISI {isi:g} ms: only one stimulus — skipped.")
                continue
            averaged = ExperimentTraces(members, data.criteria, data.metadata).grouped("isi")[0]
            onsets = list(np.mean([m.traces.stimulus_onsets_ms for m in members], axis=0))
            raw_members = [raw_of[id(m)] for m in members if id(m) in raw_of]
            baselines = _absolute_baselines(ExperimentTraces(raw_members, {}, {}), onsets) if raw_members else None
            try:
                fits.append(fit_epsp_train(template, averaged.time_ms, averaged.mean, onsets, float(isi),
                                           averaged.n_sweeps, after_ms, baselines,
                                           local_baseline_min_isi_ms, local_baseline_window_ms))
            except Exception as error:   # one bad ISI never stops the analysis
                warnings.append(f"ISI {isi:g} ms ({n_pulses} pulses): fit failed ({error}).")
    for fit in fits:
        if fit.r_squared < 0.8:
            warnings.append(f"ISI {fit.isi_ms:g} ms: poor fit (R² = {fit.r_squared:.2f}).")
    sample = {key: data.criteria.get(key) for key in ("expdate", "neuronid", "dendriteid", "spineid")}
    sample["spineid"] = sample["spineid"] if sample["spineid"] is not None else "all"
    sample["expname"] = data.metadata.get("expname", "")
    return SummatedEpspAnalysis(group=group, header=data.header_text(), template=template, fits=fits,
                                warnings=list(data.warnings) + warnings, sample=sample, unit=unit)


def deconvolve_summated_epsps(
    data: ExperimentTraces,
    threshold_isi_ms: float = 200.0,
    template_events: str = "both",
    template_length_ms: float | None = None,
    after_ms: float = 300.0,
) -> list[SummatedEpspAnalysis]:
    """
    Full pipeline for the traces of one experiment sample: keep current-clamp (mV)
    recordings, subtract each sweep's baseline, and analyse every group
    separately (e.g. pre-apv and post-apv get their own template, because a drug
    can change the EPSP shape). Any number of pulses per sweep is handled.
    Returns one ``SummatedEpspAnalysis`` per group.
    """
    current_clamp = ExperimentTraces([s for s in data.trace_sets if s.traces.unit.lower() == "mv"],
                                     data.criteria, data.metadata, data.warnings)
    if not current_clamp.trace_sets:
        raise ValueError(f"No current-clamp (mV) recordings among the selected traces (units: {data.units}).")
    analyses = []
    for group in current_clamp.values("group"):
        raw = current_clamp.subset(group=[group])
        analyses.append(analyse_summated_epsps(
            raw.baseline_subtracted(), "" if pd.isna(group) else str(group), threshold_isi_ms,
            template_events, template_length_ms, after_ms, raw=raw,
        ))
    return analyses
