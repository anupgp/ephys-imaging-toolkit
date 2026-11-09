"""
excluded_recordings_review.py
=============================

Figures to re-check the recordings that the sheet (the experiment sheet (CSV)) leaves
out of the analysis, with their EPSP fits:

* UNSELECTED files    (unselect = 1)                     -> whole file excluded
* files with BAD SWEEPS (selected, 'badsweeps' filled)    -> some sweeps excluded

The fit (``fit_recording``)
---------------------------
The excluded files are not in the batch results, so each file is fitted here with
the batch's own deconvolution (``summated_epsp_deconvolution``):
* TEMPLATE (the EPSP1 shape): built as in the batch from the recordings of the same
  input (spineid) and condition in the dendrite: its SELECTED recordings + this file
  (= what the batch would use if this file were selected again). When those hold no
  isolated EPSP (all ISIs < 200 ms), the 1st EPSPs of this file are used, each up to the
  2nd pulse (noted in the figure).
* 1st-RESPONSE FIT (green, ■): the 1st EPSP alone, every shape parameter free, from -10 ms
  to min(2nd pulse, 300 ms) - the fit the SPATIAL SUMMATION uses (measure_first_response);
  much less affected by drift later in the sweep than the whole-trace deconvolution.
* DECONVOLUTION FIT: the mean of THIS file's sweeps (the sheet's bad sweeps left out, as in the batch;
  for an unselected file normally all sweeps): EPSP1 = template shape (amplitude, latency
  free), EPSP2 free (amplitude, latency, rise, decay).

One figure per file
-------------------
    top        whole recording: every sweep (numbered from 1 as in 'badsweeps'; bad in the
               sheet = dashed grey), fitted EPSP1 / EPSP2 and their sum
    bottom     zoom on each pulse (-20 to +150 ms; one window when the pulses are closer):
               black = mean the fit was made on; blue / orange = fitted EPSP1 / EPSP2,
               faint = that EPSP alone (mean - the other fitted EPSP); ● fit peak, × trace
               peak (1 ms running mean); thick bar = 20-80 % rise (value written); dashed
               = decay curve C·exp(-t/τ) on the fitted EPSP's tail (τ written); purple = tangent
               at the max slope; grey dash-dot = mean of the SELECTED recordings of the same
               input and ISI (reference; absent when there is none)
    box        sheet entry, fit measures per EPSP (fit and trace peak, rise, τ, slope), PPR,
               R², template source, and the per-sweep measures of the PPR screen
Voltage-clamp files (pA) are shown and fitted inverted (inward EPSC up), without sag.

Nothing in the sheet is changed here.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .paired_pulse_screening import (AFTER_LAST_PULSE_MS, BASELINE_MS, EPSP_COLOURS, SUM_COLOUR, TANGENT_COLOUR,
                                     TRACE_PEAK_COLOUR, load_trace_set, measure_sweeps, trace_measures)

logger = logging.getLogger(__name__)

ZOOM_BEFORE_MS = 20.0
ZOOM_AFTER_MS = 150.0
THRESHOLD_ISI_MS = 200.0           # as the batch: isolated EPSPs = stimuli >= 200 ms apart
REFERENCE_STYLE = dict(color="0.35", linestyle="-.", linewidth=1.3)
FIRST_RESPONSE_COLOUR = "#009E73"   # 1st-response fit (the one used for spatial summation)


# =============================================================================
# 1. Which recordings
# =============================================================================
def _bad_sweeps(row) -> list[int]:
    """The sheet's bad sweeps of a catalog row (numbered from 1)."""
    text = str(row.get("badsweeps", "") or "")
    return sorted({int(float(x)) for x in text.replace(",", ";").split(";") if x.strip() not in ("", "nan", "<NA>")})


def _file_name(row) -> str:
    """'24822035.abf' from the catalog's file name (or the sheet's prefix + ephysid)."""
    name = row.get("ephys_file_name")
    if not isinstance(name, str) or not name:
        name = str(row["ephysfile_prefix"]).replace("xxx", "%03d" % int(row["ephysid"]))
    name = Path(name).name
    return name if name.lower().endswith(".abf") else name + ".abf"


def excluded_recordings(catalog, expname: str) -> pd.DataFrame:
    """
    One row per recording of ``expname`` that is unselected or has bad sweeps:
    catalog_row, category ('unselected' / 'bad sweeps'), file, set, spineid, isi_ms and the
    sheet entry (unselect, badsweeps, patchclamp, nstims, notes).
    """
    table = catalog.table
    table = table[table["expname"] == expname]
    rows = []
    for index, row in table.iterrows():
        unselected = str(row["unselect"]) == "1"
        bad = _bad_sweeps(row)
        if not unselected and not bad:
            continue
        rows.append({
            "catalog_row": index, "category": "unselected" if unselected else "bad sweeps", "file": _file_name(row),
            "set": f"{row['expdate']}_{int(row['neuronid'])}_{int(row['dendriteid'])}_{row['group']}",
            "expdate": str(row["expdate"]), "neuronid": int(row["neuronid"]), "dendriteid": int(row["dendriteid"]),
            "group": row["group"], "spineid": row["spineid"],
            "isi_ms": float(row["isi"]) if pd.notna(row["isi"]) else np.nan, "nstims": row["nstims"],
            "patchclamp": row["patchclamp"], "unselect": row["unselect"], "badsweeps": ";".join(map(str, bad)),
            "notes": row["notes"] if pd.notna(row["notes"]) else "",
        })
    return pd.DataFrame(rows).sort_values(["category", "set", "spineid", "isi_ms", "file"],
                                          ascending=[False, True, True, True, True]).reset_index(drop=True)


# =============================================================================
# 2. Data: the file, its fit, the reference mean
# =============================================================================
def _baseline(traces: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    mask = (time_ms >= -BASELINE_MS) & (time_ms < 0)
    return np.nanmean(np.asarray(traces)[:, mask], axis=1)


class SetCache:
    """Each set loaded once (all files, unselected included; the sheet's bad sweeps removed)."""

    def __init__(self, catalog):
        self.catalog, self.sets = catalog, {}

    def get(self, item: pd.Series):
        key = (item["expdate"], item["neuronid"], item["dendriteid"], item["group"])
        if key not in self.sets:
            data = self.catalog.experiment_traces(item["expdate"], item["neuronid"], item["dendriteid"],
                                                  group=item["group"], after_ms=AFTER_LAST_PULSE_MS,
                                                  remove_unselected=False, remove_bad_sweeps=True)
            data.criteria.update(expdate=item["expdate"], neuronid=item["neuronid"], dendriteid=item["dendriteid"])
            self.sets = {key: data}                      # keep only the current set in memory
        return self.sets[key]


def fit_recording(cache: SetCache, item: pd.Series):
    """
    (fit, first, polarity, note) for one excluded file (see module docstring):
    fit    = deconvolution of EPSP1 + EPSP2 (MultiEpspFit), as for the PPR
    first  = the 1st response alone, all shape parameters free, from -10 ms to min(ISI, 300 ms)
             (FirstResponse), as for the spatial summation
    Either is None (reason in the note) when it cannot be fitted.
    """
    from ..epsp_analysis.summated_epsp_deconvolution import (_absolute_baselines, build_epsp_template,
                                                             fit_epsp_train)
    from ..epsp_analysis.unitary_cluster_summation import _response_polarity, _sign_corrected, fit_first_response
    from ..experiment_catalog.experiment_traces import ExperimentTraces

    data = cache.get(item)
    this = [s for s in data.trace_sets if s.row.name == item["catalog_row"]]
    if not this:
        return None, None, 1, "no kept sweeps to fit (all bad in the sheet?)"
    this = this[0]
    unit = this.traces.unit
    same_input = [s for s in data.trace_sets
                  if s.traces.unit == unit and str(s.row["spineid"]) == str(item["spineid"])
                  and (str(s.row["unselect"]) != "1" or s.row.name == item["catalog_row"])]
    raw = ExperimentTraces(same_input, data.criteria, data.metadata, [])
    polarity = _response_polarity(raw) if unit.lower() == "pa" else 1
    raw = _sign_corrected(raw, polarity)
    corrected = raw.baseline_subtracted()
    position = [s.row.name for s in same_input].index(item["catalog_row"])
    with_sag = unit.lower() != "pa"
    n_other = len(same_input) - 1
    try:
        template = build_epsp_template(corrected, THRESHOLD_ISI_MS, "both", AFTER_LAST_PULSE_MS, with_sag=with_sag)
        note = f"template: this file + {n_other} selected recording(s) of this input"
    except Exception:                                   # no isolated EPSP: 1st EPSPs of this file
        own = ExperimentTraces([corrected.trace_sets[position]], data.criteria, data.metadata, [])
        onsets = own.trace_sets[0].traces.stimulus_onsets_ms
        gap = float(np.min(np.diff(onsets))) if len(onsets) > 1 else THRESHOLD_ISI_MS
        try:
            template = build_epsp_template(own, min(gap, THRESHOLD_ISI_MS), "first", AFTER_LAST_PULSE_MS,
                                           with_sag=with_sag)
            note = f"template: 1st EPSPs of this file only, cut at the 2nd pulse ({gap:.0f} ms)"
        except Exception as error:
            template, note = None, f"no template ({error})"
    file_set = ExperimentTraces([corrected.trace_sets[position]], data.criteria, data.metadata, [])
    averaged = file_set.grouped(None)[0]
    onsets = [float(o) for o in corrected.trace_sets[position].traces.stimulus_onsets_ms]
    baselines = _absolute_baselines(ExperimentTraces([raw.trace_sets[position]], {}, {}), onsets)
    isi = item["isi_ms"] if np.isfinite(item["isi_ms"]) else 0.0
    fit = None
    if template is not None:
        try:
            fit = fit_epsp_train(template, averaged.time_ms, averaged.mean, onsets, float(isi), averaged.n_sweeps,
                                 AFTER_LAST_PULSE_MS, baselines)
        except Exception as error:
            note += f"; deconvolution failed ({error})"
    end = min(onsets[1], AFTER_LAST_PULSE_MS) if len(onsets) > 1 else AFTER_LAST_PULSE_MS
    try:
        first = fit_first_response(averaged.time_ms, averaged.mean, float(isi), averaged.n_sweeps, end, with_sag)
    except Exception as error:
        first, note = None, note + f"; 1st-response fit failed ({error})"
    return fit, first, polarity, note


def reference_mean(catalog, item: pd.Series, polarity: int = 1):
    """
    (time, mean, n_files, n_sweeps) of the SELECTED recordings with the same expname,
    dendrite, condition, spineid, ISI and clamp as ``item`` (their bad sweeps removed),
    multiplied by ``polarity``; None if there are none.
    """
    table = catalog.table
    row = table.loc[item["catalog_row"]]
    same = table[(table["expname"] == row["expname"]) & (table["expdate"] == row["expdate"])
                 & (table["neuronid"] == row["neuronid"]) & (table["dendriteid"] == row["dendriteid"])
                 & (table["group"] == row["group"]) & (table["spineid"] == row["spineid"])
                 & (table["isi"] == row["isi"]) & (table["patchclamp"] == row["patchclamp"])
                 & (table["unselect"].astype(str) != "1")]
    same = same.drop(index=item["catalog_row"], errors="ignore")
    traces, time = [], None
    for index, other in same.iterrows():
        try:
            trace_set = load_trace_set(catalog, index, max(float(other["isi"]), 0.0))
        except Exception as error:                       # noqa: BLE001 - reference is optional
            logger.warning("reference %s not loaded (%s)", index, error)
            continue
        if trace_set is None:
            continue
        keep = [k for k, sweep in enumerate(trace_set.sweeps) if sweep + 1 not in _bad_sweeps(other)]
        if not keep or (time is not None and len(trace_set.time_ms) != len(time)):
            continue
        values = np.asarray(trace_set.traces)[keep]
        traces.append(polarity * (values - _baseline(values, trace_set.time_ms)[:, None]))
        time = trace_set.time_ms
    if not traces:
        return None
    stacked = np.vstack(traces)
    return time, np.nanmean(stacked, axis=0), len(traces), len(stacked)


# =============================================================================
# 3. Figure
# =============================================================================
def _draw_decay(axes, model, peak_t: float, peak: float, tau: float, stop: float, colour, style,
                label_offset=(4, 4)) -> None:
    """
    Decay curve C·exp(-t/τ) of a fitted EPSP. τ is the model's decay time constant (the slower
    of its two exponentials), which describes the TAIL, so the curve is anchored where the fitted
    EPSP has fallen to half its peak and drawn from halfway between the peak and that point.
    ``model``: function t -> the fitted EPSP (evaluated over the whole window, also beyond the
    range that was fitted).
    """
    if not (np.isfinite(tau) and tau > 0 and np.isfinite(peak) and peak > 0):
        return
    time_ms = np.linspace(peak_t, max(stop, peak_t + 1.0), 2000)
    curve = model(time_ms)
    after = curve <= 0.5 * peak
    anchor_t = float(time_ms[after][0]) if after.any() else min(stop, peak_t + tau)
    anchor_v = float(np.interp(anchor_t, time_ms, curve))
    decay_t = np.linspace(peak_t + 0.5 * (anchor_t - peak_t), stop, 200)
    axes.plot(decay_t, anchor_v * np.exp(-(decay_t - anchor_t) / tau), color=colour, linestyle=style,
              linewidth=1.5, zorder=6)
    axes.annotate(f"τ {tau:.0f} ms", (anchor_t, anchor_v), label_offset, textcoords="offset points", fontsize=7,
                  color=colour)


def _zoom_windows(onsets: list[float]) -> list[tuple[float, float]]:
    if len(onsets) >= 2 and onsets[1] - onsets[0] >= ZOOM_AFTER_MS:
        return [(o - ZOOM_BEFORE_MS, o + ZOOM_AFTER_MS) for o in onsets[:2]]
    return [(onsets[0] - ZOOM_BEFORE_MS, onsets[-1] + ZOOM_AFTER_MS)]


def _draw_fit_details(axes, fit, start: float, stop: float, trace_peaks) -> None:
    """Fitted EPSPs whose onset lies in [start, stop]: curve, EPSP alone, peaks, rise, decay, tangent."""
    for k in range(fit.n_epsps):
        onset, measures = fit.onsets_ms[k], fit.measures[k]
        if not start <= onset <= stop:
            continue
        colour = EPSP_COLOURS[k % len(EPSP_COLOURS)]
        alone = np.where(fit.time_ms >= onset - 5, fit.deconvolved(k), np.nan)
        axes.plot(fit.time_ms, alone, color=colour, linewidth=0.9, alpha=0.35, zorder=3)
        axes.plot(fit.time_ms, fit.curve(k), color=colour, linewidth=1.8, zorder=4, label=f"fit EPSP{k + 1}")
        peak_t = onset + measures.peak_time_ms
        axes.plot(peak_t, measures.peak_mv, "o", color=colour, markersize=6, markeredgecolor="white", zorder=7)
        value, time = trace_peaks[k]
        if np.isfinite(value):
            axes.plot(time, value, "x", color=TRACE_PEAK_COLOUR, markersize=7, markeredgewidth=2, zorder=8)
        if np.all(np.isfinite(measures.rise_points_ms)):                    # 20-80 % rise
            rise_t = np.array(measures.rise_points_ms) + onset
            axes.plot(rise_t, [0.2 * measures.peak_mv, 0.8 * measures.peak_mv], color=colour, linewidth=5,
                      alpha=0.5, solid_capstyle="butt", zorder=5)
            axes.annotate(f"rise {measures.rise_20_80_ms:.1f} ms", (rise_t[1], 0.8 * measures.peak_mv),
                          (-6, 4), textcoords="offset points", ha="right", fontsize=7, color=colour)
        _draw_decay(axes, lambda tt, p=fit.parameters[k], o=onset: p.waveform(tt, o), peak_t, measures.peak_mv,
                    measures.decay_tau_ms, stop, colour, (0, (4, 2)))
        slope = measures.max_slope_mv_per_ms                                # tangent at the max slope
        if np.isfinite(measures.max_slope_time_ms) and np.isfinite(slope) and slope > 0:
            t0 = onset + measures.max_slope_time_ms
            v0 = float(fit.parameters[k].waveform(np.array([t0]), onset)[0])
            half = 0.25 * abs(measures.peak_mv) / slope
            tangent_t = np.array([t0 - min(half, max(v0, 0.0) / slope), t0 + half])
            axes.plot(tangent_t, v0 + slope * (tangent_t - t0), color=TANGENT_COLOUR, linewidth=2, zorder=6)


def _draw_first_response(axes, first, stop: float) -> None:
    """The 1st-response fit (spatial summation): curve, peak, 20-80 % rise, decay curve with τ."""
    m = first.measures
    colour = FIRST_RESPONSE_COLOUR
    shown = first.time_ms <= stop
    axes.plot(first.time_ms[shown], first.fitted[shown], color=colour, linewidth=1.6, zorder=4,
              label="1st-response fit (spatial summation)")
    axes.plot(m.peak_time_ms, m.peak_mv, "s", color=colour, markersize=5.5, markeredgecolor="white", zorder=7)
    if np.all(np.isfinite(m.rise_points_ms)):
        axes.plot(m.rise_points_ms, [0.2 * m.peak_mv, 0.8 * m.peak_mv], color=colour, linewidth=5, alpha=0.45,
                  solid_capstyle="butt", zorder=5)
        axes.annotate(f"rise {m.rise_20_80_ms:.1f} ms", (m.rise_points_ms[1], 0.2 * m.peak_mv), (6, -2),
                      textcoords="offset points", fontsize=7, color=colour, va="top")
    _draw_decay(axes, first.parameters.waveform, m.peak_time_ms, m.peak_mv, m.decay_tau_ms, stop, colour,
                (0, (1, 1.5)), label_offset=(4, -11))


def review_figure(trace_set, measures: pd.DataFrame, item: pd.Series, fit=None, polarity: int = 1,
                  fit_note: str = "", reference=None, first=None):
    """One recording with its fit (see module docstring). ``measures``: measure_sweeps of the file."""
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    colours = colormaps["tab10"]
    unit = trace_set.unit
    label_unit = "-I (pA)" if polarity == -1 else unit
    t = trace_set.time_ms
    onsets = [float(o) for o in trace_set.stimulus_onsets_ms] or [0.0]
    bad = set(_bad_sweeps(item))
    traces = polarity * np.asarray(trace_set.traces)
    traces = traces - _baseline(traces, t)[:, None]
    windows = _zoom_windows(onsets)
    trace_peaks = trace_measures(fit) if fit is not None else []

    figure = Figure(figsize=(14.0, 8.8), layout="constrained")
    grid = figure.add_gridspec(2, len(windows) + 1, height_ratios=[0.85, 1.35],
                               width_ratios=[1.0] * len(windows) + [0.95])
    full = figure.add_subplot(grid[0, :])
    zooms = [figure.add_subplot(grid[1, k]) for k in range(len(windows))]
    text_axes = figure.add_subplot(grid[1, -1])
    text_axes.axis("off")

    sweep_lines = []
    for k, (sweep, y, m) in enumerate(zip(trace_set.sweeps, traces, measures.itertuples())):
        is_bad = sweep + 1 in bad
        style = dict(color="0.55", linestyle="--", linewidth=0.7) if is_bad else \
            dict(color=colours(k % 10), linewidth=0.7, alpha=0.6)
        full.plot(t, y, label=f"sweep {sweep + 1}" + (" [sheet: bad]" if is_bad else ""), **style)
        for one in zooms:
            one.plot(t, y, **{**style, "alpha": 0.35 if not is_bad else 0.6, "linewidth": 0.6})
        sweep_lines.append(f"{sweep + 1:>3}{'*' if is_bad else ' '} {m.peak_1:6.2f} {m.peak_2:6.2f} {m.ppr:5.2f} "
                           f"{m.max_above_baseline:6.1f} {m.max_slope:5.1f} {'AP?' if m.ap_suspect else ''}")
    if fit is not None:
        full.plot(fit.time_ms, fit.summed_curve, color=SUM_COLOUR, linestyle="--", linewidth=1.2, label="sum of fits")
        for k in range(fit.n_epsps):
            full.plot(fit.time_ms, fit.curve(k), color=EPSP_COLOURS[k % 2], linewidth=1.5, label=f"fit EPSP{k + 1}")
        for one, (start, stop) in zip(zooms, windows):
            one.plot(fit.time_ms, fit.trace, color="black", linewidth=1.4, zorder=3,
                     label=f"mean fitted ({fit.n_sweeps} sweeps)")
            one.plot(fit.time_ms, fit.summed_curve, color=SUM_COLOUR, linestyle="--", linewidth=1.1, zorder=4,
                     label="sum of fits")
            _draw_fit_details(one, fit, start, stop, trace_peaks)
    if first is not None:
        _draw_first_response(zooms[0], first, windows[0][1])
    if reference is not None:
        ref_t, ref_y, n_files, n_sweeps = reference
        label = f"selected recordings, same input & ISI ({n_files} file(s), {n_sweeps} sweeps)"
        for one in [full] + zooms:
            one.plot(ref_t, ref_y, zorder=2, label=label if one in (full, zooms[0]) else None, **REFERENCE_STYLE)
    for one in [full] + zooms:
        for onset in onsets:
            one.axvline(onset, color="0.6", linewidth=2.5, alpha=0.25, zorder=0)
        one.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        one.tick_params(labelsize=8)
    full.set_xlim(t[0], t[-1])
    full.set_ylabel(label_unit)
    full.legend(fontsize=7, frameon=False, loc="upper right", ncol=3)
    full.set_title("whole recording (baseline-subtracted)", fontsize=9)

    # zoom y range: non-spike sweeps, the fit and the reference (a spike would squash the EPSPs)
    normal = [k for k, m in enumerate(measures.itertuples()) if not m.ap_suspect] or list(range(len(measures)))
    for one, (start, stop) in zip(zooms, windows):
        mask = (t >= start) & (t <= stop)
        values = [traces[k][mask] for k in normal]
        if fit is not None:
            values.append(fit.summed_curve[(fit.time_ms >= start) & (fit.time_ms <= stop)])
        if first is not None and one is zooms[0]:
            values.append(first.fitted[(first.time_ms >= start) & (first.time_ms <= stop)])
        if reference is not None:
            values.append(reference[1][(reference[0] >= start) & (reference[0] <= stop)])
        values = np.concatenate([v for v in values if len(v)])
        low, high = np.nanpercentile(values, [0.5, 99.5])
        span = max(high - low, 0.5)
        one.set_xlim(start, stop)
        one.set_ylim(min(low, 0) - 0.15 * span, high + 0.3 * span)
        one.set_xlabel("time from 1st pulse (ms)", fontsize=8)
        one.set_ylabel(label_unit, fontsize=8)
        one.set_title(f"zoom {start:.0f} to {stop:.0f} ms" + (" (spike sweeps off scale)" if
                                                               measures["ap_suspect"].any() else ""), fontsize=9)
    zooms[0].legend(fontsize=6.5, frameon=False, loc="upper right")

    box = [f"category  {item['category']}",
           f"unselect  {item['unselect'] if pd.notna(item['unselect']) else ''}   badsweeps {item['badsweeps'] or '-'}",
           f"clamp     {item['patchclamp']}   nstims {item['nstims']}", "notes:"]
    notes = str(item["notes"]) or "-"
    box += ["  " + line for line in textwrap.wrap(notes, 48)][:4]
    box += ["", "DECONVOLUTION FIT (● fit peak, × trace peak)"]
    if fit is not None:
        for k, m in enumerate(fit.measures):
            value = trace_peaks[k][0] if k < len(trace_peaks) else np.nan
            box.append(f"EPSP{k + 1} pk {m.peak_mv:5.2f} (trace {value:5.2f})")
            box.append(f"      rise {m.rise_20_80_ms:4.1f} ms  τ {m.decay_tau_ms:4.0f} ms  slope {m.max_slope_mv_per_ms:4.2f}")
        if fit.n_epsps >= 2:
            trace_ppr = trace_peaks[1][0] / trace_peaks[0][0] if trace_peaks[0][0] > 0 else np.nan
            box.append(f"PPR fit {fit.paired_pulse_ratio:.2f}  trace {trace_ppr:.2f}   R² {fit.r_squared:.2f}")
        else:
            box.append(f"R² {fit.r_squared:.2f}")
    if fit is None:
        box.append("  no deconvolution fit")
    if first is not None:
        m = first.measures
        box += ["1st-RESPONSE FIT (■; spatial summation)",
                f"      pk {m.peak_mv:5.2f} (trace {first.raw_peak:5.2f})  R² {first.r_squared:.2f}",
                f"      rise {m.rise_20_80_ms:4.1f} ms  τ {m.decay_tau_ms:4.0f} ms  slope {m.max_slope_mv_per_ms:4.2f}"]
    box += ["  " + line for line in textwrap.wrap(fit_note or "-", 48)][:4]
    sign = " (on -I)" if polarity == -1 else ""
    box += ["", f"PER SWEEP{sign} (* = bad in sheet)", "swp   pk1    pk2   PPR    max  dV/dt", *sweep_lines]
    text_axes.text(0.0, 1.0, "\n".join(box), va="top", ha="left", family="monospace", fontsize=7.2,
                   transform=text_axes.transAxes)
    isi = f"ISI {item['isi_ms']:g} ms" if np.isfinite(item["isi_ms"]) and item["isi_ms"] > 0 else "single pulse"
    figure.suptitle(f"{item['file']} · {item['set']} · spine {item['spineid']} · {isi} · {item['category'].upper()}",
                    fontsize=11, x=0.01, ha="left")
    return figure


# =============================================================================
# 4. Run
# =============================================================================
def run_excluded_review(catalog, expname: str, output: str | Path, dpi: int = 130) -> pd.DataFrame:
    """
    Figures (with fits) of every unselected recording and every recording with bad sweeps:
        <output>/unselected/<set>_spine<id>_isi<ms>_<file>.png
        <output>/selected_with_bad_sweeps/...png
        <output>/<folder>_all.pdf          the same figures, one PDF per category
        <output>/excluded_recordings.csv   one row per file: sheet entry, fit measures, AP? sweeps,
                                           and empty 'verdict' / 'bad_sweeps_new' columns to fill in
        <output>/excluded_sweeps.csv       per-sweep measures
    """
    from matplotlib.backends.backend_pdf import PdfPages

    output = Path(output)
    folders = {"unselected": output / "unselected", "bad sweeps": output / "selected_with_bad_sweeps"}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()
    items = excluded_recordings(catalog, expname)
    new_columns = {"figure": "", "loaded": False, "sweeps_ap_suspect": "", "fit_note": "", "fit_r_squared": np.nan,
                   "fit_peak_1": np.nan, "fit_peak_2": np.nan, "fit_ppr": np.nan, "fit_rise_1": np.nan,
                   "fit_tau_1": np.nan, "fit_rise_2": np.nan, "fit_tau_2": np.nan, "first_peak": np.nan,
                   "first_trace_peak": np.nan, "first_rise": np.nan, "first_tau": np.nan, "first_r_squared": np.nan}
    for name, value in new_columns.items():
        items[name] = value
    cache = SetCache(catalog)
    all_sweeps = []
    pdfs = {name: PdfPages(output / f"{folder.name}_all.pdf") for name, folder in folders.items()}
    try:
        for index, item in items.iterrows():
            isi = item["isi_ms"] if np.isfinite(item["isi_ms"]) else 0.0
            try:
                trace_set = load_trace_set(catalog, item["catalog_row"], max(isi, 0.0))
            except Exception as error:                         # noqa: BLE001 - reported per file
                logger.warning("%s not loaded (%s)", item["file"], error)
                items.loc[index, "fit_note"] = f"NOT LOADED: {error}"
                continue
            if trace_set is None:
                continue
            try:
                fit, first, polarity, note = fit_recording(cache, item)
            except Exception as error:                         # noqa: BLE001 - figure without fit
                fit, first, polarity, note = None, None, -1 if trace_set.unit == "pA" else 1, \
                    f"fit not possible ({error})"
            for_measures = trace_set if polarity == 1 else type("Inverted", (), dict(
                time_ms=trace_set.time_ms, stimulus_onsets_ms=trace_set.stimulus_onsets_ms,
                sweeps=trace_set.sweeps, traces=-np.asarray(trace_set.traces), unit=trace_set.unit))
            measures = measure_sweeps(for_measures)
            if trace_set.unit != "mV":                       # the spike limits are in mV: not for EPSCs
                measures["ap_suspect"] = False
            all_sweeps.append(measures.assign(file=item["file"], set=item["set"], spineid=item["spineid"],
                                              isi_ms=item["isi_ms"], category=item["category"],
                                              bad_in_sheet=measures["sweep"].isin(_bad_sweeps(item))))
            figure = review_figure(trace_set, measures, item, fit, polarity, note,
                                   reference_mean(catalog, item, polarity), first)
            name = f"{item['set']}_spine{item['spineid']}_isi{isi:g}_{Path(str(item['file'])).stem}.png"
            path = folders[item["category"]] / name
            figure.savefig(path, dpi=dpi)
            pdfs[item["category"]].savefig(figure)
            items.loc[index, ["figure", "loaded", "fit_note"]] = [str(path.relative_to(output)), True, note]
            items.loc[index, "sweeps_ap_suspect"] = ";".join(map(str, measures.loc[measures["ap_suspect"], "sweep"]))
            if fit is not None:
                m = fit.measures
                items.loc[index, ["fit_r_squared", "fit_peak_1", "fit_rise_1", "fit_tau_1"]] = [
                    fit.r_squared, m[0].peak_mv, m[0].rise_20_80_ms, m[0].decay_tau_ms]
                if fit.n_epsps >= 2:
                    items.loc[index, ["fit_peak_2", "fit_ppr", "fit_rise_2", "fit_tau_2"]] = [
                        m[1].peak_mv, fit.paired_pulse_ratio, m[1].rise_20_80_ms, m[1].decay_tau_ms]
            if first is not None:
                items.loc[index, ["first_peak", "first_trace_peak", "first_rise", "first_tau", "first_r_squared"]] = [
                    first.measures.peak_mv, first.raw_peak, first.measures.rise_20_80_ms,
                    first.measures.decay_tau_ms, first.r_squared]
            logger.info("%s done", name)
    finally:
        for pdf in pdfs.values():
            pdf.close()
    # to fill in after checking: keep_excluded / reselect (unselect -> blank) / bad_sweeps (set
    # 'bad_sweeps_new', e.g. "1;3", numbered from 1). The sheet is changed only in a separate step.
    items["verdict"] = ""
    items["bad_sweeps_new"] = ""
    items.to_csv(output / "excluded_recordings.csv", index=False)
    if all_sweeps:
        pd.concat(all_sweeps, ignore_index=True).to_csv(output / "excluded_sweeps.csv", index=False)
    return items
