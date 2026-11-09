"""
paired_pulse_screening.py
=========================

Screen a batch results CSV for EPSPs with a high paired-pulse ratio (PPR =
2nd / 1st EPSP peak, from the deconvolution), so the recordings can be checked
by eye and, after checking, excluded in the sheet.

Only EPSPs (current clamp, mV) are screened: voltage-clamp EPSC (pA) rows of the
CSV are ignored.

A high PPR has three usual causes, and the screen helps to tell them apart:
* an ACTION POTENTIAL (or dendritic spike) in a sweep: a large, fast event ->
  exclude the sweep(s) ('badsweeps') or the file ('unselect');
* a TINY 1st EPSP (near noise level): the ratio of two noisy numbers is
  unreliable even when both EPSPs are normal;
* a drifting baseline (slow fluctuations of 1-2 mV under ~1 mV EPSPs).

Steps
-----
1. ``flag_high_ppr(results, catalog, threshold)``: one row per FILE behind every EPSP
   response with PPR > threshold (a response pooled from several files gives one row
   per file), with its sheet row, the current 'unselect' / 'badsweeps' and a
   ``likely_cause``.
2. ``measure_sweeps(trace_set)``: per SWEEP of one file (numbered from 1, as in
   'badsweeps'): baseline, 1st and 2nd peak (largest 1 ms running mean within
   30 ms of each pulse, minus the 10 ms baseline before the 1st pulse), their
   ratio, max above baseline, max dV/dt, and ``ap_suspect`` (see AP_* limits).
3. ``deconvolution_fits(catalog, flagged)``: every set with a flagged file is
   analysed again exactly as in the batch run (``analyse_set``; unselected rows and
   the sheet's bad sweeps removed), and the deconvolution fit of each flagged
   input x ISI is kept: the fit behind the numbers in the CSV.
4. ``screening_pages(...)``: one panel per file (see ``draw_file_panel``).
5. The user fills in ``verdict`` (keep / exclude_file / bad_sweeps) and
   ``bad_sweeps_to_add`` in the CSV written by ``run_ppr_screening``; only then is
   the sheet edited (by a separate, explicit step).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PPR_THRESHOLD = 1.2
SMALL_FIRST_RESPONSE = 0.3       # mV: 1st EPSP this small -> ratio unreliable
PEAK_WINDOW_MS = 30.0            # per-sweep peaks: within 30 ms after each pulse (as in the analysis)
SMOOTH_MS = 1.0                  # running mean for the per-sweep peaks
BASELINE_MS = 10.0               # baseline: mean of the 10 ms before the 1st pulse
AP_AMPLITUDE = 20.0              # mV above baseline ...
AP_SLOPE = 20.0                  # ... or max dV/dt in mV/ms -> 'ap_suspect' (EPSPs: < ~2 mV/ms)
SLOPE_SMOOTH_MS = 0.25           # dV/dt of a 0.25 ms running mean (raw sample-to-sample dV/dt is noise)
AFTER_LAST_PULSE_MS = 300.0      # shown / analysed after the last pulse (as the batch run's after_ms)
PANELS_PER_PAGE = (2, 3)         # rows x columns
EPSP_COLOURS = ["#0067A5", "#F38400"]   # fitted EPSP1, EPSP2 (as in the deconvolution figures)
SUM_COLOUR = "#BE0032"                  # sum of the fitted EPSPs
TRACE_PEAK_COLOUR = "#00A0B0"           # × = peak of the trace (as in the summation figures)
TANGENT_COLOUR = "#6A3D9A"              # tangent at the max slope of each fitted EPSP


# =============================================================================
# 1. Which files
# =============================================================================
def flag_high_ppr(results: pd.DataFrame, catalog, threshold: float = PPR_THRESHOLD) -> pd.DataFrame:
    """One row per file of every EPSP response with PPR > ``threshold`` (see module docstring)."""
    if "unit" in results:
        results = results[results["unit"] == "mV"]          # EPSPs only
    first = results[results["stimid"] == 1]
    high = first[first["paired_pulse_ratio"] > threshold]
    second = results[results["stimid"] == 2].set_index(["set", "input", "isi_ms"])
    rows = []
    for _, response in high.iterrows():
        try:
            peak_2 = second.loc[(response["set"], response["input"], response["isi_ms"]), "peak"]
        except KeyError:
            peak_2 = np.nan
        files = [f.strip() for f in str(response["files"]).split(";") if f.strip()]
        for file_name in files:
            sheet = catalog.rows_for_file(file_name)
            sheet = sheet[sheet["expname"] == response["expname"]] if "expname" in sheet else sheet
            sheet_row = sheet.iloc[0] if len(sheet) else None
            rows.append({
                "set": response["set"], "expdate": str(response["expdate"]), "neuronid": int(response["neuronid"]),
                "dendriteid": int(response["dendriteid"]), "group": response["group"],
                "input": response["input"], "kind": response["kind"], "isi_ms": response["isi_ms"],
                "file": file_name, "files_pooled": len(files),
                "catalog_row": int(sheet_row.name) if sheet_row is not None else None,
                "spineid": sheet_row["spineid"] if sheet_row is not None else None,
                "ppr": response["paired_pulse_ratio"], "peak_1": response["peak"], "peak_2": peak_2,
                "n_sweeps_analysed": response["n_sweeps"],
                "sheet_unselect": sheet_row["unselect"] if sheet_row is not None else None,
                "sheet_badsweeps": sheet_row["badsweeps"] if sheet_row is not None else None,
                "likely_cause": _likely_cause(response["peak"], peak_2),
            })
    return pd.DataFrame(rows)


def _likely_cause(peak_1: float, peak_2: float) -> str:
    if not np.isfinite(peak_1) or peak_1 <= SMALL_FIRST_RESPONSE:
        return f"tiny 1st EPSP (<= {SMALL_FIRST_RESPONSE:g} mV): ratio unreliable"
    if np.isfinite(peak_2) and peak_2 >= AP_AMPLITUDE:
        return "large 2nd response: possible action potential"
    return "check traces"


# =============================================================================
# 2. Per-sweep measures
# =============================================================================
def _running_mean(values: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 1:
        return values
    return np.convolve(values, np.ones(samples) / samples, mode="same")


def measure_sweeps(trace_set) -> pd.DataFrame:
    """Per sweep of one file (an ``IsiTraceSet``, mV); see module docstring."""
    t = trace_set.time_ms
    dt = float(np.median(np.diff(t)))
    onsets = list(trace_set.stimulus_onsets_ms) or [0.0]
    baseline_mask = (t >= -BASELINE_MS) & (t < 0)
    rows = []
    for sweep, trace in zip(trace_set.sweeps, trace_set.traces):
        baseline = float(np.nanmean(trace[baseline_mask])) if baseline_mask.any() else 0.0
        smooth = _running_mean(trace - baseline, max(1, int(round(SMOOTH_MS / dt))))
        peaks = []
        for onset in onsets[:2]:
            window = (t >= onset) & (t <= onset + PEAK_WINDOW_MS)
            peaks.append(float(np.nanmax(smooth[window])) if window.any() else np.nan)
        peaks += [np.nan] * (2 - len(peaks))
        after = t >= 0
        max_v = float(np.nanmax(trace[after] - baseline))
        slope_trace = _running_mean(trace, max(1, int(round(SLOPE_SMOOTH_MS / dt))))[after]
        max_slope = float(np.nanmax(np.diff(slope_trace)[2:-2] / dt))   # ends: edge effect of the mean
        rows.append({"sweep": sweep + 1, "baseline": baseline, "peak_1": peaks[0], "peak_2": peaks[1],
                     "ppr": peaks[1] / peaks[0] if peaks[0] > 0 else np.nan, "max_above_baseline": max_v,
                     "max_slope": max_slope,
                     "ap_suspect": bool(max_v >= AP_AMPLITUDE or max_slope >= AP_SLOPE)})
    return pd.DataFrame(rows)


def load_trace_set(catalog, catalog_row: int, isi_ms: float):
    """All sweeps of one file (the sheet's bad sweeps included), aligned to the 1st pulse."""
    from ..experiment_catalog.experiment_traces import collect_experiment_traces

    rows = catalog.table.loc[[catalog_row]]
    traces = collect_experiment_traces(rows, lambda r: catalog.load_ephys(r, apply_bad_sweeps=False),
                                       time_window_ms=(-50.0, float(isi_ms) + AFTER_LAST_PULSE_MS))
    return traces.trace_sets[0].traces if traces.trace_sets else None


# =============================================================================
# 3. The deconvolution fits behind the CSV
# =============================================================================
def deconvolution_fits(catalog, flagged: pd.DataFrame) -> dict:
    """
    (set, input, isi_ms) -> MultiEpspFit of the 2-pulse recording(s), from re-running the
    batch analysis of every set with a flagged file (same defaults: unselected rows and
    the sheet's bad sweeps removed, after_ms 300). Sets that fail are logged and skipped.
    """
    from ..epsp_analysis.unitary_cluster_summation import analyse_set

    fits = {}
    for set_name, rows in flagged.groupby("set", sort=False):
        first = rows.iloc[0]
        try:
            data = catalog.experiment_traces(first["expdate"], int(first["neuronid"]), int(first["dendriteid"]),
                                             group=first["group"], after_ms=AFTER_LAST_PULSE_MS,
                                             remove_unselected=True, remove_bad_sweeps=True)
            data.criteria.update(expdate=first["expdate"], neuronid=int(first["neuronid"]),
                                 dendriteid=int(first["dendriteid"]))
            analyses = [a for a in analyse_set(data) if a.unit == "mV"]
        except Exception as error:
            logger.warning("%s: deconvolution not repeated (%s)", set_name, error)
            continue
        for analysis in analyses:
            for label, deconvolution in analysis.deconvolutions.items():
                for fit in deconvolution.fits:
                    if fit.n_epsps == 2:
                        fits[(set_name, label, float(fit.isi_ms))] = fit
        logger.info("%s: deconvolution repeated", set_name)
    return fits


def trace_measures(fit) -> list[tuple[float, float]]:
    """
    (peak, its time from the 1st pulse) of the TRACE for each EPSP of a fit: largest
    1 ms running mean of that EPSP alone (recorded mean - the other fitted EPSPs),
    from its onset (fit latency) to onset + 30 ms - the same definition as the
    'raw peak' of the summation analysis.
    """
    from ..epsp_analysis.unitary_cluster_summation import _raw_peak

    values = []
    for k, (onset, measures) in enumerate(zip(fit.onsets_ms, fit.measures)):
        value, time = _raw_peak(fit.time_ms - onset, fit.deconvolved(k), measures.latency_ms)
        values.append((value, time + onset))
    return values


# =============================================================================
# 4. Figures
# =============================================================================
def draw_file_panel(axes, trace_set, flagged: pd.Series, sweep_measures: pd.DataFrame, sheet_bad: set[int],
                    fit=None) -> None:
    """
    One file:
      thin coloured lines  every sweep, numbered from 1 (legend: its own PPR, max above
                           baseline, max dV/dt, AP?); sheet-bad sweeps dashed grey
      black                averaged trace the fit was made on (the analysed sweeps of
                           all files of this input and ISI)
      blue / orange        fitted EPSP1 / EPSP2; faint: each EPSP alone (trace - other fit)
      red dashed           sum of the fitted EPSPs
      ● fit peak, × trace peak (1 ms running mean), thick bar = 20-80 % rise,
      short purple line = tangent at the max slope of the fit
      box                  per EPSP: fit peak, trace peak, rise, decay τ, max slope; PPR
                           (fit and trace); R²
    Long gaps between the pulses are cut out (//) so both EPSPs stay readable. The
    y axis fits the EPSPs (spike sweeps ignored); spike sweeps going off scale are shown
    in an inset at full scale.
    """
    from matplotlib import colormaps

    from ..interactive_viewers.epsp_deconvolution_figure import _time_axis

    colours = colormaps["tab10"]
    t = trace_set.time_ms
    if fit is not None:
        to_x, ticks, labels, gaps = _time_axis(fit)
    else:                                       # no fit: plain time axis
        to_x, ticks, labels, gaps = (lambda v: np.asarray(v, dtype=float)), None, None, []
    x = to_x(t)

    # sweeps
    sweeps = []
    for k, (trace, measures) in enumerate(zip(trace_set.traces, sweep_measures.itertuples())):
        trace = trace - measures.baseline
        sweeps.append(trace)
        flag = " AP?" if measures.ap_suspect else ""
        label = (f"sweep {measures.sweep}: PPR {measures.ppr:.2f}, max {measures.max_above_baseline:.1f}, "
                 f"dV/dt {measures.max_slope:.1f}{flag}")
        if measures.sweep in sheet_bad:
            axes.plot(x, trace, color="0.6", linestyle="--", linewidth=0.6, label=label + " [sheet: bad]")
        else:
            axes.plot(x, trace, color=colours(k % 10), linewidth=0.6, alpha=0.7, label=label)
    for onset in trace_set.stimulus_onsets_ms:
        axes.axvline(float(to_x(onset)), color="0.6", linewidth=2.5, alpha=0.3, zorder=0)
    axes.axhline(0, color="0.85", linewidth=0.8, zorder=0)

    box = []
    if fit is not None:
        fx = to_x(fit.time_ms)
        axes.plot(fx, fit.trace, color="black", linewidth=1.3, label=f"mean, analysed ({fit.n_sweeps} sweeps)")
        trace_peaks = trace_measures(fit)
        for k in range(fit.n_epsps):
            colour = EPSP_COLOURS[k % len(EPSP_COLOURS)]
            onset, measures = fit.onsets_ms[k], fit.measures[k]
            alone = np.where(fit.time_ms >= onset - 5, fit.deconvolved(k), np.nan)
            axes.plot(fx, alone, color=colour, linewidth=0.8, alpha=0.35)
            axes.plot(fx, fit.curve(k), color=colour, linewidth=1.5, label=f"fit EPSP{k + 1}")
            axes.plot(to_x(onset + measures.peak_time_ms), measures.peak_mv, "o", color=colour, markersize=5,
                      markeredgecolor="white", zorder=6)
            value, time = trace_peaks[k]
            if np.isfinite(value):
                axes.plot(to_x(time), value, "x", color=TRACE_PEAK_COLOUR, markersize=6, markeredgewidth=1.8,
                          zorder=7)
            if np.all(np.isfinite(measures.rise_points_ms)):
                rise_t = np.array(measures.rise_points_ms) + onset
                axes.plot(to_x(rise_t), [0.2 * measures.peak_mv, 0.8 * measures.peak_mv], color=colour,
                          linewidth=4, alpha=0.5, solid_capstyle="butt", zorder=5)
            slope = measures.max_slope_mv_per_ms
            if np.isfinite(measures.max_slope_time_ms) and np.isfinite(slope) and slope > 0:
                t0 = onset + measures.max_slope_time_ms
                v0 = float(fit.parameters[k].waveform(np.array([t0]), onset)[0])
                half = 0.25 * abs(measures.peak_mv) / slope     # tangent spans ±25 % of the peak in y,
                before = min(half, max(v0, 0.0) / slope)        # but never below the baseline
                tangent_t = np.array([t0 - before, t0 + half])
                axes.plot(to_x(tangent_t), v0 + slope * (tangent_t - t0), color=TANGENT_COLOUR,
                          linewidth=2.0, zorder=6)
            box.append(f"EPSP{k + 1} fit {measures.peak_mv:5.2f} · trace {value:5.2f} mV · rise "
                       f"{measures.rise_20_80_ms:4.1f} · τ {measures.decay_tau_ms:4.0f} ms · "
                       f"slope {measures.max_slope_mv_per_ms:4.2f}")
        axes.plot(fx, fit.summed_curve, color=SUM_COLOUR, linewidth=1.1, linestyle="--", label="sum of fits")
        trace_ppr = trace_peaks[1][0] / trace_peaks[0][0] if trace_peaks[0][0] > 0 else np.nan
        box.append(f"PPR fit {fit.paired_pulse_ratio:.3g} · trace {trace_ppr:.3g} · R² {fit.r_squared:.2f}")
    else:
        box.append("no deconvolution fit for this file")
    axes.text(0.99, 0.99, "\n".join(box), transform=axes.transAxes, ha="right", va="top", fontsize=5.8,
              family="monospace", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9),
              zorder=10)

    # y range: the FITTED EPSPs (a spike in the mean would squash them) plus the sweeps' median
    # peaks; anything beyond it (spikes) is shown in an inset at full scale
    reference = fit.summed_curve if fit is not None else np.nanmedian(np.array(sweeps), axis=0)
    shown = np.isfinite(to_x(fit.time_ms if fit is not None else t))
    top = float(np.nanmax(reference[shown])) if shown.any() else 1.0
    no_spike = sweep_measures[~sweep_measures["ap_suspect"]]
    if len(no_spike):
        top = max(top, float(np.nanmedian(no_spike[["peak_1", "peak_2"]].max(axis=1))))
    low = float(np.nanmin(reference[shown])) if shown.any() else -1.0
    span = max(top - min(low, 0.0), 0.5)
    upper = top + 1.1 * span                                    # room for legend and box above the EPSPs
    axes.set_ylim(min(low, 0.0) - 0.45 * span, upper)
    off_scale = [m.sweep for m in sweep_measures.itertuples() if m.ap_suspect and m.max_above_baseline > upper]
    if off_scale:
        inset = axes.inset_axes([0.62, 0.06, 0.36, 0.34])
        for trace, measures in zip(sweeps, sweep_measures.itertuples()):
            inset.plot(x, trace, color="0.6" if measures.sweep in sheet_bad else "0.25", linewidth=0.5,
                       linestyle="--" if measures.sweep in sheet_bad else "-")
        inset.set_title("full scale: off-scale sweep(s) " + ", ".join(map(str, off_scale)), fontsize=5.5)
        inset.tick_params(labelsize=5, length=2)
        inset.set_xticks([])

    if ticks is not None:
        axes.set_xticks(ticks, labels)
    for gap in gaps:
        axes.axvline(gap, color="0.3", linewidth=0.8, linestyle=(0, (2, 2)))
        axes.text(gap, 0.0, "//", transform=axes.get_xaxis_transform(), ha="center", va="center", fontsize=8,
                  backgroundcolor="white")
    axes.set_xlabel("time from 1st pulse (ms)", fontsize=7)
    axes.set_ylabel("mV", fontsize=7)
    axes.tick_params(labelsize=7)
    pooled = f" · fit on {int(flagged['files_pooled'])} files" if flagged.get("files_pooled", 1) > 1 else ""
    axes.set_title(f"{flagged['file']} · {flagged['set']} · {flagged['input']} · ISI {flagged['isi_ms']:g} ms"
                   f"{pooled}\nCSV: PPR {flagged['ppr']:.3g} (fit peaks {flagged['peak_1']:.2f} -> "
                   f"{flagged['peak_2']:.2f} mV) · {flagged['likely_cause']}", fontsize=7)
    axes.legend(fontsize=5.2, frameon=False, loc="upper left")


def load_and_measure(catalog, flagged: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Trace set of every flagged file (row index -> IsiTraceSet or None) and all per-sweep measures."""
    trace_sets, all_sweeps = {}, []
    for index, item in flagged.iterrows():
        trace_sets[index] = None
        if item["catalog_row"] is None or pd.isna(item["catalog_row"]):
            continue
        try:
            trace_sets[index] = load_trace_set(catalog, int(item["catalog_row"]), item["isi_ms"])
        except Exception as error:          # missing file etc.: shown as "could not load"
            flagged.loc[index, "likely_cause"] = f"could not load: {error}"
        if trace_sets[index] is not None:
            all_sweeps.append(measure_sweeps(trace_sets[index]).assign(
                file=item["file"], set=item["set"], input=item["input"], isi_ms=item["isi_ms"]))
    sweeps = pd.concat(all_sweeps, ignore_index=True) if all_sweeps else pd.DataFrame()
    return trace_sets, sweeps


def screening_pages(flagged: pd.DataFrame, trace_sets: dict, sweeps: pd.DataFrame, fits: dict | None = None,
                    title: str = "") -> list:
    """Figure pages (``PANELS_PER_PAGE`` files each), in the order of ``flagged``."""
    from matplotlib.figure import Figure

    from ..experiment_catalog.catalog_file_names import parse_bad_sweeps

    rows, columns = PANELS_PER_PAGE
    per_page = rows * columns
    pages = []
    n_pages = math.ceil(len(flagged) / per_page)
    for page in range(n_pages):
        chunk = flagged.iloc[page * per_page:(page + 1) * per_page]
        figure = Figure(figsize=(6.6 * columns, 4.9 * rows), layout="constrained")
        axes = figure.subplots(rows, columns, squeeze=False).ravel()
        for one, (index, item) in zip(axes, chunk.iterrows()):
            trace_set = trace_sets.get(index)
            if trace_set is None:
                one.text(0.5, 0.5, f"{item['file']}: could not load", ha="center", transform=one.transAxes)
                continue
            measures = sweeps[(sweeps["file"] == item["file"]) & (sweeps["isi_ms"] == item["isi_ms"])]
            fit = (fits or {}).get((item["set"], item["input"], float(item["isi_ms"])))
            draw_file_panel(one, trace_set, item, measures, set(parse_bad_sweeps(item["sheet_badsweeps"])), fit)
        for one in axes[len(chunk):]:
            one.set_visible(False)
        figure.suptitle(f"{title} · page {page + 1}/{n_pages}   (● fit peak, × trace peak, bar = 20-80 % rise, "
                        f"purple line = max slope of the fit)", fontsize=10, x=0.01, ha="left")
        pages.append(figure)
    return pages


# order of the causes in the outputs (spikes first)
CAUSE_ORDER = ["action potential", "large 2nd response", "check traces", "tiny 1st EPSP", "could not load"]


# =============================================================================
# 5. Everything into one folder
# =============================================================================
def run_ppr_screening(catalog, results_csv: str | Path, output: str | Path,
                      threshold: float = PPR_THRESHOLD, with_fits: bool = True) -> pd.DataFrame:
    """
    Writes into ``output``:
        ppr_above_<threshold>_files.csv     one row per file + empty 'verdict' / 'bad_sweeps_to_add'
                                            columns to fill in after checking
        ppr_above_<threshold>_sweeps.csv    per-sweep measures of those files
        ppr_above_<threshold>_traces.pdf    all pages; also page_NN.png
    ``with_fits``: repeat the deconvolution of the flagged sets to draw the fits (slow:
    ~30 s per set); False = traces only.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob(f"ppr_above_{threshold:g}_traces_page_*.png"):
        old.unlink()
    results = pd.read_csv(results_csv)
    flagged = flag_high_ppr(results, catalog, threshold).reset_index(drop=True)
    trace_sets, sweeps = load_and_measure(catalog, flagged)
    fits = deconvolution_fits(catalog, flagged) if with_fits else {}
    # per file: which sweeps look like a spike (numbered from 1), the largest deflection, the fit
    flagged["sweeps_ap_suspect"] = ""
    flagged["max_above_baseline"] = np.nan
    flagged["fit_found"] = False
    for index, item in flagged.iterrows():
        flagged.loc[index, "fit_found"] = (item["set"], item["input"], float(item["isi_ms"])) in fits
        mine = sweeps[(sweeps["file"] == item["file"]) & (sweeps["isi_ms"] == item["isi_ms"])] if len(sweeps) else sweeps
        if len(mine):
            suspect = mine.loc[mine["ap_suspect"], "sweep"].astype(int).tolist()
            flagged.loc[index, "sweeps_ap_suspect"] = ";".join(map(str, suspect))
            flagged.loc[index, "max_above_baseline"] = mine["max_above_baseline"].max()
            if suspect:
                flagged.loc[index, "likely_cause"] = (f"action potential / fast event in sweep(s) "
                                                      f"{', '.join(map(str, suspect))}")
    rank = flagged["likely_cause"].map(lambda c: next((i for i, p in enumerate(CAUSE_ORDER) if c.startswith(p)),
                                                      len(CAUSE_ORDER)))
    order = flagged.assign(_rank=rank).sort_values(["_rank", "set", "input", "isi_ms"]).index
    flagged = flagged.loc[order]
    pages = screening_pages(flagged, trace_sets, sweeps, fits, f"EPSPs with PPR > {threshold:g}: {len(flagged)} files")
    flagged = flagged.reset_index(drop=True)
    flagged["verdict"] = ""                # keep / exclude_file / bad_sweeps (filled in by the user)
    flagged["bad_sweeps_to_add"] = ""      # e.g. "2;3" (numbered from 1, as in the sheet)
    stem = f"ppr_above_{threshold:g}"
    flagged.to_csv(output / f"{stem}_files.csv", index=False)
    sweeps.to_csv(output / f"{stem}_sweeps.csv", index=False)
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output / f"{stem}_traces.pdf") as pdf:
        for number, figure in enumerate(pages, start=1):
            pdf.savefig(figure)
            figure.savefig(output / f"{stem}_traces_page_{number:02d}.png", dpi=130)
    return flagged
