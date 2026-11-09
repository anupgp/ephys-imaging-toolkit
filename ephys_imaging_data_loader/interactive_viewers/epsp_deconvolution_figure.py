"""
epsp_deconvolution_figure.py
============================

Figure of a summated-EPSP deconvolution (``epsp_analysis.summated_epsp_deconvolution``),
one figure per group; paired pulses and trains of any length. Layout:

Top rows - template and summary
    [ template: isolated EPSPs (grey), their mean (black), model fit (red) ]
    [ peak ] [ ratio to EPSP1 ] [ 20-80 % rise ] [ decay tau ]
      - paired pulses: plotted against the ISI, one line per EPSP;
      - trains (> 2 pulses): plotted against the EPSP number, one line per ISI.

Bottom rows - one panel per ISI
    black          recorded trace (average of the sweeps of that ISI)
    red dashed     sum of all fitted EPSPs (should overlap the black trace)
    coloured       fitted EPSP1, EPSP2, ... (one colour each)
    light coloured deconvolved EPSPk = recorded - all other fitted EPSPs (model-free)
    dots           peaks; thick segments = 20-80 % rise of each EPSP
    text           per EPSP: absolute baseline, peak, 20-80 % rise, decay tau; R²
    Long gaps between stimuli (> 300 ms) are cut out of the time axis (marked //),
    so every EPSP stays readable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..epsp_analysis.summated_epsp_deconvolution import MultiEpspFit, SummatedEpspAnalysis

COLOURS = {"trace": "black", "sum": "#BE0032", "template_events": "0.75"}
# One colour per EPSP (no red: red is the sum of the fits)
EPSP_COLOURS = ["#0067A5", "#F38400", "#008856", "#875692", "#882D17", "#56B4E9", "#E68FAC", "#8DB600", "#222222"]
BREAK_ABOVE_MS = 300.0     # gaps between stimuli longer than this are cut out of the time axis
FIRST_SEGMENT_MS = 200.0   # shown after a stimulus before a cut
SECOND_SEGMENT_MS = 250.0  # shown after the last stimulus
VISUAL_GAP_MS = 20.0       # width left for a cut


def _epsp_colour(index: int) -> str:
    return EPSP_COLOURS[index % len(EPSP_COLOURS)]


def _time_axis(fit: MultiEpspFit):
    """
    x positions for the trace times. Every gap between consecutive stimuli longer
    than BREAK_ABOVE_MS is cut out (from FIRST_SEGMENT_MS after the stimulus to
    20 ms before the next one). Returns (function t -> x, NaN inside cuts and after
    the shown end; tick positions; tick labels; x positions of the cuts).
    """
    onsets = list(fit.onsets_ms)
    end = onsets[-1] + SECOND_SEGMENT_MS
    cuts = [(a + FIRST_SEGMENT_MS, b - 20.0) for a, b in zip(onsets[:-1], onsets[1:]) if b - a > BREAK_ABOVE_MS]

    def to_x(t):
        t = np.asarray(t, dtype=float)
        x = t.copy()
        for cut_start, cut_end in cuts:
            x = np.where(t >= cut_end, x - (cut_end - cut_start - VISUAL_GAP_MS), x)
            x = np.where((t > cut_start) & (t < cut_end), np.nan, x)
        return np.where(t <= end, x, np.nan)

    if not cuts:
        ticks = list(np.arange(0, end + 1, 100 if end > 300 else 50))
        return to_x, ticks, [f"{t:g}" for t in ticks], []
    # ticks at every stimulus after a cut (+100, +200 ms), labelled with the nominal (sheet ISI) times
    positions, labels = [0.0, 100.0], ["0", "100"]
    for k, onset in enumerate(onsets[1:], start=1):
        if onset - onsets[k - 1] > BREAK_ABOVE_MS:
            offsets = (0.0, 100.0, 200.0) if k == len(onsets) - 1 else (0.0, 100.0)
            positions += [float(to_x(onset + o)) for o in offsets]
            labels += [f"{fit.isi_ms * k + o:g}" for o in offsets]
    return to_x, positions, labels, [float(to_x(cut_start)) + VISUAL_GAP_MS / 2 for cut_start, _ in cuts]


def _draw_isi_panel(axes, fit: MultiEpspFit, unit: str = "mV", name: str = "EPSP") -> None:
    """``unit`` / ``name``: "pA" / "EPSC" for voltage-clamp recordings (labels only)."""
    to_x, ticks, labels, gaps = _time_axis(fit)
    x = to_x(fit.time_ms)
    axes.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    for onset in fit.onsets_ms:
        axes.axvline(float(to_x(onset)), color="0.6", linewidth=2.5, alpha=0.35, zorder=0)
    for k in range(1, fit.n_epsps):          # model-free single EPSPs (from their stimulus on)
        deconvolved = np.where(fit.time_ms >= fit.onsets_ms[k] - 5, fit.deconvolved(k), np.nan)
        axes.plot(x, deconvolved, color=_epsp_colour(k), linewidth=1.0, alpha=0.4,
                  label=f"deconvolved {name} (recorded − other fits)" if k == 1 else None)
    axes.plot(x, fit.trace, color=COLOURS["trace"], linewidth=1.2, label="recorded (mean)")
    for k in range(fit.n_epsps):
        axes.plot(x, fit.curve(k), color=_epsp_colour(k), linewidth=1.2, label=f"fit {name}{k + 1}")
    axes.plot(x, fit.summed_curve, color=COLOURS["sum"], linewidth=1.3, linestyle="--", label=f"sum of fitted {name}s")

    # peaks and 20-80 % rise segments on the fitted single EPSPs
    for k, (measures, onset) in enumerate(zip(fit.measures, fit.onsets_ms)):
        colour = _epsp_colour(k)
        axes.plot(to_x(onset + measures.peak_time_ms), measures.peak_mv, "o", color=colour, markersize=4, zorder=5)
        if np.all(np.isfinite(measures.rise_points_ms)):
            rise_t = np.array(measures.rise_points_ms) + onset
            axes.plot(to_x(rise_t), [0.2 * measures.peak_mv, 0.8 * measures.peak_mv], color=colour,
                      linewidth=4, alpha=0.55, solid_capstyle="butt", zorder=4)

    lines = []
    for k, (measures, baseline) in enumerate(zip(fit.measures, fit.baselines_mv)):
        base = f"{baseline:6.1f}" if np.isfinite(baseline) else "   n/a"
        lines.append(f"{name}{k + 1} base {base} {unit} · {measures.peak_mv:.2f} {unit} · rise {measures.rise_20_80_ms:.1f}"
                     f" · τ {measures.decay_tau_ms:.0f} ms")
    lines.append((f"PPR {fit.paired_pulse_ratio:.2f} · " if fit.n_epsps == 2 else "") + f"R² {fit.r_squared:.3f}")
    axes.text(0.98, 0.97, "\n".join(lines), transform=axes.transAxes, ha="right", va="top", fontsize=6.0,
              family="monospace", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.85))
    axes.set_xticks(ticks, labels)
    for gap in gaps:
        axes.axvline(gap, color="0.3", linewidth=0.8, linestyle=(0, (2, 2)))
        axes.text(gap, 0.0, "//", transform=axes.get_xaxis_transform(), ha="center", va="center", fontsize=8,
                  backgroundcolor="white")
    pulses = f"{fit.n_epsps} pulses, " if fit.n_epsps > 2 else ""
    axes.set_title(f"ISI {fit.isi_ms:g} ms  ({pulses}{fit.n_sweeps} sweeps)", fontsize=8.5)


def _draw_template(axes, analysis: SummatedEpspAnalysis) -> None:
    template = analysis.template
    unit, name = analysis.unit, analysis.response_name
    for event in template.events:
        axes.plot(template.time_ms, event, color=COLOURS["template_events"], linewidth=0.5, alpha=0.6)
    axes.plot(template.time_ms, template.mean, color="black", linewidth=1.4, label=f"mean of isolated {name}s")
    axes.plot(template.time_ms, template.fitted, color=COLOURS["sum"], linewidth=1.3, linestyle="--",
              label="model fit")
    axes.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    m, p = template.measures, template.parameters
    sag_line = (f"sag {m.sag_mv:.2f} {unit} (τ {p.tau_sag_rise:.0f} / {p.tau_sag_decay:.0f} ms)\n"
                if p.sag_ratio > 0 else "no sag term\n")
    axes.text(0.98, 0.97,
              f"peak {m.peak_mv:.2f} {unit} · rise {m.rise_20_80_ms:.1f} ms\n"
              f"τ rise {p.tau_rise:.1f} ms · τ decay {p.tau_decay:.1f} ms\n"
              + sag_line +
              f"latency {p.latency:.1f} ms · fit RMS {template.fit_rms_mv:.3f} {unit}",
              transform=axes.transAxes, ha="right", va="top", fontsize=6.5, family="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.85))
    axes.set_title(f"Template: {template.n_first} 1st + {template.n_second} later {name}s, stimuli ≥ "
                   f"{template.threshold_isi_ms:g} ms apart", fontsize=8.5)
    axes.set_xlabel("time from stimulus (ms)")
    axes.set_ylabel(unit)
    axes.legend(fontsize=6.5, frameon=False, loc="center right")


SUMMARY_COLUMNS = {"peak": ("peak_mv", "peak (mV)"), "ratio": ("ratio_to_epsp1", "ratio to EPSP1"),
                   "rise": ("rise_20_80_ms", "rise 20–80 % (ms)"), "decay": ("decay_tau_ms", "decay τ (ms)")}


def _draw_summary(axes, analysis: SummatedEpspAnalysis, quantity: str) -> None:
    """
    Paired pulses: the quantity against the ISI, one line per EPSP.
    Trains (any fit with > 2 EPSPs): against the EPSP number, one line per ISI.
    """
    table = analysis.table()
    column, label = SUMMARY_COLUMNS[quantity]
    name = analysis.response_name
    label = label.replace("mV", analysis.unit).replace("EPSP", name)
    if quantity == "ratio":
        axes.axhline(1, color="0.7", linewidth=0.8, linestyle="--")
    if analysis.max_epsps > 2:
        numbers = np.arange(1, analysis.max_epsps + 1)
        for index, (_, row) in enumerate(table.iterrows()):
            values = [row.get(f"epsp{k}_{column}", np.nan) for k in numbers]
            axes.plot(numbers, values, "o-", color=EPSP_COLOURS[index % len(EPSP_COLOURS)], markersize=3.5,
                      linewidth=1, label=f"{row['isi_ms']:g} ms ({int(row['n_pulses'])})")
        axes.set_xlabel(f"{name} number")
        axes.set_xticks(numbers)
    else:
        first = 2 if quantity == "ratio" else 1
        for k in range(first, analysis.max_epsps + 1):
            axes.plot(table["isi_ms"], table[f"epsp{k}_{column}"], "o-", color=_epsp_colour(k - 1), markersize=3.5,
                      linewidth=1, label=f"{name}{k}" + (f" / {name}1" if quantity == "ratio" else ""))
        axes.set_xscale("log")
        axes.set_xlabel("ISI (ms)")
        axes.set_xticks([10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    axes.set_ylabel(label)
    title = label.split(" (")[0]
    axes.set_title(title[0].upper() + title[1:], fontsize=8.5)
    axes.legend(fontsize=6.5, frameon=False)


def build_epsp_deconvolution_figure(analysis: SummatedEpspAnalysis, columns: int = 4):
    """One matplotlib Figure for one group (see module docstring)."""
    import textwrap

    import matplotlib
    from matplotlib.figure import Figure

    from .save_static_figure import FIGURE_STYLE

    n_isi_rows = math.ceil(len(analysis.fits) / columns)
    width = 12 // columns                     # grid cells per ISI panel (12-cell wide grid)
    with matplotlib.rc_context({**FIGURE_STYLE, "font.size": 8}):
        figure = Figure(figsize=(4.2 * columns, 2.9 * (2 + n_isi_rows) + 1.0), layout="constrained")
        grid = figure.add_gridspec(2 + n_isi_rows, 12)
        template_axes = figure.add_subplot(grid[0:2, 0:4])
        _draw_template(template_axes, analysis)
        summary_cells = (grid[0, 4:8], grid[0, 8:12], grid[1, 4:8], grid[1, 8:12])
        for cell, quantity in zip(summary_cells, ("peak", "ratio", "rise", "decay")):
            _draw_summary(figure.add_subplot(cell), analysis, quantity)

        isi_axes = []
        for index, fit in enumerate(analysis.fits):
            row, column = divmod(index, columns)
            axes = figure.add_subplot(grid[2 + row, column * width:(column + 1) * width],
                                      sharey=isi_axes[0] if isi_axes else None)
            _draw_isi_panel(axes, fit, analysis.unit, analysis.response_name)
            if column == 0:
                axes.set_ylabel(analysis.unit)
            if index + columns >= len(analysis.fits):
                axes.set_xlabel("time from 1st stimulus (ms)")
            isi_axes.append(axes)
        if isi_axes:
            # common y-range with head-room for the value boxes above the peaks
            low = min(float(np.nanmin(f.trace)) for f in analysis.fits)
            high = max(float(np.nanmax(f.trace)) for f in analysis.fits)
            span = high - low or 1.0
            isi_axes[0].set_ylim(low - 0.08 * span, high + (0.25 + 0.1 * analysis.max_epsps) * span)
            fullest = max(range(len(analysis.fits)), key=lambda i: analysis.fits[i].n_epsps)
            handles, labels = isi_axes[fullest].get_legend_handles_labels()
            figure.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=7.5, frameon=False)

        title = f"{analysis.header}" + (f" | group: {analysis.group}" if analysis.group else "")
        lines = textwrap.wrap(title, 190, break_long_words=False)
        if analysis.warnings:
            lines += textwrap.wrap("⚠ " + " ".join(analysis.warnings), 190)
        figure.suptitle("\n".join(lines), fontsize=8.5, x=0.01, ha="left")
    return figure


def save_epsp_deconvolution_figures(analyses: list[SummatedEpspAnalysis], output_path: str | Path,
                                    dpi: int = 200, columns: int = 4) -> list[Path]:
    """
    One file per group: '<name>_<group>.<ext>' (just '<name>.<ext>' for a single
    group). PNG / SVG / PDF by extension. Returns the written paths.
    """
    from .save_static_figure import _save

    output_path = Path(output_path)
    written = []
    for analysis in analyses:
        path = output_path
        if len(analyses) > 1 and analysis.group:
            path = output_path.with_name(f"{output_path.stem}_{analysis.group}{output_path.suffix}")
        written.append(_save(build_epsp_deconvolution_figure(analysis, columns), path, dpi))
    return written
