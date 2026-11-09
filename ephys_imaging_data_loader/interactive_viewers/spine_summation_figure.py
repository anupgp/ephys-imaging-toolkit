"""
spine_summation_figure.py
=========================

Figures of the spatial summation of clustered spines
(``epsp_analysis.unitary_cluster_summation``).

Per set (``build_set_summation_figure``)
----------------------------------------
    top rows    measured cluster response (y) against the LINEAR SUM of its single
                spines (x), with the 45° line (on the line = linear, above =
                supralinear, below = sublinear):
                row 1: peak and max slope dV/dt, each twice: against the linear sum
                       with the single spines SHIFTED by their uncaging time, and
                       against the linear sum UNSHIFTED (all spines added at t = 0)
                row 2: 20-80 % rise and decay tau (shifted sum), the SPINE MAP (an
                       image of the dendrite with every single spine's laser point
                       as a red cross-hair labelled by its sheet spine id;
                       experiment_catalog/set_spine_map_image.py, only when the
                       caller passes ``spine_map``) and the PAIRED-PULSE RATIO (2nd /
                       1st response, from the deconvolution) of every cluster against
                       the ISI - only when a cluster was recorded at an ISI <= 100 ms
                       (otherwise the spine map takes its place)
                marker colour = number of spines in the cluster, a fixed scale from
                2 to 10 (colour bar); single spines are grey points at x = y (on the
                line by definition, as reference points); clusters are labelled
                "n spines"
    below       two panels per cluster:
                left  - the recorded cluster response (black), the linear sum of
                        its single spines (red dashed; shifted by uncaging time)
                        and the unshifted linear sum (grey dotted); on every trace
                        two peaks: dot = peak of the FIT (filled: measured, open:
                        linear sum), × = peak of the TRACE itself (1 ms running
                        mean); blue band = 20-80 % rise,
                        green band = decay from the peak to peak + tau, orange
                        line = tangent at the maximum slope dV/dt (all from
                        the fit, the same numbers as in the CSV)
                        (the title says if the single spines were shifted by their
                        uncaging time for the sum; the shift is used ONLY there)
                right - the single spines' 1st responses (sheet spine ids), drawn
                        as they enter the linear sum, with both peaks, on their own
                        y scale (they are much smaller than the cluster response)

Across sets (``build_batch_summation_figure``)
----------------------------------------------
The peak / max-slope (shifted and unshifted sums), rise and decay panels and the gain (measured / linear-sum peak against the
number of spines) for all sets of a batch run (from the combined
table), one row per unit (mV for EPSPs, pA for EPSCs).
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from ..epsp_analysis.ppr_recovery import fit_ppr_recovery

from ..epsp_analysis.unitary_cluster_summation import SetSummationAnalysis

# Colours of the single-spine TRACES (one per spine of a cluster): Kelly's colours of maximum contrast.
SPINE_TRACE_COLOURS = ["#0067A5", "#F38400", "#008856", "#875692", "#882D17", "#E68FAC", "#8DB600", "#A1CAF1",
                       "#222222"]
# Marker colour = number of spines in a cluster, on a FIXED scale from 2 to 10 (the same
# colour means the same cluster size in every figure): 9 steps of matplotlib's "plasma"
# map (dark blue -> purple -> pink -> orange), ordered and easy to tell apart.
# Single spines (n = 1) are grey. Clusters of more than 10 spines get the colour of 10.
SPINE_COUNT_RANGE = (2, 10)
SINGLE_SPINE_COLOUR = "#9A9A9A"

# Scatter panels: (measured column suffix, linear-sum column prefix, axis label, title);
# "{unit}" is replaced by mV / pA. "linear_sum" = spines shifted by their uncaging time,
# "linear_sum_unshifted" = all spines added at t = 0.
PEAK_AND_SLOPE_PANELS = [
    ("peak", "linear_sum", "peak ({unit})", "Peak · linear sum shifted by uncaging time"),
    ("peak", "linear_sum_unshifted", "peak ({unit})", "Peak · linear sum unshifted (all at t = 0)"),
    ("max_slope", "linear_sum", "max slope dV/dt ({unit}/ms)", "Max slope · linear sum shifted"),
    ("max_slope", "linear_sum_unshifted", "max slope dV/dt ({unit}/ms)", "Max slope · linear sum unshifted"),
]
SHAPE_PANELS = [
    ("rise_20_80_ms", "linear_sum", "rise 20–80 % (ms)", "Rise 20–80 % · linear sum shifted"),
    ("decay_tau_ms", "linear_sum", "decay τ (ms)", "Decay τ · linear sum shifted"),
]
SCATTER_PANELS = PEAK_AND_SLOPE_PANELS + SHAPE_PANELS
MAX_SLOPE_COLOUR = "#F38400"   # tangent at the maximum slope
PPR_SHORT_ISI_MS = 100.0       # the PPR panel is drawn only if a cluster was recorded at an ISI <= this


def _spine_count_colours() -> list:
    from matplotlib import colormaps

    low, high = SPINE_COUNT_RANGE
    return [colormaps["plasma"](0.85 * k / (high - low)) for k in range(high - low + 1)]


def spine_count_colour(n_spines: int):
    """Marker colour of a cluster of ``n_spines`` (grey for a single spine)."""
    n = int(n_spines)
    if n <= 1:
        return SINGLE_SPINE_COLOUR
    low, high = SPINE_COUNT_RANGE
    return _spine_count_colours()[min(max(n, low), high) - low]


def _add_spine_count_colour_bar(figure, axes_list) -> None:
    """Discrete colour bar 2 ... 10 spines next to ``axes_list``."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    low, high = SPINE_COUNT_RANGE
    norm = BoundaryNorm(np.arange(low - 0.5, high + 1.5), high - low + 1)
    bar = figure.colorbar(ScalarMappable(norm, ListedColormap(_spine_count_colours())), ax=axes_list,
                          ticks=range(low, high + 1), fraction=0.02, pad=0.01, aspect=25)
    bar.set_label("spines in cluster", fontsize=7)
    bar.ax.tick_params(labelsize=6)


def _scatter_points(table: pd.DataFrame, quantity: str, sum_prefix: str = "linear_sum") -> pd.DataFrame:
    """x (linear sum), y (measured), n_spines, label for one quantity; single spines at x = y."""
    empty = pd.DataFrame(columns=["x", "y", "n", "label"])
    measured, summed = f"measured_{quantity}", f"{sum_prefix}_{quantity}"
    if measured not in table.columns:
        return empty
    singles = table[(table["kind"] == "unitary") & table[measured].notna()]
    clusters = (table[(table["kind"] == "cluster") & table[summed].notna() & table[measured].notna()]
                if summed in table.columns else table.iloc[0:0])      # (older CSVs: no unshifted sum)
    parts = []
    if not clusters.empty:
        parts.append(pd.DataFrame({"x": clusters[summed], "y": clusters[measured],
                                   "n": clusters["n_spines"], "label": clusters["input"]}))
    if not singles.empty:
        parts.append(pd.DataFrame({"x": singles[f"measured_{quantity}"], "y": singles[f"measured_{quantity}"],
                                   "n": 1, "label": singles["input"]}))
    return pd.concat(parts, ignore_index=True) if parts else empty


def _scatter_limit(table: pd.DataFrame, quantity: str) -> float:
    """Common axis limit of a quantity over the shifted and the unshifted sum (so their panels compare)."""
    values = [_scatter_points(table, quantity, prefix)[["x", "y"]].to_numpy(dtype=float)
              for prefix in ("linear_sum", "linear_sum_unshifted")]
    values = np.concatenate([v.ravel() for v in values])
    values = values[np.isfinite(values)]
    return float(values.max()) * 1.1 if values.size and values.max() > 0 else 1.0


def _draw_scatter(axes, table: pd.DataFrame, quantity: str, axis_label: str, unit: str, annotate: bool,
                  sum_prefix: str = "linear_sum", title: str | None = None) -> None:
    points = _scatter_points(table, quantity, sum_prefix)
    axis_label = axis_label.replace("{unit}", unit)
    sum_name = "linear sum, unshifted" if sum_prefix.endswith("unshifted") else "linear sum"
    if points[points["n"] > 1].empty:
        axes.text(0.5, 0.5, f"no cluster with a {sum_name}", transform=axes.transAxes, ha="center", fontsize=7,
                  color="0.4")
    if not points.empty:
        top = _scatter_limit(table, quantity)
        axes.plot([0, top], [0, top], color="0.6", linewidth=1, linestyle="--", zorder=0, label="linear (45°)")
        for n in sorted(points["n"].unique()):
            subset = points[points["n"] == n]
            axes.scatter(subset["x"], subset["y"], s=26 if n > 1 else 12, color=spine_count_colour(n),
                         edgecolor="white", linewidth=0.5, zorder=3, label=f"{int(n)} spine{'s' if n > 1 else ''}")
        if annotate:
            for _, point in points[points["n"] > 1].iterrows():
                axes.annotate(f"{int(point['n'])} spines", (point["x"], point["y"]), fontsize=5.5,
                              xytext=(3, 3), textcoords="offset points", color="0.3")
        axes.set_xlim(0, top)
        axes.set_ylim(0, top)
        axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(f"{sum_name}: {axis_label}")
    axes.set_ylabel(f"measured: {axis_label}")
    axes.set_title(title or axis_label.split(" (")[0][0].upper() + axis_label.split(" (")[0][1:], fontsize=8)


PPR_DISPLAY_MAX = 4.0   # PPRs above this (a near-zero 1st response) are drawn as ▲ at the top edge


def _draw_ppr_vs_isi(axes, table: pd.DataFrame) -> None:
    """Paired-pulse ratio (2nd / 1st response) of every cluster against the ISI, one line per cluster."""
    column = "paired_pulse_ratio"
    clusters = table[(table["kind"] == "cluster") & pd.to_numeric(table.get(column, pd.Series(dtype=float)),
                                                                   errors="coerce").notna()]
    clusters = clusters.drop_duplicates(["input", "isi_ms"])     # one row per stimulus: the PPR is on each
    axes.axhline(1, color="0.6", linewidth=1, linestyle="--", zorder=0)
    if clusters.empty:
        axes.text(0.5, 0.5, "no paired-pulse recordings", transform=axes.transAxes, ha="center", fontsize=7,
                  color="0.4")
    else:
        values = clusters[column].astype(float)
        top = min(PPR_DISPLAY_MAX, max(2.0, float(values.max()) * 1.1))
        for label, part in clusters.groupby("input", sort=False):
            part = part.sort_values("isi_ms")
            isi, ppr = part["isi_ms"].astype(float), part[column].astype(float)
            colour = spine_count_colour(part["n_spines"].iloc[0])
            fit = fit_ppr_recovery(isi, ppr)                       # PPR = 1 - A exp(-ISI/tau)
            fit_text = f", fit R² {fit.r_squared:.2f}" if fit.ok else ""
            axes.plot(isi, ppr.clip(upper=top), color=colour, linewidth=0.8, linestyle="none" if fit.ok else "-",
                      marker="o", markersize=3.5, label=f"{int(part['n_spines'].iloc[0])} spines{fit_text}")
            if fit.ok:
                grid = np.geomspace(float(isi.min()) * 0.8, float(isi.max()) * 1.2, 150)
                axes.plot(grid, np.clip(fit.curve(grid), None, top), color=colour, linewidth=1.3,
                          linestyle="-" if fit.tau_meaningful else "--")
            above = ppr > top
            if above.any():      # (an EMPTY line drawn with clip_on=False breaks the constrained layout)
                axes.plot(isi[above], [top] * int(above.sum()), linestyle="none", marker="^", markersize=6,
                          color=colour, clip_on=False)
        axes.set_xscale("log")
        isis = sorted(clusters["isi_ms"].astype(float).unique())
        axes.set_xticks(isis, [f"{v:g}" for v in isis], rotation=90 if len(isis) > 5 else 0, fontsize=6)
        axes.minorticks_off()
        if len(isis) == 1:
            axes.set_xlim(isis[0] / 2, isis[0] * 2)
        axes.set_ylim(0, top)
        axes.legend(fontsize=5.5, frameon=False, loc="lower right", ncol=1 + clusters["input"].nunique() // 6,
                    handlelength=1.2)
    axes.set_xlabel("ISI (ms)")
    axes.set_ylabel("PPR (2nd / 1st response)")
    axes.set_title(f"Paired-pulse ratio of clusters (▲ > {PPR_DISPLAY_MAX:g})\nline: fit 1 - A·exp(-ISI/τ)",
                   fontsize=8)


def _draw_spine_map(axes, spine_map) -> None:
    """Image of the dendrite with every single spine's laser point labelled by its sheet spine id."""
    import copy

    from .image_grid_figure import draw_image_panel

    if spine_map is None:
        axes.text(0.5, 0.5, "no image with the spines' laser points", transform=axes.transAxes, ha="center",
                  fontsize=7, color="0.4")
        axes.set_axis_off()
        axes.set_title("Spine map", fontsize=8.5)
        return
    image = copy.copy(spine_map.image)          # shallow: shares the pixels, own laser points
    image.uncaging_points = spine_map.points[["x_px", "y_px"]].reset_index(drop=True)
    draw_image_panel(axes, image, "", target_labels=list(spine_map.points["label"]), show_scan_numbers=False,
                     marker="crosshair")
    axes.set_title(f"Spine map (sheet spine ids)\n{spine_map.source}", fontsize=7.5)
    if spine_map.note:
        axes.text(0.0, -0.02, textwrap.fill(spine_map.note, 70), transform=axes.transAxes, ha="left", va="top",
                  fontsize=5, color="0.35")


def _draw_gain(axes, table: pd.DataFrame) -> None:
    clusters = table[(table["kind"] == "cluster") & table.get("gain", pd.Series(dtype=float)).notna()]
    axes.axhline(1, color="0.6", linewidth=1, linestyle="--", zorder=0)
    if not clusters.empty:
        rng = np.random.default_rng(0)
        for n in sorted(clusters["n_spines"].unique()):
            gains = clusters.loc[clusters["n_spines"] == n, "gain"]
            jitter = rng.uniform(-0.12, 0.12, len(gains)) if len(gains) > 1 else 0.0
            axes.scatter(n + jitter, gains, s=18, color=spine_count_colour(n), edgecolor="white", linewidth=0.4,
                         zorder=3)
            if len(gains) > 1:
                axes.plot([n - 0.25, n + 0.25], [gains.mean()] * 2, color="black", linewidth=1.5)
        axes.set_xticks(sorted(clusters["n_spines"].unique()))
    axes.set_xlabel("number of spines in the cluster")
    axes.set_ylabel("gain (measured / linear sum)")
    axes.set_title("Gain (> 1 supralinear, < 1 sublinear)", fontsize=8.5)


PEAK_COLOUR = "red"            # peak of the fit (dot)
UNSHIFTED_COLOUR = "0.45"      # linear sum with all spines at t = 0
TRACE_PEAK_COLOUR = "#00A0B0"  # peak of the trace itself (×), 1 ms running mean
RISE_COLOUR = "#0067A5"        # 20-80 % rise of the fit
DECAY_COLOUR = "#008856"       # decay: peak -> peak + tau of the fit


def _mark_measures(axes, response, shift_ms: float = 0.0, filled: bool = True, rise_and_decay: bool = True,
                   peak_colour=None) -> None:
    """
    Peak of the fit (dot) and of the trace itself (×), 20-80 % rise, decay (peak ->
    peak + tau) and the tangent at the maximum slope (from that point over 40 % of
    the peak height) of a fitted response, drawn on its fitted curve; the values come
    from the fit (``measure_epsp``) and ``raw_peak``, the same numbers as in the CSV.
    ``shift_ms`` moves them with a shifted trace; ``peak_colour`` draws both peaks
    in one other colour (the unshifted linear sum).
    """
    m, p = response.measures, response.parameters
    if not np.isfinite(m.peak_mv) or m.peak_mv <= 0:
        return
    if rise_and_decay:
        t20, t80 = m.rise_points_ms
        if np.isfinite(t20) and np.isfinite(t80):
            t = np.linspace(t20, t80, 40)
            axes.plot(t + shift_ms, p.waveform(t), color=RISE_COLOUR, linewidth=4, alpha=0.45,
                      solid_capstyle="butt", zorder=4)
        if np.isfinite(m.decay_tau_ms):
            t = np.linspace(m.peak_time_ms, m.peak_time_ms + m.decay_tau_ms, 80)
            axes.plot(t + shift_ms, p.waveform(t), color=DECAY_COLOUR, linewidth=4, alpha=0.4,
                      solid_capstyle="butt", zorder=4)
            axes.plot([t[-1] + shift_ms], p.waveform(t[-1:]), marker="|", markersize=7, color=DECAY_COLOUR, zorder=5)
        if np.isfinite(m.max_slope_mv_per_ms):
            t0 = m.max_slope_time_ms
            v0 = float(p.waveform(np.array([t0]))[0])
            # from the max-slope point forward over 40 % of the peak height (the model's
            # steepest point is usually its onset, so a centred tangent would dip below 0)
            length = min(6.0, 0.4 * m.peak_mv / m.max_slope_mv_per_ms) if m.max_slope_mv_per_ms > 0 else 6.0
            t = np.array([t0, t0 + length])
            axes.plot(t + shift_ms, v0 + m.max_slope_mv_per_ms * (t - t0), color=MAX_SLOPE_COLOUR, linewidth=1.6,
                      zorder=5)
    colour = PEAK_COLOUR if peak_colour is None else peak_colour
    axes.plot([m.peak_time_ms + shift_ms], [m.peak_mv], marker="o", markersize=5, linestyle="none", zorder=6,
              markerfacecolor=colour if filled else "white", markeredgecolor=colour, markeredgewidth=1.2)
    # the peak measured on the trace itself (largest 1 ms running mean in the first 100 ms)
    if np.isfinite(response.raw_peak) and np.isfinite(getattr(response, "raw_peak_time_ms", np.nan)):
        axes.plot([response.raw_peak_time_ms + shift_ms], [response.raw_peak], marker="x", markersize=5.5,
                  markeredgewidth=1.5, linestyle="none", color=TRACE_PEAK_COLOUR if peak_colour is None else colour,
                  zorder=7)


def _draw_unitary_traces(axes, analysis: SetSummationAnalysis, label: str) -> None:
    """The cluster's single spines (sheet spine ids), each shifted by its uncaging time, with their peaks."""
    summed = analysis.linear_sums[label]
    time = summed.response.time_ms
    for index, (spine, trace, shift) in enumerate(zip(analysis.members[label], summed.member_traces, summed.shifts_ms)):
        colour = SPINE_TRACE_COLOURS[index % len(SPINE_TRACE_COLOURS)]
        axes.plot(time, trace, color=colour, linewidth=1.0, label=f"spine {spine}")
        single = analysis.first_responses.get(f"spine {spine}")
        if single is not None:
            _mark_measures(axes, single, shift, rise_and_decay=False)
    axes.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    axes.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=1 + len(summed.member_traces) // 5,
                handlelength=1.2)
    axes.set_xlim(-10, min(250.0, float(time[-1])))
    axes.set_title(f"single spines of {label}", fontsize=8)


def _draw_cluster_traces(axes, analysis: SetSummationAnalysis, label: str) -> None:
    measured = analysis.first_responses[label]
    summed = analysis.linear_sums[label]
    members = analysis.members[label]
    time = summed.response.time_ms
    axes.plot(measured.time_ms, measured.trace, color="black", linewidth=1.2, label="measured cluster")
    unshifted = analysis.linear_sums_unshifted.get(label)
    if unshifted is not None:
        axes.plot(unshifted.response.time_ms, unshifted.response.trace, color=UNSHIFTED_COLOUR, linewidth=1.0,
                  linestyle=":", label="linear sum, unshifted (all at t = 0)")
        _mark_measures(axes, unshifted.response, filled=False, rise_and_decay=False, peak_colour=UNSHIFTED_COLOUR)
    axes.plot(time, summed.response.trace, color="#BE0032", linewidth=1.3, linestyle="--",
              label="linear sum (shifted by uncaging time)")
    _mark_measures(axes, measured)
    _mark_measures(axes, summed.response, filled=False)
    axes.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    unit = analysis.unit
    m, e = measured.measures, summed.response.measures
    u = unshifted.response.measures if unshifted is not None else None

    def value(measures, name, fmt):
        return format(getattr(measures, name), fmt) if measures is not None else "–".rjust(len(format(0.0, fmt)))

    def gain(measures):
        return f"{m.peak_mv / measures.peak_mv:6.2f}" if measures is not None and measures.peak_mv else "     –"

    lines = [f"         measured  sum  t=0"]
    for title, name, fmt, suffix in (("peak   ", "peak_mv", "6.2f", unit), ("rise   ", "rise_20_80_ms", "6.1f", "ms"),
                                     ("τ decay", "decay_tau_ms", "6.0f", "ms"),
                                     ("slope  ", "max_slope_mv_per_ms", "6.3f", f"{unit}/ms")):
        lines.append(f"{title} {value(m, name, fmt):>7} {value(e, name, fmt):>6} {value(u, name, fmt):>6} {suffix}")
    lines.append(f"gain            {gain(e)} {gain(u)}")
    axes.text(0.98, 0.97, "\n".join(lines),
              transform=axes.transAxes, ha="right", va="top", fontsize=6, family="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.85))
    axes.set_xlim(-10, min(250.0, float(time[-1])))
    # the shift by uncaging time is used only for the linear sum, so it is said here
    shifted = any(abs(shift) > 0 for shift in summed.shifts_ms)
    how = (f"linear sum: single spines shifted by uncaging time (0–{max(summed.shifts_ms):.2f} ms)" if shifted
           else "linear sum: single spines added unshifted")
    axes.set_title(f"{label} ({len(members)} spines, ISI {measured.isi_ms:g} ms, {measured.n_sweeps} sweeps)\n{how}",
                   fontsize=8)


def _has_short_isi_ppr(table: pd.DataFrame) -> bool:
    """True when a cluster has a paired-pulse ratio at an ISI <= PPR_SHORT_ISI_MS."""
    if "paired_pulse_ratio" not in table.columns:
        return False
    clusters = table[(table["kind"] == "cluster") & pd.to_numeric(table["paired_pulse_ratio"], errors="coerce").notna()]
    return bool((clusters["isi_ms"].astype(float) <= PPR_SHORT_ISI_MS).any())


def build_set_summation_figure(analysis: SetSummationAnalysis, columns: int = 4, spine_map=None):
    """
    Figure of one set (see module docstring); None when no cluster has a linear sum.
    ``spine_map``: ``experiment_catalog.set_spine_map_image.SpineMap`` for the image panel.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from .save_static_figure import FIGURE_STYLE

    clusters = [c.label for c in analysis.clusters() if c.label in analysis.linear_sums
                and c.label in analysis.first_responses]
    if not clusters:
        return None
    table = analysis.table()
    columns = max(4, columns - columns % 2)          # pairs of panels: cluster | its single spines
    per_row = columns // 2
    rows = math.ceil(len(clusters) / per_row)
    with matplotlib.rc_context({**FIGURE_STYLE, "font.size": 8}):
        # two top rows of 4 square panels, 3.2 in high each (taller rows only add white space)
        figure = Figure(figsize=(max(4.0 * columns, 15.5), 2 * 3.2 + 2.8 * rows + 0.9), layout="constrained")
        outer = figure.add_gridspec(2, 1, height_ratios=[2 * 3.2, 2.8 * rows])
        top = outer[0].subgridspec(2, 4)
        grid = outer[1].subgridspec(rows, columns)                       # cluster | single-spine pairs

        # row 1: peak and max slope against the shifted and the unshifted linear sum
        scatter_axes = []
        for column, (quantity, prefix, label, title) in enumerate(PEAK_AND_SLOPE_PANELS):
            axes = figure.add_subplot(top[0, column])
            _draw_scatter(axes, table, quantity, label, analysis.unit, annotate=True, sum_prefix=prefix, title=title)
            scatter_axes.append(axes)
        # row 2: rise, decay, spine map, PPR vs ISI (only with short ISIs; else the map takes 2 panels)
        for column, (quantity, prefix, label, title) in enumerate(SHAPE_PANELS):
            axes = figure.add_subplot(top[1, column])
            _draw_scatter(axes, table, quantity, label, analysis.unit, annotate=True, sum_prefix=prefix, title=title)
            scatter_axes.append(axes)
        scatter_axes[0].legend(*_spine_count_legend(scatter_axes[0]), fontsize=6, frameon=False, loc="upper left")
        if _has_short_isi_ppr(table):
            _draw_spine_map(figure.add_subplot(top[1, 2]), spine_map)
            _draw_ppr_vs_isi(figure.add_subplot(top[1, 3]), table)
        else:
            _draw_spine_map(figure.add_subplot(top[1, 2:]), spine_map)
        _add_spine_count_colour_bar(figure, scatter_axes[:len(PEAK_AND_SLOPE_PANELS)])

        # cluster panels share one y scale, single-spine panels another (much smaller) one
        first_cluster_axes = first_unitary_axes = None
        for index, label in enumerate(clusters):
            row, pair = divmod(index, per_row)
            axes = figure.add_subplot(grid[row, 2 * pair], sharey=first_cluster_axes)
            first_cluster_axes = first_cluster_axes or axes
            _draw_cluster_traces(axes, analysis, label)
            unitary_axes = figure.add_subplot(grid[row, 2 * pair + 1], sharey=first_unitary_axes)
            first_unitary_axes = first_unitary_axes or unitary_axes
            _draw_unitary_traces(unitary_axes, analysis, label)
            for panel in (axes, unitary_axes):
                panel.set_ylabel(analysis.unit)
                if index + per_row >= len(clusters):
                    panel.set_xlabel("time from 1st stimulus (ms)")
        handles, labels = first_cluster_axes.get_legend_handles_labels()
        markers = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=PEAK_COLOUR, markeredgecolor=PEAK_COLOUR),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=PEAK_COLOUR),
            Line2D([], [], marker="x", linestyle="none", color=TRACE_PEAK_COLOUR, markeredgewidth=1.5),
            Line2D([], [], color=RISE_COLOUR, linewidth=4, alpha=0.45),
            Line2D([], [], color=DECAY_COLOUR, linewidth=4, alpha=0.4),
            Line2D([], [], color=MAX_SLOPE_COLOUR, linewidth=1.6),
        ]
        figure.legend(handles + markers,
                      labels + ["peak of the fit (measured, single spines)", "peak of the fit (linear sum)",
                                "peak of the trace (1 ms mean)", "20–80 % rise", "decay: peak → peak + τ",
                                "max slope dV/dt (tangent)"],
                      loc="outside lower center", ncol=5, fontsize=7, frameon=False)
        info = analysis.set_info
        title = (f"Spatial summation · {info['expdate']} | neuron {info['neuronid']} | dendrite {info['dendriteid']}"
                 f" | group {info['group']} | {analysis.response_name}s ({analysis.unit})")
        lines = [title]
        notes = sorted({n for c in analysis.inputs for n in c.notes if "missing" in n or "assumed" in n})
        if notes:
            lines += textwrap.wrap("⚠ " + " · ".join(notes), 190)
        figure.suptitle("\n".join(lines), fontsize=8.5, x=0.01, ha="left")
    return figure


def _spine_count_legend(axes) -> tuple[list, list]:
    """Legend of a scatter panel without the cluster sizes (those are on the colour bar)."""
    handles, labels = axes.get_legend_handles_labels()
    keep = [i for i, text in enumerate(labels) if not text.endswith("spines")]
    return [handles[i] for i in keep], [labels[i] for i in keep]


def build_batch_summation_figure(table: pd.DataFrame):
    """Scatter + gain panels of a combined batch table, one row per unit; None without clusters."""
    import matplotlib
    from matplotlib.figure import Figure

    from .save_static_figure import FIGURE_STYLE

    if table.empty or "linear_sum_peak" not in table.columns:
        return None
    units = [u for u in table["unit"].dropna().unique()
             if table[(table["unit"] == u) & (table["kind"] == "cluster")]["linear_sum_peak"].notna().any()]
    if not units:
        return None
    with matplotlib.rc_context({**FIGURE_STYLE, "font.size": 8}):
        n_panels = len(SCATTER_PANELS) + 1
        figure = Figure(figsize=(3.6 * n_panels, 4.0 * len(units) + 0.6), layout="constrained")
        grid = figure.add_gridspec(len(units), n_panels)
        scatter_axes = []
        for row, unit in enumerate(units):
            part = table[table["unit"] == unit]
            for column, (quantity, prefix, label, title) in enumerate(SCATTER_PANELS):
                axes = figure.add_subplot(grid[row, column])
                name = "EPSC" if unit.lower() == "pa" else "EPSP"
                _draw_scatter(axes, part, quantity, label, unit, annotate=False, sum_prefix=prefix,
                              title=f"{name} {title[0].lower()}{title[1:]}" if column == 0 else title)
                scatter_axes.append(axes)
                if column == 0:
                    axes.legend(*_spine_count_legend(axes), fontsize=6, frameon=False, loc="upper left")
            _draw_gain(figure.add_subplot(grid[row, len(SCATTER_PANELS)]), part)
        _add_spine_count_colour_bar(figure, scatter_axes)
        n_sets = table[["expdate", "neuronid", "dendriteid", "group"]].drop_duplicates().shape[0]
        n_clusters = int(((table["kind"] == "cluster") & table["linear_sum_peak"].notna()).sum())
        figure.suptitle(f"Spatial summation, all sets: {n_sets} sets, {n_clusters} clusters with a linear sum",
                        fontsize=9, x=0.01, ha="left")
    return figure


def save_figure(figure, path: str | Path, dpi: int = 200) -> Path | None:
    """Save a figure (PNG / SVG / PDF by extension); nothing when ``figure`` is None."""
    from .save_static_figure import _save

    return None if figure is None else _save(figure, Path(path), dpi)
