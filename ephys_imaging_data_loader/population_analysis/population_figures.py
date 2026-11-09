"""
population_figures.py
=====================

Step 3 of the population analysis: figures across sets, drawn from the tables
of ``results_tables`` (one value per set). Two levels:

* ``draw_*(axes, table, ...)``    ONE panel; reusable in any figure layout
* ``*_figure(...)``               a whole figure built from those panels

Figures
-------
    ppr_vs_isi_figure            rows: PPR of the peak, PPR of the max slope (2nd / 1st
                                 response) against the ISI; columns: one per condition
                                 (thin line = one set, thick line + band = median and
                                 interquartile range per ISI), last column: the medians
                                 of all conditions together
    measured_vs_linear_sum_figure
                                 measured cluster response (y) against the linear sum of
                                 its single spines (x): left with the single spines
                                 shifted by their uncaging time, right unshifted (all
                                 at t = 0); 45° line = linear summation
    pre_post_figure              the dendrites recorded both pre- and post-apv:
                                 top - PPR of peak and of slope against ISI, one column per
                                 dendrite; bottom - paired plots (one line per dendrite)
                                 of the cluster's 1st-response measures, short-ISI PPR and
                                 gain, each with its paired test

Ratios (PPR, gain) are drawn on a log2 axis, so a halving and a doubling are the
same distance from 1. Values outside the shown range are drawn at its edge as ▲ / ▼.
Colours follow the condition (``GROUP_COLOURS``), the same in every figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .population_statistics import format_p
from .results_tables import GROUP_ORDER

# Condition colours (Okabe-Ito, colour-blind safe): fixed per condition, never by rank.
GROUP_COLOURS = {"ctrl": "#333333", "pre-apv": "#E69F00", "post-apv": "#0072B2"}
OTHER_GROUP_COLOUR = "#999999"
RATIO_LIMITS = (0.25, 4.0)          # shown range of PPR axes (log2)
RATIO_TICKS = [0.25, 0.5, 1, 2, 4]
ISI_TICKS = [30, 50, 100, 200, 500, 1000]
N_SPINES_MARKERS = {5: "o", 6: "s"}  # other cluster sizes: diamond

RATIO_COLUMNS = {"ppr_peak": "PPR of peak (2nd / 1st)", "ppr_slope": "PPR of max slope (2nd / 1st)"}


def group_colour(group) -> str:
    return GROUP_COLOURS.get(str(group), OTHER_GROUP_COLOUR)


def present_groups(table: pd.DataFrame) -> list[str]:
    """The conditions in the table, in GROUP_ORDER."""
    present = set(table["group"].astype(str))
    return [g for g in GROUP_ORDER if g in present] + sorted(present - set(GROUP_ORDER))


def _ratio_axis(axes, label: str | None = None) -> None:
    axes.set_yscale("log", base=2)
    axes.set_ylim(*RATIO_LIMITS)
    axes.set_yticks(RATIO_TICKS, [f"{t:g}" for t in RATIO_TICKS])
    axes.minorticks_off()
    axes.axhline(1.0, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
    if label:
        axes.set_ylabel(label)


def _isi_axis(axes, label: bool = True) -> None:
    axes.set_xscale("log")
    axes.set_xticks(ISI_TICKS, [f"{t:g}" for t in ISI_TICKS])
    axes.minorticks_off()
    axes.set_xlim(25, 1200)
    if label:
        axes.set_xlabel("ISI (ms)")


def _clip_ratio(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values clipped to the shown range, and masks of those above / below it."""
    values = values.to_numpy(dtype=float)
    low, high = RATIO_LIMITS
    return np.clip(values, low, high), values > high, values < low


def _mark_out_of_range(axes, x, values, colour) -> None:
    """▲ / ▼ at the edge for values outside the shown range (only drawn when there are any)."""
    clipped, above, below = _clip_ratio(values)
    for mask, marker in ((above, "^"), (below, "v")):
        if mask.any():
            axes.plot(np.asarray(x)[mask], clipped[mask], linestyle="none", marker=marker, markersize=6,
                      color=colour, markeredgecolor="white", markeredgewidth=0.6, zorder=4)


def median_and_iqr(table: pd.DataFrame, column: str, by: str = "isi_ms") -> pd.DataFrame:
    """Per value of ``by``: median, 25th / 75th percentile and number of sets."""
    grouped = table.groupby(by)[column]
    return pd.DataFrame({"median": grouped.median(), "q1": grouped.quantile(0.25), "q3": grouped.quantile(0.75),
                         "n": grouped.count()}).reset_index()


def draw_ratio_vs_isi(axes, pulses: pd.DataFrame, column: str, groups=None, show_sets: bool = True,
                      labels: dict | None = None) -> None:
    """
    A ratio column of a ``paired_pulse_table`` (``ppr_peak`` / ``ppr_slope``) against the
    ISI. Per condition: thin lines = single sets (``show_sets``), thick line + band =
    median and interquartile range over sets. ``labels``: condition -> legend text.
    """
    groups = present_groups(pulses) if groups is None else groups
    for group in groups:
        part = pulses[pulses["group"].astype(str) == group]
        if part.empty:
            continue
        colour = group_colour(group)
        if show_sets:
            for _, one_set in part.groupby("set", observed=True):
                one_set = one_set.sort_values("isi_ms")
                axes.plot(one_set["isi_ms"], _clip_ratio(one_set[column])[0], color=colour, linewidth=0.7,
                          alpha=0.35, zorder=1)
            _mark_out_of_range(axes, part["isi_ms"], part[column], colour)
        summary = median_and_iqr(part, column)
        if (summary["q3"] > summary["q1"]).any():
            axes.fill_between(summary["isi_ms"], _clip_ratio(summary["q1"])[0], _clip_ratio(summary["q3"])[0],
                              color=colour, alpha=0.15, linewidth=0, zorder=2)
        n_sets = part["set"].nunique()
        label = (labels or {}).get(group, f"{group} (n = {n_sets} set{'s' if n_sets != 1 else ''})")
        axes.plot(summary["isi_ms"], _clip_ratio(summary["median"])[0], color=colour, linewidth=2.0, marker="o",
                  markersize=3.5, label=label, zorder=3)
        if not show_sets:
            _mark_out_of_range(axes, summary["isi_ms"], summary["median"], colour)
    _ratio_axis(axes)
    _isi_axis(axes)


def ppr_vs_isi_figure(pulses: pd.DataFrame, title: str = ""):
    """See module docstring. ``pulses``: ``results_tables.paired_pulse_table``."""
    from matplotlib.figure import Figure

    groups = present_groups(pulses)
    columns = len(groups) + 1
    figure = Figure(figsize=(3.4 * columns, 6.6), layout="constrained")
    axes = figure.subplots(2, columns, sharex=True, sharey=True, squeeze=False)
    for row, (column, label) in enumerate(RATIO_COLUMNS.items()):
        for index, group in enumerate(groups):
            draw_ratio_vs_isi(axes[row, index], pulses, column, [group])
            if row == 0:
                n_sets = pulses.loc[pulses["group"].astype(str) == group, "set"].nunique()
                axes[row, index].set_title(f"{group} · {n_sets} sets", fontsize=10)
        draw_ratio_vs_isi(axes[row, -1], pulses, column, groups, show_sets=False)
        if row == 0:
            axes[row, -1].set_title("medians (band: 25-75 %)", fontsize=10)
            axes[row, -1].legend(fontsize=7, frameon=False, loc="lower right")
        axes[row, 0].set_ylabel(label)
    for one in axes[0]:
        one.set_xlabel("")
    if title:
        figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    return figure


def draw_measured_vs_linear_sum(axes, summation: pd.DataFrame, quantity: str = "peak",
                                sum_prefix: str = "linear_sum", unit: str = "mV", limit: float | None = None) -> None:
    """
    Measured cluster response (y) against the linear sum (x) of a ``summation_table``,
    one point per cluster, colour = condition, marker = cluster size; legend with the
    median measured / linear-sum ratio per condition.
    """
    x_column, y_column = f"{sum_prefix}_{quantity}", f"measured_{quantity}"
    data = summation.dropna(subset=[x_column, y_column])
    if limit is None:
        limit = 1.08 * float(np.nanmax(data[[x_column, y_column]].to_numpy())) if len(data) else 1.0
    axes.plot([0, limit], [0, limit], color="0.6", linestyle="--", linewidth=0.8, zorder=0)
    for group in present_groups(summation):
        part = data[data["group"].astype(str) == group]
        if part.empty:
            continue
        gain = np.median(part[y_column] / part[x_column])
        first = True
        for n_spines, by_size in part.groupby("n_spines"):
            axes.scatter(by_size[x_column], by_size[y_column], s=36, color=group_colour(group),
                         marker=N_SPINES_MARKERS.get(int(n_spines), "D"), edgecolor="white", linewidth=0.7,
                         zorder=3, label=f"{group} (n = {len(part)}, median ratio {gain:.2f})" if first else None)
            first = False
    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_aspect("equal")
    axes.set_xlabel(f"linear sum, {quantity.replace('_', ' ')} ({unit})")
    axes.set_ylabel(f"measured, {quantity.replace('_', ' ')} ({unit})")


def measured_vs_linear_sum_figure(summation: pd.DataFrame, quantity: str = "peak", unit: str = "mV",
                                  title: str = ""):
    """See module docstring. ``summation``: ``results_tables.summation_table``."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9.6, 5.0), layout="constrained")
    axes = figure.subplots(1, 2, squeeze=False)[0]
    columns = [f"{p}_{quantity}" for p in ("measured", "linear_sum", "linear_sum_unshifted")]
    limit = 1.08 * float(np.nanmax(summation[columns].to_numpy()))
    panels = [("linear_sum", "linear sum: single spines shifted by uncaging time"),
              ("linear_sum_unshifted", "linear sum: single spines added at t = 0 (unshifted)")]
    for one, (prefix, panel_title) in zip(axes, panels):
        draw_measured_vs_linear_sum(one, summation, quantity, prefix, unit, limit)
        one.set_title(panel_title, fontsize=10)
        one.legend(fontsize=7, frameon=False, loc="upper left")
    sizes = sorted(summation["n_spines"].dropna().astype(int).unique())
    if len(sizes) > 1:
        figure.text(0.99, 0.995, "marker: " + ", ".join(
            f"{'●' if N_SPINES_MARKERS.get(n) == 'o' else '■' if N_SPINES_MARKERS.get(n) == 's' else '◆'} "
            f"{n} spines" for n in sizes), ha="right", va="top", fontsize=8)
    if title:
        figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    return figure


def draw_paired(axes, summary_pairs: pd.DataFrame, column: str, label: str, test: dict | None = None,
                before: str = "pre-apv", after: str = "post-apv") -> None:
    """
    Paired plot: one grey line per dendrite from ``before`` to ``after``, points coloured by
    condition, black bar = mean. Title: the paired test (``population_statistics.paired_test``).
    """
    wide = summary_pairs.pivot_table(index="dendrite", columns="group", values=column, observed=True)
    for condition in (before, after):
        if condition not in wide.columns:
            wide[condition] = np.nan
    for dendrite, values in wide[[before, after]].iterrows():
        axes.plot([0, 1], values.to_numpy(), color="0.7", linewidth=1.0, zorder=1)
    for position, condition in enumerate((before, after)):
        values = wide[condition].dropna()
        axes.scatter(np.full(len(values), position), values, s=36, color=group_colour(condition),
                     edgecolor="white", linewidth=0.7, zorder=3)
        if len(values):
            axes.plot([position - 0.18, position + 0.18], [values.mean()] * 2, color="black", linewidth=2, zorder=4)
    axes.set_xticks([0, 1], [before, after])
    axes.set_xlim(-0.5, 1.5)
    axes.set_ylabel(label)
    if test is not None:
        axes.set_title(f"paired t: {format_p(test['p'])} (n = {test['n']})", fontsize=9)
    if column.startswith(("ppr", "gain")):
        axes.axhline(1.0, color="0.6", linestyle="--", linewidth=0.8, zorder=0)


# Bottom row of pre_post_figure: cluster_summary column -> axis label
PRE_POST_MEASURES = {
    "peak_1": "1st-response peak (mV)",
    "slope_1": "1st-response max slope (mV/ms)",
    "rise_20_80_ms_1": "20-80 % rise (ms)",
    "decay_tau_ms_1": "decay τ (ms)",
    "ppr_peak_short": "PPR of peak, ISI ≤ 50 ms",
    "gain": "measured / linear sum (shifted)",
}


def pre_post_figure(pulse_pairs: pd.DataFrame, summary_pairs: pd.DataFrame, tests: dict | None = None,
                    measures: dict | None = None, title: str = ""):
    """
    See module docstring. ``pulse_pairs`` / ``summary_pairs``: ``paired_pulse_table`` /
    ``cluster_summary`` passed through ``pre_post_pairs``; ``tests``: measure -> row of
    ``population_statistics.paired_test`` (shown in the panel titles).
    """
    from matplotlib.figure import Figure

    measures = PRE_POST_MEASURES if measures is None else measures
    dendrites = sorted(pulse_pairs["dendrite"].unique())
    figure = Figure(figsize=(max(3.3 * len(dendrites), 2.8 * len(measures)), 11.0), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[2.0, 1.1])
    curves = top.subplots(2, len(dendrites), sharex=True, sharey=True, squeeze=False)
    for index, dendrite in enumerate(dendrites):
        part = pulse_pairs[pulse_pairs["dendrite"] == dendrite]
        for row, (column, label) in enumerate(RATIO_COLUMNS.items()):
            draw_ratio_vs_isi(curves[row, index], part, column, ["pre-apv", "post-apv"], show_sets=False,
                              labels={"pre-apv": "pre-apv", "post-apv": "post-apv"})
            if index == 0:
                curves[row, index].set_ylabel(label)
            if row == 0:
                curves[row, index].set_xlabel("")
        n_spines = ", ".join(str(int(n)) for n in sorted(part["n_spines"].unique()))
        curves[0, index].set_title(f"{dendrite} · {n_spines} spines", fontsize=10)
    curves[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    paired = bottom.subplots(1, len(measures), squeeze=False)[0]
    for one, (column, label) in zip(paired, measures.items()):
        draw_paired(one, summary_pairs, column, label, (tests or {}).get(column))
    bottom.suptitle("one value per dendrite and condition (1st-response measures: median over ISIs); "
                    "grey line = same dendrite, black bar = mean", fontsize=9)
    if title:
        figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    return figure
