"""
spatiotemporal_figures.py
=========================

Population figures of spatial and temporal summation over all sets (ctrl,
pre-apv, post-apv), drawn from the tables of ``spatiotemporal_tables`` (one
value per set; per ISI for the curves). Panels reuse ``population_figures``
(same colours per condition, log2 ratio axes, ▲/▼ for values off the axis).

    spatial_summation_figure     measured cluster EPSP vs the linear sum of its single
                                 spines (shifted / unshifted), for the peak and the max
                                 dV/dt, gain per condition, cluster and single-spine peaks
                                 per condition, gain vs spread
    temporal_summation_figure    rows: PPR of peak, PPR of max slope against ISI;
                                 columns: conditions + medians
    pre_post_temporal_figure     dendrites recorded pre- AND post-apv: curves per dendrite,
                                 paired plots of the temporal / kinetic measures
    pre_post_spatial_figure      same dendrites: cluster EPSP, single spines, linear sum,
                                 gain; single spines recorded in both conditions
    kinetics_figure              decay tau of EPSP2 at every ISI followed by EPSP1's (one
                                 panel per condition); decay and 20-80 % rise ratio
                                 EPSP2 / EPSP1 against ISI
    relationships_figure         PPR vs depolarisation left at the 2nd pulse, PPR vs EPSP1
                                 size, decay vs size, decay / resting potential / fit
                                 quality per condition

Every per-condition strip shows one dot per set and a black bar at the median.
Test results (``tests``: list of rows from ``spatiotemporal_statistics``) are
written into the panel titles where a panel has one main test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .population_figures import (N_SPINES_MARKERS, _clip_ratio, _isi_axis, _mark_out_of_range, _ratio_axis,
                                 draw_measured_vs_linear_sum, draw_paired, draw_ratio_vs_isi, group_colour,
                                 median_and_iqr, present_groups)
from .population_statistics import format_p

# temporal_summation_figure rows: column -> axis label
TEMPORAL_ROWS = {
    "ppr_peak": "PPR of peak\n(EPSP2 / EPSP1)",
    "ppr_slope": "PPR of max slope\n(EPSP2 / EPSP1)",
}
KINETIC_ROWS = {"tau_ratio": "decay τ ratio\n(EPSP2 / EPSP1)", "rise_ratio": "20-80 % rise ratio\n(EPSP2 / EPSP1)"}


def _title(figure, title: str, note: str = "") -> None:
    """Figure title, left; an explanatory note as a smaller second line."""
    if note:
        figure.suptitle(f"{title}\n{note}", fontsize=9.5, x=0.01, ha="left")
    elif title:
        figure.suptitle(title, fontsize=11, x=0.01, ha="left")


def _find_test(tests, **match) -> dict | None:
    """The first test row whose fields equal ``match``."""
    for row in tests or []:
        if all(str(row.get(k)) == str(v) for k, v in match.items()):
            return row
    return None


def _test_text(row: dict | None, prefix: str = "") -> str:
    if row is None:
        return ""
    if row.get("comparison") == "two groups":
        return f"{prefix}ctrl vs post-apv: MW {format_p(row['p'])}"
    if row.get("comparison") == "correlation":
        return f"{prefix}Spearman ρ = {row['centre']:.2f}, {format_p(row['p'])} (n = {row['n']})"
    return f"{prefix}{format_p(row['p'])}"


def draw_strip(axes, table: pd.DataFrame, column: str, label: str, groups=None, ratio: bool = False,
               marker_column: str | None = None, offset: float = 0.0, hollow: bool = False, size: float = 30) -> None:
    """
    One dot per row (set) per condition, jittered; black bar = median; 'n' under each
    condition. ``ratio``: log2 ratio axis with values off it at the edge (▲/▼).
    ``offset`` / ``hollow``: to draw a second measure next to the first one.
    """
    groups = present_groups(table) if groups is None else groups
    rng = np.random.default_rng(0)
    for position, group in enumerate(groups):
        values = table.loc[table["group"].astype(str) == group, column].dropna()
        if values.empty:
            continue
        x = position + offset + rng.uniform(-0.12, 0.12, len(values))
        shown = _clip_ratio(values)[0] if ratio else values.to_numpy()
        colour = group_colour(group)
        axes.scatter(x, shown, s=size, facecolor="white" if hollow else colour, edgecolor=colour,
                     linewidth=1.0 if hollow else 0.6, zorder=3)
        if ratio:
            _mark_out_of_range(axes, x, values, colour)
        median = float(np.median(values))
        axes.plot([position + offset - 0.2, position + offset + 0.2], [_clip_ratio(pd.Series([median]))[0][0]
                                                                       if ratio else median] * 2,
                  color="black", linewidth=2, zorder=4)
    axes.set_xticks(range(len(groups)), [f"{g}\nn = {table.loc[table['group'].astype(str) == g, column].notna().sum()}"
                                        for g in groups], fontsize=8)
    axes.set_xlim(-0.6, len(groups) - 0.4)
    if ratio:
        _ratio_axis(axes)
    axes.set_ylabel(label)


def _legend_groups(axes, groups, loc="best") -> None:
    from matplotlib.lines import Line2D

    axes.legend([Line2D([], [], marker="o", linestyle="none", color=group_colour(g)) for g in groups], groups,
                fontsize=7, frameon=False, loc=loc)


def _gain_strip(axes, with_sum: pd.DataFrame, quantity: str, groups, tests, label: str) -> None:
    """Gain = measured / linear sum per condition: ● shifted, ○ unshifted; title: shifted vs 1."""
    table = with_sum.assign(
        _shifted=with_sum[f"measured_{quantity}"] / with_sum[f"linear_sum_{quantity}"],
        _unshifted=with_sum[f"measured_{quantity}"] / with_sum[f"linear_sum_unshifted_{quantity}"])
    draw_strip(axes, table, "_shifted", label, groups, ratio=True, offset=-0.15)
    draw_strip(axes, table, "_unshifted", label, groups, ratio=True, offset=0.15, hollow=True)
    lowest = float(np.nanmin(table[["_shifted", "_unshifted"]].to_numpy())) if len(table) else 1.0
    if lowest >= 0.5:
        axes.set_ylim(0.5, 2.0)
        axes.set_yticks([0.5, 0.71, 1, 1.41, 2], ["0.5", "0.71", "1", "1.41", "2"])
    else:                                       # every value visible (values < 0.25 at the edge as ▼)
        axes.set_ylim(0.25, 2.0)
        axes.set_yticks([0.25, 0.5, 0.71, 1, 1.41, 2], ["0.25", "0.5", "0.71", "1", "1.41", "2"])
    lines = [f"{g}: {format_p(row['p'])}" for g in groups
             if (row := _find_test(tests, measure=f"measured / linear sum (shifted), {quantity}", group=g)) is not None
             and np.isfinite(row["p"])]
    axes.set_title("gain (● shifted, ○ unshifted)\nshifted vs 1, Wilcoxon: " + "; ".join(lines), fontsize=8.5)


def spatial_summation_figure(summary: pd.DataFrame, unitary: pd.DataFrame, tests=None, title: str = ""):
    """
    ``summary``: set_summary_table; ``unitary``: unitary_table.
    Row 1: peak, row 2: max dV/dt - measured vs linear sum shifted / unshifted + gain.
    Row 3: cluster EPSP (all sets), single spines, gain vs cluster spread.
    """
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.5, 14.0), layout="constrained")
    axes = figure.subplots(3, 3)
    groups = present_groups(summary)
    with_sum = summary.dropna(subset=["linear_sum_peak"])
    n_text = f"{len(with_sum)} of {len(summary)} sets"
    for row, quantity, unit, label in ((0, "peak", "mV", "peak"), (1, "max_slope", "mV/ms", "max dV/dt")):
        columns = [f"{p}_{quantity}" for p in ("measured", "linear_sum", "linear_sum_unshifted")]
        limit = 1.08 * float(np.nanmax(with_sum[columns].to_numpy())) if len(with_sum) else 1.0
        for one, (prefix, name) in zip(axes[row, :2], (("linear_sum", "shifted by uncaging time"),
                                                       ("linear_sum_unshifted", "unshifted (all at t = 0)"))):
            draw_measured_vs_linear_sum(one, with_sum, quantity, prefix, unit, limit)
            one.set_xlabel(f"linear sum, {label} ({unit})")
            one.set_ylabel(f"measured, {label} ({unit})")
            one.set_title(f"{label}: measured vs linear sum, {name}\n({n_text} have every single spine)", fontsize=9.5)
            one.legend(fontsize=7, frameon=False, loc="upper left")
        _gain_strip(axes[row, 2], with_sum, quantity, groups, tests, f"measured / linear sum, {label}")

    draw_strip(axes[2, 0], summary, "measured_peak", "cluster EPSP peak (mV), 1st response", groups)
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 0].set_title("cluster EPSP, all sets\n" + _test_text(_find_test(tests, measure="measured_peak",
                                                                            comparison="two groups")), fontsize=9)

    # single spines: small dots = spines, big hollow = mean of the set
    one = axes[2, 1]
    rng = np.random.default_rng(1)
    for position, group in enumerate(groups):
        spines = unitary[unitary["group"].astype(str) == group]["peak"].dropna()
        one.scatter(position - 0.15 + rng.uniform(-0.08, 0.08, len(spines)), spines, s=10, color=group_colour(group),
                    alpha=0.5, linewidth=0, zorder=2)
    draw_strip(one, summary, "unitary_peak_mean", "single-spine EPSP peak (mV)", groups, offset=0.15, hollow=True)
    one.set_ylim(bottom=0)
    n_spines = unitary.groupby(unitary["group"].astype(str))["peak"].count()
    one.set_title("single spines (small: each spine; ○ mean per set)\n"
                  + ", ".join(f"{g}: {n_spines.get(g, 0)} spines" for g in groups) + "\n"
                  + _test_text(_find_test(tests, measure="unitary_peak_mean", comparison="two groups"), "set means, "),
                  fontsize=8.5)

    one = axes[2, 2]
    gain = with_sum["measured_peak"] / with_sum["linear_sum_peak"]
    for group in groups:
        part = with_sum[with_sum["group"].astype(str) == group]
        for n, by_size in part.groupby("n_spines"):
            one.scatter(by_size["cluster_spread_um"], gain[by_size.index], s=36, color=group_colour(group),
                        marker=N_SPINES_MARKERS.get(int(n), "D"), edgecolor="white", linewidth=0.7, zorder=3)
    one.axhline(1, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
    one.set_xlabel("cluster spread (µm)")
    one.set_ylabel("gain of peak (shifted)")
    _legend_groups(one, [g for g in groups if (with_sum["group"].astype(str) == g).any()])
    one.set_title(f"gain vs cluster spread ({n_text})\n"
                  + _test_text(_find_test(tests, measure="gain vs cluster spread")), fontsize=9)
    n_without = len(summary) - len(with_sum)
    note = (f"Linear sum needs a response from every single spine of the cluster: {n_text}; the other {n_without} "
            "lack ≥ 1 single spine (unselected in the sheet or never recorded). ● 5 spines, ■ 6 spines.")
    _title(figure, title, note)
    return figure


def _isi_rows_figure(pulses: pd.DataFrame, rows: dict, title: str, dashed: dict | None = None,
                     ylims: dict | None = None):
    """Rows = columns of ``pulses`` against ISI; columns = conditions + medians of all."""
    from matplotlib.figure import Figure

    groups = present_groups(pulses)
    figure = Figure(figsize=(3.3 * (len(groups) + 1), 2.9 * len(rows) + 0.6), layout="constrained")
    axes = figure.subplots(len(rows), len(groups) + 1, sharex=True, squeeze=False)
    for r, (column, label) in enumerate(rows.items()):
        for c, group in enumerate(groups + [None]):
            one = axes[r, c]
            draw_ratio_vs_isi(one, pulses, column, [group] if group else groups, show_sets=group is not None)
            if dashed and column in dashed:
                for g in [group] if group else groups:
                    part = pulses[pulses["group"].astype(str) == g]
                    summary = median_and_iqr(part, dashed[column])
                    one.plot(summary["isi_ms"], summary["median"], color=group_colour(g), linestyle="--",
                             linewidth=1.2, zorder=3)
            if ylims and column in ylims:
                one.set_ylim(*ylims[column])
            if r == 0:
                n = pulses.loc[pulses["group"].astype(str) == group, "set"].nunique() if group else None
                one.set_title(f"{group} · {n} sets" if group else "medians (band: 25-75 %)", fontsize=10)
            if r < len(rows) - 1:
                one.set_xlabel("")
            if c > 0:
                one.set_yticklabels([])
        axes[r, 0].set_ylabel(label, fontsize=9)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="lower right")
    figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    return figure


def temporal_summation_figure(pulses: pd.DataFrame, title: str = ""):
    """``pulses``: pulse_pair_table. Thin lines = sets; thick = median per ISI."""
    return _isi_rows_figure(pulses, TEMPORAL_ROWS, title)


DECAY_TICKS = [10, 20, 50, 100, 200, 400]


def draw_decay_vs_isi_then_epsp1(axes, pulses: pd.DataFrame, groups, show_sets: bool = True) -> None:
    """
    Decay tau of EPSP2 at every ISI (equally spaced, short to long), then - after a gap
    and a dashed line - the decay tau of EPSP1 (the set's isolated-EPSP template, one value
    per set). Thin lines / small dots = sets; thick line + band and ◆ + bar = median and
    25-75 %.
    """
    isis = sorted(pulses["isi_ms"].unique())
    position = {isi: k for k, isi in enumerate(isis)}
    epsp1_x = len(isis) + 0.6
    rng = np.random.default_rng(2)
    for group in groups:
        part = pulses[pulses["group"].astype(str) == group]
        if part.empty:
            continue
        colour = group_colour(group)
        tau_1 = part.groupby("set", observed=True)["tau_1"].first()
        if show_sets:
            for _, one_set in part.groupby("set", observed=True):
                one_set = one_set.sort_values("isi_ms")
                axes.plot(one_set["isi_ms"].map(position), one_set["tau_2"], color=colour, linewidth=0.7, alpha=0.35)
            axes.scatter(epsp1_x + rng.uniform(-0.15, 0.15, len(tau_1)), tau_1, s=12, color=colour, alpha=0.5,
                         linewidth=0, zorder=2)
        summary = median_and_iqr(part, "tau_2")
        x = summary["isi_ms"].map(position)
        axes.fill_between(x, summary["q1"], summary["q3"], color=colour, alpha=0.15, linewidth=0, zorder=1)
        axes.plot(x, summary["median"], color=colour, linewidth=2, marker="o", markersize=3.5, zorder=3,
                  label=f"{group} (n = {part['set'].nunique()} sets)")
        q1, median, q3 = np.percentile(tau_1, [25, 50, 75])
        axes.errorbar(epsp1_x, median, yerr=[[median - q1], [q3 - median]], color=colour, marker="D", markersize=6,
                      capsize=3, linewidth=1.5, zorder=4)
    axes.axvline(len(isis) - 0.2, color="0.5", linestyle="--", linewidth=0.8)
    axes.set_xticks(list(range(len(isis))) + [epsp1_x], [f"{isi:g}" for isi in isis] + ["EPSP1"], fontsize=8)
    axes.set_xlim(-0.5, epsp1_x + 0.6)
    axes.set_xlabel("ISI (ms) of EPSP2  |  EPSP1")
    axes.set_yscale("log")
    axes.set_ylim(DECAY_TICKS[0], DECAY_TICKS[-1])
    axes.set_yticks(DECAY_TICKS, [f"{t:g}" for t in DECAY_TICKS])
    axes.minorticks_off()


def kinetics_figure(pulses: pd.DataFrame, title: str = ""):
    """
    Top: decay tau of EPSP2 against ISI followed by EPSP1's, one panel per condition +
    medians. Below: decay tau ratio and 20-80 % rise ratio (EPSP2 / EPSP1) against ISI.
    """
    from matplotlib.figure import Figure

    groups = present_groups(pulses)
    figure = Figure(figsize=(3.6 * (len(groups) + 1), 10.0), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[1.0, 1.9])
    decay = top.subplots(1, len(groups) + 1, sharey=True, squeeze=False)[0]
    for c, group in enumerate(groups + [None]):
        draw_decay_vs_isi_then_epsp1(decay[c], pulses, [group] if group else groups, show_sets=group is not None)
        n = pulses.loc[pulses["group"].astype(str) == group, "set"].nunique() if group else None
        decay[c].set_title(f"{group} · {n} sets" if group else "medians (band / bar: 25-75 %)", fontsize=10)
    decay[0].set_ylabel("decay τ (ms)")
    decay[-1].legend(fontsize=7, frameon=False, loc="lower left")
    ratios = bottom.subplots(len(KINETIC_ROWS), len(groups) + 1, sharex=True, sharey=True, squeeze=False)
    for r, (column, label) in enumerate(KINETIC_ROWS.items()):
        for c, group in enumerate(groups + [None]):
            draw_ratio_vs_isi(ratios[r, c], pulses, column, [group] if group else groups, show_sets=group is not None)
            if r < len(KINETIC_ROWS) - 1:
                ratios[r, c].set_xlabel("")
        ratios[r, 0].set_ylabel(label, fontsize=9)
    ratios[0, -1].legend(fontsize=7, frameon=False, loc="lower right")
    _title(figure, title, "EPSP1 has the set's template shape at every ISI (only amplitude and latency fitted), "
                          "so its τ and rise are one value per set; EPSP2 is fitted freely at each ISI.")
    return figure


def pre_post_temporal_figure(pulse_pairs: pd.DataFrame, summary_pairs: pd.DataFrame, measures: dict,
                             tests=None, title: str = ""):
    """Top: per dendrite, pre vs post curves (PPR of peak, temporal gain, decay ratio).
    Bottom: paired plots of ``measures`` (summary column -> label)."""
    from matplotlib.figure import Figure

    curve_rows = {"ppr_peak": "PPR of peak", "temporal_gain": "temporal gain", "tau_ratio": "decay τ ratio\n(2nd/1st)"}
    dendrites = sorted(pulse_pairs["dendrite"].unique())
    figure = Figure(figsize=(max(3.2 * len(dendrites), 2.6 * len(measures)), 13.0), layout="constrained")
    top, bottom = figure.subfigures(2, 1, height_ratios=[2.6, 1.0])
    curves = top.subplots(len(curve_rows), len(dendrites), sharex=True, sharey="row", squeeze=False)
    for c, dendrite in enumerate(dendrites):
        part = pulse_pairs[pulse_pairs["dendrite"] == dendrite]
        for r, (column, label) in enumerate(curve_rows.items()):
            draw_ratio_vs_isi(curves[r, c], part, column, ["pre-apv", "post-apv"], show_sets=False,
                              labels={"pre-apv": "pre-apv", "post-apv": "post-apv"})
            if c == 0:
                curves[r, c].set_ylabel(label, fontsize=9)
            if r < len(curve_rows) - 1:
                curves[r, c].set_xlabel("")
        sizes = ", ".join(str(int(n)) for n in sorted(part["n_spines"].unique()))
        curves[0, c].set_title(f"{dendrite} · {sizes} spines", fontsize=10)
    curves[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    paired = bottom.subplots(1, len(measures), squeeze=False)[0]
    for one, (column, label) in zip(paired, measures.items()):
        draw_paired(one, summary_pairs, column, label, _find_test(tests, measure=column))
    bottom.suptitle("one value per dendrite and condition; grey line = same dendrite, black bar = mean; "
                    "short = ISI ≤ 50 ms, long = ISI ≥ 500 ms", fontsize=9)
    figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    return figure


def pre_post_spatial_figure(summary_pairs: pd.DataFrame, unitary_pairs: pd.DataFrame, measures: dict,
                            tests=None, title: str = ""):
    """Paired plots of the spatial measures + the single spines recorded in both conditions."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2.9 * (len(measures) + 1.4), 4.6), layout="constrained")
    axes = figure.subplots(1, len(measures) + 1, squeeze=False, width_ratios=[1] * len(measures) + [1.4])[0]
    for one, (column, label) in zip(axes, measures.items()):
        draw_paired(one, summary_pairs, column, label, _find_test(tests, measure=column))
        n = summary_pairs.pivot_table(index="dendrite", columns="group", values=column, observed=True).dropna()
        if len(n) < 2:
            one.set_title(f"{len(n)} dendrite(s) with both: no test", fontsize=9)
    one = axes[-1]
    markers = ["o", "s", "^", "D", "v", "P"]
    shared = unitary_pairs.pivot_table(index=["dendrite", "spineid"], columns="group", values="peak",
                                       observed=True).dropna()
    shared = shared.sort_values("post-apv")
    span = max(float(shared.to_numpy().max()) if len(shared) else 1.0, 1e-9)
    label_y = -np.inf
    for (dendrite, spine), values in shared.iterrows():
        marker = markers[sorted(shared.index.get_level_values(0).unique()).index(dendrite) % len(markers)]
        one.plot([0, 1], values[["pre-apv", "post-apv"]].to_numpy(), color="0.7", linewidth=1, zorder=1)
        for x, g in enumerate(("pre-apv", "post-apv")):
            one.scatter(x, values[g], color=group_colour(g), marker=marker, s=30, edgecolor="white", zorder=3)
        label_y = max(values["post-apv"], label_y + 0.06 * span)      # labels never overlap
        one.annotate(f"{dendrite} spine {spine}", (1, values["post-apv"]), (1.08, label_y), fontsize=6.5,
                     va="center", arrowprops=dict(arrowstyle="-", color="0.75", linewidth=0.6))
    one.set_xticks([0, 1], ["pre-apv", "post-apv"])
    one.set_xlim(-0.4, 1.9)
    one.set_ylim(bottom=0)
    one.set_ylabel("single-spine EPSP peak (mV)")
    one.set_title(f"single spines recorded in both: {len(shared)}\n(descriptive, no test)", fontsize=9)
    figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    return figure


def relationships_figure(summary: pd.DataFrame, pulses: pd.DataFrame, tests=None, title: str = ""):
    """Per-set relationships (see module docstring); one dot per set, colour = condition."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14.5, 9.0), layout="constrained")
    axes = figure.subplots(2, 3)
    groups = present_groups(summary)
    scatter_panels = [(axes[0, 0], "residual_mv_short", "ppr_peak_short",
                       "depolarisation left at the 2nd pulse (mV)\nmean of ISI ≤ 50 ms", "PPR of peak, ISI ≤ 50 ms", True),
                      (axes[0, 1], "peak_1", "ppr_peak_short", "EPSP1 peak (mV)", "PPR of peak, ISI ≤ 50 ms", True),
                      (axes[0, 2], "peak_1", "decay_tau_ms_1", "EPSP1 peak (mV)", "EPSP1 decay τ (ms)", False)]
    for one, x, y, xlabel, ylabel, ratio in scatter_panels:
        for group in groups:
            part = summary[summary["group"].astype(str) == group]
            one.scatter(part[x], _clip_ratio(part[y])[0] if ratio else part[y], s=34, color=group_colour(group),
                        edgecolor="white", linewidth=0.6, zorder=3, label=group)
        if ratio:
            _ratio_axis(one)
            one.set_ylim(0.35, 1.6)
            one.set_yticks([0.5, 0.71, 1, 1.41], ["0.5", "0.71", "1", "1.41"])
        one.set_xlabel(xlabel)
        one.set_ylabel(ylabel)
        lines = [f"{g}: ρ = {row['centre']:.2f}, {format_p(row['p'])}" for g in ["all"] + groups
                 if (row := _find_test(tests, measure=f"{y} vs {x}", group=g)) is not None and np.isfinite(row["p"])]
        one.set_title("Spearman " + "\n".join(lines), fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    draw_strip(axes[1, 0], summary, "decay_tau_ms_1", "EPSP1 decay τ (ms)", groups)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].set_title("EPSP1 decay τ (isolated EPSP template)\n" + _test_text(
        _find_test(tests, measure="decay_tau_ms_1", comparison="two groups")), fontsize=9)
    draw_strip(axes[1, 1], summary, "baseline_rest_mv", "resting potential (mV)", groups)
    axes[1, 1].set_title("potential before the 1st stimulus\n" + _test_text(
        _find_test(tests, measure="resting potential (mV)")), fontsize=9)
    one = axes[1, 2]
    for group in groups:
        part = pulses[pulses["group"].astype(str) == group]
        stats_ = median_and_iqr(part, "r_squared")
        one.plot(stats_["isi_ms"], stats_["median"], color=group_colour(group), marker="o", markersize=3.5,
                 linewidth=1.8, label=group)
        one.fill_between(stats_["isi_ms"], stats_["q1"], stats_["q3"], color=group_colour(group), alpha=0.12,
                         linewidth=0)
    _isi_axis(one)
    one.set_ylim(0.5, 1.0)
    one.set_ylabel("deconvolution fit R²")
    one.set_title("fit quality (median, band 25-75 %)", fontsize=9)
    one.legend(fontsize=7, frameon=False, loc="lower right")
    figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    return figure
