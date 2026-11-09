"""
traces_by_isi_viewer.py
=======================

Shows the traces of ONE spine grouped by inter-stimulus interval (ISI):
one panel per ISI, shortest to longest.

In every panel
--------------
* thin lines  : every good sweep (bad sweeps from the sheet are excluded),
* thick line  : the average of those sweeps, one colour per group
                (e.g. pre-apv / post-apv),
* shaded bands: the stimulus pulses; time 0 = first pulse,
* title       : ISI and number of averaged sweeps per group, e.g.
                "30 ms — pre-apv (3) · post-apv (3)".
All panels share the same y-axis by default, so amplitudes can be compared.

Header above the panels
-----------------------
    date 20260322 | neuron 1 | dendrite 1 | spine 0 | location hc ca3 basal | stim power 30 | ...
    pre-apv: 30 ms (3), 40 ms (3), ... | post-apv: 30 ms (3), ...
and any warnings (files not found, sheet ISI different from the file, ...).

The data come from ``experiment_catalog.traces_by_isi.collect_traces_by_isi``;
this module only draws. Use ``ExperimentCatalog.show_traces_by_isi(...)`` /
``save_traces_by_isi_figure(...)`` rather than calling these functions directly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..experiment_catalog.traces_by_isi import TracesByIsi
from .display_helpers import COLORBLIND_SAFE_COLORS, min_max_downsample


def _group_colors(groups: list[str]) -> dict[str, str]:
    return {group: COLORBLIND_SAFE_COLORS[i % len(COLORBLIND_SAFE_COLORS)] for i, group in enumerate(groups)}


def _panel_title(data: TracesByIsi, isi_ms: float, groups: list[str]) -> str:
    counts = " · ".join(
        f"{group or 'all'} ({data.sweep_count(isi_ms, group)})" for group in groups if data.sweep_count(isi_ms, group)
    )
    return f"{data.format_isi(isi_ms)} — {counts}"


def _y_label(data: TracesByIsi) -> str:
    labels = dict.fromkeys(f"{s.channel} ({s.unit})" for s in data.trace_sets)
    return " / ".join(labels)


def _shared_y_limits(data: TracesByIsi, isis: list[float], groups: list[str], show_sweeps: bool) -> tuple[float, float]:
    """Common y-range for all panels: covers every drawn line plus 5 % padding."""
    values = []
    for isi in isis:
        for group in groups:
            for trace_set in data.sets_for(isi, group):
                values.append(trace_set.traces if show_sweeps else trace_set.traces.mean(axis=0, keepdims=True))
    if not values:
        return -1.0, 1.0
    low = min(float(v.min()) for v in values)
    high = max(float(v.max()) for v in values)
    padding = 0.05 * (high - low or 1.0)
    return low - padding, high + padding


def _selected(data: TracesByIsi, isis, groups) -> tuple[list[float], list[str]]:
    """Requested ISIs/groups that actually have traces (defaults: all)."""
    isis = data.isis if not isis else [isi for isi in sorted(float(i) for i in isis) if isi in data.isis]
    groups = data.groups if not groups else [group for group in groups if group in data.groups]
    return isis, groups


# =============================================================================
# Interactive (Bokeh + Panel)
# =============================================================================
def build_traces_by_isi_figure(
    data: TracesByIsi,
    isis: list[float] | None = None,
    groups: list[str] | None = None,
    show_sweeps: bool = True,
    shared_y_axis: bool = True,
    columns: int = 3,
    panel_width: int = 330,
    panel_height: int = 230,
):
    """Grid of Bokeh panels, one per ISI (see module docstring)."""
    from bokeh.layouts import gridplot
    from bokeh.models import BoxAnnotation, Range1d
    from bokeh.plotting import figure

    isis, groups = _selected(data, isis, groups)
    colors = _group_colors(data.groups)
    shared_y = Range1d(*_shared_y_limits(data, isis, groups, show_sweeps)) if shared_y_axis else None

    panels = []
    for panel_number, isi in enumerate(isis):
        sets = [s for s in data.sets_for(isi) if s.group in groups]
        time_ms = sets[0].time_ms
        panel = figure(
            width=panel_width,
            height=panel_height,
            title=_panel_title(data, isi, groups),
            x_axis_label="time from 1st stimulus (ms)",
            y_axis_label=_y_label(data) if panel_number % columns == 0 else None,
            x_range=Range1d(float(time_ms[0]), float(time_ms[-1])),
            y_range=shared_y if shared_y is not None else None,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            output_backend="webgl",
        )

        # Stimulus pulses (drawn at least 0.8 % of the window wide to be visible).
        band_width = 0.008 * float(time_ms[-1] - time_ms[0])
        for onset in sets[0].stimulus_onsets_ms:
            panel.add_layout(BoxAnnotation(left=float(onset), right=float(onset) + band_width,
                                           fill_color="grey", fill_alpha=0.25, line_alpha=0))

        for group in groups:
            group_sets = [s for s in sets if s.group == group]
            if not group_sets:
                continue
            if show_sweeps:
                for trace_set in group_sets:
                    for trace in trace_set.traces:
                        t_plot, y_plot = min_max_downsample(trace_set.time_ms, trace, 2000)
                        panel.line(t_plot, y_plot, line_color=colors[group], line_alpha=0.25, line_width=1)
            mean_time, mean_trace, n_sweeps = data.mean_trace(isi, group)
            t_plot, y_plot = min_max_downsample(mean_time, mean_trace, 2000)
            # Legend only in the first panel (Bokeh refuses legend_label=None).
            legend = {"legend_label": group or "mean"} if panel_number == 0 else {}
            panel.line(t_plot, y_plot, line_color=colors[group], line_width=2.2, **legend)
        if panel_number == 0 and panel.legend:
            panel.legend.location = "top_right"
            panel.legend.background_fill_alpha = 0.6
            panel.legend.label_text_font_size = "9pt"
        panel.title.text_font_size = "10pt"
        panel.grid.grid_line_alpha = 0.3
        panels.append(panel)

    return gridplot(panels, ncols=columns, toolbar_location="right")


def _header_markdown(data: TracesByIsi, isis, groups) -> str:
    text = f"**{data.metadata_text()}**\n\n**Averaged sweeps per ISI:** {data.isi_summary_text(isis, groups)}"
    if data.warnings:
        text += f"\n\n⚠ {len(data.warnings)} warning(s):\n" + "\n".join(f"- {w}" for w in data.warnings)
    return text


def build_traces_by_isi_viewer(
    data: TracesByIsi,
    isis: list[float] | None = None,
    groups: list[str] | None = None,
    show_sweeps: bool = True,
    shared_y_axis: bool = True,
    columns: int = 3,
):
    """
    Interactive view: header + controls (ISIs, groups, sweeps on/off, shared
    y-axis, number of columns) + panel grid. The arguments set the initial state.
    """
    import panel as pn

    pn.extension()

    initial_isis, initial_groups = _selected(data, isis, groups)
    isi_selector = pn.widgets.MultiChoice(
        label="ISIs (ms)",
        options={data.format_isi(isi): isi for isi in data.isis},
        value=initial_isis,
        width=420,
    )
    group_selector = pn.widgets.CheckBoxGroup(label="Groups", options=data.groups, value=initial_groups, inline=True)
    sweeps_checkbox = pn.widgets.Checkbox(label="Show individual sweeps", value=show_sweeps)
    shared_y_checkbox = pn.widgets.Checkbox(label="Same y-axis for all ISIs", value=shared_y_axis)
    columns_input = pn.widgets.IntInput(label="Panels per row", value=columns, start=1, end=6, width=120)

    def header_for(selected_isis, selected_groups):
        return pn.pane.Markdown(_header_markdown(data, selected_isis, selected_groups))

    def grid_for(selected_isis, selected_groups, sweeps_on, shared_on, n_columns):
        if not selected_isis or not selected_groups:
            return pn.pane.Markdown("*Select at least one ISI and one group.*")
        return build_traces_by_isi_figure(
            data, selected_isis, selected_groups, sweeps_on, shared_on, n_columns
        )

    return pn.Column(
        pn.bind(header_for, isi_selector, group_selector),
        pn.Row(isi_selector, pn.Column(group_selector, sweeps_checkbox, shared_y_checkbox), columns_input),
        pn.bind(grid_for, isi_selector, group_selector, sweeps_checkbox, shared_y_checkbox, columns_input),
    )


# =============================================================================
# Static figure (matplotlib)
# =============================================================================
def save_traces_by_isi_figure(
    data: TracesByIsi,
    output_path: str | Path,
    isis: list[float] | None = None,
    groups: list[str] | None = None,
    show_sweeps: bool = True,
    shared_y_axis: bool = True,
    columns: int = 4,
    panel_size_in: tuple[float, float] = (3.0, 2.2),
    dpi: int = 300,
) -> Path:
    """Same content as the interactive view, saved as PNG / SVG / PDF."""
    import textwrap

    import matplotlib
    from matplotlib.figure import Figure

    from .save_static_figure import FIGURE_STYLE, _save

    isis, groups = _selected(data, isis, groups)
    if not isis:
        raise ValueError("No traces to plot for the selected ISIs / groups.")
    colors = _group_colors(data.groups)
    n_rows = math.ceil(len(isis) / columns)
    y_limits = _shared_y_limits(data, isis, groups, show_sweeps) if shared_y_axis else None

    with matplotlib.rc_context(FIGURE_STYLE):
        # "constrained" layout reserves exactly the space the header needs.
        figure = Figure(figsize=(panel_size_in[0] * columns, panel_size_in[1] * n_rows + 0.8), layout="constrained")
        axes_grid = figure.subplots(n_rows, columns, squeeze=False)
        for index, axes in enumerate(axes_grid.flat):
            if index >= len(isis):
                axes.set_visible(False)
                continue
            isi = isis[index]
            sets = [s for s in data.sets_for(isi) if s.group in groups]
            for onset in sets[0].stimulus_onsets_ms:
                axes.axvline(onset, color="0.6", linewidth=3, alpha=0.4, zorder=0)
            for group in groups:
                group_sets = [s for s in sets if s.group == group]
                if not group_sets:
                    continue
                if show_sweeps:
                    for trace_set in group_sets:
                        axes.plot(trace_set.time_ms, trace_set.traces.T, color=colors[group], alpha=0.25, linewidth=0.5)
                mean_time, mean_trace, _ = data.mean_trace(isi, group)
                axes.plot(mean_time, mean_trace, color=colors[group], linewidth=1.3, label=group or "mean")
            axes.set_title(_panel_title(data, isi, groups), fontsize=8)
            axes.set_xlim(sets[0].time_ms[0], sets[0].time_ms[-1])
            if y_limits:
                axes.set_ylim(y_limits)
            if index % columns == 0:
                axes.set_ylabel(_y_label(data))
            if index >= len(isis) - columns:
                axes.set_xlabel("time from 1st stimulus (ms)")
            if index == 0:
                axes.legend(fontsize=7, frameon=False)

        header_lines = [data.metadata_text(), "Averaged sweeps per ISI: " + data.isi_summary_text(isis, groups)]
        wrapped = [piece for line in header_lines for piece in textwrap.wrap(line, 160, break_long_words=False)]
        figure.suptitle("\n".join(wrapped), fontsize=8, x=0.01, ha="left")
        return _save(figure, output_path, dpi)
