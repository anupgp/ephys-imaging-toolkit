"""
experiment_traces_viewer.py
===========================

Shows ALL traces of one experiment sample (expdate + neuron + dendrite, and
optionally one spine), aligned to the first stimulus (time 0), with optional
averaging by sheet columns such as the ISI.

What is drawn
-------------
* average_by = None / []      : every good sweep as a thin line, coloured by
                                ``color_by`` (default: ISI); when several groups
                                (e.g. ctrl / ctz) are present they get different
                                line styles (solid, dashed, ...).
* average_by = "isi"          : one thick MEAN line per ISI (optionally with the
                                single sweeps faintly behind it).
* average_by = ["isi","group"]: one mean per ISI and group; colour = first column,
                                line style = second column.
* panels                      : one per value of ``panel_by``; by default one per
                                spine when no spine was given, else a single panel.
                                Voltage- and current-clamp recordings (pA / mV)
                                always get separate panels.
* baseline_subtract=True      : each sweep minus its mean before the first stimulus
                                (for sweeps recorded at different holding levels).
* stimulus                    : grey band at 0 (first pulse); the later pulses of
                                each line as thin dashed lines in its colour.
* legend                      : "<values> (<number of sweeps>)"; click an entry
                                to hide / show it.

The data come from ``experiment_catalog.experiment_traces.collect_experiment_traces``;
use ``ExperimentCatalog.show_experiment_traces(...)`` /
``save_experiment_traces_figure(...)`` rather than calling these functions directly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..experiment_catalog.experiment_traces import AveragedTraces, ExperimentTraces, _sort_key, format_value
from .display_helpers import min_max_downsample

# Columns offered in the "average by" / "colour by" / "panels" choices of the interactive view.
CHOICE_COLUMNS = ["isi", "group", "spineid", "dendriteid", "stim_power_percent", "expname"]
BOKEH_DASHES = ["solid", "dashed", "dotted", "dotdash", "dashdot"]
MPL_DASHES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]


# =============================================================================
# Shared helpers
# =============================================================================
def _as_list(columns) -> list[str]:
    if columns is None:
        return []
    return [columns] if isinstance(columns, str) else list(columns)


def _line_columns(data: ExperimentTraces, average_by, color_by: str) -> list[str]:
    """
    Columns that define the drawn lines: the ``average_by`` columns, or (no
    averaging) the colour column plus "group" when several groups are present.
    """
    average_by = _as_list(average_by)
    if average_by:
        return average_by
    columns = [color_by]
    if color_by != "group" and len(data.values("group")) > 1:
        columns.append("group")
    return columns


# Distinct colours for telling many lines apart (e.g. 11-12 ISIs), after Kelly's
# "colours of maximum contrast", ordered so that NEIGHBOURING values (10 / 20 / 30 ms)
# never get similar hues, and without pale colours that vanish on white.
DISTINCT_LINE_COLORS = [
    "#0067A5",  # blue
    "#F38400",  # orange
    "#008856",  # green
    "#BE0032",  # red
    "#875692",  # purple
    "#56B4E9",  # sky blue
    "#882D17",  # brown
    "#E68FAC",  # pink
    "#8DB600",  # yellow-green
    "#222222",  # black
    "#D4A900",  # dark yellow
    "#848482",  # grey
    "#2B3D26",  # dark olive
    "#F99379",  # salmon
    "#604E97",  # violet
    "#B3446C",  # raspberry
    "#C2B280",  # buff
    "#E25822",  # red-orange
    "#654522",  # olive brown
    "#A1CAF1",  # light blue
]


def _palette(n: int) -> list[str]:
    """n clearly different colours (repeats only beyond 20 values)."""
    return [DISTINCT_LINE_COLORS[i % len(DISTINCT_LINE_COLORS)] for i in range(n)]


def _styles(data: ExperimentTraces, columns: list[str]) -> tuple[dict, dict]:
    """value -> colour (first column) and value -> dash index (second column), fixed over all panels."""
    colours = {}
    if columns:
        values = data.values(columns[0])
        colours = dict(zip(values, _palette(len(values))))
    dashes = {}
    if len(columns) > 1:
        dashes = {value: i % len(BOKEH_DASHES) for i, value in enumerate(data.values(columns[1]))}
    return colours, dashes


def _style_of(line: AveragedTraces, columns: list[str], colours: dict, dashes: dict) -> tuple[str, int]:
    colour = colours.get(line.key.get(columns[0]), "black") if columns else "black"
    dash = dashes.get(line.key.get(columns[1]), 0) if len(columns) > 1 else 0
    return colour, dash


def _panels(data: ExperimentTraces, panel_by: str | None) -> list[tuple[str, ExperimentTraces]]:
    """
    (title, subset) per panel. Recordings with different units (voltage clamp in
    pA, current clamp in mV) always get separate panels.
    """
    panels = [("", data)] if not panel_by else [
        (format_value(panel_by, value), data.subset(**{panel_by: [value]})) for value in data.values(panel_by)
    ]
    if len(data.units) < 2:
        return panels
    split = []
    for title, subset in panels:
        for unit in subset.units:
            only_unit = ExperimentTraces([s for s in subset.trace_sets if s.traces.unit == unit],
                                         subset.criteria, subset.metadata, subset.warnings)
            split.append((f"{title} · {unit}" if title else unit, only_unit))
    return split


def _resolve_panel_by(data: ExperimentTraces, panel_by):
    """'auto' -> one panel per spine when the data contain several spines, else one panel."""
    if panel_by == "auto":
        return "spineid" if len(data.values("spineid")) > 1 else None
    return panel_by


def _panel_title(title: str, subset: ExperimentTraces) -> str:
    counts = f"{len(subset.trace_sets)} files, {subset.n_sweeps} sweeps"
    return f"{title} — {counts}" if title else counts


def _y_label(data: ExperimentTraces) -> str:
    return " / ".join(dict.fromkeys(f"{s.traces.channel} ({s.traces.unit})" for s in data.trace_sets))


def _y_limits(lines_per_panel: list[list[AveragedTraces]], draw_sweeps: bool, averaging: bool) -> tuple[float, float]:
    """Common y-range: every drawn line plus 5 % padding."""
    values = [line.sweeps if (draw_sweeps or not averaging) else line.mean[None, :]
              for lines in lines_per_panel for line in lines]
    values = [v[np.isfinite(v)] for v in values]
    values = [v for v in values if v.size]
    if not values:
        return -1.0, 1.0
    low, high = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
    padding = 0.05 * (high - low or 1.0)
    return low - padding, high + padding


def _limits_per_panel(panels_data, lines_per_panel, draw_sweeps: bool, averaging: bool) -> list[tuple[float, float]]:
    """Shared y-range per panel, computed over all panels with the SAME unit (pA and mV never share an axis)."""
    by_unit: dict[str, list] = {}
    for (_, subset), lines in zip(panels_data, lines_per_panel):
        by_unit.setdefault("/".join(subset.units), []).append(lines)
    limits = {unit: _y_limits(groups, draw_sweeps, averaging) for unit, groups in by_unit.items()}
    return [limits["/".join(subset.units)] for _, subset in panels_data]


def summary_text(data: ExperimentTraces, average_by) -> str:
    """'Averaged by isi: 10 ms (4), 20 ms (4), ...' or 'All sweeps: ...'."""
    average_by = _as_list(average_by)
    lines = data.grouped(average_by) if average_by else []
    if not average_by:
        return "No averaging: every sweep is shown"
    return f"Averaged by {' + '.join(average_by)}: " + ", ".join(f"{l.label} ({l.n_sweeps})" for l in lines)


# =============================================================================
# Interactive (Bokeh + Panel)
# =============================================================================
def build_experiment_traces_figure(
    data: ExperimentTraces,
    average_by=None,
    color_by: str = "isi",
    panel_by="auto",
    show_sweeps: bool | None = None,
    baseline_subtract: bool = False,
    shared_y_axis: bool = True,
    columns: int = 2,
    panel_width: int = 520,
    panel_height: int = 300,
):
    """Grid of Bokeh panels (see module docstring)."""
    if baseline_subtract:
        data = data.baseline_subtracted()
    from bokeh.layouts import gridplot
    from bokeh.models import BoxAnnotation, Range1d, Span
    from bokeh.plotting import figure

    averaging = bool(_as_list(average_by))
    draw_sweeps = (not averaging) if show_sweeps is None else show_sweeps
    line_columns = _line_columns(data, average_by, color_by)
    colours, dashes = _styles(data, line_columns)
    panel_by = _resolve_panel_by(data, panel_by)
    panels_data = _panels(data, panel_by)
    lines_per_panel = [subset.grouped(line_columns) for _, subset in panels_data]
    shared_ranges = {}   # one Range1d per y-range, so panels with the same unit zoom together
    for limits in _limits_per_panel(panels_data, lines_per_panel, draw_sweeps, averaging):
        shared_ranges.setdefault(limits, Range1d(*limits))
    panel_limits = _limits_per_panel(panels_data, lines_per_panel, draw_sweeps, averaging)

    figures = []
    for number, ((title, subset), lines) in enumerate(zip(panels_data, lines_per_panel)):
        if not lines:
            continue
        start = min(float(l.time_ms[0]) for l in lines)
        stop = max(float(l.time_ms[-1]) for l in lines)
        panel = figure(
            width=panel_width, height=panel_height, title=_panel_title(title, subset),
            x_axis_label="time from 1st stimulus (ms)",
            y_axis_label=_y_label(subset) if number % columns == 0 or len(data.units) > 1 else None,
            x_range=Range1d(start, stop), y_range=shared_ranges[panel_limits[number]] if shared_y_axis else None,
            tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom", output_backend="webgl",
        )
        panel.add_layout(BoxAnnotation(left=0, right=max(0.004 * (stop - start), 0.5),
                                       fill_color="grey", fill_alpha=0.3, line_alpha=0))
        marked = set()
        for line in lines:
            colour, dash = _style_of(line, line_columns, colours, dashes)
            legend = {"legend_label": f"{line.label} ({line.n_sweeps})"}
            for onset in line.stimulus_onsets_ms:          # later pulses, in the line's colour
                if onset > 0 and (onset, colour) not in marked:
                    marked.add((onset, colour))
                    panel.add_layout(Span(location=onset, dimension="height", line_color=colour,
                                          line_dash="dashed", line_alpha=0.5, line_width=1))
            if draw_sweeps:
                for sweep in line.sweeps:
                    t, y = min_max_downsample(line.time_ms, sweep, 2000)
                    panel.line(t, y, line_color=colour, line_dash=BOKEH_DASHES[dash],
                               line_alpha=0.25 if averaging else 0.6, line_width=1, **({} if averaging else legend))
            if averaging:
                t, y = min_max_downsample(line.time_ms, line.mean, 2000)
                panel.line(t, y, line_color=colour, line_dash=BOKEH_DASHES[dash], line_width=2.5, **legend)
        legend_box = panel.legend[0]
        legend_box.click_policy = "hide"
        legend_box.label_text_font_size = "8pt"
        legend_box.spacing = 0
        panel.add_layout(legend_box, "right")
        panel.title.text_font_size = "10pt"
        panel.grid.grid_line_alpha = 0.3
        figures.append(panel)
    return gridplot(figures, ncols=columns, toolbar_location="right")


def _header_markdown(data: ExperimentTraces, average_by) -> str:
    text = f"**{data.header_text()}**\n\n{summary_text(data, average_by)}"
    if data.warnings:
        text += f"\n\n⚠ {len(data.warnings)} warning(s):\n" + "\n".join(f"- {w}" for w in data.warnings)
    return text


def build_experiment_traces_viewer(
    data: ExperimentTraces,
    average_by=None,
    color_by: str = "isi",
    panel_by="auto",
    show_sweeps: bool | None = None,
    baseline_subtract: bool = False,
    shared_y_axis: bool = True,
    columns: int = 2,
):
    """
    Interactive view: header + controls + panel grid. The arguments set the
    initial state; everything can be changed with the controls without reloading files.
    """
    import panel as pn

    pn.extension()
    if not data.trace_sets:
        return pn.pane.Markdown(_header_markdown(data, None) + "\n\n*No traces could be loaded.*")

    # Only offer columns that actually vary in these data (plus isi/group always).
    offered = [c for c in CHOICE_COLUMNS if c in ("isi", "group") or len(data.values(c)) > 1]
    average_input = pn.widgets.MultiChoice(label="Average by (empty = all sweeps)", options=offered,
                                           value=_as_list(average_by), width=260)
    colour_input = pn.widgets.Select(label="Colour by (no averaging)", options=offered,
                                     value=color_by if color_by in offered else "isi", width=170)
    panel_options = {"one panel": None, **{c: c for c in offered}}
    resolved_panel = _resolve_panel_by(data, panel_by)
    panel_input = pn.widgets.Select(label="Panels by", options=panel_options, value=resolved_panel, width=140)
    isi_input = pn.widgets.MultiChoice(label="ISIs (empty = all)", options={format_value("isi", v): v for v in data.values("isi")},
                                       value=[], width=330)
    group_input = pn.widgets.CheckBoxGroup(label="Groups", options=[str(g) for g in data.values("group")],
                                           value=[str(g) for g in data.values("group")], inline=True)
    sweeps_input = pn.widgets.Checkbox(label="Show single sweeps behind averages",
                                       value=bool(show_sweeps) if _as_list(average_by) else True)
    baseline_input = pn.widgets.Checkbox(label="Subtract baseline (mean before 1st stimulus)", value=baseline_subtract)
    shared_input = pn.widgets.Checkbox(label="Same y-axis for all panels", value=shared_y_axis)
    columns_input = pn.widgets.IntInput(label="Panels per row", value=columns, start=1, end=6, width=110)

    def filtered(isis, groups) -> ExperimentTraces:
        return data.subset(isi=isis or None, group=groups)

    def header_for(isis, groups, average):
        return pn.pane.Markdown(_header_markdown(filtered(isis, groups), average))

    def grid_for(isis, groups, average, colour, panels, sweeps, baseline, shared, n_columns):
        subset = filtered(isis, groups)
        if not subset.trace_sets:
            return pn.pane.Markdown("*Nothing selected.*")
        return build_experiment_traces_figure(subset, average, colour, panels, sweeps if average else True,
                                              baseline, shared, n_columns)

    controls = pn.Row(
        pn.Column(average_input, isi_input),
        pn.Column(colour_input, panel_input),
        pn.Column(group_input, sweeps_input, baseline_input, shared_input, columns_input),
    )
    return pn.Column(
        pn.bind(header_for, isi_input, group_input, average_input),
        controls,
        pn.bind(grid_for, isi_input, group_input, average_input, colour_input, panel_input,
                sweeps_input, baseline_input, shared_input, columns_input),
    )


# =============================================================================
# Static figure (matplotlib)
# =============================================================================
def save_experiment_traces_figure(
    data: ExperimentTraces,
    output_path: str | Path,
    average_by=None,
    color_by: str = "isi",
    panel_by="auto",
    show_sweeps: bool | None = None,
    baseline_subtract: bool = False,
    shared_y_axis: bool = True,
    columns: int = 2,
    panel_size_in: tuple[float, float] = (5.0, 3.0),
    dpi: int = 300,
) -> Path:
    """Same content as the interactive view, saved as PNG / SVG / PDF."""
    import textwrap

    import matplotlib
    from matplotlib.figure import Figure

    from .save_static_figure import FIGURE_STYLE, _save

    if not data.trace_sets:
        raise ValueError("No traces to plot.")
    if baseline_subtract:
        data = data.baseline_subtracted()
    averaging = bool(_as_list(average_by))
    draw_sweeps = (not averaging) if show_sweeps is None else show_sweeps
    line_columns = _line_columns(data, average_by, color_by)
    colours, dashes = _styles(data, line_columns)
    panels_data = _panels(data, _resolve_panel_by(data, panel_by))
    lines_per_panel = [subset.grouped(line_columns) for _, subset in panels_data]
    panel_limits = _limits_per_panel(panels_data, lines_per_panel, draw_sweeps, averaging) if shared_y_axis else None
    columns = min(columns, len(panels_data))
    n_rows = math.ceil(len(panels_data) / columns)

    with matplotlib.rc_context(FIGURE_STYLE):
        figure = Figure(figsize=(panel_size_in[0] * columns + 1.5, panel_size_in[1] * n_rows + 0.9),
                        layout="constrained")
        axes_grid = figure.subplots(n_rows, columns, squeeze=False)
        for index, axes in enumerate(axes_grid.flat):
            if index >= len(panels_data):
                axes.set_visible(False)
                continue
            (title, subset), lines = panels_data[index], lines_per_panel[index]
            axes.axvline(0, color="0.6", linewidth=3, alpha=0.4, zorder=0)
            for line in lines:
                colour, dash = _style_of(line, line_columns, colours, dashes)
                for onset in line.stimulus_onsets_ms:
                    if onset > 0:
                        axes.axvline(onset, color=colour, linewidth=0.6, linestyle="--", alpha=0.5, zorder=0)
                label = f"{line.label} ({line.n_sweeps})"
                if draw_sweeps:
                    axes.plot(line.time_ms, line.sweeps.T, color=colour, linestyle=MPL_DASHES[dash],
                              alpha=0.25 if averaging else 0.6, linewidth=0.5)
                    if not averaging:   # one legend entry per line, not per sweep
                        axes.plot([], [], color=colour, linestyle=MPL_DASHES[dash], label=label)
                if averaging:
                    axes.plot(line.time_ms, line.mean, color=colour, linestyle=MPL_DASHES[dash], linewidth=1.6, label=label)
            axes.set_title(_panel_title(title, subset), fontsize=8)
            if panel_limits:
                axes.set_ylim(panel_limits[index])
            if index % columns == 0 or len(data.units) > 1:
                axes.set_ylabel(_y_label(subset))
            axes.set_xlabel("time from 1st stimulus (ms)")
            axes.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))

        header = [data.header_text(), summary_text(data, average_by)]
        wrapped = [piece for line in header for piece in textwrap.wrap(line, 170, break_long_words=False)]
        figure.suptitle("\n".join(wrapped), fontsize=8, x=0.01, ha="left")
        return _save(figure, output_path, dpi)
