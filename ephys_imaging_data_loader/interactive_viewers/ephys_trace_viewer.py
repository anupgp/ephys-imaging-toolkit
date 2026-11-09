"""
ephys_trace_viewer.py
=====================

Interactive viewer for patch-clamp recordings (``EphysRecording``).

What you see
------------
* One panel per channel, stacked vertically, all sharing the same time axis
  (zooming in one panel zooms all of them).
* Every selected sweep as a thin coloured line, and optionally the average of
  the selected sweeps as a thick black line.
* The digital stimulus pulses of the protocol as shaded vertical bands, one
  colour per digital output, so you can see exactly when each stimulator /
  uncaging laser was triggered.

Controls (widgets above the plot)
---------------------------------
* Channels to show, sweeps to show, vertical offset between sweeps
  (to "fan out" overlapping traces), mean on/off, stimulus bands on/off.

Speed
-----
Traces are reduced with a min/max method before drawing (see
``display_helpers.min_max_downsample``): peaks are preserved but the browser
receives ~4000 points per trace instead of e.g. 80,000.

Usage
-----
In a notebook, the returned object displays itself::

    recording.show()                                   # all defaults
    recording.show(channels=["ImLEFT"], sweeps=[0, 1], offset=-20)

In a script, open it in the browser with ``viewer.show()``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..recording_data_containers.ephys_recording import EphysRecording
from .display_helpers import COLORBLIND_SAFE_COLORS, min_max_downsample
from .ephys_recording_title import ephys_title_parts


def build_ephys_trace_figure(
    recording: EphysRecording,
    channels: Sequence[str | int] | None = None,
    sweeps: Sequence[int] | None = None,
    offset: float = 0.0,
    show_mean: bool = True,
    show_stimuli: bool = True,
    max_points_per_trace: int = 4000,
    width: int = 950,
    height_per_channel: int = 230,
):
    """
    Build the (non-interactive-widget) Bokeh figure: stacked channel panels.

    This is used by ``build_ephys_trace_viewer`` every time a widget changes,
    and can also be used on its own (e.g. ``bokeh.io.show(figure)``).

    Parameters
    ----------
    recording : the EphysRecording to plot
    channels : channel names or indices (default: all channels)
    sweeps : sweep indices (default: all good sweeps)
    offset : vertical shift added per sweep (in the channel's unit)
    show_mean : draw the average of the selected sweeps
    show_stimuli : draw the digital stimulus pulses as shaded bands
    max_points_per_trace : downsampling target per trace (see module docstring)
    width, height_per_channel : figure size in screen pixels
    """
    from bokeh.layouts import column
    from bokeh.models import BoxAnnotation, HoverTool, Range1d
    from bokeh.palettes import Viridis256
    from bokeh.plotting import figure

    channel_indices = (
        list(range(len(recording.channel_names)))
        if not channels
        else [recording.channel_index(c) for c in channels]
    )
    sweep_indices = recording.good_sweeps if sweeps is None else list(sweeps)
    time_s = recording.time_s

    # One colour per sweep, spread over the viridis colour map.
    sweep_colors = (
        [Viridis256[int(i)] for i in np.linspace(0, 230, len(sweep_indices))]
        if sweep_indices
        else []
    )

    # All panels share one x-range object -> zooming is linked.
    shared_time_range = Range1d(start=0.0, end=float(time_s[-1]))

    # Stimulus pulses of the FIRST selected sweep (protocols may vary per sweep).
    stimulus_pulses = recording.stimulus_pulses
    if show_stimuli and sweep_indices and not stimulus_pulses.empty:
        stimulus_pulses = stimulus_pulses[stimulus_pulses["sweep"] == sweep_indices[0]]
    else:
        stimulus_pulses = stimulus_pulses.iloc[0:0]

    panels = []
    for panel_number, channel_index in enumerate(channel_indices):
        channel_name = recording.channel_names[channel_index]
        unit = recording.channel_units[channel_index]
        is_last_panel = panel_number == len(channel_indices) - 1

        panel = figure(
            width=width,
            height=height_per_channel,
            x_range=shared_time_range,
            x_axis_label="Time (s)" if is_last_panel else None,
            y_axis_label=f"{channel_name} ({unit})",
            tools="pan,xwheel_zoom,box_zoom,reset,save",
            active_drag="box_zoom",
            active_scroll="xwheel_zoom",
            output_backend="webgl",  # GPU drawing: much faster for many lines
        )
        if panel_number == 0:
            # file | protocol | 3/3 sweeps | 2 pulses | IPI: 200 ms | CA1 | apical | ...
            # (see ephys_recording_title.py for how the stimulus is identified)
            panel.title.text = ephys_title_parts(recording, sweep_indices)[0]

        # ---- the sweeps -------------------------------------------------------
        for position, (sweep, color) in enumerate(zip(sweep_indices, sweep_colors)):
            t_plot, y_plot = min_max_downsample(
                time_s, recording.data[sweep, channel_index], max_points_per_trace
            )
            panel.line(
                t_plot,
                y_plot + position * offset,
                line_color=color,
                line_width=1,
                alpha=0.8,
                name=f"sweep {sweep}",
            )

        # ---- the average ------------------------------------------------------
        if show_mean and len(sweep_indices) > 1:
            mean_trace = recording.data[sweep_indices, channel_index].mean(axis=0)
            t_plot, y_plot = min_max_downsample(time_s, mean_trace, max_points_per_trace)
            panel.line(t_plot, y_plot, line_color="black", line_width=2, name="mean")

        # ---- the stimulus bands -----------------------------------------------
        for pulse in stimulus_pulses.itertuples():
            # Very short TTL pulses (0.1 ms) would be thinner than one screen
            # pixel; draw them at least 0.4 % of the sweep duration wide.
            visible_width_s = max(pulse.duration_s, 0.004 * recording.sweep_duration_s)
            panel.add_layout(
                BoxAnnotation(
                    left=pulse.onset_s,
                    right=pulse.onset_s + visible_width_s,
                    fill_color=COLORBLIND_SAFE_COLORS[int(pulse.digital_output_bit) % len(COLORBLIND_SAFE_COLORS)],
                    fill_alpha=0.25,
                    line_alpha=0,
                )
            )

        panel.add_tools(HoverTool(tooltips=[("sweep", "$name"), ("t (s)", "$x{0.0000}"), ("value", "$y{0.00}")]))
        if not is_last_panel:
            panel.xaxis.major_label_text_font_size = "0pt"  # hide repeated time labels
        panel.grid.grid_line_alpha = 0.3
        panels.append(panel)

    return column(*panels)


def build_ephys_trace_viewer(
    recording: EphysRecording,
    channels: Sequence[str | int] | None = None,
    sweeps: Sequence[int] | None = None,
    offset: float = 0.0,
    show_mean: bool = True,
    show_stimuli: bool = True,
    max_points_per_trace: int = 4000,
    width: int = 950,
    height_per_channel: int = 230,
):
    """
    Full interactive viewer: widgets + the stacked-channel figure.

    Returns a Panel layout. In a notebook it displays itself; in a script call
    ``.show()`` on it to open a browser tab. The arguments set the initial
    state of the widgets (see ``build_ephys_trace_figure`` for their meaning).
    """
    import panel as pn

    pn.extension()  # loads Panel's JavaScript in notebooks; harmless elsewhere

    # ---- widgets ---------------------------------------------------------------
    channel_options = {
        f"{name} ({unit})": index
        for index, (name, unit) in enumerate(zip(recording.channel_names, recording.channel_units))
    }
    initial_channels = (
        list(channel_options.values())
        if not channels
        else [recording.channel_index(c) for c in channels]
    )
    channel_selector = pn.widgets.CheckBoxGroup(
        label="Channels", options=channel_options, value=initial_channels, inline=True
    )

    sweep_options = {
        f"sweep {s}" + (" (bad)" if s in recording.bad_sweeps else ""): s
        for s in range(recording.n_sweeps)
    }
    sweep_selector = pn.widgets.MultiChoice(
        label="Sweeps",
        options=sweep_options,
        value=recording.good_sweeps if sweeps is None else list(sweeps),
        width=450,
    )
    offset_input = pn.widgets.FloatInput(label="Offset between sweeps", value=offset, step=1.0, width=170)
    mean_checkbox = pn.widgets.Checkbox(label="Show mean", value=show_mean)
    stimulus_checkbox = pn.widgets.Checkbox(label="Show stimulus pulses", value=show_stimuli)

    # ---- figure that is rebuilt whenever a widget changes -----------------------
    def figure_for_widgets(selected_channels, selected_sweeps, offset_value, mean_on, stimuli_on):
        if not selected_channels or not selected_sweeps:
            return pn.pane.Markdown("*Select at least one channel and one sweep.*")
        return build_ephys_trace_figure(
            recording,
            channels=selected_channels,
            sweeps=sorted(selected_sweeps),
            offset=offset_value,
            show_mean=mean_on,
            show_stimuli=stimuli_on,
            max_points_per_trace=max_points_per_trace,
            width=width,
            height_per_channel=height_per_channel,
        )

    live_figure = pn.bind(
        figure_for_widgets,
        channel_selector,
        sweep_selector,
        offset_input,
        mean_checkbox,
        stimulus_checkbox,
    )

    # Caption: the title line (bold), then details: how the stimulus was
    # identified, date/neuron/dendrite/spine from the sheet, recording layout, source.
    title, details = ephys_title_parts(recording)
    header = pn.pane.Markdown(
        f"**{title}**\n\n{details} | {recording.acquisition_mode}, "
        f"{recording.sweep_duration_s:.2f} s sweeps at {recording.sampling_rate_hz:g} Hz | "
        f"source: `{recording.source_description}`"
    )
    controls = pn.Row(
        channel_selector,
        sweep_selector,
        pn.Column(offset_input, mean_checkbox, stimulus_checkbox),
    )
    return pn.Column(header, controls, live_figure)
