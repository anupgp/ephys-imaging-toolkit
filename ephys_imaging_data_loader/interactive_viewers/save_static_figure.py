"""
save_static_figure.py
=====================

Save publication-style STATIC figures (PNG, SVG or PDF) with matplotlib.

Why not export the interactive Bokeh plots?
-------------------------------------------
The old ``olympus_image_class.py`` exported Bokeh plots through selenium and
a headless Chrome browser, which is fragile to install and slow. Here,
matplotlib writes the file directly, and the format follows the file
extension:

    ".png" -> raster image (``dpi`` controls resolution)
    ".svg" -> vector image; text stays editable in Illustrator/Inkscape
    ".pdf" -> vector image

The figures are created with ``matplotlib.figure.Figure`` directly (not
``pyplot``), so saving figures never opens windows and never interferes with
other plots in a notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.figure import Figure

from ..recording_data_containers import EphysRecording, ImageRecording
from .display_helpers import (
    COLORBLIND_SAFE_COLORS,
    contrast_limits,
    format_scale_bar_label,
    nice_scale_bar_length,
)
from .linescan_kymograph_viewer import baseline_end_time_s, fluorescence_trace

# Keep text as text in SVG files (editable later), and use a common font.
FIGURE_STYLE = {"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}


def _save(figure: Figure, output_path: str | Path, dpi: int) -> Path:
    """Create the folder if needed, save, and return the path."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def save_ephys_traces_figure(
    recording: EphysRecording,
    output_path: str | Path,
    channels: Sequence[str | int] | None = None,
    sweeps: Sequence[int] | None = None,
    time_window_s: tuple[float, float] | None = None,
    show_mean: bool = True,
    show_stimuli: bool = True,
    figure_size_in: tuple[float, float] | None = None,
    dpi: int = 300,
) -> Path:
    """
    Save stacked channel panels (sweeps in grey, mean in black, stimuli shaded).

    Parameters
    ----------
    time_window_s : (start, end) in seconds to show, e.g. (0.7, 1.2); default: whole sweep
    figure_size_in : (width, height) in inches; default scales with channel count
    Other parameters as in ``ephys_trace_viewer.build_ephys_trace_figure``.
    """
    import matplotlib

    channel_indices = (
        list(range(len(recording.channel_names))) if not channels else [recording.channel_index(c) for c in channels]
    )
    sweep_indices = recording.good_sweeps if sweeps is None else list(sweeps)
    time_s = recording.time_s
    in_window = np.ones_like(time_s, dtype=bool)
    if time_window_s:
        in_window = (time_s >= time_window_s[0]) & (time_s <= time_window_s[1])

    with matplotlib.rc_context(FIGURE_STYLE):
        figure = Figure(figsize=figure_size_in or (7, 1.8 * len(channel_indices) + 0.6))
        axes_list = figure.subplots(len(channel_indices), 1, sharex=True, squeeze=False)[:, 0]

        pulses = recording.stimulus_pulses
        if show_stimuli and sweep_indices and not pulses.empty:
            pulses = pulses[pulses["sweep"] == sweep_indices[0]]
        else:
            pulses = pulses.iloc[0:0]

        # A 0.1 ms TTL pulse is far thinner than one screen pixel in a 4 s window.
        # Draw every pulse at least 0.4 % of the visible time span wide.
        shown_span_s = float(time_s[in_window][-1] - time_s[in_window][0])
        minimum_band_width_s = 0.004 * shown_span_s

        for axes, channel_index in zip(axes_list, channel_indices):
            traces = recording.data[sweep_indices, channel_index][:, in_window]
            for trace in traces:
                axes.plot(time_s[in_window], trace, color="0.65", linewidth=0.6)
            if show_mean and len(sweep_indices) > 1:
                axes.plot(time_s[in_window], traces.mean(axis=0), color="black", linewidth=1.2)
            for pulse in pulses.itertuples():
                axes.axvspan(pulse.onset_s, pulse.onset_s + max(pulse.duration_s, minimum_band_width_s),
                             color=COLORBLIND_SAFE_COLORS[int(pulse.digital_output_bit) % 8], alpha=0.3, linewidth=0)
            axes.set_ylabel(f"{recording.channel_names[channel_index]}\n({recording.channel_units[channel_index]})")
        axes_list[-1].set_xlabel("Time (s)")
        if time_window_s:
            axes_list[-1].set_xlim(time_window_s)
        axes_list[0].set_title(f"{recording.file_name} ({len(sweep_indices)} sweeps)", fontsize=10)
        figure.align_ylabels(axes_list)
        return _save(figure, output_path, dpi)


def save_image_figure(
    recording: ImageRecording,
    output_path: str | Path,
    channel: str | int = 0,
    view: str = "auto",
    z_index: int = 0,
    time_index: int = 0,
    contrast_percentiles: tuple[float, float] = (0.5, 99.5),
    colormap: str = "gray",
    show_uncaging_points: bool = True,
    scale_bar_um: float | None = None,
    use_reference_image: bool = False,
    smoothing_sigma_px: float = 1.0,
    target_color: str = "red",
    target_labels: list[str] | None = None,
    scale_bar_color: str = "black",
    figure_width_in: float = 4.0,
    dpi: int = 300,
) -> Path:
    """
    Save one image (plane or projection) with scale bar and uncaging targets.

    Parameters
    ----------
    view : "plane", "max", "mean", or "auto" (= "max" for z-stacks, else "plane")
    smoothing_sigma_px : Gaussian smoothing of each slice before a projection
        (ignored for single planes); 0 = none
    target_color : colour of the uncaging-target markers and labels (default red)
    scale_bar_color : colour of the scale bar and its label (default black)
    target_labels : labels for the targets; default: spine ids from the
        experiment sheet (see experiment_catalog/uncaging_target_labels.py)
    scale_bar_um : bar length; None chooses a round length (~20 % of the width)
    colormap : any matplotlib colormap name, e.g. "gray", "magma", "inferno"
    use_reference_image : draw the overview image stored in the file instead
    """
    import matplotlib

    channel_index = recording.channel_index(channel)
    if use_reference_image:
        if recording.reference_image is None:
            raise ValueError(f"{recording.file_name} has no reference image.")
        image = recording.reference_image[min(channel_index, recording.reference_image.shape[0] - 1)]
        pixel_size_um = recording.reference_image_pixel_size_um
        points = None
    else:
        if view == "auto":
            view = "max" if recording.n_z_slices > 1 else "plane"
        image = (
            recording.get_plane(time_index, channel_index, z_index)
            if view == "plane"
            else recording.projection(
                channel_index, view, time_index=time_index, smoothing_sigma_px=smoothing_sigma_px
            )
        )
        pixel_size_um = recording.pixel_size_x_um
        points = recording.uncaging_points if show_uncaging_points else None

    low, high = contrast_limits(image, *contrast_percentiles)
    n_rows, n_columns = image.shape

    with matplotlib.rc_context(FIGURE_STYLE):
        figure = Figure(figsize=(figure_width_in, figure_width_in * n_rows / n_columns))
        axes = figure.add_axes([0, 0, 1, 1])  # image fills the whole figure
        # origin="upper": pixel (0, 0) at the top-left, as on the microscope.
        axes.imshow(image, cmap=colormap, vmin=low, vmax=high, origin="upper", interpolation="nearest")
        axes.set_axis_off()

        if points is not None and len(points):
            if target_labels is None:
                from ..experiment_catalog.uncaging_target_labels import label_uncaging_targets

                target_labels = label_uncaging_targets(recording).labels
            # on-spine targets: circles + spine id; moved-away targets (label None): grey crosses
            for label, point in zip(target_labels, points.itertuples()):
                if label is None:
                    axes.scatter([point.x_px], [point.y_px], s=30, marker="x", color="grey", linewidths=1.2)
                    continue
                axes.scatter([point.x_px], [point.y_px], s=80, facecolors="none", edgecolors=target_color, linewidths=1.5)
                axes.annotate(label, (point.x_px, point.y_px), xytext=(6, -6),
                              textcoords="offset points", color=target_color, fontsize=9, fontweight="bold")

        if pixel_size_um:
            bar_um = scale_bar_um or nice_scale_bar_length(n_columns * pixel_size_um)
            bar_px = bar_um / pixel_size_um
            margin = 0.05 * n_columns
            y = n_rows - margin
            axes.plot([n_columns - margin - bar_px, n_columns - margin], [y, y], color=scale_bar_color, linewidth=3)
            axes.text(n_columns - margin - bar_px / 2, y - 0.02 * n_rows, format_scale_bar_label(bar_um),
                      color=scale_bar_color, ha="center", va="bottom", fontsize=9)
        return _save(figure, output_path, dpi)


def save_kymograph_figure(
    recording: ImageRecording,
    output_path: str | Path,
    channel: str | int = 0,
    x_range_px: tuple[int, int] | None = None,
    as_delta_f_over_f: bool = True,
    colormap: str = "inferno",
    figure_size_in: tuple[float, float] = (7, 4),
    dpi: int = 300,
) -> Path:
    """
    Save a line-scan kymograph (top) with its fluorescence trace (bottom).

    ``x_range_px`` = (first, last) pixel along the line, last EXCLUDED;
    default: the middle 20 % of the line.
    """
    import matplotlib

    channel_index = recording.channel_index(channel)
    n_positions = recording.width_px
    x_range_px = x_range_px or (int(n_positions * 0.4), int(n_positions * 0.6))
    times_s = recording.line_times_s
    kymograph = recording.kymograph(channel_index).T  # (position, time)
    trace = fluorescence_trace(recording, channel_index, x_range_px, as_delta_f_over_f)
    low, high = contrast_limits(kymograph)

    with matplotlib.rc_context(FIGURE_STYLE):
        figure = Figure(figsize=figure_size_in)
        kymo_axes, trace_axes = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
        kymo_axes.imshow(kymograph, aspect="auto", cmap=colormap, vmin=low, vmax=high,
                         extent=(0, times_s[-1], n_positions, 0), interpolation="nearest")
        kymo_axes.axhspan(x_range_px[0], x_range_px[1], fill=False, edgecolor="cyan", linestyle="--", linewidth=1)
        kymo_axes.set_ylabel("position (px)")
        kymo_axes.set_title(f"{recording.file_name} — {recording.channel_names[channel_index]}", fontsize=10)

        trace_axes.plot(times_s, trace, color=COLORBLIND_SAFE_COLORS[2], linewidth=0.8)
        if as_delta_f_over_f:
            trace_axes.axvspan(0, baseline_end_time_s(recording), color="0.9", zorder=0)
        trace_axes.set_ylabel("ΔF/F₀" if as_delta_f_over_f else "intensity (a.u.)")
        trace_axes.set_xlabel("Time (s)")
        for stim_time in recording.event_times_s("stim"):
            for axes in (kymo_axes, trace_axes):
                axes.axvline(stim_time, color=COLORBLIND_SAFE_COLORS[5], linestyle="--", linewidth=1)
        figure.align_ylabels([kymo_axes, trace_axes])
        return _save(figure, output_path, dpi)
