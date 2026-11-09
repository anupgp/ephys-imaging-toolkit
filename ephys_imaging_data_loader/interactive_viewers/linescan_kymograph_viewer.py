"""
linescan_kymograph_viewer.py
============================

Interactive viewer for line-scan time series (e.g. Zeiss LSM880 line scans
across a spine and its dendrite, used for iGluSNFR / calcium imaging).

What you see
------------
Top    : the kymograph — time runs LEFT -> RIGHT (like an ephys trace),
         position along the scanned line runs top -> bottom.
         The pixel range selected for the fluorescence trace is outlined.
Bottom : the fluorescence trace = mean intensity over the selected pixel
         range for every scanned line, either raw or as ΔF/F₀.
Both   : the microscope's stimulus events ("stim" markers from ZEN) as
         dashed vertical lines. Both panels share the time axis.
Right  : the reference image (inverted) with the scanned line (black, dashed), an arrow
         from position 0 towards the last position (= top -> bottom of the
         kymograph), the selected pixel range in cyan, and a black scale bar. Time
         series have no reference of their own; then the reference of another
         file in the same folder with the same focus and rotation is borrowed
         (named in the title; see file_readers/zeiss_line_scan_reference.py).

ΔF/F₀
-----
F₀ (baseline) = mean of the trace before the first stimulus event (or the
first 10 % of the recording if there are no events). ΔF/F₀ = (F − F₀) / F₀.
This is a display aid for quick inspection, not a replacement for a proper
analysis pipeline (no background subtraction or bleaching correction).

Controls
--------
Channel, pixel range along the line (range slider), raw / ΔF/F₀ switch.
"""

from __future__ import annotations

import numpy as np

from ..file_readers.zeiss_line_scan_reference import LineScanReference, find_line_scan_reference
from ..recording_data_containers.image_recording import ImageRecording
from .display_helpers import COLORBLIND_SAFE_COLORS, contrast_limits, format_scale_bar_label, nice_scale_bar_length

# Beyond this many time points, the kymograph IMAGE is shown averaged over
# small time bins (the trace below always uses every line).
MAX_KYMOGRAPH_COLUMNS = 4000


def baseline_end_time_s(recording: ImageRecording) -> float:
    """End of the baseline window: first stim event, else 10 % of the recording."""
    stim_times = recording.event_times_s("stim")
    stim_times = stim_times[stim_times > 0]
    if len(stim_times):
        return float(stim_times.min())
    return float(recording.line_times_s[-1] * 0.1)


def fluorescence_trace(
    recording: ImageRecording, channel: int, x_range_px: tuple[int, int], as_delta_f_over_f: bool
) -> np.ndarray:
    """Mean over the pixel range per line, optionally converted to ΔF/F₀."""
    trace = recording.line_profile_over_time(channel, x_range_px)
    if not as_delta_f_over_f:
        return trace
    in_baseline = recording.line_times_s < baseline_end_time_s(recording)
    baseline = trace[in_baseline].mean() if in_baseline.any() else trace.mean()
    return (trace - baseline) / baseline if baseline else trace * np.nan


def _bin_columns(kymograph_x_by_t: np.ndarray, max_columns: int) -> np.ndarray:
    """Average neighbouring time columns so at most ``max_columns`` remain."""
    n_positions, n_times = kymograph_x_by_t.shape
    if n_times <= max_columns:
        return kymograph_x_by_t
    per_bin = int(np.ceil(n_times / max_columns))
    usable = (n_times // per_bin) * per_bin
    return kymograph_x_by_t[:, :usable].reshape(n_positions, -1, per_bin).mean(axis=2)


def build_reference_figure(
    recording: ImageRecording,
    reference: LineScanReference,
    channel: int,
    x_range_px: tuple[int, int],
    size: int = 430,
):
    """Reference image with the scanned line, its direction, the selected range and a scale bar."""
    from bokeh.models import Arrow, LinearColorMapper, Range1d, Title, VeeHead
    from bokeh.palettes import Greys256
    from bokeh.plotting import figure

    image = reference.image[channel if channel < reference.image.shape[0] else 0].astype(np.float32)
    height, width = image.shape
    n_positions = recording.width_px
    low, high = contrast_limits(image, 0.5, 99.8)
    plot = figure(
        title=reference.describe(), width=size, height=size,
        x_range=Range1d(0, width, bounds=(0, width)), y_range=Range1d(height, 0, bounds=(0, height)),  # row 0 at the top
        tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom",
    )
    plot.title.text_font_size = "9pt"
    for note in reference.notes:
        plot.add_layout(Title(text=note, text_font_size="8pt", text_font_style="italic"), "above")
    plot.image(image=[image], x=0, y=0, dw=width, dh=height,
               color_mapper=LinearColorMapper(palette=Greys256[::-1], low=low, high=high))

    # the whole line (may extend beyond a borrowed, higher-zoom reference)
    (x0, y0), (x1, y1) = reference.line_start_xy, reference.line_end_xy
    plot.line([x0, x1], [y0, y1], line_color="black", line_dash="dashed", line_width=1.5)
    # direction: arrow over the visible part, labelled with kymograph positions
    visible = reference.positions_inside(n_positions)
    if visible:
        (ax, ay), (bx, by) = reference.point_at(visible[0], n_positions), reference.point_at(visible[1], n_positions)
        plot.add_layout(Arrow(end=VeeHead(size=12, fill_color="yellow", line_color="yellow"),
                              x_start=ax, y_start=ay, x_end=bx, y_end=by, line_color="yellow", line_width=1.5))
        offset = -0.04 * height
        plot.text([ax, bx], [ay + offset, by + offset], text=[str(visible[0]), str(visible[1])],
                  text_color="yellow", text_font_size="9pt", text_align="center", text_baseline="bottom")
    # the pixel range used for the fluorescence trace (same as the cyan box on the kymograph)
    (sx, sy), (ex, ey) = reference.point_at(x_range_px[0], n_positions), reference.point_at(x_range_px[1] - 1, n_positions)
    plot.line([sx, ex], [sy, ey], line_color="cyan", line_width=4, line_alpha=0.8)

    # scale bar, bottom right
    bar_um = nice_scale_bar_length(width * reference.pixel_size_um)
    bar_px = bar_um / reference.pixel_size_um
    margin = 0.05 * width
    # black: the reference is shown inverted (dark structures on a white background)
    plot.line([width - margin - bar_px, width - margin], [height - margin] * 2, line_color="black", line_width=4)
    plot.text([width - margin - bar_px / 2], [height - margin - 0.02 * height], text=[format_scale_bar_label(bar_um)],
              text_color="black", text_font_size="10pt", text_font_style="bold", text_align="center", text_baseline="bottom")
    plot.axis.visible = False
    plot.grid.visible = False
    return plot


def build_kymograph_figure(
    recording: ImageRecording,
    channel: int,
    x_range_px: tuple[int, int],
    as_delta_f_over_f: bool = True,
    colormap: str = "Inferno256",
    width: int = 950,
    reference: LineScanReference | None = None,
):
    """
    Kymograph + fluorescence trace, stacked with a shared time axis; with a
    ``reference`` (``find_line_scan_reference``) the reference image with the
    scanned line is shown on the right.
    """
    import bokeh.palettes as bokeh_palettes
    from bokeh.layouts import column, row
    from bokeh.models import BoxAnnotation, LinearColorMapper, Range1d, Span
    from bokeh.plotting import figure

    times_s = recording.line_times_s
    duration_s = float(times_s[-1]) if len(times_s) > 1 else 1.0
    kymograph = recording.kymograph(channel)  # (T, X)
    n_positions = kymograph.shape[1]
    display_image = _bin_columns(kymograph.T.astype(np.float32), MAX_KYMOGRAPH_COLUMNS)  # (X, T)

    shared_time_range = Range1d(0, duration_s, bounds=(0, duration_s))
    low, high = contrast_limits(display_image, 0.5, 99.5)

    # ---- top: kymograph image -----------------------------------------------------
    kymo_plot = figure(
        title=f"{recording.file_name} — channel {recording.channel_names[channel]}",
        width=width,
        height=260,
        x_range=shared_time_range,
        y_range=Range1d(n_positions, 0),  # position 0 at the top
        y_axis_label="position along line (px)",
        tools="pan,xwheel_zoom,box_zoom,reset,save",
        active_scroll="xwheel_zoom",
    )
    kymo_plot.image(
        image=[display_image],
        x=0, y=0, dw=duration_s, dh=n_positions,
        color_mapper=LinearColorMapper(palette=getattr(bokeh_palettes, colormap, bokeh_palettes.Inferno256), low=low, high=high),
    )
    # Outline the pixel range used for the trace.
    kymo_plot.add_layout(
        BoxAnnotation(bottom=x_range_px[0], top=x_range_px[1], fill_alpha=0.0,
                      line_color="cyan", line_width=1.5, line_dash="dashed")
    )
    kymo_plot.xaxis.major_label_text_font_size = "0pt"
    kymo_plot.grid.visible = False

    # ---- bottom: fluorescence trace ------------------------------------------------
    trace = fluorescence_trace(recording, channel, x_range_px, as_delta_f_over_f)
    trace_plot = figure(
        width=width,
        height=230,
        x_range=shared_time_range,
        x_axis_label="Time (s)",
        y_axis_label="ΔF/F₀" if as_delta_f_over_f else "mean intensity (a.u.)",
        tools="pan,xwheel_zoom,ywheel_zoom,box_zoom,reset,save",
        active_scroll="xwheel_zoom",
    )
    trace_plot.line(times_s, trace, line_color=COLORBLIND_SAFE_COLORS[2], line_width=1.2)
    if as_delta_f_over_f:
        trace_plot.add_layout(BoxAnnotation(right=baseline_end_time_s(recording), fill_color="grey", fill_alpha=0.08))
    trace_plot.grid.grid_line_alpha = 0.3

    # ---- stimulus events on both panels -----------------------------------------------
    for stim_time in recording.event_times_s("stim"):
        for plot in (kymo_plot, trace_plot):
            plot.add_layout(Span(location=float(stim_time), dimension="height",
                                 line_color=COLORBLIND_SAFE_COLORS[5], line_dash="dashed", line_width=1.5))
    if reference is None:
        return column(kymo_plot, trace_plot)
    return row(column(kymo_plot, trace_plot), build_reference_figure(recording, reference, channel, x_range_px))


def build_linescan_kymograph_viewer(
    recording: ImageRecording,
    channel: str | int = 0,
    x_range_px: tuple[int, int] | None = None,
    as_delta_f_over_f: bool = True,
    colormap: str = "Inferno256",
    show_reference: bool = True,
    width: int = 950,
):
    """
    Full interactive viewer for a line-scan time series (see module docstring).

    Parameters
    ----------
    channel : initial channel (name or index)
    x_range_px : initial pixel range (first, last) along the line;
        default: the middle 20 % of the line
    as_delta_f_over_f : start in ΔF/F₀ mode (else raw intensity)
    colormap : Bokeh palette for the kymograph (e.g. "Inferno256", "Greys256")
    show_reference : show the reference image with the scanned line on the right
        (the file's own, else one borrowed from the same folder)
    width : width of the kymograph and trace panels (px)
    """
    import panel as pn

    pn.extension()
    if not (recording.is_line_scan and recording.n_timepoints > 1):
        raise ValueError(f"{recording.file_name} is not a line-scan time series.")

    n_positions = recording.width_px
    default_range = (int(n_positions * 0.4), int(n_positions * 0.6))

    channel_selector = pn.widgets.Select(
        label="Channel",
        options={name: index for index, name in enumerate(recording.channel_names)},
        value=recording.channel_index(channel),
        width=180,
    )
    range_slider = pn.widgets.IntRangeSlider(
        label="Pixel range along the line", start=0, end=n_positions - 1,
        value=tuple(x_range_px) if x_range_px else default_range, width=350,
    )
    mode_selector = pn.widgets.RadioButtonGroup(
        label="Trace", options=["ΔF/F₀", "raw"], value="ΔF/F₀" if as_delta_f_over_f else "raw"
    )

    reference = find_line_scan_reference(recording) if show_reference else None

    def figure_for_widgets(channel_index, pixel_range, mode):
        first, last = pixel_range
        return build_kymograph_figure(
            recording, channel_index, (first, last + 1), mode == "ΔF/F₀", colormap, width, reference
        )

    live_figure = pn.bind(figure_for_widgets, channel_selector, range_slider, mode_selector)
    stim_count = len(recording.event_times_s("stim"))
    reference_text = ""
    if show_reference:
        reference_text = (f" — {reference.describe()}" if reference is not None
                          else " — no reference image found (none in the file or its folder with the same focus and rotation)")
    header = pn.pane.Markdown(
        f"**{recording.file_name}** — {recording.n_timepoints} lines, "
        f"{(recording.time_interval_s or 0) * 1000:.4g} ms/line, {n_positions} px "
        f"({(recording.pixel_size_x_um or 0):.3g} µm/px), {stim_count} stim events — "
        f"source: `{recording.source_description}`{reference_text}"
    )
    return pn.Column(header, pn.Row(channel_selector, range_slider, mode_selector), live_figure)
