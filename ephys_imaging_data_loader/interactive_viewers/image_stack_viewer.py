"""
image_stack_viewer.py
=====================

Interactive viewer for frames, z-stacks and time series (``ImageRecording``),
from Zeiss or Olympus — the viewer only sees the standard (T, C, Z, Y, X) data.

What you see
------------
* The image in true orientation: pixel (0, 0) at the TOP-left, as on the
  microscope screen. (Bokeh normally puts y = 0 at the bottom; instead of
  flipping the pixel array — which the old code did and which made marker
  coordinates confusing — the y-axis itself is reversed, so pixel rows and
  marker coordinates can be used exactly as stored in the file.)
* Uncaging / photostimulation targets (Olympus "Bleach" points) as circles
  with their index; hover over them to see coordinates.
* A scale bar with a round length (e.g. 5 µm), computed from the pixel size.

Controls
--------
* Channel; z-slice and time-point sliders (only when the file has them).
* View: a single plane, or a maximum / mean projection over a z-range.
* Contrast: black and white levels as intensity percentiles.
* "Show reference image": switch to the overview image stored in the file.

Only the plane being shown is sent to the browser, so even large z-stacks
(e.g. 2 x 173 x 512 x 512) stay fast. (The old viewer sent the whole
5-D array to the browser at once.)
"""

from __future__ import annotations

import numpy as np

from ..recording_data_containers.image_recording import ImageRecording
from .display_helpers import contrast_limits, format_scale_bar_label, nice_scale_bar_length


def _as_hex_color(color: str) -> str:
    """'red' -> '#ff0000' (the colour picker widget needs hex codes)."""
    from matplotlib.colors import to_hex

    try:
        return to_hex(color)
    except ValueError:
        return "#ff0000"


def build_image_figure(
    image_2d: np.ndarray,
    title: str,
    pixel_size_um: float | None = None,
    uncaging_points=None,
    black_and_white_levels: tuple[float, float] | None = None,
    colormap: str = "Greys256",
    show_scale_bar: bool = True,
    scale_bar_color: str = "black",
    max_width_px: int = 650,
    target_labels: list[str] | None = None,
    target_color: str = "red",
):
    """
    A Bokeh figure of one 2-D image (Y, X) with optional markers and scale bar.

    Parameters
    ----------
    image_2d : array (rows, columns) to show
    title : figure title
    pixel_size_um : µm per pixel, for the scale bar (None -> no scale bar)
    uncaging_points : DataFrame with columns point, x_px, y_px (or None)
    target_labels : text next to each target (e.g. spine ids "1".."5");
        default: target order 1, 2, 3 ...
    target_color : colour of the target circles and labels (any CSS colour name
        or hex code, e.g. "red", "yellow", "#ff00ff"); default red
    black_and_white_levels : display range; default = 0.5-99.5 percentiles
    colormap : any Bokeh palette name, e.g. "Greys256", "Viridis256", "Inferno256"
    show_scale_bar : draw the scale bar (needs pixel_size_um)
    scale_bar_color : colour of the scale bar and its label (default "black")
    max_width_px : on-screen width of the figure
    """
    import bokeh.palettes as bokeh_palettes
    from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, Range1d
    from bokeh.plotting import figure

    n_rows, n_columns = image_2d.shape
    low, high = black_and_white_levels or contrast_limits(image_2d)
    # Look the palette up by name, e.g. "Viridis256"; unknown names fall back to grey.
    palette = getattr(bokeh_palettes, colormap, bokeh_palettes.Greys256)
    if colormap == "Greys256":
        palette = tuple(reversed(palette))  # Bokeh's Greys runs white->black; we want black->white

    # Keep square pixels: figure height follows the image aspect ratio.
    # Always use the full width (small images are enlarged) so the one-line caption fits.
    screen_width = max_width_px
    screen_height = max(150, int(screen_width * n_rows / n_columns))

    plot = figure(
        title=title,
        width=screen_width,
        height=screen_height,
        x_range=Range1d(0, n_columns, bounds=(0, n_columns)),
        # Reversed y-axis: row 0 at the top, like the microscope display.
        y_range=Range1d(n_rows, 0, bounds=(0, n_rows)),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
    )
    image_renderer = plot.image(
        image=[image_2d],
        x=0,
        y=0,
        dw=n_columns,
        dh=n_rows,
        color_mapper=LinearColorMapper(palette=palette, low=low, high=high),
    )
    plot.add_tools(
        HoverTool(renderers=[image_renderer], tooltips=[("x", "$x{0}"), ("y", "$y{0}"), ("value", "@image")])
    )
    plot.grid.visible = False
    # No pixel axes: physical size is given by the scale bar (bottom right).
    # (Pixel coordinates are still shown when hovering over the image.)
    plot.axis.visible = False
    plot.min_border = 5
    # Smaller title so the one-line caption fits the image width.
    plot.title.text_font_size = "9pt"

    # ---- uncaging / stimulation targets -----------------------------------------
    # Targets on spines: circles labelled with the spine id. Targets whose label
    # is None were moved away from the dendrite (shrinking-cluster series; the
    # laser still fires there): small grey crosses without a label.
    if uncaging_points is not None and len(uncaging_points) > 0:
        labels = (
            list(target_labels) if target_labels is not None and len(target_labels) == len(uncaging_points)
            else [str(int(p) + 1) for p in uncaging_points["point"]]
        )
        # +0.5 puts the marker in the centre of the target pixel.
        x_all = uncaging_points["x_px"].to_numpy() + 0.5
        y_all = uncaging_points["y_px"].to_numpy() + 0.5
        on = [label is not None for label in labels]
        on_source = ColumnDataSource({
            "x": x_all[on], "y": y_all[on], "label": [l for l in labels if l is not None],
        })
        markers = plot.scatter(
            "x", "y", source=on_source, size=14, marker="circle",
            fill_alpha=0, line_color=target_color, line_width=2, legend_label="uncaging target (spine id)",
        )
        plot.text("x", "y", text="label", source=on_source, x_offset=9, y_offset=-9,
                  text_color=target_color, text_font_size="11pt", text_font_style="bold")
        plot.add_tools(HoverTool(renderers=[markers], tooltips=[("spine", "@label"), ("x, y (px)", "@x{0.0}, @y{0.0}")]))
        if not all(on):
            off = [not flag for flag in on]
            off_source = ColumnDataSource({
                "x": x_all[off], "y": y_all[off],
                "position": [str(i + 1) for i, flag in enumerate(off) if flag],
            })
            moved = plot.scatter("x", "y", source=off_source, size=9, marker="x", line_color="grey",
                                 line_width=1.5, legend_label="target moved off the dendrite")
            plot.add_tools(HoverTool(renderers=[moved], tooltips=[("moved-away target #", "@position")]))
        plot.legend.location = "top_right"
        plot.legend.background_fill_alpha = 0.4

    # ---- scale bar (bottom-right corner) ---------------------------------------------
    if show_scale_bar and pixel_size_um:
        bar_um = nice_scale_bar_length(n_columns * pixel_size_um)
        bar_px = bar_um / pixel_size_um
        margin_px = 0.05 * n_columns
        bar_right = n_columns - margin_px
        bar_y = n_rows - margin_px  # near the bottom (y axis is reversed)
        plot.line([bar_right - bar_px, bar_right], [bar_y, bar_y], line_color=scale_bar_color, line_width=4)
        plot.text([bar_right - bar_px / 2], [bar_y - 0.02 * n_rows], text=[format_scale_bar_label(bar_um)],
                  text_color=scale_bar_color, text_align="center", text_baseline="bottom", text_font_size="10pt")
    return plot


def build_image_stack_viewer(
    recording: ImageRecording,
    channel: str | int = 0,
    view: str | None = None,
    colormap: str = "Greys256",
    contrast_percentiles: tuple[float, float] = (0.5, 99.5),
    smoothing_sigma_px: float = 1.0,
    show_uncaging_points: bool = True,
    show_scale_bar: bool = True,
    scale_bar_color: str = "black",
    target_color: str = "red",
    target_labels: list[str] | None = None,
):
    """
    Full interactive viewer for an ImageRecording (see module docstring).

    Parameters
    ----------
    channel : initial channel (name or index)
    view : "plane", "max projection" or "mean projection";
        default: "max projection" for z-stacks, "plane" otherwise
    colormap : Bokeh palette name
    contrast_percentiles : initial (black, white) level percentiles
    smoothing_sigma_px : initial Gaussian smoothing applied to each slice
        before a projection (0 = none)
    show_uncaging_points, show_scale_bar : initial state of these overlays
    scale_bar_color : colour of the scale bar and its label (default "black")
    target_color : colour of the uncaging-target markers (default "red"; can
        also be changed with the colour picker in the viewer)
    target_labels : labels for the targets; default: spine ids from the
        experiment sheet (see experiment_catalog/uncaging_target_labels.py)
    """
    import panel as pn

    pn.extension()

    n_t, n_c, n_z, _, _ = recording.shape_tczyx
    is_stack = n_z > 1

    # ---- widgets (sliders only appear for dimensions that exist) -------------------
    channel_selector = pn.widgets.Select(
        label="Channel",
        options={name: index for index, name in enumerate(recording.channel_names)},
        value=recording.channel_index(channel),
        width=180,
    )
    view_selector = pn.widgets.RadioButtonGroup(
        label="View",
        options=["plane", "max projection", "mean projection"] if is_stack else ["plane"],
        value=view or ("max projection" if is_stack else "plane"),
    )
    z_slider = pn.widgets.IntSlider(label="z-slice", start=0, end=max(n_z - 1, 1), value=0, disabled=not is_stack)
    z_range_slider = pn.widgets.IntRangeSlider(
        label="z-range for projection", start=0, end=max(n_z - 1, 1), value=(0, max(n_z - 1, 1)), disabled=not is_stack
    )
    t_slider = pn.widgets.IntSlider(label="time point", start=0, end=max(n_t - 1, 1), value=0, disabled=n_t <= 1)
    contrast_slider = pn.widgets.RangeSlider(
        label="Contrast (percentiles)", start=0, end=100, step=0.1, value=contrast_percentiles
    )
    # Blur each slice before projecting (see ImageRecording.projection); 0 = off.
    smoothing_slider = pn.widgets.FloatSlider(
        label="Smoothing before projection (px)", start=0, end=4, step=0.25,
        value=smoothing_sigma_px, disabled=not is_stack,
    )
    markers_checkbox = pn.widgets.Checkbox(
        label="Show uncaging targets", value=show_uncaging_points, disabled=recording.uncaging_points.empty
    )
    reference_checkbox = pn.widgets.Checkbox(
        label="Show reference image", value=False, disabled=recording.reference_image is None
    )
    target_color_picker = pn.widgets.ColorPicker(
        label="Target colour", value=_as_hex_color(target_color), disabled=recording.uncaging_points.empty
    )

    # Title line from the experiment sheet (computed once): file | ephys file |
    # n1d1s1 | date | age | area | uncaging power (see image_recording_title.py)
    from .image_recording_title import image_title_parts

    image_title, image_details = image_title_parts(recording)

    # Target labels = spine ids from the experiment sheet (computed once).
    label_note = ""
    if target_labels is None and not recording.uncaging_points.empty:
        from ..experiment_catalog.uncaging_target_labels import label_uncaging_targets

        found = label_uncaging_targets(recording)
        target_labels, label_note = found.labels, found.note

    def image_for_widgets(
        channel_index, view_mode, z_index, z_range, t_index, percentiles, smoothing_px, markers_on, reference_on,
        marker_color,
    ):
        # 1) Pick the 2-D image to show.
        if reference_on and recording.reference_image is not None:
            image = recording.reference_image[min(channel_index, recording.reference_image.shape[0] - 1)]
            pixel_size = recording.reference_image_pixel_size_um
            title = f"{recording.file_name} — reference image"
            # The reference image is the overview on which FluoView's stimulation
            # points were set, so draw them at their original (bleach-GUI) coordinates.
            points = None
            if markers_on and {"x_px_bleach_gui", "y_px_bleach_gui"} <= set(recording.uncaging_points.columns):
                points = recording.uncaging_points.assign(
                    x_px=recording.uncaging_points["x_px_bleach_gui"],
                    y_px=recording.uncaging_points["y_px_bleach_gui"],
                )
        else:
            if view_mode == "plane" or not is_stack:
                image = recording.data[t_index, channel_index, z_index]
                title = f"{recording.file_name} — z {z_index}, t {t_index}"
            else:
                method = "max" if view_mode.startswith("max") else "mean"
                image = recording.projection(
                    channel_index, method, (z_range[0], z_range[1] + 1), t_index, smoothing_px
                )
                title = f"{recording.file_name} — {method} projection z {z_range[0]}–{z_range[1]}"
            pixel_size = recording.pixel_size_x_um
            points = recording.uncaging_points if markers_on else None

        # 2) Line scans have a single row: make them visible as a thin strip.
        if image.ndim == 1 or image.shape[0] == 1:
            image = np.repeat(np.atleast_2d(image), 16, axis=0)

        return build_image_figure(
            image,
            # Only the caption line: date | eid | oid | nds | age sex | area | side | LP | temp
            title=image_title,
            pixel_size_um=pixel_size,
            uncaging_points=points,
            black_and_white_levels=contrast_limits(image, *percentiles),
            colormap=colormap,
            show_scale_bar=show_scale_bar,
            scale_bar_color=scale_bar_color,
            target_labels=target_labels,
            target_color=marker_color,
        )

    live_image = pn.bind(
        image_for_widgets,
        channel_selector, view_selector, z_slider, z_range_slider, t_slider,
        contrast_slider, smoothing_slider, markers_checkbox, reference_checkbox, target_color_picker,
    )

    pixel_text = (
        f"{recording.pixel_size_x_um:.4g} µm/pixel" if recording.pixel_size_x_um else "pixel size unknown"
    )
    header = pn.pane.Markdown(
        f"**{image_title}**\n\n"
        + (f"{image_details} — " if image_details else "")
        + f"{recording.microscope}, {recording.acquisition_type}, "
        f"TCZYX = {recording.shape_tczyx}, {pixel_text} — source: `{recording.source_description}`"
        + (f"\n\nUncaging targets labelled by: {label_note}" if label_note else "")
    )
    controls = pn.Column(
        channel_selector, view_selector, z_slider, z_range_slider, t_slider,
        contrast_slider, smoothing_slider, markers_checkbox, reference_checkbox, target_color_picker, width=300,
    )
    return pn.Column(header, pn.Row(controls, live_image))
