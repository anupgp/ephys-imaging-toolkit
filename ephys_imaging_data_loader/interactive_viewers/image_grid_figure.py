"""
image_grid_figure.py
====================

Several imaging files side by side in ONE figure, each with its uncaging
(laser) points — to check by eye which spines were stimulated in a series of
recordings (e.g. a shrinking cluster: 12345678, 1234567, ... 12).

Every panel shows
-----------------
* the image (max projection for z-stacks, else the plane), auto contrast;
* every laser point with its SCAN NUMBER (P1, P2 ... = order in which the laser
  visits them, as stored in the file), in small white text;
* points on spines: red circle + spine id (from the sheet, via
  experiment_catalog/uncaging_target_labels.py); points moved off the dendrite
  (shrinking clusters, compared with the full-cluster image): grey cross;
* title: expdate, image name, and for every sheet row that lists the image its
  ephys id and spineid; "(shared)" when the image is listed for several ephys
  files (then it is a reference image, not taken during the recording).

Use it through ``ExperimentCatalog.show_image_grid(...)``.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np

from .display_helpers import contrast_limits, format_scale_bar_label, nice_scale_bar_length


def crosshair_marker():
    """A '+' with an open centre (marker path), so the spine under it stays visible."""
    from matplotlib.path import Path as MarkerPath

    gap = 0.4
    vertices = [(-1, 0), (-gap, 0), (gap, 0), (1, 0), (0, -1), (0, -gap), (0, gap), (0, 1)]
    codes = [MarkerPath.MOVETO, MarkerPath.LINETO] * 4
    return MarkerPath(vertices, codes)


def draw_image_panel(axes, recording, title: str, target_labels=None, contrast_percentiles=(1.0, 99.9),
                     colormap: str = "gray", show_scan_numbers: bool = True, smoothing_sigma_px: float = 1.0,
                     marker: str = "circle") -> None:
    """
    One image with its laser points (see module docstring) on ``axes``.
    ``marker``: "circle" (open red circle) or "crosshair" (small red cross-hair
    with an open centre) for the points on spines.
    """
    from matplotlib import patheffects
    from scipy.ndimage import gaussian_filter

    if recording.n_z_slices > 1:
        image = recording.projection(0, "max", smoothing_sigma_px=smoothing_sigma_px)
    else:
        image = recording.get_plane(0, 0, 0).astype(float)
        if smoothing_sigma_px:
            image = gaussian_filter(image, smoothing_sigma_px)   # single scans are noisy
    low, high = contrast_limits(image, *contrast_percentiles)
    outline = [patheffects.withStroke(linewidth=2, foreground="black")]   # text readable on white dendrites
    axes.imshow(image, cmap=colormap, vmin=low, vmax=high, origin="upper", interpolation="nearest")
    axes.set_axis_off()
    n_rows, n_columns = image.shape

    points = recording.uncaging_points
    if points is not None and len(points):
        labels = list(target_labels) if target_labels is not None else [str(k + 1) for k in range(len(points))]
        for number, (label, point) in enumerate(zip(labels, points.itertuples()), start=1):
            if label is None:
                axes.scatter([point.x_px], [point.y_px], s=28, marker="x", color="0.6", linewidths=1.2)
            else:
                if marker == "crosshair":
                    axes.scatter([point.x_px], [point.y_px], s=60, marker=crosshair_marker(), color="red",
                                 linewidths=1.3)
                else:
                    axes.scatter([point.x_px], [point.y_px], s=70, facecolors="none", edgecolors="red",
                                 linewidths=1.4)
                axes.annotate(label, (point.x_px, point.y_px), xytext=(6, -9), textcoords="offset points",
                              color="red", fontsize=8, fontweight="bold", path_effects=outline)
            if show_scan_numbers:
                axes.annotate(f"P{number}", (point.x_px, point.y_px), xytext=(-6, 5), textcoords="offset points",
                              color="yellow", fontsize=6, ha="right", path_effects=outline)
    else:
        axes.text(0.5, 0.5, "no uncaging points", transform=axes.transAxes, ha="center", color="yellow", fontsize=8)

    if recording.pixel_size_x_um:
        bar_um = nice_scale_bar_length(n_columns * recording.pixel_size_x_um)
        bar_px = bar_um / recording.pixel_size_x_um
        margin = 0.05 * n_columns
        y = n_rows - margin
        axes.plot([n_columns - margin - bar_px, n_columns - margin], [y, y], color="white", linewidth=2.5)
        axes.text(n_columns - margin - bar_px / 2, y - 0.02 * n_rows, format_scale_bar_label(bar_um), color="white",
                  ha="center", va="bottom", fontsize=7)
    axes.set_title(title, fontsize=7.5)


def build_image_grid_figure(panels: list[dict], columns: int = 4, panel_size_in: float = 3.2, title: str = ""):
    """
    ``panels``: dicts with "recording" (ImageRecording or None), "title",
    "labels" (target labels or None) and optionally "error" (text shown instead).
    """
    import matplotlib
    from matplotlib.figure import Figure

    from .save_static_figure import FIGURE_STYLE

    columns = max(1, min(columns, len(panels)))
    rows = math.ceil(len(panels) / columns)
    with matplotlib.rc_context({**FIGURE_STYLE, "font.size": 8}):
        figure = Figure(figsize=(panel_size_in * columns, (panel_size_in + 0.45) * rows + 0.4), layout="constrained")
        for index, panel in enumerate(panels):
            axes = figure.add_subplot(rows, columns, index + 1)
            if panel.get("recording") is None:
                axes.set_axis_off()
                axes.text(0.5, 0.5, "\n".join(textwrap.wrap(panel.get("error", "not loaded"), 40)),
                          transform=axes.transAxes, ha="center", va="center", fontsize=7, color="firebrick")
                axes.set_title(panel["title"], fontsize=7.5)
                continue
            draw_image_panel(axes, panel["recording"], panel["title"], panel.get("labels"))
        if title:
            figure.suptitle(title, fontsize=9, x=0.01, ha="left")
    return figure


def parse_image_requests(images) -> list[tuple[str, int, int | None]]:
    """
    [(expdate, ophysid), ...] or [(expdate, neuronid, ophysid), ...], or text
    "20270408 31" / "20270408,1,31", or {expdate: [ophysids]} -> list of
    (expdate, ophysid, neuronid or None).
    """
    if isinstance(images, dict):
        return [(str(date), int(i), None) for date, ids in images.items() for i in ids]
    requests = []
    for item in images:
        fields = item.replace(",", " ").split() if isinstance(item, str) else list(item)
        if len(fields) == 2:
            requests.append((str(fields[0]).replace(".0", ""), int(fields[1]), None))
        elif len(fields) == 3:
            requests.append((str(fields[0]).replace(".0", ""), int(fields[2]), int(fields[1])))
        else:
            raise ValueError(f"Use (expdate, ophysid) or (expdate, neuronid, ophysid); got {item!r}.")
    return requests


def save_figure(figure, path: str | Path, dpi: int = 200) -> Path:
    from .save_static_figure import _save

    return _save(figure, Path(path), dpi)
