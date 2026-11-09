"""
set_spine_map_image.py
======================

One image of a set's dendrite with every single (unitary) spine marked by its
SHEET spine id — the "spine map" panel of the summation figure
(``interactive_viewers/spine_summation_figure.py``).

The labels are never inferred from the image: they always come from the
sheet's ``spineid`` (see ``uncaging_target_labels.py``).

How the image is chosen
-----------------------
1. The image of a CLUSTER recording whose laser points are named by the sheet
   (e.g. spineid 12345678 for an image with 8 points -> points 1..8 in scan
   order), the cluster with the most spines first. It is used when its named
   points cover every single spine of the set. All points are in one image, so
   their positions relative to each other are exact.
2. Otherwise (e.g. the cluster is spineid 0 = "all spines", so its points have
   no ids): the image of the first single spine, with the ONE laser point of
   every single-spine image drawn on it, each labelled with the spineid of
   its own sheet row. Only images of the same size and pixel size are used.
   The dendrite can drift between images (by up to about one spine spacing),
   so these positions are approximate; the panel's note says so.
3. Whichever of the two names more single spines when neither names all;
   None when no image could be read.

Cluster spread (``cluster_spreads``)
------------------------------------
How spread out the spines of a cluster are along the dendrite: the MEAN
DISTANCE (µm) of the member spines' laser points from their centroid, in the
image plane (x, y; the images are single planes, so z is not known):

    spread = mean_i |p_i - mean(p)|  ×  pixel size

The positions come from
1. the cluster's OWN image: its laser points that are on spines (all points, or
   for a shrinking cluster the points still on the dendrite, see
   ``uncaging_target_labels``), when their number equals the cluster's number
   of spines. Ids are not needed for this, so it also works for spineid 0
   clusters. Exact (one image). ``cluster_spread_source`` = "cluster image ImageNNNN".
2. otherwise the single-spine images of its member spines (one point each,
   same image size and pixel size as the first one); approximate, since the
   dendrite can drift between images. Source = "single-spine images (approximate)".
3. otherwise NaN, source "no positions".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .uncaging_target_labels import label_uncaging_targets

logger = logging.getLogger(__name__)


@dataclass
class SpineMap:
    """What the spine-map panel draws."""

    image: object                 # ImageRecording used as background
    points: pd.DataFrame          # columns x_px, y_px, label (sheet spine id)
    source: str                   # e.g. "cluster 1-2-3-4-5-6-7-8 · Image0021"
    note: str = ""                # how the points were obtained / caveats

    @property
    def labels(self) -> set[str]:
        return set(self.points["label"])


def _sheet_id(value) -> str:
    """The sheet's spineid as text ('3', never '3.0')."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _image_name(image) -> str:
    return str(getattr(image, "file_name", "")).rsplit(".", 1)[0]


def _load(catalog, recording, cache: dict):
    """The image of a recording (its sheet row's first ophysfile), or None; cached per row."""
    row_label = recording.traces.catalog_row
    if row_label not in cache:
        cache[row_label] = None
        if pd.notna(recording.value("ophysfile")):
            try:
                cache[row_label] = catalog.load_imaging(row_label)
            except Exception as error:     # missing / unreadable image: just not used
                logger.debug("Image of row %s not usable: %s", row_label, error)
    return cache[row_label]


def _from_cluster_images(catalog, analysis, wanted: set[str], cache: dict) -> SpineMap | None:
    """Rule 1: the best-named cluster image (most single spines covered)."""
    best = None
    for cluster in sorted(analysis.clusters(), key=lambda c: -c.n_spines):
        for recording in analysis.recordings_of(cluster):
            image = _load(catalog, recording, cache)
            if image is None or not len(image.uncaging_points):
                continue
            labels = label_uncaging_targets(image, catalog)
            if not labels.resolved:
                continue
            points = image.uncaging_points.reset_index(drop=True)
            named = [(float(p.x_px), float(p.y_px), label) for p, label in zip(points.itertuples(), labels.labels)
                     if label]                 # None = moved off the dendrite, "" = no id in the sheet
            if not named:
                continue
            spine_map = SpineMap(image, pd.DataFrame(named, columns=["x_px", "y_px", "label"]),
                                 f"{cluster.label} · {_image_name(image)}", labels.note)
            if wanted <= spine_map.labels:
                return spine_map
            if best is None or len(spine_map.labels & wanted) > len(best.labels & wanted):
                best = spine_map
            break                              # one image per cluster is enough
    return best


def _from_single_spine_images(catalog, analysis, cache: dict) -> SpineMap | None:
    """Rule 2: each single spine's own laser point, drawn on the first single spine's image."""
    background, points, sources = None, [], []
    for single in analysis.inputs:
        if single.kind != "unitary":
            continue
        for recording in analysis.recordings_of(single):
            image = _load(catalog, recording, cache)
            if image is None or len(image.uncaging_points) != 1:
                continue
            if background is None:
                background = image
            same_frame = (image.width_px == background.width_px and image.height_px == background.height_px
                          and np.isclose(image.pixel_size_x_um or 0, background.pixel_size_x_um or 0, rtol=0.01))
            if not same_frame:
                continue
            point = image.uncaging_points.iloc[0]
            points.append((float(point["x_px"]), float(point["y_px"]), _sheet_id(single.sheet_spineid)))
            sources.append(_image_name(image))
            break
    if background is None:
        return None
    return SpineMap(background, pd.DataFrame(points, columns=["x_px", "y_px", "label"]),
                    f"single-spine images · background {_image_name(background)}",
                    "each spine's point from its own single-spine image (" + ", ".join(sources)
                    + "); positions approximate (the dendrite can drift between images)")


def spine_map_for_set(catalog, analysis, cache: dict | None = None) -> SpineMap | None:
    """
    The spine map of one analysed set (``SetSummationAnalysis``); see module docstring.
    ``cache``: images already loaded (row -> image), shared with ``cluster_spreads``.
    """
    wanted = {_sheet_id(s.sheet_spineid) for s in analysis.inputs if s.kind == "unitary"}
    cache = {} if cache is None else cache
    from_cluster = _from_cluster_images(catalog, analysis, wanted, cache)
    if from_cluster is not None and wanted <= from_cluster.labels:
        return from_cluster
    from_singles = _from_single_spine_images(catalog, analysis, cache)
    candidates = [m for m in (from_cluster, from_singles) if m is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda m: len(m.labels & wanted))


# =============================================================================
# Cluster spread (see module docstring)
# =============================================================================
@dataclass
class ClusterSpread:
    spread_um: float              # mean distance of the spines from their centroid (µm)
    source: str                   # where the positions came from
    n_positions: int
    note: str = ""


def spread_um(x_px, y_px, pixel_size_x_um: float, pixel_size_y_um: float | None = None) -> float:
    """Mean distance (µm) of the points from their centroid."""
    x = np.asarray(x_px, dtype=float) * pixel_size_x_um
    y = np.asarray(y_px, dtype=float) * (pixel_size_y_um or pixel_size_x_um)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.hypot(x - x.mean(), y - y.mean())))


def _spread_from_cluster_image(catalog, analysis, cluster, cache: dict) -> ClusterSpread | None:
    """Rule 1: the on-spine laser points of the cluster's own image."""
    for recording in analysis.recordings_of(cluster):
        image = _load(catalog, recording, cache)
        if image is None or len(image.uncaging_points) < 2 or not image.pixel_size_x_um:
            continue
        labels = label_uncaging_targets(image, catalog)
        points = image.uncaging_points.reset_index(drop=True)
        on_spine = points[np.asarray(labels.on_spine, dtype=bool)] if labels.resolved else points
        if len(on_spine) != cluster.n_spines:
            continue                     # e.g. a shared reference image: not this recording's points
        return ClusterSpread(spread_um(on_spine["x_px"], on_spine["y_px"], image.pixel_size_x_um,
                                       image.pixel_size_y_um),
                             f"cluster image {_image_name(image)}", len(on_spine))
    return None


def _spread_from_single_spine_images(catalog, analysis, members, cache: dict) -> ClusterSpread | None:
    """Rule 2: one point from each member spine's own single-spine image."""
    singles = {s.spines[0]: s for s in analysis.inputs if s.kind == "unitary"}
    first, xs, ys = None, [], []
    for spine in members:
        if spine not in singles:
            return None
        for recording in analysis.recordings_of(singles[spine]):
            image = _load(catalog, recording, cache)
            if image is None or len(image.uncaging_points) != 1 or not image.pixel_size_x_um:
                continue
            first = first or image
            if (image.width_px, image.height_px) != (first.width_px, first.height_px) or not np.isclose(
                    image.pixel_size_x_um, first.pixel_size_x_um, rtol=0.01):
                continue
            xs.append(float(image.uncaging_points.iloc[0]["x_px"]))
            ys.append(float(image.uncaging_points.iloc[0]["y_px"]))
            break
        else:
            return None                  # no usable image for this spine
    return ClusterSpread(spread_um(xs, ys, first.pixel_size_x_um, first.pixel_size_y_um),
                         "single-spine images (approximate)", len(xs),
                         "cluster spread from the single-spine images: approximate (drift between images)")


def cluster_spreads(catalog, analysis, cache: dict | None = None) -> dict[str, ClusterSpread]:
    """Input label -> ``ClusterSpread`` for every cluster of the set (see module docstring)."""
    cache = {} if cache is None else cache
    result = {}
    for cluster in analysis.clusters():
        members = analysis.members.get(cluster.label, cluster.spines or ())
        spread = _spread_from_cluster_image(catalog, analysis, cluster, cache)
        if spread is None and len(members) == cluster.n_spines:
            spread = _spread_from_single_spine_images(catalog, analysis, members, cache)
        result[cluster.label] = spread or ClusterSpread(np.nan, "no positions", 0,
                                                        "cluster spread: no image with the spines' positions")
    return result
