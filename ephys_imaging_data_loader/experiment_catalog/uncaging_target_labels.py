"""
uncaging_target_labels.py
=========================

Names the uncaging (photostimulation) targets of an Olympus image after the
SPINES they belong to, using the experiment sheet (``spineid`` column),
instead of the meaningless point index 0, 1, 2 ... stored in the file.

How targets are labelled (since 15 Oct 2026: ALWAYS the sheet's spine ids)
-------------------------------------------------------------------------
The labels are the spine ids exactly as written in the sheet's ``spineid``;
they are never inferred from positions (matching to the single-spine images
proved unreliable: the dendrite moves between images by about the spacing of
spines). A label "" = a point on a spine for which the sheet gives no id.

1. Find the sheet rows of the image: rows whose ``ophysfile`` contains the
   image name, on the image's date (and cell folder, e.g. ".../c1/").
2. ONE target and one spine id in those rows  -> label = that spineid.
3. As many targets as digits in spineid (12345 for 5 points) -> the digits,
   in scan order (point 1 = first digit ...).
4. spineid 0 ("all spines")                   -> every point "" (no id per point).
5. Fewer digits than points (shrinking cluster) -> which points are still on
   the dendrite is found by comparison with the session's full-cluster image
   (see ``_shrinking_cluster_labels``); they get the spineid digits in scan
   order, the moved-off points get None.
6. Rows of different spines share the image (a reference image), or the
   image is not in the sheet -> every point "", ``resolved=False``.

The result carries a short ``note`` explaining which rule was used, shown in
the viewer caption, so a label is never silently guessed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..recording_data_containers.image_recording import ImageRecording

logger = logging.getLogger(__name__)

# Targets of a cluster image and of the single-spine image of the same spine
# may differ by tissue drift; observed differences were <= ~15 px (512 px images).
MAX_MATCH_DISTANCE_PX = 40.0
# In shrinking-cluster series (e.g. spines 12345678, 1234567, ..., 12) the
# targets that are no longer wanted are MOVED AWAY from the dendrite (they stay
# active). Compared with the full-cluster image of the same session, a target
# still on its spine moved <= ~22 px (drift), a moved-away target >= ~113 px.
MOVED_AWAY_DISTANCE_PX = 60.0


@dataclass
class TargetLabels:
    labels: list[str | None]   # one label per target (file order); None = moved away from the spines
    note: str                  # how the labels were obtained (shown in the viewer)
    on_spine: list[bool] | None = None  # per target: still on a spine? (None = all on spines)
    resolved: bool = True      # False when it is unknown which targets were stimulated

    def __post_init__(self):
        if self.on_spine is None:
            self.on_spine = [True] * len(self.labels)

    @property
    def n_on_spine(self) -> int:
        return sum(self.on_spine)

    @property
    def last_on_spine_position(self) -> int | None:
        """1-based scan position of the last on-spine target (None if none)."""
        positions = [i + 1 for i, on in enumerate(self.on_spine) if on]
        return max(positions) if positions else None


def _id_text(value) -> str:
    return str(int(value)) if pd.notna(value) else "?"


def _image_identity(image: ImageRecording) -> tuple[str | None, int | None]:
    """(expdate 'YYYYMMDD', neuron id) of an image, from the catalog row, its folder or its date."""
    row = image.metadata.get("catalog_row")
    if row:
        return str(row.get("expdate")), int(row["neuronid"]) if pd.notna(row.get("neuronid")) else None

    # Folder layout <expdate>/c<neuron>/<ImageNNNN>.oib (local path or Box cache path).
    for text in (image.source_description, str(image.local_file_path)):
        match = re.search(r"(\d{8})/[cC](\d+)/[^/]+$", text.replace("box: ", "").split(" (")[0])
        if match:
            return match.group(1), int(match.group(2))
    date = image.acquisition_datetime.strftime("%Y%m%d") if image.acquisition_datetime else None
    return date, None


def _rows_for_image(catalog, image: ImageRecording) -> pd.DataFrame:
    """Sheet rows that list this image file (any number of ';'-separated names per cell)."""
    expdate, neuron = _image_identity(image)
    if expdate is None:
        return catalog.table.iloc[0:0]
    stem = Path(image.file_name).stem.lower()
    table = catalog.table
    candidates = table[(table["expdate"] == expdate) & table["ophysfile"].notna()]
    if neuron is not None:
        candidates = candidates[candidates["neuronid"] == neuron]
    listed = candidates["ophysfile"].map(
        lambda value: stem in [Path(name.strip()).stem.lower() for name in str(value).split(";")]
    )
    return candidates[listed]


def _single_spine_positions(catalog, rows: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    spineid -> (x, y) target position from the single-target images of the same
    date / neuron / dendrite, preferring the same group as ``rows``.
    """
    first = rows.iloc[0]
    table = catalog.table
    same_place = table[
        (table["expdate"] == first["expdate"]) & (table["neuronid"] == first["neuronid"])
        & (table["dendriteid"] == first["dendriteid"]) & (table["spineid"] > 0) & table["ophysfile"].notna()
    ]
    groups = set(rows["group"].dropna())
    # same group first (positions drift between e.g. pre-apv and post-apv)
    ordered = pd.concat([same_place[same_place["group"].isin(groups)], same_place[~same_place["group"].isin(groups)]])

    positions: dict[str, tuple[float, float]] = {}
    for label, row in ordered.iterrows():
        spine = _id_text(row["spineid"])
        if spine in positions:
            continue
        try:
            single = catalog.load_imaging(label)
        except Exception as error:  # missing image: just not used for matching
            logger.debug("Single-spine image of row %s not usable: %s", label, error)
            continue
        if len(single.uncaging_points) == 1:
            point = single.uncaging_points.iloc[0]
            positions[spine] = (float(point["x_px_bleach_gui"]), float(point["y_px_bleach_gui"]))
    return positions


def _match_by_position(targets: np.ndarray, positions: dict[str, tuple[float, float]]) -> list[str] | None:
    """One-to-one nearest-neighbour matching; None if any match is too far or ambiguous."""
    if len(positions) < len(targets):
        return None
    spines = list(positions)
    reference = np.array([positions[s] for s in spines])
    distances = np.linalg.norm(targets[:, None, :] - reference[None, :, :], axis=2)
    labels: list[str | None] = [None] * len(targets)
    used: set[int] = set()
    # assign the closest pairs first
    for flat_index in np.argsort(distances, axis=None):
        target, spine = divmod(int(flat_index), len(spines))
        if labels[target] is not None or spine in used:
            continue
        if distances[target, spine] > MAX_MATCH_DISTANCE_PX:
            return None
        labels[target] = spines[spine]
        used.add(spine)
    return labels if all(labels) else None


def label_uncaging_targets(image: ImageRecording, catalog=None) -> TargetLabels:
    """
    Spine-id labels for the uncaging targets of ``image`` (see module docstring).

    ``catalog`` defaults to the experiment sheet configured in the settings.
    """
    points = image.uncaging_points
    n_targets = len(points)
    # "" = on a spine, but the sheet gives no id for this point (circle without number)
    unnamed = [""] * n_targets
    if n_targets == 0:
        return TargetLabels([], "")

    if catalog is None:
        from .ephys_experiment_catalog import get_default_catalog

        catalog = get_default_catalog()
    if catalog is None:
        return TargetLabels(unnamed, "no experiment sheet configured: points not named", resolved=False)

    rows = _rows_for_image(catalog, image)
    if rows.empty:
        return TargetLabels(unnamed, "image not found in the experiment sheet: points not named", resolved=False)

    spine_ids = sorted({_id_text(v) for v in rows["spineid"].dropna()})
    if n_targets == 1 and len(spine_ids) == 1:
        return TargetLabels(spine_ids, "spine id from the experiment sheet")

    # What each row says was stimulated: spineid 0, or a list covering as many
    # spines as the image has targets (e.g. 12345 for 5 targets), both mean "all
    # targets" (the same recording is sometimes listed under two experiments,
    # once as 0 and once as 12345). Anything else is the list of spines.
    def stimulated(value) -> str:
        value = int(value)
        return "all" if value == 0 or len(str(value)) == n_targets else str(value)

    meanings = sorted({stimulated(v) for v in rows["spineid"].dropna()})
    # Rows that disagree (e.g. 20251221: spines 1..9 each list one 9-target image):
    # which targets were stimulated in each recording cannot be told from the image.
    if len(meanings) > 1:
        return TargetLabels(
            unnamed,
            f"several sheet rows (spines {', '.join(spine_ids)}) share this image (a reference image), "
            "so which points were stimulated is unknown; points not named",
            resolved=False,
        )

    # The labels are ALWAYS the sheet's spine ids, never inferred: matching
    # positions to the single-spine images is unreliable (the dendrite moves
    # between images by about the spacing of spines).
    full_list = [r for r in rows["spineid"].dropna() if int(r) != 0 and len(str(int(r))) == n_targets]
    if full_list:                       # e.g. 12345 for 5 points (also when the same file is listed as 0)
        return TargetLabels(list(str(int(full_list[0]))), "spine ids from the sheet's spineid, in scan order")
    spine_value = 0 if not meanings or meanings[0] == "all" else int(meanings[0])
    if spine_value == 0:
        return TargetLabels(unnamed, "spineid 0 (all spines): the sheet gives no id per point")

    # Fewer spines than points (shrinking cluster): the others were moved off the dendrite.
    targets = points[["x_px_bleach_gui", "y_px_bleach_gui"]].to_numpy(dtype=float)
    listed = [int(ch) for ch in str(spine_value)]
    return _shrinking_cluster_labels(catalog, rows, targets, listed, unnamed)


def _shrinking_cluster_labels(catalog, rows, targets: np.ndarray, listed: list[int], order_labels) -> TargetLabels:
    """
    Which targets are still on spines, judged against the FULL-cluster image of
    the same session (same date / neuron / dendrite, preferably same group,
    nearest in the sheet): a target that stayed (<= MOVED_AWAY_DISTANCE_PX) is
    on its spine, the others were moved away. The number of targets that stayed
    must equal the number of spines in spineid, otherwise nothing is guessed.
    """
    n_targets = len(targets)
    first = rows.iloc[0]
    table = catalog.table
    session = table[
        (table["expdate"] == first["expdate"]) & (table["neuronid"] == first["neuronid"])
        & (table["dendriteid"] == first["dendriteid"]) & table["ophysfile"].notna()
    ]

    def spine_count(value) -> int:
        return 0 if pd.isna(value) or int(value) == 0 else len(str(int(value)))

    full = session[[spine_count(v) == n_targets for v in session["spineid"]]]
    if full.empty:
        return TargetLabels(order_labels, f"{len(listed)} of {n_targets} targets on spines, but no full-cluster image "
                            "of this session to tell which; all targets drawn", resolved=False)
    same_group = full[full["group"] == first["group"]]
    candidates = same_group if not same_group.empty else full
    reference_label = min(candidates.index, key=lambda label: abs(label - rows.index[0]))
    try:
        reference = catalog.load_imaging(reference_label).uncaging_points[["x_px_bleach_gui", "y_px_bleach_gui"]]
    except Exception as error:
        return TargetLabels(order_labels, f"full-cluster image not readable ({error})", resolved=False)
    if len(reference) != n_targets:
        return TargetLabels(order_labels, "full-cluster image has a different number of targets", resolved=False)

    # A point is on the dendrite when it lies near ANY point of the full cluster
    # (not only the point with the same number): a point may also have been moved
    # onto another spine (e.g. point 1 placed on the spine of full-cluster point 4).
    reference_xy = reference.to_numpy(dtype=float)
    moved = np.linalg.norm(targets[:, None, :] - reference_xy[None, :, :], axis=2).min(axis=1)
    on_spine = [bool(distance <= MOVED_AWAY_DISTANCE_PX) for distance in moved]
    if sum(on_spine) != len(listed):
        return TargetLabels(order_labels, f"{sum(on_spine)} targets stayed on the dendrite but spineid lists "
                            f"{len(listed)} spines; all targets drawn", resolved=False)

    # the sheet's spine ids, in the sheet's order, on the points that stayed (in scan order)
    spine_labels = iter(str(x) for x in listed)
    labels = [next(spine_labels) if on else None for on in on_spine]
    note = (f"{len(listed)} of {n_targets} targets on spines (others moved off the dendrite, compared with the "
            "full cluster); spine ids from the sheet, in scan order")
    return TargetLabels(labels, note, on_spine=on_spine)
