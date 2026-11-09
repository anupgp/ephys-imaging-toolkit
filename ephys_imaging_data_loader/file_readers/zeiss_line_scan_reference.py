"""
zeiss_line_scan_reference.py
============================

Where was a Zeiss LSM880 line scan placed? This module finds a reference
image (frame) for a line scan and the position of the scanned line on it, so
a viewer can show which part of the kymograph is the spine and which the
dendrite.

The reference image
-------------------
ZEN embeds a reference frame (an "Image" attachment, 512 x 512 px) only in
SINGLE line scans, not in line-scan TIME SERIES. For a file without one, ``find_line_scan_reference``
borrows the reference of another file in the SAME folder that imaged the same
field:
  * same objective,
  * same scan rotation (within 1 degree)       - otherwise the line would be
                                                 tilted on the reference,
  * same focus (OffsetZ, within 1 um),
  * the line overlaps the reference's field,
and, among those, the one acquired closest in time. The viewer says which
file the reference came from and how many minutes apart they were.

Geometry
--------
A line scan is the horizontal centre line of a rotated scan field. Its
reference frame has its own zoom, rotation and scanner offset
(SampleOffsetX/Y, um in the sample). When the two rotations are equal, the line
is horizontal on the reference and its centre lies at

    centre_px = image centre + R(rotation_ref) . (offset_line - offset_ref) / ref_pixel_um

with R the usual 2-D rotation matrix; it runs left -> right (position 0 of the
kymograph on the left). The prediction was checked by matching the
time-averaged line profile against the reference image. The line length is
n_px * line pixel size, so a low-zoom line can be longer than a high-zoom
reference: then only part of it is inside the reference.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MAX_ROTATION_DIFFERENCE_DEG = 1.0
MAX_FOCUS_DIFFERENCE_UM = 1.0


@dataclass
class LineScanReference:
    """A reference frame for a line scan and where the line lies on it."""

    image: np.ndarray                   # (C, Y, X)
    pixel_size_um: float
    geometry: dict                      # zoom, scan_rotation_deg, scan_offset_x_um, scan_offset_y_um
    source_file: str                    # file the reference image comes from
    borrowed: bool                      # True: from another file of the same folder
    line_start_xy: tuple[float, float] = (0.0, 0.0)   # kymograph position 0, reference pixels
    line_end_xy: tuple[float, float] = (0.0, 0.0)     # last kymograph position
    time_difference_s: float | None = None             # reference file time - line scan time
    notes: list[str] = field(default_factory=list)

    def point_at(self, position_px: float, n_positions: int) -> tuple[float, float]:
        """Reference-image (x, y) of a kymograph position (0 .. n_positions - 1)."""
        fraction = position_px / max(n_positions - 1, 1)
        (x0, y0), (x1, y1) = self.line_start_xy, self.line_end_xy
        return x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)

    def positions_inside(self, n_positions: int) -> tuple[int, int] | None:
        """First and last kymograph positions that fall inside the reference image (None: none)."""
        height, width = self.image.shape[-2:]
        # tolerance: half a line pixel (a line pixel covers several reference pixels)
        (x0, y0), (x1, y1) = self.line_start_xy, self.line_end_xy
        slack = 0.5 * math.hypot(x1 - x0, y1 - y0) / max(n_positions - 1, 1)
        inside = [p for p in range(n_positions)
                  if -slack <= self.point_at(p, n_positions)[0] <= width - 1 + slack
                  and -slack <= self.point_at(p, n_positions)[1] <= height - 1 + slack]
        return (inside[0], inside[-1]) if inside else None

    def describe(self) -> str:
        """One line for a figure title."""
        if not self.borrowed:
            return f"reference image of {self.source_file}"
        minutes = (self.time_difference_s or 0) / 60
        when = f"{abs(minutes):.1f} min {'later' if minutes >= 0 else 'earlier'}" if self.time_difference_s is not None else ""
        return f"reference borrowed from {self.source_file} ({when}; same focus and rotation)"


def scan_line_on_reference(
    line_geometry: dict,
    line_pixel_um: float,
    n_line_px: int,
    reference_geometry: dict,
    reference_pixel_um: float,
    reference_shape_yx: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Start (position 0) and end (last position) of the scanned line, in
    reference-image pixels (x = column, y = row; see the module docstring).
    """
    height, width = reference_shape_yx
    angle = math.radians(reference_geometry.get("scan_rotation_deg") or 0.0)
    dx = (line_geometry.get("scan_offset_x_um") or 0.0) - (reference_geometry.get("scan_offset_x_um") or 0.0)
    dy = (line_geometry.get("scan_offset_y_um") or 0.0) - (reference_geometry.get("scan_offset_y_um") or 0.0)
    centre_x = (width - 1) / 2 + (math.cos(angle) * dx - math.sin(angle) * dy) / reference_pixel_um
    centre_y = (height - 1) / 2 + (math.sin(angle) * dx + math.cos(angle) * dy) / reference_pixel_um
    # a small rotation difference tilts the line (equal rotations: horizontal)
    tilt = math.radians((line_geometry.get("scan_rotation_deg") or 0.0) - (reference_geometry.get("scan_rotation_deg") or 0.0))
    half_length = (n_line_px - 1) / 2 * line_pixel_um / reference_pixel_um
    step_x, step_y = math.cos(tilt) * half_length, math.sin(tilt) * half_length
    return (centre_x - step_x, centre_y - step_y), (centre_x + step_x, centre_y + step_y)


def _reference_of(recording) -> LineScanReference | None:
    """The recording's own embedded reference (None if it has none)."""
    geometry = recording.metadata.get("reference_image_geometry")
    if recording.reference_image is None or not recording.reference_image_pixel_size_um or not geometry:
        return None
    return LineScanReference(image=recording.reference_image, pixel_size_um=recording.reference_image_pixel_size_um,
                             geometry=geometry, source_file=recording.file_name, borrowed=False)


@functools.lru_cache(maxsize=8)
def _folder_references(folder: str, folder_signature: tuple) -> list:
    """(file name, recording) of every .czi in the folder that embeds a reference image (cached)."""
    from .zeiss_czi_file_reader import read_zeiss_czi_file

    found = []
    for name, _ in folder_signature:
        try:
            other = read_zeiss_czi_file(Path(folder) / name)
        except Exception as error:      # an unreadable file must not stop the search
            logger.info("Skipped %s while looking for reference images: %s", name, error)
            continue
        if other.reference_image is not None and other.metadata.get("reference_image_geometry"):
            found.append((name, other))
    return found


def find_line_scan_reference(recording, borrow_from_folder: bool = True) -> LineScanReference | None:
    """
    Reference image for a line scan with the line's position on it: the file's
    own reference, else (``borrow_from_folder``) the best match in the same
    local folder (see the module docstring). None when nothing fits.
    """
    if not recording.is_line_scan or not recording.pixel_size_x_um:
        return None
    meta = recording.metadata
    n_positions = recording.width_px
    reference = _reference_of(recording)

    if reference is None and borrow_from_folder and recording.local_file_path:
        folder = Path(recording.local_file_path).parent
        signature = tuple(sorted((p.name, p.stat().st_mtime) for p in folder.glob("*.czi")
                                 if p.resolve() != Path(recording.local_file_path).resolve()))
        candidates = []
        for name, other in _folder_references(str(folder), signature):
            other_meta = other.metadata
            geometry = other_meta["reference_image_geometry"]
            # whitespace differs between files ("20x/1.0  DIC" vs "20x/1.0 DIC")
            if " ".join(str(other_meta.get("objective_model")).split()) != " ".join(str(meta.get("objective_model")).split()):
                continue
            if abs((geometry.get("scan_rotation_deg") or 0) - (meta.get("scan_rotation_deg") or 0)) > MAX_ROTATION_DIFFERENCE_DEG:
                continue
            focus, other_focus = meta.get("focus_offset_z_um"), other_meta.get("focus_offset_z_um")
            if focus is None or other_focus is None or abs(focus - other_focus) > MAX_FOCUS_DIFFERENCE_UM:
                continue
            candidate = LineScanReference(image=other.reference_image, pixel_size_um=other.reference_image_pixel_size_um,
                                          geometry=geometry, source_file=name, borrowed=True)
            candidate.line_start_xy, candidate.line_end_xy = scan_line_on_reference(
                meta, recording.pixel_size_x_um, n_positions, geometry, candidate.pixel_size_um, candidate.image.shape[-2:])
            if candidate.positions_inside(n_positions) is None:
                continue
            if recording.acquisition_datetime and other.acquisition_datetime:
                candidate.time_difference_s = (other.acquisition_datetime - recording.acquisition_datetime).total_seconds()
            candidates.append(candidate)
        if candidates:
            reference = min(candidates, key=lambda c: abs(c.time_difference_s) if c.time_difference_s is not None else math.inf)

    if reference is None:
        return None
    if not reference.borrowed:
        reference.line_start_xy, reference.line_end_xy = scan_line_on_reference(
            meta, recording.pixel_size_x_um, n_positions, reference.geometry, reference.pixel_size_um,
            reference.image.shape[-2:])
    inside = reference.positions_inside(n_positions)
    if inside and inside != (0, n_positions - 1):
        reference.notes.append(f"line positions {inside[0]}-{inside[1]} of 0-{n_positions - 1} lie inside this image")
    return reference
