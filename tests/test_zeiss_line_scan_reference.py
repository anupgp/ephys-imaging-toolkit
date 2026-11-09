"""Tests for locating a Zeiss line scan on a reference image (zeiss_line_scan_reference.py)."""

import math

import pytest

from ephys_imaging_data_loader.file_readers.zeiss_czi_file_reader import read_zeiss_czi_file
from ephys_imaging_data_loader.file_readers.zeiss_line_scan_reference import (
    find_line_scan_reference,
    scan_line_on_reference,
)


def test_line_geometry_offset_rotation_and_length():
    reference = {"scan_rotation_deg": -20.0, "scan_offset_x_um": 0.0, "scan_offset_y_um": 0.0}
    # same offset: horizontal line through the centre, 128 px of 0.0832 um = 4 reference px each
    line = dict(reference)
    (x0, y0), (x1, y1) = scan_line_on_reference(line, 0.0832, 128, reference, 0.0208, (512, 512))
    assert (y0, y1) == pytest.approx((255.5, 255.5)) and (x0 + x1) / 2 == pytest.approx(255.5)
    assert x1 - x0 == pytest.approx(127 * 4)
    # synthetic offset (0.50, -0.40) um at -20 deg: R(-20 deg) . offset / 0.0439 um per px -> (+7.6, -12.5) px
    line = {**reference, "scan_offset_x_um": 0.50, "scan_offset_y_um": -0.40}
    (x0, y0), (x1, y1) = scan_line_on_reference(line, 0.176, 128, reference, 0.0439, (512, 512))
    assert y0 - 255.5 == pytest.approx(-12.46, abs=0.05) and (x0 + x1) / 2 - 255.5 == pytest.approx(7.59, abs=0.05)
    # a rotation difference tilts the line by that angle
    line = {**reference, "scan_rotation_deg": -10.0}
    (x0, y0), (x1, y1) = scan_line_on_reference(line, 0.0832, 128, reference, 0.0208, (512, 512))
    assert math.degrees(math.atan2(y1 - y0, x1 - x0)) == pytest.approx(10.0)


