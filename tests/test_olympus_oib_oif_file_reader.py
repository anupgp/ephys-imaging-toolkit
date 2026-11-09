"""Tests for reading Olympus .oib/.oif files into ImageRecording."""

import re

import numpy as np
import pytest

from ephys_imaging_data_loader.file_readers.olympus_oib_oif_file_reader import (
    read_olympus_oib_oif_file,
    reorder_to_tczyx,
)


@pytest.mark.parametrize(
    "axes, shape, expected",
    [
        ("CYX", (2, 4, 5), (1, 2, 1, 4, 5)),
        ("CZYX", (2, 3, 4, 5), (1, 2, 3, 4, 5)),
        ("ZCYX", (3, 2, 4, 5), (1, 2, 3, 4, 5)),
        ("TZCYX", (6, 3, 2, 4, 5), (6, 2, 3, 4, 5)),
        ("LYX", (2, 4, 5), (1, 2, 1, 4, 5)),  # spectral axis L -> channels
    ],
)
def test_reorder_to_tczyx(axes, shape, expected):
    array = np.arange(np.prod(shape)).reshape(shape)
    reordered = reorder_to_tczyx(array, axes)
    assert reordered.shape == expected


def test_reorder_keeps_values_in_the_right_place():
    array = np.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)  # Z C Y X
    reordered = reorder_to_tczyx(array, "ZCYX")
    assert np.array_equal(reordered[0, 1, 2], array[2, 1])  # channel 1, slice 2


def test_reorder_refuses_unknown_non_trivial_axis():
    with pytest.raises(ValueError):
        reorder_to_tczyx(np.zeros((3, 4, 5)), "QYX")


# =============================================================================
# Regression tests: series order and axes (bug found on 2026-10-01)
#   * oiffile lists the "-R001" reference series FIRST, so the main image and
#     the reference image were swapped;
#   * the settings-file axes ("ZCYX") differ from the array's axes ("CZYX"),
#     which swapped channels and z-slices and broke partial z-projections;
#   * in line scans each TIFF holds time x position ("TX"), not "YX".
# =============================================================================
class FakeSeries(list):
    """Stand-in for oiffile's TiffSequence: iterable file names + axes/shape."""

    def __init__(self, names, axes, shape):
        super().__init__(names)
        self.axes, self.shape = axes, shape


class FakeOlympusFile:
    def __init__(self, series, settings_axes, sizes):
        self.series = series
        self.axes = settings_axes
        self.mainfile = {
            f"Axis {i} Parameters Common": {"AxisCode": code, "MaxSize": size}
            for i, (code, size) in enumerate(sizes.items())
        }


def test_series_are_identified_by_file_name():
    from ephys_imaging_data_loader.file_readers.olympus_oib_oif_file_reader import main_and_reference_series

    reference = FakeSeries(["Storage00001/s_C001-R001.tif", "Storage00001/s_C002-R002.tif"], "C", (2,))
    main = FakeSeries(["Storage00001/s_C001.tif", "Storage00001/s_C002.tif"], "C", (2,))
    assert main_and_reference_series(FakeOlympusFile([reference, main], "CYX", {})) == (1, 0)
    assert main_and_reference_series(FakeOlympusFile([main], "CYX", {})) == (0, None)


@pytest.mark.parametrize(
    "series_axes, series_shape, settings_axes, sizes, expected_axes, expected_shape, tczyx",
    [
        # z-stack: settings say ZCYX, files are s_C001Z001.tif ... -> array is C, Z, Y, X
        ("CZ", (1, 109), "ZCYX", {"X": 512, "Y": 512, "C": 1, "Z": 109}, "CZYX", (1, 109, 512, 512), (1, 1, 109, 512, 512)),
        # line scan: settings CTX, each TIFF is time x position
        ("C", (2,), "CTX", {"X": 110, "Y": 1, "C": 2, "Z": 0, "T": 5000}, "CTX", (2, 5000, 110), (5000, 2, 1, 1, 110)),
        # single frame
        ("C", (1,), "CYX", {"X": 512, "Y": 512, "C": 1}, "CYX", (1, 512, 512), (1, 1, 1, 512, 512)),
    ],
)
def test_series_axes_and_shape(series_axes, series_shape, settings_axes, sizes, expected_axes, expected_shape, tczyx):
    from ephys_imaging_data_loader.file_readers.olympus_oib_oif_file_reader import series_axes_and_shape

    fake = FakeOlympusFile([FakeSeries([], series_axes, series_shape)], settings_axes, sizes)
    axes, shape = series_axes_and_shape(fake, 0)
    assert (axes, shape) == (expected_axes, expected_shape)
    placeholder = np.broadcast_to(np.zeros((), dtype=np.uint8), shape)
    assert reorder_to_tczyx(placeholder, axes).shape == tczyx


def test_wrong_objective_calibration_is_corrected():
    """Files set to the wrong objective in FluoView store a calibration ~12x too large."""
    from ephys_imaging_data_loader.file_readers.olympus_oib_oif_file_reader import _wrong_objective_correction

    def settings(span_um, zoom):
        return {"Acquisition Parameters Common": {"ZoomValue": zoom},
                "Axis 0 Parameters Common": {"AxisCode": "X", "StartPosition": 0.0, "EndPosition": span_um,
                                             "MaxSize": 512, "PixUnit": "um"}}

    factor, note = _wrong_objective_correction(settings(507.94, 5.0), {"pixel_size_x_um": 507.94 / 512})
    assert 507.94 / 512 * factor == pytest.approx(209.5 / 5 / 512, rel=1e-3) and "wrong objective" in note
    assert _wrong_objective_correction(settings(26.1875, 8.0), {"pixel_size_x_um": 26.1875 / 512}) == (1.0, "")


def test_objective_stored_in_file_decides_the_correction():
    """The objective set in FluoView (.pty) decides; the 60x was always used."""
    from ephys_imaging_data_loader.file_readers.olympus_oib_oif_file_reader import _wrong_objective_correction

    settings = {"Acquisition Parameters Common": {"ZoomValue": 7.0}}
    assert _wrong_objective_correction(settings, {"pixel_size_x_um": 0.7086}, 60.0) == (1.0, "")
    factor, note = _wrong_objective_correction(settings, {"pixel_size_x_um": 0.7086}, 5.0)
    assert 0.7086 * factor == pytest.approx(0.0585, abs=2e-4) and "5x objective" in note
    assert _wrong_objective_correction(settings, {"pixel_size_x_um": 0.3}, 20.0)[0] == pytest.approx(1 / 3)
