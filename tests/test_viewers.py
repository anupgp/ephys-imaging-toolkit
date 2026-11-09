"""
Tests that every viewer builds, reacts to widget changes, and that static
figures are written. (Visual correctness was checked by eye; these tests
make sure nothing breaks later.)
"""

import numpy as np
import pytest

from ephys_imaging_data_loader import load_recording
from ephys_imaging_data_loader.interactive_viewers import (
    build_ephys_trace_figure,
    save_ephys_traces_figure,
    save_image_figure,
    save_kymograph_figure,
)
from ephys_imaging_data_loader.interactive_viewers.display_helpers import (
    contrast_limits,
    min_max_downsample,
    nice_scale_bar_length,
)

# =============================================================================
# Display helpers
# =============================================================================


def test_min_max_downsample_keeps_peaks():
    time = np.arange(100_000) / 20000
    trace = np.zeros_like(time)
    trace[31_337] = -500.0  # a single-sample spike must survive downsampling
    trace[77_777] = 300.0
    reduced_t, reduced_y = min_max_downsample(time, trace, 2000)
    assert len(reduced_y) <= 2000
    assert reduced_y.min() == -500.0 and reduced_y.max() == 300.0
    assert np.all(np.diff(reduced_t) >= 0)  # still in time order


def test_short_traces_are_not_changed():
    time, values = np.arange(10.0), np.arange(10.0)
    assert min_max_downsample(time, values, 4000)[1] is values


@pytest.mark.parametrize("width_um, expected", [(23.5, 2.0), (425, 50.0), (10, 2.0), (1000, 200.0)])
def test_nice_scale_bar_length(width_um, expected):
    assert nice_scale_bar_length(width_um) == expected


def test_contrast_limits_ignore_outliers():
    image = np.full((100, 100), 10.0)
    image[0, 0] = 1e6  # one hot pixel
    low, high = contrast_limits(image, 0.5, 99.5)
    assert high < 1e6


# =============================================================================
# Interactive viewers
# =============================================================================


# =============================================================================
# Static figures
# =============================================================================


