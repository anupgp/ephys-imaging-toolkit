"""Tests for reading Zeiss .czi files into ImageRecording."""

import collections

import czifile
import numpy as np
import pytest

from ephys_imaging_data_loader.file_readers.zeiss_czi_file_reader import (
    read_zeiss_czi_file,
    reorder_zeiss_array_to_tczyx,
    standard_shape_from_zeiss_sizes,
)
from ephys_imaging_data_loader.recording_data_containers.image_recording import STANDARD_AXIS_ORDER


def test_standard_shape_from_sizes():
    sizes = {"B": 1, "V": 1, "C": 2, "T": 5200, "Z": 1, "Y": 1, "X": 128, "S": 1}
    assert standard_shape_from_zeiss_sizes(sizes) == (5200, 2, 1, 1, 128)
    with pytest.raises(ValueError):
        standard_shape_from_zeiss_sizes({"B": 2, "C": 1, "Y": 4, "X": 4})


def test_rgb_samples_become_channels():
    """A colour camera image (S = 3) is shown as 3 channels."""
    rgb = np.arange(4 * 5 * 3).reshape(1, 4, 5, 3)  # C Y X S
    reordered = reorder_zeiss_array_to_tczyx(rgb, ("C", "Y", "X", "S"))
    assert reordered.shape == (1, 3, 1, 4, 5)
    assert np.array_equal(reordered[0, 2, 0], rgb[0, :, :, 2])
