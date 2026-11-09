"""Tests for reading Axon .abf files into EphysRecording."""

import numpy as np
import pytest

from ephys_imaging_data_loader.file_readers.axon_abf_file_reader import read_axon_abf_file


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_axon_abf_file(tmp_path / "nope.abf")
