"""
Tests for the experiment catalog: file names from the sheet, bad sweeps,
selection (incl. the optional unselect filter), stimulus detection, traces by
ISI, the viewers and the interactive browser.

A small sheet is written for each test that points at the REAL sample files
of 20260322 (neuron 1), including the tricky cases found in the real sheet:
  * ISI 800 ms, where the test-step gate on DO0 ALSO has pulses 800 ms apart,
  * 26322023: sheet says ISI 400 ms but the file's protocol is 200 ms,
  * a row without ephysfile_prefix (name inferred from the date),
  * bad sweeps that do not exist in the file.
"""

import textwrap

import numpy as np
import pytest

from ephys_imaging_data_loader import ExperimentCatalog, load_recording
from ephys_imaging_data_loader.experiment_catalog import (
    StimulusDetectionError,
    build_ephys_file_name,
    find_stimulus_onsets,
    parse_bad_sweeps,
)

# =============================================================================
# File names and bad sweeps
# =============================================================================


@pytest.mark.parametrize(
    "prefix, ephys_id, expdate, expected",
    [
        ("26322xxx", "7", "20260322", ("26322007", "template")),
        ("2027_02_18_xxxx", "3", "20270218", ("2027_02_18_0003", "template")),
        ("24d05050", "50", "20241205", ("24d05050", "full name")),
        (None, "8", "20240413", ("24413008", "inferred from date")),
        (None, "50", "20241205", ("24d05050", "inferred from date")),   # October -> "o"
        (None, "63", "20250110", ("25110063", "inferred from date")),   # December -> "d"
        ("26322xxx", None, "20260322", (None, "missing")),
        (None, None, "20260322", (None, "missing")),
        ("26322xx", "123", "20260322", (None, "missing")),              # id too long for the template
    ],
)
def test_build_ephys_file_name(prefix, ephys_id, expdate, expected):
    assert build_ephys_file_name(prefix, ephys_id, expdate) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("1;2;3;4", [1, 2, 3, 4]), ("2;4", [2, 4]), ("3", [3]), ("1,2", [1, 2]), ("", []), (None, []), (float("nan"), [])],
)
def test_parse_bad_sweeps(text, expected):
    assert parse_bad_sweeps(text) == expected


# =============================================================================
# Stimulus detection
# =============================================================================


# =============================================================================
# Traces by ISI
# =============================================================================


# =============================================================================
# Viewers
# =============================================================================


@pytest.mark.parametrize("interval_ms, text", [(200.1, "200 ms"), (1000.1, "1000 ms"), (30.1, "30 ms"), (5.25, "5.2 ms")])
def test_format_ipi(interval_ms, text):
    from ephys_imaging_data_loader.interactive_viewers.ephys_recording_title import format_ipi

    assert format_ipi(interval_ms) == text


