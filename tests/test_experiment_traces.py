"""Tests for experiment_traces (all traces of one experiment sample) and its viewer, on synthetic traces."""

import numpy as np
import pandas as pd
import pytest

from ephys_imaging_data_loader.experiment_catalog.experiment_traces import (
    ExperimentTraces,
    ExperimentTraceSet,
    format_value,
)
from ephys_imaging_data_loader.experiment_catalog.traces_by_isi import IsiTraceSet


def _trace_set(isi, group, spine, values, unit="pA", n_sweeps=2, before=-50.0):
    """One fake file: n_sweeps sweeps of constant ``values`` from ``before`` to isi + 150 ms at 10 kHz."""
    time_ms = np.arange(before, isi + 150, 0.1)
    traces = np.array([np.full(len(time_ms), values + k) for k in range(n_sweeps)], dtype=float)
    row = pd.Series({"isi": float(isi), "group": group, "spineid": spine, "expname": "test"}, name=spine * 100 + isi)
    isi_set = IsiTraceSet(
        isi_ms=float(isi), group=group, file_name=f"f{spine}_{isi}_{group}", catalog_row=int(row.name),
        channel="IN0", unit=unit, sweeps=list(range(n_sweeps)), time_ms=time_ms, traces=traces,
        stimulus_onsets_ms=np.array([0.0, float(isi)]), decoded_interval_ms=float(isi), stimulus_source="test",
    )
    return ExperimentTraceSet(row=row, traces=isi_set)


@pytest.fixture
def data():
    sets = [
        _trace_set(10, "ctrl", 1, -100), _trace_set(10, "ctz", 1, -200),
        _trace_set(100, "ctrl", 1, -110), _trace_set(100, "ctrl", 2, -120),
    ]
    return ExperimentTraces(sets, criteria={}, metadata={"expdate": "20250125"})


def test_values_sorted_numerically(data):
    assert data.values("isi") == [10.0, 100.0]
    assert data.values("group") == ["ctrl", "ctz"]
    assert data.n_sweeps == 8


def test_average_by_isi(data):
    parts = data.grouped("isi")
    assert [p.label for p in parts] == ["10 ms", "100 ms"]
    assert [p.n_sweeps for p in parts] == [4, 4]
    # 10 ms: sweeps -100, -99, -200, -199 -> mean -149.5
    assert np.allclose(parts[0].mean, -149.5)


def test_average_by_isi_and_group(data):
    labels = [p.label for p in data.grouped(["isi", "group"])]
    assert labels == ["10 ms · ctrl", "10 ms · ctz", "100 ms · ctrl"]


def test_no_averaging_is_one_part(data):
    parts = data.grouped(None)
    assert len(parts) == 1 and parts[0].n_sweeps == 8


def test_different_lengths_padded_with_nan(data):
    # group "ctrl" mixes 10 ms (window to 160 ms) and 100 ms files (window to 250 ms)
    ctrl = [p for p in data.grouped("group") if p.label == "ctrl"][0]
    late = ctrl.time_ms > 200
    # after 160 ms only the 100-ms sweeps (-110, -109, -120, -119) have data
    assert np.allclose(ctrl.mean[late], np.mean([-110, -109, -120, -119]))
    assert np.isnan(ctrl.sweeps[0, late]).all()


def test_units_never_averaged_together():
    sets = [_trace_set(50, "ctrl", 1, -70, unit="mV"), _trace_set(50, "ctrl", 1, -100, unit="pA")]
    parts = ExperimentTraces(sets, {}, {}).grouped("isi")
    assert len(parts) == 2
    assert {p.unit for p in parts} == {"mV", "pA"}


def test_baseline_subtraction(data):
    zeroed = data.baseline_subtracted()
    for trace_set in zeroed.trace_sets:
        assert np.allclose(trace_set.traces.traces, 0.0)
    assert data.trace_sets[0].traces.traces[0, 0] == -100   # original unchanged


def test_subset(data):
    assert len(data.subset(isi=[10]).trace_sets) == 2
    assert len(data.subset(group=["ctrl"], spineid=[2]).trace_sets) == 1
    assert len(data.subset(isi=None).trace_sets) == 4


def test_format_value():
    assert format_value("isi", 30.0) == "30 ms"
    assert format_value("spineid", 3) == "spine 3"
    assert format_value("group", "ctz") == "ctz"


def test_viewer_builds_and_saves(data, tmp_path):
    from ephys_imaging_data_loader.interactive_viewers.experiment_traces_viewer import (
        _panels,
        build_experiment_traces_figure,
        build_experiment_traces_viewer,
        save_experiment_traces_figure,
    )

    build_experiment_traces_figure(data)
    build_experiment_traces_figure(data, average_by=["isi", "group"], panel_by="spineid", baseline_subtract=True)
    build_experiment_traces_viewer(data, average_by="isi")
    assert [title for title, _ in _panels(data, "spineid")] == ["spine 1", "spine 2"]
    path = save_experiment_traces_figure(data, tmp_path / "traces.png", average_by="isi", dpi=50)
    assert path.exists()


def test_mixed_units_get_separate_panels():
    from ephys_imaging_data_loader.interactive_viewers.experiment_traces_viewer import _panels

    sets = [_trace_set(50, "ctrl", 1, -70, unit="mV"), _trace_set(50, "ctrl", 1, -100, unit="pA")]
    titles = [title for title, _ in _panels(ExperimentTraces(sets, {}, {}), None)]
    assert titles == ["mV", "pA"]
