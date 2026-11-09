"""
Tests for the per-set analysis (epsp_analysis/unitary_cluster_summation.py) on
synthetic sets built from KNOWN responses: single spines 1-3 and clusters whose
response is a known multiple of the sum of its spines.
"""

import numpy as np
import pandas as pd
import pytest

from ephys_imaging_data_loader.epsp_analysis import EpspParameters, measure_epsp
from ephys_imaging_data_loader.epsp_analysis.unitary_cluster_summation import (
    analyse_set,
    classify_row,
    SET_TABLE_COLUMNS,
    set_summation_table,
)
from ephys_imaging_data_loader.experiment_catalog.experiment_traces import ExperimentTraces, ExperimentTraceSet
from ephys_imaging_data_loader.experiment_catalog.traces_by_isi import IsiTraceSet

STEP_MS = 0.1
SHAPE = EpspParameters(amplitude=1.0, latency=2.0, tau_rise=2.0, tau_decay=25.0, sag_ratio=0.1,
                       tau_sag_rise=60.0, tau_sag_decay=150.0)
NO_SAG = SHAPE.with_values(sag_ratio=0.0)
SPINE_AMPLITUDES = {1: 0.5, 2: 0.7, 3: 0.6}


def _recording(spineid, targets, isi, amplitude, shape=SHAPE, unit="mV", sign=1.0, offset=-65.0, label=0,
               group="ctrl", n_sweeps=4, step_total=None):
    rng = np.random.default_rng(label)
    onsets = np.array([0.0, float(isi)])
    t = np.arange(-50.0, isi + 300.0, STEP_MS)
    clean = sum(shape.with_values(amplitude=amplitude).waveform(t, o) for o in onsets)
    traces = np.array([offset + sign * clean + rng.normal(0, 0.01, len(t)) for _ in range(n_sweeps)])
    row = pd.Series({"spineid": spineid, "uncaging_targets": str(targets), "isi": float(isi), "group": group,
                     "expname": "synthetic", "stimdur_total": step_total, "stimdur": None,
                     "stimpower": 50.0}, name=label)
    traces_set = IsiTraceSet(isi_ms=float(isi), group=group, file_name=f"f{label}", catalog_row=label, channel="IN0",
                             unit=unit, sweeps=list(range(n_sweeps)), time_ms=t, traces=traces,
                             stimulus_onsets_ms=onsets, decoded_interval_ms=float(isi), stimulus_source="test")
    return ExperimentTraceSet(row=row, traces=traces_set)


def _set(cluster_gain=1.5, **options):
    recordings = [_recording(s, 1, 1000, a, label=s, **options) for s, a in SPINE_AMPLITUDES.items()]
    summed = sum(SPINE_AMPLITUDES[s] for s in (1, 2))
    recordings.append(_recording(12, 3, 1000, cluster_gain * summed, label=12, **options))
    recordings.append(_recording(12, 3, 30, cluster_gain * summed, label=13, **options))
    return ExperimentTraces(recordings, {"expdate": "20990101", "neuronid": 1, "dendriteid": 2}, {})


def test_classify_rows():
    assert classify_row(pd.Series({"spineid": 3, "uncaging_targets": "1"}))[:3] == ("unitary", (3,), 1)
    assert classify_row(pd.Series({"spineid": 10, "uncaging_targets": "1"}))[:3] == ("unitary", (10,), 1)
    assert classify_row(pd.Series({"spineid": 234, "uncaging_targets": "8"}))[:3] == ("cluster", (2, 3, 4), 3)
    assert classify_row(pd.Series({"spineid": 0, "uncaging_targets": "5"}))[:3] == ("cluster", None, 5)
    kind, spines, n, note = classify_row(pd.Series({"spineid": 4, "uncaging_targets": "8"}))
    assert (kind, spines, n) == ("unitary", (4,), 1) and "single spine" in note


def test_supralinear_cluster_gain():
    (result,) = analyse_set(_set(cluster_gain=1.5), uncaging_offset=False)
    table = result.table()
    cluster = table[(table.kind == "cluster") & (table.isi_ms == 1000) & (table.stimid == 1)].iloc[0]
    expected_peak = measure_epsp(SHAPE.with_values(amplitude=1.2)).peak_mv
    assert cluster["linear_sum_peak"] == pytest.approx(expected_peak, rel=0.05)
    assert cluster["gain"] == pytest.approx(1.5, rel=0.05)
    assert cluster["measured_decay_tau_ms"] == pytest.approx(25.0, rel=0.15)
    assert cluster["member_spines"] == "1-2" and cluster["n_spines"] == 2
    # the cluster's 30 ms recording is deconvolved too (same input, another ISI)
    assert set(table[table.kind == "cluster"].isi_ms) == {30, 1000}
    # one row per stimulus: stimid 1 and 2 at each ISI, each with its own peak
    clusters = table[table.kind == "cluster"]
    assert sorted(zip(clusters.isi_ms, clusters.stimid)) == [(30, 1), (30, 2), (1000, 1), (1000, 2)]
    assert clusters["peak"].notna().all() and (clusters["laser_power_percent"] == 50).all()
    # the 1st-response / linear-sum values only once, on stimid 1 of the summation ISI
    assert clusters["linear_sum_peak"].notna().sum() == 1
    unitary = table[table.input == "spine 2"].iloc[0]
    assert unitary["measured_peak"] == pytest.approx(measure_epsp(SHAPE.with_values(amplitude=0.7)).peak_mv,
                                                     rel=0.05)


def test_uncaging_offset_shifts_members():
    (result,) = analyse_set(_set(step_total=10.0), uncaging_offset=True)
    summed = result.linear_sums["cluster 1-2"]
    # 3 laser targets for 2 spines (shrinking cluster): spine k = target k, 10 ms / 3 targets each
    assert summed.shifts_ms == pytest.approx([0.0, 10.0 / 3])
    # shifted responses peak less together than aligned ones
    assert summed.response.measures.peak_mv < summed.sum_of_peaks


def test_image_positions_override_spine_order():
    # image says the cluster's 2 on-spine points are scan positions 2 and 3 (of 3)
    positions = lambda recording: [2, 3] if recording.row["spineid"] == 12 else None
    (result,) = analyse_set(_set(step_total=9.0), on_spine_positions=positions)
    assert result.linear_sums["cluster 1-2"].shifts_ms == pytest.approx([3.0, 6.0])
    notes = result.table().query("kind == 'cluster'")["notes"].iloc[0]
    assert "from the image" in notes and "≠ spineid" in notes


def test_inward_epscs_are_made_positive():
    (result,) = analyse_set(_set(cluster_gain=1.0, shape=NO_SAG, unit="pA", sign=-1.0, offset=-50.0),
                            uncaging_offset=False)
    assert result.polarity == -1 and result.unit == "pA"
    table = result.table()
    cluster = table[(table.kind == "cluster") & (table.isi_ms == 1000) & (table.stimid == 1)].iloc[0]
    assert cluster["measured_peak"] > 0
    assert cluster["gain"] == pytest.approx(1.0, rel=0.05)
    assert cluster["baseline"] == pytest.approx(-50.0, abs=0.5)       # real holding current


def test_missing_single_spine_is_reported():
    data = _set()
    data = ExperimentTraces([s for s in data.trace_sets if s.row.name != 2], data.criteria, {})
    (result,) = analyse_set(data)
    cluster = result.table().query("kind == 'cluster'").iloc[0]
    assert cluster["missing_spines"] == "2" and "missing" in cluster["notes"]


def test_table_of_several_sets(tmp_path):
    table = set_summation_table(analyse_set(_set()) + analyse_set(_set(group="other")))
    assert len(table) == 2 * 5 * 2  # (3 spines + cluster at 2 ISIs) x 2 stimuli, per set
    assert list(table.columns) == SET_TABLE_COLUMNS      # fixed columns: tables stack directly
    table.to_csv(tmp_path / "t.csv", index=False)
