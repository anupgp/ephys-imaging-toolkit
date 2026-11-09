"""Tests for the baseline corrections of 2026-11-02 (baseline at the stimulus; local baseline of EPSP2)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ephys_imaging_data_loader.epsp_analysis.epsp_waveform_model import EpspParameters
from ephys_imaging_data_loader.epsp_analysis.summated_epsp_deconvolution import fit_epsp_train
from ephys_imaging_data_loader.experiment_catalog.experiment_traces import ExperimentTraces, ExperimentTraceSet
from ephys_imaging_data_loader.experiment_catalog.traces_by_isi import IsiTraceSet

SHAPE = EpspParameters(amplitude=2.0, latency=2.0, tau_rise=2.0, tau_decay=40.0, sag_ratio=0.0,
                       tau_sag_rise=100.0, tau_sag_decay=400.0)


def _traces(time_ms, values) -> ExperimentTraces:
    trace_set = IsiTraceSet(isi_ms=0.0, group="ctrl", file_name="x.abf", catalog_row=0, channel="IN0", unit="mV",
                            sweeps=[0], time_ms=time_ms, traces=np.atleast_2d(values),
                            stimulus_onsets_ms=np.array([0.0]), decoded_interval_ms=None, stimulus_source="test")
    return ExperimentTraces([ExperimentTraceSet(pd.Series(dtype=object), trace_set)], {}, {})


def test_line_baseline_zeroes_a_drifting_sweep_at_the_stimulus():
    t = np.arange(-50.0, 20.0, 0.1)
    drift = -70.0 + 0.01 * t                       # 10 mV/s drift, -70 mV at the stimulus
    line = _traces(t, drift).baseline_subtracted("line", 30.0).trace_sets[0].traces.traces[0]
    mean = _traces(t, drift).baseline_subtracted().trace_sets[0].traces.traces[0]
    at_zero = np.argmin(np.abs(t))
    assert line[at_zero] == pytest.approx(0.0, abs=1e-6)          # zero where the response starts
    assert mean[at_zero] == pytest.approx(0.25, abs=0.01)         # old mean lags by half the window's drift


def test_local_baseline_step_absorbs_drift_before_the_2nd_epsp():
    t = np.arange(-10.0, 1300.0, 0.1)
    onsets = [0.0, 1000.0]
    true_epsp2 = 1.5
    trace = (SHAPE.waveform(t, 0.0) + SHAPE.with_values(amplitude=true_epsp2).waveform(t, 1000.0)
             + 0.8 * (t >= 900.0))                 # baseline drifted up by 0.8 mV before pulse 2
    template = SimpleNamespace(parameters=SHAPE, measures=SimpleNamespace(peak_time_ms=12.0))
    without = fit_epsp_train(template, t, trace, onsets, 1000.0, 1)
    with_step = fit_epsp_train(template, t, trace, onsets, 1000.0, 1, local_baseline_min_isi_ms=200.0)
    assert with_step.baseline_steps_mv[1] == pytest.approx(0.8, abs=0.02)
    assert with_step.measures[1].peak_mv == pytest.approx(true_epsp2, rel=0.03)
    assert without.measures[1].peak_mv - true_epsp2 > 0.1              # without the step: EPSP2 inflated (~+14 %)
    short = fit_epsp_train(template, t, trace, [0.0, 50.0], 50.0, 1, local_baseline_min_isi_ms=200.0)
    assert short.baseline_steps_mv[1] == 0.0                           # no step at short intervals


def test_add_sheet_columns_uses_the_files_of_each_row():
    from ephys_imaging_data_loader.batch_analysis.sheet_columns import add_sheet_columns

    sheet = pd.DataFrame({"ephysfile_prefix": ["26326xxx", "26326xxx"], "ephysid": ["5", "6"],
                          "region": ["hc", "hc"], "area": ["ca1", "ca1"], "temp": ["25", "27"]})
    results = pd.DataFrame({"files": ["26326005.abf", "26326005.abf;26326006.abf", "99999999.abf"], "peak": [1, 2, 3]})
    out = add_sheet_columns(results, sheet)
    assert out["sheet_region"].tolist()[:2] == ["hc", "hc"] and pd.isna(out["sheet_region"].iloc[2])
    assert out["sheet_temp"].tolist()[:2] == [25.0, 26.0]            # pooled files: mean
    assert out["peak"].tolist() == [1, 2, 3]


def test_ppr_recovery_fit_recovers_known_parameters():
    from ephys_imaging_data_loader.epsp_analysis.ppr_recovery import add_ppr_recovery_columns, fit_ppr_recovery, recovery

    isi = np.array([30, 40, 50, 100, 200, 400, 600, 800, 1000], dtype=float)
    fit = fit_ppr_recovery(isi, recovery(isi, 0.3, 80.0))
    assert fit.A == pytest.approx(0.3, abs=1e-3) and fit.tau_ms == pytest.approx(80.0, rel=1e-3)
    assert fit.r_squared == pytest.approx(1.0) and fit.tau_meaningful
    table = pd.DataFrame({"set": "s", "input": "cluster", "unit": "mV", "n_pulses": 2, "isi_ms": np.repeat(isi, 2),
                          "paired_pulse_ratio": np.repeat(recovery(isi, 0.3, 80.0), 2)})
    out = add_ppr_recovery_columns(table)
    assert out["ppr_fit_tau_ms"].round(1).unique().tolist() == [80.0] and len(out) == len(table)
    assert not fit_ppr_recovery([1000.0], [1.0]).ok                     # one ISI: no fit
