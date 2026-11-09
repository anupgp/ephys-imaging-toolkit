"""
Tests for the EPSP model and the summated-EPSP deconvolution, on synthetic
paired-pulse traces built from KNOWN EPSPs (with sag) plus a little noise:
the algorithm must recover the 2nd EPSP even when it rides on the 1st.
"""

import numpy as np
import pandas as pd
import pytest

from ephys_imaging_data_loader.epsp_analysis import (
    EpspParameters,
    build_epsp_template,
    deconvolve_summated_epsps,
    fit_epsp,
    measure_epsp,
)
from ephys_imaging_data_loader.experiment_catalog.experiment_traces import ExperimentTraces, ExperimentTraceSet
from ephys_imaging_data_loader.experiment_catalog.traces_by_isi import IsiTraceSet

TRUE_SHAPE = EpspParameters(amplitude=3.0, latency=2.0, tau_rise=3.0, tau_decay=30.0, sag_ratio=0.15,
                            tau_sag_rise=60.0, tau_sag_decay=150.0)
STEP_MS = 0.1


def _paired_trace_set(isi, amplitude2, decay2=30.0, n_sweeps=3, seed=0, group="ctrl"):
    """Sweeps of EPSP1 (amplitude 3) + EPSP2 (given amplitude / decay) at ``isi``, -50 .. isi+300 ms."""
    rng = np.random.default_rng(seed + int(isi))
    t = np.arange(-50.0, isi + 300.0, STEP_MS)
    clean = TRUE_SHAPE.waveform(t) + TRUE_SHAPE.with_values(amplitude=amplitude2, tau_decay=decay2).waveform(t, isi)
    traces = np.array([clean + rng.normal(0, 0.03, len(t)) for _ in range(n_sweeps)])
    row = pd.Series({"isi": float(isi), "group": group, "spineid": 0}, name=int(isi))
    isi_set = IsiTraceSet(isi_ms=float(isi), group=group, file_name=f"f{isi}", catalog_row=int(isi), channel="IN0",
                          unit="mV", sweeps=list(range(n_sweeps)), time_ms=t, traces=traces,
                          stimulus_onsets_ms=np.array([0.0, float(isi)]), decoded_interval_ms=float(isi),
                          stimulus_source="test")
    return ExperimentTraceSet(row=row, traces=isi_set)


@pytest.fixture(scope="module")
def synthetic():
    # 2nd EPSP depressed (2.0 mV) at short ISIs, equal (3.0 mV) at long ISIs
    sets = [_paired_trace_set(isi, 2.0 if isi < 100 else 3.0) for isi in (20, 30, 50, 100, 200, 400, 800)]
    return ExperimentTraces(sets, criteria={}, metadata={"expdate": "20990101"})


def test_model_measures():
    measures = measure_epsp(TRUE_SHAPE)
    assert 0 < measures.rise_20_80_ms < 10
    assert measures.decay_tau_ms == 30.0
    assert measures.sag_mv < 0                      # the sag goes below baseline
    assert measures.peak_mv == pytest.approx(TRUE_SHAPE.waveform(np.arange(0, 100, 0.02)).max(), rel=1e-6)


def test_fit_recovers_isolated_epsp():
    t = np.arange(-10, 300, STEP_MS)
    fitted, rms = fit_epsp(t, TRUE_SHAPE.waveform(t))
    assert rms < 0.01
    assert measure_epsp(fitted).peak_mv == pytest.approx(measure_epsp(TRUE_SHAPE).peak_mv, rel=0.02)
    assert measure_epsp(fitted).rise_20_80_ms == pytest.approx(measure_epsp(TRUE_SHAPE).rise_20_80_ms, rel=0.05)


def test_template_uses_only_isolated_epsps(synthetic):
    template = build_epsp_template(synthetic, threshold_isi_ms=200)
    # ISIs 200, 400, 800 (3 files x 3 sweeps): their 1st and 2nd EPSPs; ISIs 20-100 are not used
    assert template.n_first == 9 and template.n_second == 9
    assert template.measures.sag_mv < 0
    assert template.time_ms[-1] > 290          # long enough for the slow end of the sag


def test_deconvolution_recovers_second_epsp(synthetic):
    (analysis,) = deconvolve_summated_epsps(synthetic, threshold_isi_ms=200)
    true_peak_long = measure_epsp(TRUE_SHAPE).peak_mv
    true_peak_short = measure_epsp(TRUE_SHAPE.with_values(amplitude=2.0)).peak_mv
    for fit in analysis.fits:
        expected = true_peak_short if fit.isi_ms < 100 else true_peak_long
        assert fit.second_measures.peak_mv == pytest.approx(expected, rel=0.06), fit.isi_ms
        assert fit.first_measures.peak_mv == pytest.approx(true_peak_long, rel=0.06), fit.isi_ms
        assert fit.r_squared > 0.99
        # the fitted sum overlaps the recorded trace
        assert np.sqrt(np.mean((fit.summed_curve - fit.trace) ** 2)) < 0.05
    table = analysis.table()
    assert list(table["isi_ms"]) == [20, 30, 50, 100, 200, 400, 800]
    assert table.loc[table.isi_ms == 20, "paired_pulse_ratio"].iloc[0] == pytest.approx(2 / 3, rel=0.08)


def test_second_epsp_decay_recovered_at_short_isi():
    sets = [_paired_trace_set(isi, 3.0, decay2=50.0 if isi == 30 else 30.0) for isi in (30, 200, 400)]
    (analysis,) = deconvolve_summated_epsps(ExperimentTraces(sets, {}, {}), threshold_isi_ms=200)
    fit_30 = [f for f in analysis.fits if f.isi_ms == 30][0]
    assert fit_30.second_measures.decay_tau_ms == pytest.approx(50.0, rel=0.15)


def test_groups_and_units_are_separated():
    sets = [_paired_trace_set(isi, 3.0, group=group) for isi in (30, 300) for group in ("pre", "post")]
    voltage_clamp = _paired_trace_set(300, 3.0)
    voltage_clamp.traces.unit = "pA"
    analyses = deconvolve_summated_epsps(ExperimentTraces(sets + [voltage_clamp], {}, {}), threshold_isi_ms=200)
    assert sorted(a.group for a in analyses) == ["post", "pre"]
    assert all(len(a.fits) == 2 for a in analyses)


def test_no_isolated_epsps_raises():
    sets = [_paired_trace_set(isi, 3.0) for isi in (20, 30)]
    with pytest.raises(ValueError, match="Lower threshold_isi_ms"):
        deconvolve_summated_epsps(ExperimentTraces(sets, {}, {}), threshold_isi_ms=200)


def test_figure(synthetic, tmp_path):
    from ephys_imaging_data_loader.interactive_viewers.epsp_deconvolution_figure import (
        save_epsp_deconvolution_figures,
    )

    analyses = deconvolve_summated_epsps(synthetic, threshold_isi_ms=200)
    (path,) = save_epsp_deconvolution_figures(analyses, tmp_path / "deconvolution.png", dpi=40)
    assert path.exists()


def test_table_has_measures_for_both_epsps_and_all_groups(tmp_path):
    from ephys_imaging_data_loader.epsp_analysis import epsp_deconvolution_table

    sets = [_paired_trace_set(isi, 3.0, group=group) for isi in (30, 300) for group in ("pre", "post")]
    data = ExperimentTraces(sets, {"expdate": "20990101", "neuronid": 1, "dendriteid": 2, "spineid": None},
                            {"expname": "synthetic"})
    table = epsp_deconvolution_table(deconvolve_summated_epsps(data, threshold_isi_ms=200))
    assert len(table) == 4                                     # 2 groups x 2 ISIs
    for epsp in ("epsp1", "epsp2"):
        for measure in ("peak_mv", "rise_20_80_ms", "decay_tau_ms"):
            assert table[f"{epsp}_{measure}"].notna().all()
    assert set(table["group"]) == {"pre", "post"}
    assert (table["expdate"] == "20990101").all() and (table["spineid"] == "all").all()
    table.to_csv(tmp_path / "table.csv", index=False)
    assert (tmp_path / "table.csv").exists()


def _train_trace_set(isi, amplitudes, resting_mv=0.0, n_sweeps=3, seed=1, group="ctrl"):
    """Sweeps of a train: EPSP k with amplitude amplitudes[k] at k * isi, on a resting potential."""
    rng = np.random.default_rng(seed + int(isi) + len(amplitudes))
    onsets = np.arange(len(amplitudes)) * float(isi)
    t = np.arange(-50.0, onsets[-1] + 300.0, STEP_MS)
    clean = sum(TRUE_SHAPE.with_values(amplitude=a).waveform(t, o) for a, o in zip(amplitudes, onsets))
    traces = np.array([resting_mv + clean + rng.normal(0, 0.03, len(t)) for _ in range(n_sweeps)])
    row = pd.Series({"isi": float(isi), "group": group, "spineid": 0}, name=int(isi) * 10 + len(amplitudes))
    isi_set = IsiTraceSet(isi_ms=float(isi), group=group, file_name=f"train{isi}", catalog_row=int(row.name),
                          channel="IN0", unit="mV", sweeps=list(range(n_sweeps)), time_ms=t, traces=traces,
                          stimulus_onsets_ms=onsets, decoded_interval_ms=float(isi), stimulus_source="test")
    return ExperimentTraceSet(row=row, traces=isi_set)


def test_train_of_five_epsps_is_deconvolved():
    train = [3.0, 2.4, 2.0, 1.8, 1.6]                         # depressing train at 50 ms
    sets = [_train_trace_set(50, train, resting_mv=-70.0), _train_trace_set(1000, [3.0, 3.0], resting_mv=-70.0)]
    (analysis,) = deconvolve_summated_epsps(ExperimentTraces(sets, {}, {}), threshold_isi_ms=200)
    fit = [f for f in analysis.fits if f.n_epsps == 5][0]
    for k, amplitude in enumerate(train):
        expected = measure_epsp(TRUE_SHAPE.with_values(amplitude=amplitude)).peak_mv
        assert fit.measures[k].peak_mv == pytest.approx(expected, rel=0.06), k
    assert fit.r_squared > 0.99
    table = analysis.table()
    assert {"epsp5_peak_mv", "epsp5_baseline_mv", "epsp5_ratio_to_epsp1"} <= set(table.columns)
    paired = table[table.n_pulses == 2].iloc[0]
    assert np.isnan(paired["epsp3_peak_mv"])                  # a pair has no 3rd EPSP


def test_absolute_baselines():
    sets = [_train_trace_set(30, [3.0, 3.0], resting_mv=-65.0), _train_trace_set(400, [3.0, 3.0], resting_mv=-65.0)]
    (analysis,) = deconvolve_summated_epsps(ExperimentTraces(sets, {}, {}), threshold_isi_ms=200)
    short = [f for f in analysis.fits if f.isi_ms == 30][0]
    assert short.baselines_mv[0] == pytest.approx(-65.0, abs=0.05)              # resting potential
    # the 2nd EPSP starts on the decaying 1st EPSP: absolute potential just before the 2nd stimulus
    t = np.arange(28.0, 30.0, STEP_MS)
    expected = -65.0 + TRUE_SHAPE.waveform(t).mean()
    assert short.baselines_mv[1] == pytest.approx(expected, abs=0.05)


def test_max_slope_is_the_largest_dv_dt_between_onset_and_peak():
    """measure_epsp: max dV/dt of the model curve, at a time on the rise, in mV/ms."""
    import numpy as np

    measures = measure_epsp(TRUE_SHAPE)
    t = np.arange(0.0, 1000.0, 0.001)
    dv_dt = np.diff(TRUE_SHAPE.waveform(t)) / 0.001
    assert measures.max_slope_mv_per_ms == pytest.approx(dv_dt.max(), rel=1e-2)   # 0.02 ms sampling
    assert measures.latency_ms <= measures.max_slope_time_ms <= measures.peak_time_ms


def test_peak_is_searched_only_up_to_30_ms_after_onset():
    """A slowly rising curve (true maximum ~70 ms after onset) is measured at onset + 30 ms."""
    from ephys_imaging_data_loader.epsp_analysis.epsp_waveform_model import PEAK_WINDOW_MS

    slow = TRUE_SHAPE.with_values(tau_rise=20.0, tau_decay=60.0, sag_ratio=0.0)
    measures = measure_epsp(slow)
    assert measures.peak_capped
    assert measures.peak_time_ms == pytest.approx(slow.latency + PEAK_WINDOW_MS, abs=0.05)
    assert not measure_epsp(TRUE_SHAPE).peak_capped
