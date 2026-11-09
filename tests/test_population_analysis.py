"""Tests for population_analysis (tables, statistics, figures) on a tiny synthetic results table."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ephys_imaging_data_loader.population_analysis import (
    cluster_summary, paired_pulse_table, pre_post_pairs, summation_table)
from ephys_imaging_data_loader.population_analysis.population_statistics import (
    compare_to_one, holm_adjust, paired_test)


def _results():
    """Two dendrites: A (pre + post apv), B (ctrl). One 5-spine cluster each, ISIs 30 and 1000 ms."""
    rows = []
    for expdate, group, scale in (("20260209", "pre-apv", 1.0), ("20260209", "post-apv", 0.5),
                                  ("20260313", "ctrl", 2.0)):
        for isi in (30.0, 1000.0):
            for stimid in (1, 2):
                ppr = 0.8 if isi == 30 else 1.0
                peak = scale * (ppr if stimid == 2 else 1.0)
                row = dict(expdate=expdate, neuronid=1, dendriteid=1, group=group, set=f"{expdate}_1_1_{group}",
                           input="all 5 spines", kind="cluster", n_spines=5, isi_ms=isi, stimid=stimid,
                           peak=peak, max_slope_per_ms=peak / 5, rise_20_80_ms=5.0, decay_tau_ms=40.0,
                           measured_peak=np.nan, cluster_spread_um=2.0, summation_isi_ms=np.nan)
                if isi == 1000 and stimid == 1:
                    row.update(measured_peak=peak, summation_isi_ms=1000.0, linear_sum_peak=peak / 1.2,
                               linear_sum_unshifted_peak=peak / 1.1, measured_max_slope=peak / 5,
                               linear_sum_max_slope=peak / 6, linear_sum_unshifted_max_slope=peak / 5.5,
                               gain=1.2, gain_unshifted=1.1)
                rows.append(row)
    results = pd.DataFrame(rows)
    results["dendrite"] = results["expdate"] + "_1_1"
    return results


def test_paired_pulse_table_one_row_per_cluster_and_isi():
    pulses = paired_pulse_table(_results())
    assert len(pulses) == 6
    assert pulses.loc[pulses["isi_ms"] == 30, "ppr_peak"].tolist() == pytest.approx([0.8] * 3)
    assert list(pulses["group"].cat.categories[:3]) == ["ctrl", "pre-apv", "post-apv"]


def test_summary_and_pre_post_pairs():
    summary = cluster_summary(_results())
    assert summary.set_index("group").loc["post-apv", "peak_1"] == pytest.approx(0.5)
    assert summary.set_index("group").loc["ctrl", "ppr_peak_short"] == pytest.approx(0.8)
    pairs = pre_post_pairs(summary)
    assert sorted(pairs["group"].astype(str)) == ["post-apv", "pre-apv"]      # ctrl dendrite dropped
    assert len(summation_table(_results())) == 3


def test_statistics():
    assert compare_to_one([0.8, 0.85, 0.9, 0.7, 0.75, 0.8, 0.9])["p"] < 0.05
    row = paired_test([5, 6, 7, 8], [2, 3, 3, 4])
    assert row["n"] == 4 and row["p"] < 0.05 and "min p = 0.125" in row["note"]
    assert holm_adjust([0.01, 0.04, np.nan]).tolist()[:2] == pytest.approx([0.02, 0.04])


def test_figures_draw_without_layout_warnings():
    from ephys_imaging_data_loader.population_analysis import (
        measured_vs_linear_sum_figure, ppr_vs_isi_figure, pre_post_figure)

    results = _results()
    summary_pairs = pre_post_pairs(cluster_summary(results))
    figures = [ppr_vs_isi_figure(paired_pulse_table(results)), measured_vs_linear_sum_figure(summation_table(results)),
               pre_post_figure(pre_post_pairs(paired_pulse_table(results)), summary_pairs)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for figure in figures:
            figure.canvas.draw()


def test_sweep_screen_finds_a_spike():
    from types import SimpleNamespace

    from ephys_imaging_data_loader.batch_analysis.paired_pulse_screening import measure_sweeps

    t = np.arange(-50, 200, 0.05)
    epsp = lambda onset: np.where(t >= onset, np.exp(-(t - onset) / 30) - np.exp(-(t - onset) / 3), 0) * 2
    clean = epsp(0) + epsp(100)
    spike = clean + np.where((t > 110) & (t < 111), 100.0, 0.0)
    trace_set = SimpleNamespace(time_ms=t, stimulus_onsets_ms=np.array([0.0, 100.0]), sweeps=[0, 1],
                                traces=np.vstack([clean, spike]), unit="mV")
    measures = measure_sweeps(trace_set)
    assert measures["sweep"].tolist() == [1, 2]                       # numbered from 1, as 'badsweeps'
    assert measures["ap_suspect"].tolist() == [False, True]
    assert measures.loc[0, "ppr"] == pytest.approx(1.0, abs=0.1)


def test_results_keep_only_epsps_by_default(tmp_path):
    from ephys_imaging_data_loader.population_analysis import load_results

    results = _results().drop(columns="dendrite")
    results["unit"] = "mV"
    results.loc[0, "unit"] = "pA"
    results.to_csv(tmp_path / "r.csv", index=False)
    assert len(load_results(tmp_path / "r.csv")) == len(results) - 1
    assert len(load_results(tmp_path / "r.csv", unit=None)) == len(results)


def test_spatiotemporal_tables_and_figures():
    from ephys_imaging_data_loader.population_analysis.spatiotemporal_figures import (
        kinetics_figure, relationships_figure, spatial_summation_figure, temporal_summation_figure)
    from ephys_imaging_data_loader.population_analysis.spatiotemporal_tables import (
        pulse_pair_table, set_summary_table, unitary_table)

    results = _results()
    # EPSP1 starts at -70 mV; at 30 ms the 2nd stimulus starts from -68 mV (2 mV left of EPSP1)
    results["baseline"] = np.where((results["stimid"] == 2) & (results["isi_ms"] == 30), -68.0, -70.0)
    results = results.assign(latency_ms=2.0, sag=0.0, r_squared=0.99, n_sweeps=3, spineid="12345")
    pulses = pulse_pair_table(results)
    short = pulses[pulses["isi_ms"] == 30].set_index("group")
    # pre-apv: peak_1 = 1, peak_2 = 0.8, residual 2 -> compound 2.8, linear 3, gain 2.8 / 3
    assert short.loc["pre-apv", "residual_mv"] == pytest.approx(2.0)
    assert short.loc["pre-apv", "temporal_gain"] == pytest.approx(2.8 / 3.0)
    assert short.loc["pre-apv", "residual_fraction"] == pytest.approx(2.0)
    assert pulses["tau_ratio"].tolist() == pytest.approx([1.0] * len(pulses))
    summary = set_summary_table(results)
    assert len(summary) == 3 and summary["baseline_rest_mv"].tolist() == pytest.approx([-70.0] * 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for figure in (spatial_summation_figure(summary, unitary_table(results)), temporal_summation_figure(pulses),
                       kinetics_figure(pulses), relationships_figure(summary, pulses)):
            figure.canvas.draw()


def test_restrict_to_location(tmp_path):
    from ephys_imaging_data_loader.population_analysis.spatiotemporal_tables import restrict_to_location

    pd.DataFrame({"expname": ["x"] * 3, "expdate": ["20260209", "20260209", "20260313"], "neuronid": ["1"] * 3,
                  "dendriteid": ["1"] * 3, "region": ["hc", "HC", "pfc"], "area": ["ca1", "CA1", "l5"],
                  "side": ["apical"] * 3}).to_csv(tmp_path / "sheet.csv", index=False)
    kept, location = restrict_to_location(_results(), tmp_path / "sheet.csv", "x")
    assert set(kept["dendrite"]) == {"20260209_1_1"}
    assert location.set_index("dendrite")["kept"].to_dict() == {"20260209_1_1": True, "20260313_1_1": False}
