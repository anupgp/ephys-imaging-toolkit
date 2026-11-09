"""
spatiotemporal_tables.py
========================

More tables for the population analysis of spatial and temporal summation,
built on ``results_tables`` (same rules: ONE row per set and x value, the set
being the experimental unit).

    pulse_pair_table(results)       one row per cluster x ISI: everything of
                                    ``paired_pulse_table`` + the kinetics of both EPSPs
                                    and the temporal-summation measures below
    set_summary_table(results)      one row per set (= per cluster): 1st-response
                                    measures (``cluster_summary``), the spatial-summation
                                    columns (``summation_table``), the single spines'
                                    peaks, and mean ratios at short / long ISIs
    unitary_table(results)          one row per single spine: its 1st-response peak
                                    (median over its recordings)

Temporal-summation measures (per cluster and ISI)
-------------------------------------------------
From the deconvolution, per EPSP k: ``peak_k`` (of that EPSP alone) and
``baseline_k`` (ABSOLUTE potential: EPSP1 = the 10 ms before the 1st stimulus,
EPSP2 = the 2 ms before the 2nd stimulus, i.e. including what is left of EPSP1).

    residual_mv           baseline_2 - baseline_1: depolarisation still left from EPSP1
                          when the 2nd stimulus arrives
    residual_fraction     residual_mv / peak_1: the same as a fraction of EPSP1 (shows the
                          decay of EPSP1 independent of its size)
    compound_ratio        (residual + peak_2) / peak_1: the summed response at the 2nd
                          pulse relative to EPSP1 (approximate: EPSP1 keeps decaying
                          while EPSP2 rises, so this slightly overestimates the
                          compound peak - equally for the linear expectation below)
    linear_compound_ratio (residual + peak_1) / peak_1: the same if EPSP2 were an exact
                          copy of EPSP1 (linear temporal summation)
    temporal_gain         compound / linear expectation = (residual + peak_2) /
                          (residual + peak_1): < 1 sublinear, > 1 supralinear
    tau_ratio, rise_ratio decay tau / 20-80 % rise of EPSP2 over EPSP1

NOTE on EPSP1 kinetics: in the deconvolution EPSP1 has the set's TEMPLATE shape
at every ISI (only its amplitude and latency are fitted), so decay_tau_1 and
rise_1 are one value per set; EPSP2's are fitted freely at each ISI. tau_ratio
is therefore "EPSP2 at this ISI vs the set's isolated EPSP".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .results_tables import (PAIRED_PULSE_KEYS, SHORT_ISI_MS, _clusters, _ordered_groups, cluster_summary,
                             summation_table)

LONG_ISI_MS = 500.0
# CSV column -> short name; the table gets <name>_1 and <name>_2
PULSE_COLUMNS = {"peak": "peak", "max_slope_per_ms": "slope", "rise_20_80_ms": "rise", "decay_tau_ms": "tau",
                 "latency_ms": "latency", "baseline": "baseline", "sag": "sag"}


def pulse_pair_table(results: pd.DataFrame, n_spines=None) -> pd.DataFrame:
    """One row per cluster x ISI; see module docstring for the columns."""
    clusters = _clusters(results, n_spines)
    first = clusters[clusters["stimid"] == 1].set_index(PAIRED_PULSE_KEYS)
    second = clusters[clusters["stimid"] == 2].set_index(PAIRED_PULSE_KEYS)
    for name, part in (("1st", first), ("2nd", second)):
        if part.index.duplicated().any():
            raise ValueError(f"More than one {name}-response row for a cluster and ISI; check the CSV.")
    columns = {}
    for column, short in PULSE_COLUMNS.items():
        columns[f"{short}_1"] = first[column]
        columns[f"{short}_2"] = second[column]
    columns["r_squared"] = first["r_squared"]
    columns["n_sweeps"] = first["n_sweeps"]
    table = pd.DataFrame(columns).reset_index()
    table["ppr_peak"] = table["peak_2"] / table["peak_1"]
    table["ppr_slope"] = table["slope_2"] / table["slope_1"]
    table["tau_ratio"] = table["tau_2"] / table["tau_1"]
    table["rise_ratio"] = table["rise_2"] / table["rise_1"]
    table["residual_mv"] = table["baseline_2"] - table["baseline_1"]
    table["residual_fraction"] = table["residual_mv"] / table["peak_1"]
    table["compound_ratio"] = (table["residual_mv"] + table["peak_2"]) / table["peak_1"]
    table["linear_compound_ratio"] = (table["residual_mv"] + table["peak_1"]) / table["peak_1"]
    table["temporal_gain"] = table["compound_ratio"] / table["linear_compound_ratio"]
    table = _ordered_groups(table)
    return table.sort_values(["group", "set", "isi_ms"]).reset_index(drop=True)


def unitary_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per single spine (set x input): median 1st-response peak / slope over its recordings."""
    singles = results[(results["kind"] == "unitary") & (results["stimid"] == 1)]
    table = singles.groupby(["set", "dendrite", "group", "input", "spineid"], observed=True).agg(
        peak=("peak", "median"), slope=("max_slope_per_ms", "median"), decay_tau=("decay_tau_ms", "median"),
        n_isis=("isi_ms", "nunique")).reset_index()
    return _ordered_groups(table).sort_values(["group", "set", "input"]).reset_index(drop=True)


RATIO_SUMMARIES = ["ppr_peak", "ppr_slope", "temporal_gain", "tau_ratio", "rise_ratio"]


def set_summary_table(results: pd.DataFrame, n_spines=None, short_isi_ms: float = SHORT_ISI_MS,
                      long_isi_ms: float = LONG_ISI_MS) -> pd.DataFrame:
    """
    One row per set. Columns: ``cluster_summary`` (peak_1, slope_1, rise/decay of EPSP1,
    n_isis, ppr_*_short, gain*), the ``summation_table`` columns (measured_peak,
    linear_sum_peak, ..., cluster_spread_um), the single spines (n_singles,
    unitary_peak_mean, unitary_peak_sum), and for every ratio in RATIO_SUMMARIES its
    mean over the short ISIs (``<ratio>_short``) and the long ISIs (``<ratio>_long``),
    plus residual_mv_short, baseline_rest_mv (absolute potential before the 1st stimulus)
    and r_squared_median (deconvolution fit), both median over ISIs.
    """
    summary = cluster_summary(results, n_spines, short_isi_ms)
    if summary["set"].duplicated().any():
        raise ValueError("More than one cluster in a set; set_summary_table expects one cluster per set.")
    summary = summary.drop(columns=[c for c in summary.columns if c.startswith(("ppr_", "gain"))])
    spatial = summation_table(results, n_spines).drop(columns=["dendrite", "group", "input", "n_spines"])
    summary = summary.merge(spatial, on="set", how="left")
    singles = unitary_table(results).groupby("set", observed=True).agg(
        n_singles=("peak", "count"), unitary_peak_mean=("peak", "mean"), unitary_peak_sum=("peak", "sum"))
    summary = summary.merge(singles, left_on="set", right_index=True, how="left")
    pulses = pulse_pair_table(results, n_spines)
    # resting potential (absolute baseline before the 1st stimulus) and fit quality, median over ISIs
    quality = pulses.groupby("set", observed=True).agg(baseline_rest_mv=("baseline_1", "median"),
                                                        r_squared_median=("r_squared", "median"))
    summary = summary.merge(quality, left_on="set", right_index=True, how="left")
    for name, selected in (("short", pulses["isi_ms"] <= short_isi_ms), ("long", pulses["isi_ms"] >= long_isi_ms)):
        means = pulses[selected].groupby("set", observed=True)[RATIO_SUMMARIES + ["residual_mv", "residual_fraction"]].mean()
        summary = summary.merge(means.add_suffix(f"_{name}"), left_on="set", right_index=True, how="left")
    return _ordered_groups(summary).sort_values(["group", "set"]).reset_index(drop=True)


def restrict_to_location(results: pd.DataFrame, sheet_csv, expname: str, region: str = "hc", area: str = "ca1",
                         side: str = "apical") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Only the dendrites whose rows in the experiment sheet (``sheet_csv``, experiment
    ``expname``) ALL have this region / area / side (case-insensitive), e.g. hc / ca1 /
    apical. Returns (kept results, table of every dendrite with its location and kept yes/no)
    so the exclusions can be reported.
    """
    sheet = pd.read_csv(sheet_csv, dtype=str)
    sheet = sheet[sheet["expname"] == expname]
    sheet["dendrite"] = sheet["expdate"] + "_" + sheet["neuronid"] + "_" + sheet["dendriteid"]
    location = sheet.groupby("dendrite").agg(
        **{name: (name, lambda x: ";".join(sorted(set(x.dropna().str.strip().str.lower()))))
           for name in ("region", "area", "side")})
    location["kept"] = ((location["region"] == region.lower()) & (location["area"] == area.lower())
                        & (location["side"] == side.lower()))
    location = location[location.index.isin(results["dendrite"])]
    kept = results[results["dendrite"].isin(location.index[location["kept"]])].reset_index(drop=True)
    return kept, location.reset_index()
