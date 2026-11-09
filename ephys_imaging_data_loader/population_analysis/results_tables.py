"""
results_tables.py
=================

Step 1 of the population analysis: read a combined batch results CSV (one row
per set x input x ISI x stimid, written by ``batch_analysis.run_batch_analysis``)
and reshape it into the small tables that every figure and test uses.

The experimental UNIT is one set = one dendrite in one condition (ctrl,
pre-apv, post-apv). Every table here has at most one value per set and per
x value (ISI), so plots and statistics never count sweeps or pulses as
independent samples.

    load_results(path)              the CSV, EPSP (mV) rows only by default, + a 'dendrite' key
                                    (expdate_neuron_dendrite), which pairs the pre- and
                                    post-apv sets of one dendrite
    paired_pulse_table(results)     one row per cluster x ISI: peak and max slope of the
                                    1st and 2nd response; PPR of the peak and of the slope
    summation_table(results)        one row per cluster with a linear sum: measured vs
                                    linear-sum peak / max slope (shifted and unshifted sums)
    cluster_summary(results)        one row per cluster: its 1st-response measures averaged
                                    over all ISIs (the 1st response does not depend on the
                                    ISI), mean PPR at short ISIs, gain
    pre_post_pairs(table)           only the dendrites recorded both pre- and post-apv
    flag_suspect_responses(...)     rows whose size suggests a spike / artefact (reported,
                                    never removed silently)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Conditions in plotting / legend order.
GROUP_ORDER = ["ctrl", "pre-apv", "post-apv"]
# "Short" ISIs for one-number summaries of paired-pulse behaviour (per set: mean over these ISIs).
SHORT_ISI_MS = 50.0
# Responses above these are flagged as possible spikes or artefacts (a unitary-cluster EPSP of
# 5-6 spines is normally a few mV with a max slope below ~2 mV/ms). Flagged rows stay in the
# tables; the statistics report lists them so they can be checked in the traces.
SUSPECT_PEAK_MV = 20.0
SUSPECT_SLOPE_MV_PER_MS = 5.0


def load_results(path: str | Path, unit: str | None = "mV") -> pd.DataFrame:
    """
    The combined results CSV, with ``dendrite`` = 'expdate_neuronid_dendriteid'.
    ``unit``: keep only rows in this unit; "mV" (default) = EPSPs only (current clamp),
    which drops the voltage-clamp EPSC (pA) test recordings; None = keep all.
    """
    results = pd.read_csv(path)
    if unit is not None and "unit" in results:
        results = results[results["unit"] == unit].reset_index(drop=True)
    results["dendrite"] = (results["expdate"].astype(str) + "_" + results["neuronid"].astype(int).astype(str)
                           + "_" + results["dendriteid"].astype(int).astype(str))
    return results


def _clusters(results: pd.DataFrame, n_spines=None) -> pd.DataFrame:
    """Cluster rows only; ``n_spines`` = None (all), a number or a list of numbers."""
    clusters = results[results["kind"] == "cluster"]
    if n_spines is not None:
        clusters = clusters[clusters["n_spines"].isin(np.atleast_1d(n_spines))]
    return clusters


def _ordered_groups(table: pd.DataFrame) -> pd.DataFrame:
    """'group' as an ordered category (ctrl, pre-apv, post-apv, then any others)."""
    others = sorted(set(table["group"].dropna()) - set(GROUP_ORDER))
    table["group"] = pd.Categorical(table["group"], categories=GROUP_ORDER + others, ordered=True)
    return table


PAIRED_PULSE_KEYS = ["set", "dendrite", "group", "input", "n_spines", "isi_ms"]


def paired_pulse_table(results: pd.DataFrame, n_spines=None) -> pd.DataFrame:
    """
    One row per cluster x ISI with
        peak_1, peak_2      peak of the 1st / 2nd response (deconvolution fit)
        slope_1, slope_2    max slope dV/dt of the 1st / 2nd response
        ppr_peak            peak_2 / peak_1 (= the CSV's paired_pulse_ratio)
        ppr_slope           slope_2 / slope_1
    """
    clusters = _clusters(results, n_spines)
    first = clusters[clusters["stimid"] == 1].set_index(PAIRED_PULSE_KEYS)
    second = clusters[clusters["stimid"] == 2].set_index(PAIRED_PULSE_KEYS)
    for name, part in (("1st", first), ("2nd", second)):
        if part.index.duplicated().any():
            raise ValueError(f"More than one {name}-response row for a cluster and ISI; check the CSV.")
    table = pd.DataFrame({
        "peak_1": first["peak"], "peak_2": second["peak"],
        "slope_1": first["max_slope_per_ms"], "slope_2": second["max_slope_per_ms"],
    }).reset_index()
    table["ppr_peak"] = table["peak_2"] / table["peak_1"]
    table["ppr_slope"] = table["slope_2"] / table["slope_1"]
    table = _ordered_groups(table)
    return table.sort_values(["group", "set", "isi_ms"]).reset_index(drop=True)


SUMMATION_COLUMNS = ["measured_peak", "linear_sum_peak", "linear_sum_unshifted_peak",
                     "measured_max_slope", "linear_sum_max_slope", "linear_sum_unshifted_max_slope",
                     "gain", "gain_unshifted"]


def summation_table(results: pd.DataFrame, n_spines=None) -> pd.DataFrame:
    """One row per cluster that has a measured response (linear-sum columns NaN when the set
    has no single-spine recordings)."""
    clusters = _clusters(results, n_spines)
    table = clusters[clusters["measured_peak"].notna()]
    table = table[["set", "dendrite", "group", "input", "n_spines", "cluster_spread_um", "summation_isi_ms"]
                  + SUMMATION_COLUMNS].copy()
    if table["set"].duplicated().any():
        raise ValueError("More than one cluster with a summation row in a set; check the CSV.")
    return _ordered_groups(table).sort_values(["group", "set"]).reset_index(drop=True)


# 1st-response measures averaged over ISIs in cluster_summary: CSV column -> summary column
FIRST_RESPONSE_MEASURES = {"peak": "peak_1", "max_slope_per_ms": "slope_1",
                           "rise_20_80_ms": "rise_20_80_ms_1", "decay_tau_ms": "decay_tau_ms_1"}


def cluster_summary(results: pd.DataFrame, n_spines=None, short_isi_ms: float = SHORT_ISI_MS) -> pd.DataFrame:
    """
    One row per cluster (= per set):
        peak_1, slope_1, rise_20_80_ms_1, decay_tau_ms_1
                            1st-response measures, MEDIAN over all ISIs of the set (the 1st
                            response does not depend on the ISI, so every ISI is a repeat;
                            the median ignores the odd bad fit)
        n_isis              number of ISIs behind those medians
        ppr_peak_short, ppr_slope_short
                            mean PPR over the ISIs <= ``short_isi_ms``
        gain, gain_unshifted   measured / linear-sum peak (NaN without single spines)
    """
    clusters = _clusters(results, n_spines)
    first = clusters[clusters["stimid"] == 1]
    keys = ["set", "dendrite", "group", "input", "n_spines"]
    summary = first.groupby(keys, observed=True)[list(FIRST_RESPONSE_MEASURES)].median().rename(
        columns=FIRST_RESPONSE_MEASURES)
    summary["n_isis"] = first.groupby(keys, observed=True)["isi_ms"].nunique()
    pulses = paired_pulse_table(results, n_spines)
    short = pulses[pulses["isi_ms"] <= short_isi_ms].groupby("set", observed=True)[["ppr_peak", "ppr_slope"]].mean()
    summary = summary.reset_index().merge(short.add_suffix("_short"), left_on="set", right_index=True, how="left")
    gains = summation_table(results, n_spines)[["set", "gain", "gain_unshifted"]]
    summary = summary.merge(gains, on="set", how="left")
    return _ordered_groups(summary).sort_values(["group", "set"]).reset_index(drop=True)


def pre_post_pairs(table: pd.DataFrame, before: str = "pre-apv", after: str = "post-apv") -> pd.DataFrame:
    """Rows of the dendrites that have BOTH a ``before`` and an ``after`` set, nothing else."""
    conditions = table["group"].astype(str)
    has_before = set(table.loc[conditions == before, "dendrite"])
    has_after = set(table.loc[conditions == after, "dendrite"])
    paired = has_before & has_after
    selected = table[table["dendrite"].isin(paired) & conditions.isin([before, after])].copy()
    selected["group"] = pd.Categorical(selected["group"].astype(str), categories=[before, after], ordered=True)
    return selected.reset_index(drop=True)


def flag_suspect_responses(pulses: pd.DataFrame, peak_mv: float = SUSPECT_PEAK_MV,
                           slope_mv_per_ms: float = SUSPECT_SLOPE_MV_PER_MS) -> pd.DataFrame:
    """Rows of a ``paired_pulse_table`` with a peak or max slope above the limits (either pulse)."""
    too_big = ((pulses[["peak_1", "peak_2"]] > peak_mv).any(axis=1)
               | (pulses[["slope_1", "slope_2"]] > slope_mv_per_ms).any(axis=1))
    return pulses[too_big]
