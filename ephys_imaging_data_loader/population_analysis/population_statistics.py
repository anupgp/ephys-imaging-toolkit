"""
population_statistics.py
========================

Step 2 of the population analysis: the tests behind the figures. Every test
takes ONE value per set (the experimental unit; see ``results_tables``), and
every test returns one row of a results table, so all statistics of a run end
up in one CSV (``statistics.csv``) next to the figures:

    figure, measure, comparison, group(s), n, centre values, test, statistic, p, note

Tests (two-sided)
-----------------
    compare_to_one(values)      is a ratio (PPR, gain) different from 1?
                                Wilcoxon signed-rank on log(ratio) [main], and a
                                one-sample t-test on log(ratio)
    paired_test(before, after)  pre- vs post-apv of the same dendrites:
                                paired t-test [main] and Wilcoxon signed-rank
    two_group_test(a, b)        two independent groups (e.g. ctrl vs post-apv):
                                Mann-Whitney U [main] and Welch's t-test

With very few dendrites the Wilcoxon test cannot reach p < 0.05 at all (n = 4:
smallest possible two-sided p = 0.125); the 'note' column says so. No
correction for multiple comparisons is applied here; the table lists every
test made, so a correction (e.g. Holm, ``holm_adjust``) can be applied per
family of tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _clean(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _small_n_note(n: int, minimum_for_wilcoxon: int = 6) -> str:
    if n < minimum_for_wilcoxon:
        return f"n = {n}: Wilcoxon cannot reach p < 0.05 (min p = {2 / 2 ** n:.3g}); read the t-test with care"
    return ""


def compare_to_one(values, **context) -> dict:
    """A ratio against 1 (tests on log ratio, so 0.5 and 2 are equally far from 1)."""
    ratios = _clean(values)
    ratios = ratios[ratios > 0]
    logs = np.log(ratios)
    n = len(logs)
    row = dict(context, comparison="vs 1", n=n, centre=float(np.median(ratios)) if n else np.nan,
               centre_kind="median", test="Wilcoxon signed-rank on log ratio", statistic=np.nan, p=np.nan,
               t_test_p=np.nan, note=_small_n_note(n))
    if n >= 2:
        result = stats.wilcoxon(logs)
        row.update(statistic=float(result.statistic), p=float(result.pvalue),
                   t_test_p=float(stats.ttest_1samp(logs, 0.0).pvalue))
    return row


def paired_test(before, after, **context) -> dict:
    """Paired values (same dendrites, same order); pairs with a NaN are dropped."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    keep = np.isfinite(before) & np.isfinite(after)
    before, after = before[keep], after[keep]
    n = len(before)
    row = dict(context, comparison="paired", n=n,
               centre=float(np.mean(before)) if n else np.nan, centre_2=float(np.mean(after)) if n else np.nan,
               centre_kind="mean", test="paired t-test", statistic=np.nan, p=np.nan, wilcoxon_p=np.nan,
               note=_small_n_note(n))
    if n >= 2:
        result = stats.ttest_rel(before, after)
        row.update(statistic=float(result.statistic), p=float(result.pvalue),
                   wilcoxon_p=float(stats.wilcoxon(before, after).pvalue) if np.any(before != after) else np.nan)
    return row


def two_group_test(first, second, **context) -> dict:
    """Two independent groups."""
    first, second = _clean(first), _clean(second)
    row = dict(context, comparison="two groups", n=len(first), n_2=len(second),
               centre=float(np.median(first)) if len(first) else np.nan,
               centre_2=float(np.median(second)) if len(second) else np.nan, centre_kind="median",
               test="Mann-Whitney U", statistic=np.nan, p=np.nan, welch_t_p=np.nan, note="")
    if len(first) >= 2 and len(second) >= 2:
        result = stats.mannwhitneyu(first, second, alternative="two-sided")
        row.update(statistic=float(result.statistic), p=float(result.pvalue),
                   welch_t_p=float(stats.ttest_ind(first, second, equal_var=False).pvalue))
    return row


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p values (NaN stays NaN)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def statistics_table(rows: list[dict]) -> pd.DataFrame:
    """The rows of all tests as one table, the main columns first."""
    table = pd.DataFrame(rows)
    first = [c for c in ["figure", "measure", "comparison", "group", "group_2", "n", "n_2", "centre_kind",
                         "centre", "centre_2", "test", "statistic", "p"] if c in table.columns]
    return table[first + [c for c in table.columns if c not in first]]


# =============================================================================
# The tests of each population figure (one list of rows per figure)
# =============================================================================
LONG_ISI_MS = 500.0


def ppr_statistics(pulses: pd.DataFrame, short_isi_ms: float = 50.0, long_isi_ms: float = LONG_ISI_MS,
                   figure: str = "PPR vs ISI") -> list[dict]:
    """
    Per set, the mean PPR over the SHORT ISIs (<= short_isi_ms) and over the LONG ISIs
    (>= long_isi_ms); then, per condition, each against 1, and ctrl against post-apv
    (independent dendrites; pre-apv shares its dendrites with post-apv, so pre vs post is
    tested pairwise in ``pre_post_statistics``).
    """
    rows = []
    windows = {f"ISI <= {short_isi_ms:g} ms": pulses["isi_ms"] <= short_isi_ms,
               f"ISI >= {long_isi_ms:g} ms": pulses["isi_ms"] >= long_isi_ms}
    for window, selected in windows.items():
        per_set = pulses[selected].groupby(["set", "group"], observed=True)[["ppr_peak", "ppr_slope"]].mean()
        per_set = per_set.reset_index()
        for measure in ("ppr_peak", "ppr_slope"):
            name = f"{measure}, mean {window}"
            for group, values in per_set.groupby("group", observed=True)[measure]:
                rows.append(compare_to_one(values, figure=figure, measure=name, group=str(group)))
            ctrl, post = (per_set.loc[per_set["group"] == g, measure] for g in ("ctrl", "post-apv"))
            if len(ctrl) and len(post):
                rows.append(two_group_test(ctrl, post, figure=figure, measure=name, group="ctrl", group_2="post-apv"))
    return rows


def summation_statistics(summation: pd.DataFrame, quantity: str = "peak",
                         figure: str = "measured vs linear sum") -> list[dict]:
    """
    measured / linear sum (= gain) against 1 per condition and for all clusters together,
    for the shifted and the unshifted sum; and the shifted against the unshifted gain
    (paired, same clusters).
    """
    rows = []
    gains = {}
    for prefix, name in (("linear_sum", "shifted"), ("linear_sum_unshifted", "unshifted")):
        gains[name] = summation[f"measured_{quantity}"] / summation[f"{prefix}_{quantity}"]
        measure = f"measured / linear sum ({name}), {quantity}"
        for group, values in gains[name].groupby(summation["group"], observed=True):
            rows.append(compare_to_one(values, figure=figure, measure=measure, group=str(group)))
        rows.append(compare_to_one(gains[name], figure=figure, measure=measure, group="all"))
    rows.append(paired_test(gains["shifted"], gains["unshifted"], figure=figure,
                            measure=f"gain shifted vs unshifted, {quantity}", group="shifted", group_2="unshifted"))
    return rows


def pre_post_statistics(summary_pairs: pd.DataFrame, measures, figure: str = "pre vs post apv") -> list[dict]:
    """Paired pre- vs post-apv test of every measure (one value per dendrite and condition)."""
    rows = []
    for measure in measures:
        wide = summary_pairs.pivot_table(index="dendrite", columns="group", values=measure, observed=True)
        if {"pre-apv", "post-apv"} - set(wide.columns):
            continue
        rows.append(paired_test(wide["pre-apv"], wide["post-apv"], figure=figure, measure=measure,
                                group="pre-apv", group_2="post-apv"))
    return rows


def format_p(p: float) -> str:
    """'p = 0.032' / 'p < 0.001' / 'p = n/a' for figure titles."""
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
