"""
spatiotemporal_statistics.py
============================

Tests for the spatial / temporal summation figures, in the same one-row-per-test
format as ``population_statistics`` (so every test of a run goes into one CSV).

    correlation(x, y)           Spearman rank correlation [main] + Pearson, one point per set
    isi_mixed_model(pulses, y)  ALL ISIs at once: linear mixed model
                                    log(y) ~ condition * log2(ISI) + (1 | dendrite)
                                one row per fixed-effect term. The random intercept per
                                DENDRITE handles that each set gives several ISIs and that
                                pre- and post-apv sets of one dendrite are paired.
                                Terms: Intercept = log(y) of ctrl at ISI = 1 ms (not
                                interesting by itself); C(group)[T.x] = shift of condition x
                                vs ctrl; log2(ISI) = change of log(y) per doubling of the
                                ISI in ctrl; ...:log2(ISI) = how much that slope differs in x.
    spatial_statistics(...), temporal_statistics(...), kinetics_statistics(...),
    relationship_statistics(...)
                                the tests of each figure
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .population_statistics import compare_to_one, ppr_statistics, summation_statistics, two_group_test


def correlation(x, y, **context) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    row = dict(context, comparison="correlation", n=len(x), centre_kind="Spearman rho", centre=np.nan,
               test="Spearman rank correlation", statistic=np.nan, p=np.nan, pearson_r=np.nan, pearson_p=np.nan,
               note="")
    if len(x) >= 4:
        rho, p = stats.spearmanr(x, y)
        r, pearson_p = stats.pearsonr(x, y)
        row.update(centre=float(rho), statistic=float(rho), p=float(p), pearson_r=float(r), pearson_p=float(pearson_p))
    return row


def isi_mixed_model(pulses: pd.DataFrame, column: str, figure: str, reference: str = "ctrl") -> list[dict]:
    """See module docstring. Rows with a non-positive or missing value are left out (log)."""
    import statsmodels.formula.api as smf

    data = pulses[["dendrite", "set", "group", "isi_ms", column]].copy()
    data["group"] = data["group"].astype(str)
    data = data[np.isfinite(data[column]) & (data[column] > 0)]
    data["log_y"] = np.log(data[column])
    data["log2_isi"] = np.log2(data["isi_ms"])
    formula = f"log_y ~ C(group, Treatment('{reference}')) * log2_isi"
    base = dict(figure=figure, measure=f"log {column}, all ISIs", comparison="mixed model",
                test="linear mixed model, random intercept per dendrite", n=data["set"].nunique(),
                n_2=len(data), centre_kind="coefficient (log units)")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.mixedlm(formula, data, groups=data["dendrite"]).fit(reml=True)
        note = "; ".join(sorted({str(w.message)[:80] for w in caught})) or ""
    except Exception as error:                                   # noqa: BLE001 - reported, not hidden
        return [dict(base, group="model failed", note=f"{type(error).__name__}: {error}")]
    ci = fit.conf_int()
    rows = []
    for term in fit.fe_params.index:
        label = (term.replace(f"C(group, Treatment('{reference}'))[T.", "").replace("]", "")
                 .replace("log2_isi", "log2(ISI)"))
        rows.append(dict(base, group=label, centre=float(fit.fe_params[term]), statistic=float(fit.tvalues[term]),
                         p=float(fit.pvalues[term]), ci_low=float(ci.loc[term, 0]), ci_high=float(ci.loc[term, 1]),
                         note=note + ("; n = sets, n_2 = set x ISI rows" if not rows else "")))
    return rows


def _ctrl_vs_post(summary: pd.DataFrame, column: str, figure: str, measure: str | None = None) -> list[dict]:
    a, b = (summary.loc[summary["group"].astype(str) == g, column] for g in ("ctrl", "post-apv"))
    if a.notna().sum() and b.notna().sum():
        return [two_group_test(a, b, figure=figure, measure=measure or column, group="ctrl", group_2="post-apv")]
    return []


def _ratio_vs_one(summary: pd.DataFrame, column: str, figure: str) -> list[dict]:
    rows = [compare_to_one(values, figure=figure, measure=column, group=str(group))
            for group, values in summary.groupby("group", observed=True)[column] if values.notna().any()]
    return rows + _ctrl_vs_post(summary, column, figure)


def spatial_statistics(summary: pd.DataFrame, figure: str = "fig1 spatial summation") -> list[dict]:
    with_sum = summary.dropna(subset=["linear_sum_peak"])
    rows = summation_statistics(with_sum, "peak", figure=figure) + summation_statistics(with_sum, "max_slope",
                                                                                         figure=figure)
    for column in ("gain", "measured_peak", "unitary_peak_mean"):
        rows += _ctrl_vs_post(summary, column, figure)
    rows.append(correlation(with_sum["cluster_spread_um"], with_sum["gain"], figure=figure,
                            measure="gain vs cluster spread", group="all"))
    return rows


def temporal_statistics(pulses: pd.DataFrame, summary: pd.DataFrame, figure: str = "fig2 temporal summation"):
    rows = ppr_statistics(pulses, figure=figure)
    for column in ("ppr_peak", "ppr_slope"):
        rows += isi_mixed_model(pulses, column, figure)
    return rows


def kinetics_statistics(pulses: pd.DataFrame, summary: pd.DataFrame, figure: str = "fig4 decay 2nd vs 1st"):
    rows = []
    for column in ("tau_ratio_short", "tau_ratio_long", "rise_ratio_short", "rise_ratio_long"):
        rows += _ratio_vs_one(summary, column, figure)
    rows += _ctrl_vs_post(summary, "decay_tau_ms_1", figure)
    for column in ("tau_ratio", "rise_ratio"):
        rows += isi_mixed_model(pulses, column, figure)
    return rows


# fig5 panels: (x, y) per set
RELATIONSHIPS = [("residual_mv_short", "ppr_peak_short"), ("peak_1", "ppr_peak_short"),
                 ("peak_1", "decay_tau_ms_1")]


def relationship_statistics(summary: pd.DataFrame, figure: str = "fig5 relationships") -> list[dict]:
    rows = []
    for x, y in RELATIONSHIPS:
        for group in ["all"] + [str(g) for g in summary["group"].cat.categories if (summary["group"] == g).any()]:
            part = summary if group == "all" else summary[summary["group"].astype(str) == group]
            rows.append(correlation(part[x], part[y], figure=figure, measure=f"{y} vs {x}", group=group))
    rows += _ctrl_vs_post(summary, "baseline_rest_mv", figure, "resting potential (mV)")
    return rows
