"""
ppr_recovery.py
===============

Fit of the paired-pulse ratio against the ISI for one input (cluster or spine):

    PPR(ISI) = 1 - A * exp(-ISI / tau)

    A    depression extrapolated to ISI = 0 (A > 0: depression, A < 0: facilitation)
    tau  recovery time constant (ms)

    fit_ppr_recovery(isi_ms, ppr)          -> PprRecoveryFit
    add_ppr_recovery_columns(table)        a set / batch results table + ppr_fit_* columns
    draw_ppr_recovery(axes, isi, ppr, fit) points + fitted curve + A, tau, R² in the legend

Goodness of fit
---------------
r_squared  1 - SS_residual / SS_total. NOTE: a set whose PPR is ~1 at every ISI has
           almost no variance to explain, so its R² is low even when the curve passes
           through the points; read it together with rms (the scatter of the points around
           the curve) and with the standard errors.
rms        root-mean-square residual (PPR units)
A_se, tau_se   standard errors of the parameters (from the fit's covariance)
tau_meaningful A >= MIN_A (clearly depressed) and tau within TAU_MEANINGFUL (half the
           shortest to the longest measured ISI) and tau_se < tau (relative error < 100 %)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

A_BOUNDS = (-2.0, 1.0)
TAU_BOUNDS = (5.0, 3000.0)          # ms
TAU_MEANINGFUL = (15.0, 1000.0)     # ms
MIN_A = 0.05
MIN_ISIS = 3                         # fewer ISIs: no fit
PPR_FIT_COLUMNS = ["ppr_fit_A", "ppr_fit_A_se", "ppr_fit_tau_ms", "ppr_fit_tau_se_ms", "ppr_fit_r_squared",
                   "ppr_fit_rms", "ppr_fit_n_isis", "ppr_fit_tau_meaningful"]


def recovery(isi_ms, a, tau):
    return 1.0 - a * np.exp(-np.asarray(isi_ms, dtype=float) / tau)


@dataclass
class PprRecoveryFit:
    A: float = np.nan
    A_se: float = np.nan
    tau_ms: float = np.nan
    tau_se_ms: float = np.nan
    r_squared: float = np.nan
    rms: float = np.nan
    n_isis: int = 0

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.A) and np.isfinite(self.tau_ms))

    @property
    def tau_meaningful(self) -> bool:
        return bool(self.ok and self.A >= MIN_A and TAU_MEANINGFUL[0] <= self.tau_ms <= TAU_MEANINGFUL[1]
                    and np.isfinite(self.tau_se_ms) and self.tau_se_ms < self.tau_ms)

    def curve(self, isi_ms) -> np.ndarray:
        return recovery(isi_ms, self.A, self.tau_ms)

    def as_columns(self) -> dict:
        return dict(zip(PPR_FIT_COLUMNS, [self.A, self.A_se, self.tau_ms, self.tau_se_ms, self.r_squared,
                                          self.rms, self.n_isis, self.tau_meaningful]))


def fit_ppr_recovery(isi_ms, ppr) -> PprRecoveryFit:
    """Least-squares fit of the model (see module docstring); empty fit when < MIN_ISIS points."""
    isi, values = np.asarray(isi_ms, dtype=float), np.asarray(ppr, dtype=float)
    keep = np.isfinite(isi) & np.isfinite(values) & (isi > 0)
    isi, values = isi[keep], values[keep]
    if len(np.unique(isi)) < MIN_ISIS:
        return PprRecoveryFit(n_isis=len(np.unique(isi)))
    start = [float(np.clip(1.0 - values[np.argmin(isi)], -0.5, 0.9)), 100.0]
    try:
        (a, tau), covariance = curve_fit(recovery, isi, values, p0=start, maxfev=20000,
                                         bounds=([A_BOUNDS[0], TAU_BOUNDS[0]], [A_BOUNDS[1], TAU_BOUNDS[1]]))
    except (RuntimeError, ValueError):
        return PprRecoveryFit(n_isis=len(np.unique(isi)))
    errors = np.sqrt(np.diag(covariance)) if np.all(np.isfinite(covariance)) else [np.nan, np.nan]
    residual = values - recovery(isi, a, tau)
    total = np.sum((values - values.mean()) ** 2)
    return PprRecoveryFit(A=float(a), A_se=float(errors[0]), tau_ms=float(tau), tau_se_ms=float(errors[1]),
                          r_squared=float(1.0 - np.sum(residual ** 2) / total) if total > 0 else np.nan,
                          rms=float(np.sqrt(np.mean(residual ** 2))), n_isis=len(np.unique(isi)))


def add_ppr_recovery_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of a set (or batch) results table with the ppr_fit_* columns: one fit per set x
    input x unit, from its paired_pulse_ratio at every ISI (2-pulse recordings), written on
    every row of that input.
    """
    table = table.drop(columns=[c for c in PPR_FIT_COLUMNS if c in table.columns]).copy()
    for column in PPR_FIT_COLUMNS:
        table[column] = np.nan
    table["ppr_fit_tau_meaningful"] = table["ppr_fit_tau_meaningful"].astype(object)
    if "paired_pulse_ratio" not in table.columns:
        return table
    keys = [k for k in ("set", "input", "unit") if k in table.columns]
    for _, rows in table.groupby(keys, sort=False, dropna=False):
        points = rows.drop_duplicates("isi_ms")
        if "n_pulses" in points:
            points = points[points["n_pulses"] == 2]
        fit = fit_ppr_recovery(points["isi_ms"], points["paired_pulse_ratio"])
        for column, value in fit.as_columns().items():
            table.loc[rows.index, column] = value
    return table


def draw_ppr_recovery(axes, isi_ms, ppr, fit: PprRecoveryFit, colour="black", label: str = "",
                      top: float | None = None) -> None:
    """Points + fitted curve; the legend entry gives A, tau (± SE) and R²."""
    isi_ms, ppr = np.asarray(isi_ms, dtype=float), np.asarray(ppr, dtype=float)
    order = np.argsort(isi_ms)
    shown = np.clip(ppr, None, top) if top is not None else ppr
    axes.plot(isi_ms[order], shown[order], linestyle="none", marker="o", markersize=4, color=colour, alpha=0.8)
    if fit.ok:
        grid = np.geomspace(max(isi_ms.min() * 0.8, 1.0), isi_ms.max() * 1.2, 200)
        text = (f"{label}A = {fit.A:.2f} ± {fit.A_se:.2f}, τ = {fit.tau_ms:.0f} ± {fit.tau_se_ms:.0f} ms, "
                f"R² = {fit.r_squared:.2f}" + ("" if fit.tau_meaningful else " (τ not meaningful)"))
        axes.plot(grid, fit.curve(grid), color=colour, linewidth=1.6,
                  linestyle="-" if fit.tau_meaningful else "--", label=text)
    elif label:
        axes.plot([], [], color=colour, label=f"{label}no fit ({fit.n_isis} ISIs)")


def ppr_recovery_figure(table: pd.DataFrame, title: str = ""):
    """
    One panel per input (and unit) with >= MIN_ISIS ISIs: its PPR at every ISI, the fitted
    recovery curve (solid: tau meaningful, dashed: not) and the fit's A, tau (± SE), R² and
    rms. None when no input has enough ISIs.
    """
    from matplotlib.figure import Figure

    table = table if "ppr_fit_r_squared" in table.columns else add_ppr_recovery_columns(table)
    keys = [k for k in ("input", "unit") if k in table.columns]
    inputs = []
    for key, rows in table.groupby(keys, sort=False):
        points = rows.drop_duplicates("isi_ms")
        if "n_pulses" in points:
            points = points[points["n_pulses"] == 2]
        points = points[pd.to_numeric(points["paired_pulse_ratio"], errors="coerce").notna()]
        if points["isi_ms"].nunique() >= MIN_ISIS:
            inputs.append((key if isinstance(key, tuple) else (key,), points))
    if not inputs:
        return None
    columns = min(3, len(inputs))
    rows_n = int(np.ceil(len(inputs) / columns))
    figure = Figure(figsize=(4.6 * columns, 3.6 * rows_n + 0.6), layout="constrained")
    axes = np.atleast_1d(figure.subplots(rows_n, columns, squeeze=False)).ravel()
    for one, (key, points) in zip(axes, inputs):
        isi = points["isi_ms"].astype(float).to_numpy()
        ppr = points["paired_pulse_ratio"].astype(float).to_numpy()
        fit = fit_ppr_recovery(isi, ppr)
        draw_ppr_recovery(one, isi, ppr, fit, colour="#0072B2")
        one.axhline(1, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
        one.set_xscale("log")
        ticks = sorted(np.unique(isi))
        one.set_xticks(ticks, [f"{t:g}" for t in ticks], fontsize=7, rotation=90 if len(ticks) > 6 else 0)
        one.minorticks_off()
        one.set_ylim(min(0.4, float(np.nanmin(ppr)) - 0.1), max(1.6, float(np.nanmax(ppr)) + 0.1))
        one.set_xlabel("ISI (ms)", fontsize=8)
        one.set_ylabel("PPR (2nd / 1st)", fontsize=8)
        name = " · ".join(str(k) for k in key)
        if fit.ok:
            stats = (f"A = {fit.A:.2f} ± {fit.A_se:.2f}\nτ = {fit.tau_ms:.0f} ± {fit.tau_se_ms:.0f} ms\n"
                     f"R² = {fit.r_squared:.2f}   rms = {fit.rms:.3f}\n{fit.n_isis} ISIs"
                     + ("" if fit.tau_meaningful else "\nτ not meaningful"))
        else:
            stats = f"no fit ({fit.n_isis} ISIs)"
        one.text(0.98, 0.04, stats, transform=one.transAxes, ha="right", va="bottom", fontsize=7.5,
                 family="monospace", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"))
        one.set_title(name, fontsize=9)
    for one in axes[len(inputs):]:
        one.axis("off")
    figure.suptitle((title + "\n" if title else "") + "PPR = 1 - A·exp(-ISI/τ)   (solid: τ meaningful; "
                    "dashed: weak depression or τ poorly determined)", fontsize=9.5, x=0.01, ha="left")
    return figure
