"""
epsp_waveform_model.py
======================

A small analytic model of ONE EPSP including the sag (h-current) that pulls
the membrane potential below baseline before it recovers, and the tools to fit
it and to measure peak, 20-80 % rise time and decay time constant.

The model
---------
For time t (ms) after the stimulus and onset latency ``latency``
(tau = t - latency, the EPSP is 0 before its onset):

    v(t) = amplitude * [ k(tau; tau_rise, tau_decay)
                         - sag_ratio * k(tau; tau_sag_rise, tau_sag_decay) ]

    k(tau; a, b) = (exp(-tau/slow) - exp(-tau/fast)) / (peak value of that difference)
                   with fast = min(a, b), slow = max(a, b)

* k is a difference of two exponentials ("double exponential") normalised to a
  peak of 1, so ``amplitude`` is the peak of the depolarising component (mV);
* the first term is the EPSP (fast rise ``tau_rise``, decay ``tau_decay``);
* the second, slower term with weight ``sag_ratio`` is the sag: it develops
  after the EPSP and makes the trace dip below baseline before returning.

Identifiability: without constraints the two terms can become two large,
nearly cancelling double exponentials with similar time constants (good fit,
meaningless parameters). The fit therefore requires the sag to be SLOWER than
the EPSP: tau_sag_rise >= tau_decay and tau_sag_decay >= 1.5 x tau_sag_rise
(fitted as multiples of tau_decay), and sag_ratio <= 1.

Measured on the model curve (``measure_epsp``):
* peak         : maximum of v(t) (mV) and its time;
* rise 20-80 % : time between 20 % and 80 % of the peak on the rising phase (ms);
* decay tau    : ``tau_decay`` of the fit (ms);
* sag          : most negative value after the peak (mV, <= 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

PARAMETER_NAMES = ("amplitude", "latency", "tau_rise", "tau_decay", "sag_ratio", "tau_sag_rise", "tau_sag_decay")

# Allowed ranges (ms, mV). Wide, but they keep the fit physiological.
PARAMETER_BOUNDS = {
    "amplitude": (0.0, 200.0),
    "latency": (0.0, 20.0),
    "tau_rise": (0.1, 30.0),
    "tau_decay": (1.0, 1000.0),
    "sag_ratio": (0.0, 1.0),
    "tau_sag_rise": (2.0, 1000.0),
    "tau_sag_decay": (10.0, 5000.0),
}


def unit_peak_double_exponential(tau: np.ndarray, time_constant_a: float, time_constant_b: float) -> np.ndarray:
    """exp(-tau/slow) - exp(-tau/fast) scaled to a peak of 1 (0 for tau <= 0)."""
    fast, slow = sorted((float(time_constant_a), float(time_constant_b)))
    if slow - fast < 1e-3 * fast:           # equal time constants: avoid 0/0
        slow = fast * 1.001
    peak_time = slow * fast / (slow - fast) * np.log(slow / fast)
    peak_value = np.exp(-peak_time / slow) - np.exp(-peak_time / fast)
    tau = np.maximum(tau, 0.0)
    return (np.exp(-tau / slow) - np.exp(-tau / fast)) / peak_value


def epsp_waveform(t_ms: np.ndarray, amplitude, latency, tau_rise, tau_decay, sag_ratio, tau_sag_rise, tau_sag_decay):
    """The model EPSP (see module docstring); t_ms relative to the stimulus."""
    tau = np.asarray(t_ms, dtype=float) - latency
    epsp = unit_peak_double_exponential(tau, tau_rise, tau_decay)
    sag = unit_peak_double_exponential(tau, tau_sag_rise, tau_sag_decay)
    return np.where(tau > 0, amplitude * (epsp - sag_ratio * sag), 0.0)


@dataclass
class EpspParameters:
    amplitude: float
    latency: float
    tau_rise: float
    tau_decay: float
    sag_ratio: float
    tau_sag_rise: float
    tau_sag_decay: float

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in PARAMETER_NAMES}

    def waveform(self, t_ms: np.ndarray, onset_ms: float = 0.0) -> np.ndarray:
        """Model curve for a stimulus at ``onset_ms``."""
        return epsp_waveform(np.asarray(t_ms) - onset_ms, **self.as_dict())

    def with_values(self, **changes) -> "EpspParameters":
        return EpspParameters(**{**self.as_dict(), **changes})


@dataclass
class EpspMeasures:
    """What is reported for one EPSP."""

    peak_mv: float
    peak_time_ms: float          # from the stimulus
    rise_20_80_ms: float
    decay_tau_ms: float
    sag_mv: float                # most negative value after the peak (<= 0)
    latency_ms: float
    rise_points_ms: tuple[float, float] = field(default=(np.nan, np.nan))   # times of 20 % and 80 %
    max_slope_mv_per_ms: float = np.nan   # largest dV/dt (mV/ms, pA/ms for EPSCs), onset -> peak
    max_slope_time_ms: float = np.nan     # when it occurs (from the stimulus)
    peak_capped: bool = False             # the curve still rises at onset + PEAK_WINDOW_MS: peak taken there


# The peak is searched only up to PEAK_WINDOW_MS after the EPSP onset (the fit's
# latency): uncaging EPSPs rise in < ~15 ms, so a later maximum is more likely a
# polysynaptic / late component than the EPSP itself. Applies to the fitted
# curve here and to the peak of the trace (``unitary_cluster_summation._raw_peak``).
PEAK_WINDOW_MS = 30.0


# The maximum slope is searched from the EPSP onset (the fit's latency) to the
# EPSP peak: the largest dV/dt of an EPSP is on its rise, and searching the long
# decay of a small response on the recorded trace only picks up noise/artefacts.


def max_slope_window_ms(measures: "EpspMeasures") -> tuple[float, float]:
    """(start, end) of the max-slope search: EPSP onset -> EPSP peak."""
    return measures.latency_ms, measures.peak_time_ms


def measure_epsp(parameters: EpspParameters, duration_ms: float = 1000.0, step_ms: float = 0.02) -> EpspMeasures:
    """
    Peak, 20-80 % rise time, decay tau, sag and max slope (dV/dt) of a model EPSP
    (stimulus at t = 0). The peak is the maximum between the onset and onset +
    PEAK_WINDOW_MS; ``peak_capped`` when the curve's own maximum is later.
    """
    t = np.arange(0.0, duration_ms, step_ms)
    v = parameters.waveform(t)
    in_window = t <= parameters.latency + PEAK_WINDOW_MS
    peak_index = int(np.argmax(np.where(in_window, v, -np.inf)))
    peak = float(v[peak_index])
    capped = bool(peak > 0 and v.max() > peak * (1 + 1e-6) and int(np.argmax(v)) > peak_index)
    rise = (np.nan, np.nan)
    if peak > 0 and peak_index > 0:
        rising_t, rising_v = t[: peak_index + 1], v[: peak_index + 1]
        # first crossing of 20 % / 80 % on the rising phase (linear interpolation)
        rise = tuple(float(np.interp(fraction * peak, rising_v, rising_t)) for fraction in (0.2, 0.8))
    decay_tau = max(parameters.tau_rise, parameters.tau_decay)
    # max dV/dt of the curve between onset and peak
    slope, slope_time = np.nan, np.nan
    # forward differences, at the midpoints: the model's rise starts with a kink at
    # the onset (often the steepest point), which a central difference would halve
    dv_dt, t_mid = np.diff(v) / step_ms, t[:-1] + step_ms / 2
    window = (t_mid >= parameters.latency) & (t_mid <= float(t[peak_index]))
    if peak > 0 and window.sum() > 2:
        best = int(np.argmax(np.where(window, dv_dt, -np.inf)))
        slope, slope_time = float(dv_dt[best]), float(t_mid[best])
    return EpspMeasures(
        peak_mv=peak,
        peak_time_ms=float(t[peak_index]),
        rise_20_80_ms=rise[1] - rise[0],
        decay_tau_ms=decay_tau,
        sag_mv=min(0.0, float(v[peak_index:].min())),
        latency_ms=parameters.latency,
        rise_points_ms=rise,
        max_slope_mv_per_ms=slope,
        max_slope_time_ms=slope_time,
        peak_capped=capped,
    )


def initial_guess(t_ms: np.ndarray, v: np.ndarray) -> EpspParameters:
    """Rough starting values read off an isolated EPSP (stimulus at t = 0)."""
    after = t_ms >= 0
    t, y = t_ms[after], v[after]
    peak_index = int(np.argmax(y))
    peak = max(float(y[peak_index]), 1e-3)
    onset = t[np.argmax(y >= 0.1 * peak)]
    latency = max(0.0, float(onset) - 1.0)
    tau_rise = max(0.3, (float(t[peak_index]) - latency) / 3)
    below = np.nonzero(y[peak_index:] <= peak / np.e)[0]
    tau_decay = float(t[peak_index + below[0]] - t[peak_index]) if below.size else 30.0
    sag = max(0.0, -float(y[peak_index:].min()))
    return EpspParameters(peak, latency, tau_rise, max(tau_decay, 2 * tau_rise), min(sag / peak * 2, 1.0), 40.0, 200.0)


def fit_waveform(
    model,                     # function(t, parameter dict) -> curve
    t_ms: np.ndarray,
    v: np.ndarray,
    starts: list[dict[str, float]],
    bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], float]:
    """
    Least-squares fit of ``model`` with the free parameters named in ``bounds``,
    tried from every start in ``starts``; returns (best parameters, residual RMS).
    """
    names = list(bounds)
    low = np.array([bounds[n][0] for n in names])
    high = np.array([bounds[n][1] for n in names])
    best, best_cost = None, np.inf
    for start in starts:
        x0 = np.clip([start[n] for n in names], low + 1e-9, high - 1e-9)
        result = least_squares(lambda x: model(t_ms, dict(zip(names, x))) - v, x0, bounds=(low, high),
                               x_scale="jac", max_nfev=4000)
        if result.cost < best_cost:
            best, best_cost = dict(zip(names, result.x)), result.cost
    return best, float(np.sqrt(2 * best_cost / len(v)))


# The sag time constants are fitted as multiples of the EPSP decay (see module docstring).
SAG_RISE_FACTOR_BOUNDS = (1.0, 30.0)      # tau_sag_rise  = factor x tau_decay
SAG_DECAY_FACTOR_BOUNDS = (1.5, 50.0)     # tau_sag_decay = factor x tau_sag_rise


def _from_factors(p: dict) -> dict:
    """Fit parameters (with sag factors) -> model parameters (with sag time constants)."""
    model = {k: v for k, v in p.items() if k not in ("sag_rise_factor", "sag_decay_factor")}
    model["tau_sag_rise"] = p["tau_decay"] * p["sag_rise_factor"]
    model["tau_sag_decay"] = model["tau_sag_rise"] * p["sag_decay_factor"]
    return model


def fit_epsp(t_ms: np.ndarray, v: np.ndarray, with_sag: bool = True) -> tuple[EpspParameters, float]:
    """
    Fit the 7-parameter model to ONE isolated EPSP (e.g. the template), with the
    sag constrained to be slower than the EPSP; several starting values for the
    decay and sag are tried. Returns (parameters, residual RMS in mV).

    ``with_sag=False``: no sag term (sag_ratio fixed at 0) — for synaptic CURRENTS
    (EPSCs in voltage clamp), which have no h-current sag; only amplitude,
    latency, rise and decay are fitted.
    """
    guess = initial_guess(t_ms, v)
    if not with_sag:
        bounds = {name: PARAMETER_BOUNDS[name] for name in ("amplitude", "latency", "tau_rise", "tau_decay")}
        no_sag = {"sag_ratio": 0.0, "tau_sag_rise": 100.0, "tau_sag_decay": 500.0}
        starts = [{k: value for k, value in guess.with_values(tau_decay=guess.tau_decay * scale).as_dict().items()
                   if k in bounds} for scale in (0.7, 1.0, 1.5)]
        fitted, rms = fit_waveform(lambda t, p: epsp_waveform(t, **p, **no_sag), t_ms, v, starts, bounds)
        return EpspParameters(**fitted, **no_sag), rms
    bounds = {name: PARAMETER_BOUNDS[name] for name in ("amplitude", "latency", "tau_rise", "tau_decay", "sag_ratio")}
    bounds.update(sag_rise_factor=SAG_RISE_FACTOR_BOUNDS, sag_decay_factor=SAG_DECAY_FACTOR_BOUNDS)
    starts = []
    for decay_scale in (0.7, 1.0, 1.5):
        for rise_factor, decay_factor in ((1.5, 3.0), (3.0, 3.0), (6.0, 2.0)):
            start = guess.with_values(tau_decay=guess.tau_decay * decay_scale).as_dict()
            start.update(sag_rise_factor=rise_factor, sag_decay_factor=decay_factor,
                         sag_ratio=max(start["sag_ratio"], 0.05))
            del start["tau_sag_rise"], start["tau_sag_decay"]
            starts.append(start)
    fitted, rms = fit_waveform(lambda t, p: epsp_waveform(t, **_from_factors(p)), t_ms, v, starts, bounds)
    return EpspParameters(**_from_factors(fitted)), rms
