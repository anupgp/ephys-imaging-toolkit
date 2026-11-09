"""
display_helpers.py
==================

Small, well-tested helper functions shared by all viewers.

None of these change the recorded data; they only prepare it for display.
"""

from __future__ import annotations

import math

import numpy as np

# Colour-blind friendly palette (Okabe & Ito) used for channels / digital outputs.
COLORBLIND_SAFE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def min_max_downsample(
    time_s: np.ndarray, values: np.ndarray, max_points: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce a long trace to about ``max_points`` points WITHOUT losing peaks.

    Why: a 4 s sweep at 20 kHz has 80,000 samples; 33 sweeps x 3 channels is
    ~8 million points, which makes the browser slow. A screen is only ~1000
    pixels wide, so drawing every sample is pointless — but simply taking every
    20th sample could skip a fast EPSC peak or a stimulus artefact.

    How: the trace is split into ``max_points // 2`` equal bins, and for every
    bin both the minimum and the maximum are kept (in their time order). Every
    peak and trough therefore stays visible at any zoom level of the overview.

    Returns (time_s, values) of the reduced trace. Short traces are returned
    unchanged.
    """
    n_samples = len(values)
    if n_samples <= max_points or max_points < 4:
        return time_s, values

    n_bins = max_points // 2
    samples_per_bin = n_samples // n_bins
    usable = n_bins * samples_per_bin  # drop the few samples that do not fill a bin

    binned = values[:usable].reshape(n_bins, samples_per_bin)
    index_of_min = binned.argmin(axis=1)
    index_of_max = binned.argmax(axis=1)

    # Keep min and max of every bin in the order they occur in time.
    first = np.minimum(index_of_min, index_of_max)
    second = np.maximum(index_of_min, index_of_max)
    bin_starts = np.arange(n_bins) * samples_per_bin
    keep = np.empty(2 * n_bins, dtype=np.int64)
    keep[0::2] = bin_starts + first
    keep[1::2] = bin_starts + second
    return time_s[keep], values[keep]


def contrast_limits(
    image: np.ndarray, low_percentile: float = 0.5, high_percentile: float = 99.5
) -> tuple[float, float]:
    """
    Display range (black level, white level) from intensity percentiles.

    Using percentiles instead of min/max stops a few very bright or dead
    pixels from making the whole image look dark or washed out.
    """
    finite = image[np.isfinite(image)] if np.issubdtype(image.dtype, np.floating) else image
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if high <= low:  # flat image: avoid a zero display range
        high = low + 1.0
    return float(low), float(high)


def nice_scale_bar_length(image_width_um: float, target_fraction: float = 0.2) -> float:
    """
    A "round" scale-bar length (1, 2 or 5 x 10^n um) close to
    ``target_fraction`` of the image width.

    Example: an image 23.5 um wide -> 5 um bar; 425 um wide -> 100 um bar.
    """
    if not image_width_um or image_width_um <= 0:
        return 0.0
    target = image_width_um * target_fraction
    power_of_ten = 10 ** math.floor(math.log10(target))
    for multiplier in (5, 2, 1):
        if multiplier * power_of_ten <= target:
            return float(multiplier * power_of_ten)
    return float(power_of_ten)


def format_scale_bar_label(length_um: float) -> str:
    """'10 µm' or '0.5 µm' (no useless decimals)."""
    return f"{length_um:g} µm"


def display_figure_in_notebook(figure, dpi: int = 100) -> bool:
    """
    Show a matplotlib Figure in a notebook as a PNG image. Figures made with
    ``matplotlib.figure.Figure`` (not pyplot) are only shown by ``display(figure)``
    when matplotlib's inline display is active, which depends on the notebook
    (VS Code, JupyterLab ...); a PNG is shown everywhere. Returns False outside IPython.
    """
    import io

    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return False
    if get_ipython() is None:
        return False
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    display(Image(data=buffer.getvalue()))
    return True
