"""Tests for the extra panels of the set summation figure (spine_summation_figure.py)."""

import warnings

import numpy as np
import pytest

import pandas as pd

from ephys_imaging_data_loader.interactive_viewers.spine_summation_figure import _draw_ppr_vs_isi, _draw_spine_map


def _render(draw):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4, 3), layout="constrained")
    axes = figure.add_subplot()
    draw(axes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # e.g. "constrained_layout not applied"
        figure.canvas.draw()
    return axes


def test_ppr_panel_one_line_per_cluster_and_no_layout_collapse():
    table = pd.DataFrame({
        "input": ["spine 1"] + ["cluster 1-2"] * 3 + ["cluster 1-2-3"],
        "kind": ["unitary"] + ["cluster"] * 4,
        "n_spines": [1, 2, 2, 2, 3],
        "isi_ms": [1000, 50, 100, 1000, 1000],
        "paired_pulse_ratio": [1e12, 0.8, 0.9, 1.0, 1.1],     # single spines are not drawn
    })
    axes = _render(lambda a: _draw_ppr_vs_isi(a, table))
    lines = [line for line in axes.get_lines() if len(line.get_xdata()) and line.get_marker() == "o"]
    assert sorted(len(line.get_xdata()) for line in lines) == [1, 3]                  # data: one per cluster
    assert any(len(line.get_xdata()) == 150 for line in axes.get_lines())            # + the PPR fit (3 ISIs)
    assert axes.get_ylim()[1] == 2.0


def test_ppr_above_display_range_is_marked_and_spine_map_may_be_missing():
    table = pd.DataFrame({"input": ["cluster 1-2"] * 2, "kind": ["cluster"] * 2, "n_spines": [2, 2],
                          "isi_ms": [50, 1000], "paired_pulse_ratio": [9.0, 1.0]})
    axes = _render(lambda a: _draw_ppr_vs_isi(a, table))
    assert any(line.get_marker() == "^" for line in axes.get_lines())
    _render(lambda a: _draw_spine_map(a, None))


def test_ppr_panel_only_with_a_cluster_isi_up_to_100_ms_and_fixed_spine_count_colours():
    from ephys_imaging_data_loader.interactive_viewers.spine_summation_figure import (
        SINGLE_SPINE_COLOUR, _has_short_isi_ppr, spine_count_colour)

    table = pd.DataFrame({"input": ["cluster 1-2"] * 2 + ["spine 1"], "kind": ["cluster"] * 2 + ["unitary"],
                          "n_spines": [2, 2, 1], "isi_ms": [500, 1000, 50], "paired_pulse_ratio": [0.9, 1.0, 1.0]})
    assert not _has_short_isi_ppr(table)              # a single spine at 50 ms does not count
    table.loc[0, "isi_ms"] = 100
    assert _has_short_isi_ppr(table)
    colours = [spine_count_colour(n) for n in range(2, 11)]
    assert len(set(map(tuple, colours))) == 9 and spine_count_colour(12) == colours[-1]
    assert spine_count_colour(1) == SINGLE_SPINE_COLOUR


def test_scatter_points_for_shifted_and_unshifted_sums():
    from ephys_imaging_data_loader.interactive_viewers.spine_summation_figure import _scatter_points

    table = pd.DataFrame({"input": ["spine 1", "cluster 1-2"], "kind": ["unitary", "cluster"], "n_spines": [1, 2],
                          "measured_peak": [0.3, 0.5], "linear_sum_peak": [float("nan"), 0.6],
                          "linear_sum_unshifted_peak": [float("nan"), 0.65]})
    assert _scatter_points(table, "peak", "linear_sum_unshifted").set_index("n").loc[2, "x"] == 0.65
    # older CSVs without the unshifted columns: only the single spines
    assert list(_scatter_points(table.drop(columns="linear_sum_unshifted_peak"), "peak",
                                "linear_sum_unshifted")["n"]) == [1]


def test_cluster_spread_is_mean_distance_from_centroid_in_um():
    from ephys_imaging_data_loader.experiment_catalog.set_spine_map_image import spread_um

    # 4 points on a square of side 2 px: each sqrt(2) px from the centre; 0.5 um per px
    assert spread_um([0, 2, 0, 2], [0, 0, 2, 2], 0.5) == pytest.approx(np.sqrt(2) * 0.5)
    assert np.isnan(spread_um([3], [4], 0.5))       # one spine: no spread
