"""Tests for interactive_viewers/image_grid_figure.py: the ways images can be asked for."""

import pytest

from ephys_imaging_data_loader.interactive_viewers.image_grid_figure import (
    build_image_grid_figure,
    parse_image_requests,
)


def test_image_requests():
    assert parse_image_requests([[20270408, 31], [20270408, 1, 32]]) == [("20270408", 31, None), ("20270408", 32, 1)]
    assert parse_image_requests(["20270408 31", "20270408,1,32"]) == [("20270408", 31, None), ("20270408", 32, 1)]
    assert parse_image_requests({20270401: range(9, 11)}) == [("20270401", 9, None), ("20270401", 10, None)]
    with pytest.raises(ValueError):
        parse_image_requests([[20270408]])


def test_grid_with_missing_images():
    figure = build_image_grid_figure([{"recording": None, "title": "a", "error": "not listed"},
                                      {"recording": None, "title": "b"}], columns=4)
    assert len(figure.axes) == 2
