"""
Tests for naming uncaging targets after spine ids (experiment_catalog/uncaging_target_labels.py).

A small fake catalog stands in for the sheet + image files: each "image" only
needs uncaging-point coordinates. Positions mimic 20260322, dendrite 1: the
5-target cluster image has each target within ~15 px of the single-spine target.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ephys_imaging_data_loader.experiment_catalog.uncaging_target_labels import (
    _match_by_position,
    label_uncaging_targets,
)
from ephys_imaging_data_loader.recording_data_containers import ImageRecording

SINGLE = {1: (213, 221), 2: (286, 283), 3: (294, 210), 4: (183, 200), 5: (152, 189)}
CLUSTER = [(213, 206), (285, 278), (291, 203), (187, 192), (150, 182)]


def make_image(name, points, date="20260322", folder="c1"):
    table = pd.DataFrame(
        {"point": range(len(points)), "x_px": [p[0] for p in points], "y_px": [p[1] for p in points],
         "x_px_bleach_gui": [p[0] for p in points], "y_px_bleach_gui": [p[1] for p in points]}
    )
    return ImageRecording(
        file_name=f"{name}.oib", source_description=f"local: /data/{date}/{folder}/{name}.oib",
        local_file_path=Path(f"/data/{date}/{folder}/{name}.oib"), microscope="Olympus FV1000",
        acquisition_type="single frame", acquisition_datetime=datetime.datetime(2022, 2, 11, 18),
        shape_tczyx=(1, 1, 1, 512, 512), dtype=np.dtype("uint16"), bits_per_pixel=12, channel_names=["RXD2"],
        uncaging_points=table,
    )


class FakeCatalog:
    """Rows of one dendrite: spines 1-5 with single-target images, spine 0 with the cluster image."""

    def __init__(self, cluster_points=CLUSTER, with_single_images=True):
        rows = [{"expdate": "20260322", "neuronid": 1, "dendriteid": 1, "spineid": s, "group": "pre-apv",
                 "ophysfile": f"Image000{4 + s}"} for s in SINGLE]
        rows.append({"expdate": "20260322", "neuronid": 1, "dendriteid": 1, "spineid": 0, "group": "pre-apv",
                     "ophysfile": "Image0010.oib"})
        self.table = pd.DataFrame(rows)
        self.table["neuronid"] = self.table["neuronid"].astype("Int64")
        self.table["spineid"] = self.table["spineid"].astype("Int64")
        self.images = {f"Image000{4 + s}": make_image(f"Image000{4 + s}", [xy]) for s, xy in SINGLE.items()}
        self.with_single_images = with_single_images
        self.cluster = make_image("Image0010", cluster_points)

    def load_imaging(self, label):
        if not self.with_single_images:
            raise FileNotFoundError("not available")
        return self.images[self.table.loc[label, "ophysfile"]]


def test_single_target_gets_its_spine_id():
    catalog = FakeCatalog()
    result = label_uncaging_targets(catalog.images["Image0007"], catalog)
    assert result.labels == ["3"] and "experiment sheet" in result.note


def test_spineid_zero_cluster_points_are_not_named():
    """spineid 0: the sheet gives no id per point -> no numbers are invented (not even by position)."""
    catalog = FakeCatalog()
    result = label_uncaging_targets(catalog.cluster, catalog)
    assert result.labels == [""] * 5 and "spineid 0" in result.note and result.n_on_spine == 5


def test_listed_cluster_uses_the_sheet_digits_in_scan_order():
    catalog = FakeCatalog()
    catalog.table.loc[len(catalog.table) - 1, "spineid"] = 31524
    result = label_uncaging_targets(catalog.cluster, catalog)
    assert result.labels == ["3", "1", "5", "2", "4"] and "sheet" in result.note


def test_image_not_in_sheet():
    catalog = FakeCatalog()
    result = label_uncaging_targets(make_image("Image0099", [(10, 10)], date="20240209"), catalog)
    assert result.labels == [""] and "not found" in result.note and not result.resolved


def test_matching_refuses_far_or_insufficient_positions():
    targets = np.array([[0.0, 0.0], [100.0, 100.0]])
    assert _match_by_position(targets, {"1": (2, 2), "2": (101, 99)}) == ["1", "2"]
    assert _match_by_position(targets, {"1": (2, 2), "2": (300, 300)}) is None   # too far
    assert _match_by_position(targets, {"1": (2, 2)}) is None                    # not enough spines


# =============================================================================
# Image title: file | ephys file | n d s | date | age | area | uncaging power
# =============================================================================
class TitleCatalog(FakeCatalog):
    def __init__(self):
        super().__init__()
        t = self.table
        t["ephys_file_name"] = [f"2632201{s}" for s in t["spineid"]]
        t["expname"], t["age_days"], t["area"], t["side"], t["sex"], t["temp"] = "photostim_test", 40, "ca3", "basal", "female", 31.5
        t["ephysid"] = [10 + s for s in t["spineid"]]
        t["stim_power_percent"], t["zstack"] = 30.0, None
        t.loc[len(t)] = {**t.iloc[0].to_dict(), "spineid": 0, "ophysfile": None, "zstack": "16;17;18", "dendriteid": 2}


def test_image_title_for_paired_image():
    from ephys_imaging_data_loader.interactive_viewers.image_recording_title import image_title_parts

    catalog = TitleCatalog()
    image = catalog.images["Image0007"]
    image.metadata["uncaging_laser_power_percent"] = 48.0          # file metadata wins over the sheet (30)
    title, details = image_title_parts(image, catalog)
    assert title == "20260322 | e/o ids: 13/7 | nds: 113 | 40 d Female | ca3 | basal | LP: 48% | 31.5 °C"
    assert "photostim_test" in details


def test_image_title_for_zstack_and_unlisted_image():
    from ephys_imaging_data_loader.interactive_viewers.image_recording_title import image_title_parts

    catalog = TitleCatalog()
    zstack = make_image("Image0017", [])
    assert image_title_parts(zstack, catalog)[0] == "20260322 | o id: 17 | nd: 12 | 40 d Female | ca3 | basal | 31.5 °C"
    unlisted = make_image("Image0099", [])
    title, details = image_title_parts(unlisted, catalog)
    assert title == "20260322 | o id: 99 | n: 1 | 40 d Female | ca3 | basal | 31.5 °C" and "not listed" in details


def test_two_digit_ids_are_separated_by_dots():
    from ephys_imaging_data_loader.interactive_viewers.image_recording_title import image_title_parts

    catalog = TitleCatalog()
    catalog.table.loc[2, "spineid"] = 12                     # Image0007's row now spine 12
    title, _ = image_title_parts(catalog.images["Image0007"], catalog)
    assert "| nds: 1.1.12 |" in title


def test_laser_power_and_duration_in_caption():
    from ephys_imaging_data_loader.interactive_viewers.image_recording_title import image_title_parts

    catalog = TitleCatalog()
    image = catalog.images["Image0007"]
    image.metadata.update(uncaging_laser_power_percent=60.0, uncaging_duration_ms=0.2)
    assert "| LP: 60%, 0.2ms / 0.2ms |" in image_title_parts(image, catalog)[0]


# =============================================================================
# Shrinking clusters: unwanted targets are MOVED OFF the dendrite (stay active)
# =============================================================================
FULL8 = [(272, 138), (213, 114), (202, 210), (194, 265), (248, 278), (186, 300), (174, 419), (174, 363)]


class ShrinkingCatalog:
    """One session: full cluster 12345678 (Image0021), then 123 (Image0026) with targets 4-8 moved away."""

    def __init__(self, shrunk_spineid=123, kept=(0, 1, 2)):
        moved = [p if i in kept else (440 + 5 * i, 400) for i, p in enumerate(FULL8)]
        self.images = {"Image0021": make_image("Image0021", FULL8, date="20270313"),
                       "Image0026": make_image("Image0026", moved, date="20270313")}
        self.table = pd.DataFrame([
            {"expdate": "20270313", "neuronid": 1, "dendriteid": 1, "spineid": 12345678, "group": "ctrl", "ophysfile": "Image0021"},
            {"expdate": "20270313", "neuronid": 1, "dendriteid": 1, "spineid": shrunk_spineid, "group": "ctrl", "ophysfile": "Image0026"},
        ])
        self.table["neuronid"] = self.table["neuronid"].astype("Int64")
        self.table["spineid"] = self.table["spineid"].astype("Int64")

    def load_imaging(self, label):
        return self.images[self.table.loc[label, "ophysfile"]]


def test_shrinking_cluster_marks_moved_targets():
    catalog = ShrinkingCatalog()
    result = label_uncaging_targets(catalog.images["Image0026"], catalog)
    assert result.labels == ["1", "2", "3", None, None, None, None, None]
    assert result.n_on_spine == 3 and result.last_on_spine_position == 3 and result.resolved


def test_kept_spines_at_the_end_of_the_scan_order():
    """20270409-style: spines 6,7,8 kept, 1-5 moved -> last on-spine target is #8."""
    catalog = ShrinkingCatalog(shrunk_spineid=678, kept=(5, 6, 7))
    result = label_uncaging_targets(catalog.images["Image0026"], catalog)
    assert result.labels == [None] * 5 + ["6", "7", "8"] and result.last_on_spine_position == 8


def test_shrinking_cluster_keeps_the_sheet_ids_and_order():
    """Sheet 234 with points 1-3 kept: labels are the sheet's 2, 3, 4 (not re-numbered)."""
    catalog = ShrinkingCatalog(shrunk_spineid=234, kept=(0, 1, 2))
    result = label_uncaging_targets(catalog.images["Image0026"], catalog)
    assert result.labels == ["2", "3", "4"] + [None] * 5


def test_count_mismatch_is_not_guessed():
    catalog = ShrinkingCatalog(shrunk_spineid=12, kept=(0, 1, 2))      # 3 stayed, sheet says 2 spines
    assert label_uncaging_targets(catalog.images["Image0026"], catalog).resolved is False


def test_zero_and_full_list_mean_the_same_cluster():
    """Same recording listed as spineid 0 and as 12345 (5 targets): all targets on spines."""
    catalog = FakeCatalog()
    catalog.table.loc[len(catalog.table)] = {**catalog.table.iloc[-1].to_dict(), "spineid": 12345}
    assert label_uncaging_targets(catalog.cluster, catalog).labels == ["1", "2", "3", "4", "5"]


def test_different_single_spines_sharing_a_cluster_image_are_unresolved():
    catalog = FakeCatalog()
    catalog.table.loc[len(catalog.table)] = {**catalog.table.iloc[-1].to_dict(), "spineid": 3}
    assert label_uncaging_targets(catalog.cluster, catalog).resolved is False
