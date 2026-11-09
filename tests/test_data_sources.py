"""
Tests for local folders, the Box data source (with a fake Box client), the
download cache, the "auto" source, the settings file and load_recording().
"""

import hashlib
import textwrap

import pytest

from ephys_imaging_data_loader import (
    BoxCloudDataSource,
    BoxDownloadCache,
    EphysRecording,
    ImageRecording,
    LocalFolderDataSource,
    LocalThenBoxDataSource,
    load_recording,
    open_data_source,
)
from ephys_imaging_data_loader.data_sources import box_cloud_data_source
from ephys_imaging_data_loader.data_sources.base_data_source import matches_pattern, normalize_relative_path

from .conftest import FakeBoxEntry

# =============================================================================
# Path helpers
# =============================================================================


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("/20260114/c1/26114040.abf", "20260114/c1/26114040.abf"),  # leading slash (old notebooks)
        ("20260114\\c1\\x.abf", "20260114/c1/x.abf"),               # Windows separators
        ("./a//b/", "a/b"),
        ("", ""),
    ],
)
def test_normalize_relative_path(raw, clean):
    assert normalize_relative_path(raw) == clean


def test_parent_folder_is_refused():
    with pytest.raises(ValueError):
        normalize_relative_path("../secret.txt")


def test_pattern_matching_is_case_insensitive_and_supports_lists():
    assert matches_pattern("26322010.ABF", "*.abf")
    assert matches_pattern("Image0004.oib", "*.oif;*.oib")
    assert not matches_pattern("notes.txt", "*.abf;*.czi")


# =============================================================================
# Local folders
# =============================================================================


# =============================================================================
# Box (fake client) + download cache
# =============================================================================


def add_file(client, folder_id, file_id, name, content: bytes):
    client.folders_content.setdefault(folder_id, []).append(
        FakeBoxEntry(file_id, name, "file", hashlib.sha1(content).hexdigest(), len(content))
    )
    client.file_bytes[file_id] = content


def add_folder(client, parent_id, folder_id, name):
    client.folders_content.setdefault(parent_id, []).append(FakeBoxEntry(folder_id, name, "folder"))
    client.folders_content.setdefault(folder_id, [])


def make_box_source(client, tmp_path, size_limit=None):
    return BoxCloudDataSource({"ephys": "100"}, BoxDownloadCache(tmp_path / "cache", size_limit), client=client)


def test_all_pages_of_a_big_folder_are_read(fake_box_client, tmp_path, monkeypatch):
    """The old code only looked at the first page; a file on page 3 must be found."""
    monkeypatch.setattr(box_cloud_data_source, "ITEMS_PER_PAGE", 10)
    for number in range(25):
        add_file(fake_box_client, "100", f"{500 + number}", f"file_{number:02d}.abf", b"x")
    source = make_box_source(fake_box_client, tmp_path)
    assert source.file_exists("ephys", "file_24.abf")
    assert len(source.list_files("ephys", "", "*.abf")) == 25
    assert fake_box_client.listing_calls == 3  # 3 pages, listing remembered afterwards


def test_oif_companion_folder_is_downloaded(fake_box_client, tmp_path):
    add_file(fake_box_client, "100", "700", "Image0001.oif", b"[oif]")
    add_folder(fake_box_client, "100", "701", "Image0001.oif.files")
    add_file(fake_box_client, "701", "702", "s_C001.tif", b"tif-bytes")
    source = make_box_source(fake_box_client, tmp_path)
    fetched = source.fetch_file("ephys", "Image0001.oif")
    companion = fetched.local_path.parent / "Image0001.oif.files" / "s_C001.tif"
    assert companion.read_bytes() == b"tif-bytes"


def test_cache_size_limit_removes_least_recently_used(fake_box_client, tmp_path):
    for number in range(3):
        add_file(fake_box_client, "100", f"80{number}", f"f{number}.bin", bytes(1000))
    source = make_box_source(fake_box_client, tmp_path, size_limit=2500)
    for number in range(3):
        source.fetch_file("ephys", f"f{number}.bin")
    cached = {p.name for p in (tmp_path / "cache" / "box_files").rglob("*.bin")}
    assert cached == {"f1.bin", "f2.bin"}  # f0 was used longest ago


# =============================================================================
# "auto" = local first, then Box
# =============================================================================


# =============================================================================
# Settings file + open_data_source + load_recording
# =============================================================================


