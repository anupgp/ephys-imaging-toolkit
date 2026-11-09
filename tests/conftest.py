"""
Shared test fixtures.

All tests use synthetic data generated in the tests themselves.
No recorded data is part of this repository.

Run all tests from the project folder with:

    python -m pytest -v
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pytest


# =============================================================================
# A fake Box client, so the Box code can be tested without network access.
# It mimics only the parts of box_sdk_gen.BoxClient that the package uses.
# =============================================================================
@dataclass
class FakeBoxEntry:
    """Same attribute names as box_sdk_gen's FileFull / FolderMini objects."""

    id: str
    name: str
    type: str
    sha_1: str | None = None  # the real SDK uses "sha_1" (verified against Box)
    size: int | None = None


@dataclass
class FakeBoxPage:
    entries: list
    total_count: int


@dataclass
class FakeBoxClient:
    """
    folders : folder id -> list of FakeBoxEntry
    file_bytes : file id -> content
    Counts calls so tests can check that caching avoids repeated work.
    """

    folders_content: dict = field(default_factory=dict)
    file_bytes: dict = field(default_factory=dict)
    listing_calls: int = 0
    download_calls: int = 0

    def __post_init__(self):
        client = self

        class Folders:
            def get_folder_items(self, folder_id, fields=None, offset=0, limit=100):
                client.listing_calls += 1
                everything = client.folders_content.get(folder_id, [])
                return FakeBoxPage(everything[offset : offset + limit], len(everything))

        class Downloads:
            def download_file(self, file_id):
                client.download_calls += 1
                return io.BytesIO(client.file_bytes[file_id])

        self.folders = Folders()
        self.downloads = Downloads()


@pytest.fixture
def fake_box_client() -> FakeBoxClient:
    return FakeBoxClient()
