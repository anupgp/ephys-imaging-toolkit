"""
box_cloud_data_source.py
========================

Data source for files stored on Box (box.com).

What it does
------------
* Finds files by root name + relative path, e.g.
  ("ephys_olympus2p", "20260322/c1/26322010.abf"). Each root name maps to a
  Box folder id in the settings file.
* Downloads a file only when needed, into a ``BoxDownloadCache`` (so the
  second time it is instant, and only re-downloaded if it changed on Box).
* Lists folder contents with wildcard patterns, optionally recursively.

Improvements over the older ``BoxAuthApp.get_file_stream``
----------------------------------------------------------
* **All pages are read.** Box returns folder contents in pages; the old
  file lookup only checked the first page (100 items by default), so files in
  big folders could be reported as missing. Here every page is read.
* **Folder listings are remembered** during a session, so walking the same
  path again (e.g. loading 50 files from one cell folder) costs no extra
  requests. Call ``forget_folder_listings()`` to see new uploads.
* **Case-insensitive fallback.** Folder names on Box are sometimes "C1" and
  sometimes "c1"; if the exact name is not found, a case-insensitive match is
  used (and a message is logged).
* **Streaming download to disk** (no full copy of the file in memory).
* **Olympus .oif files** are downloaded together with their companion
  "<name>.oif.files" folder, which the reader needs.

Login happens lazily: nothing contacts Box until a file or listing is needed.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .base_data_source import DataSource, FetchedFile, matches_pattern, normalize_relative_path
from .box_download_cache import COPY_BLOCK_SIZE, BoxDownloadCache

logger = logging.getLogger(__name__)

# Box allows at most 1000 items per page.
ITEMS_PER_PAGE = 1000
# Only these fields are requested per item: fewer fields = faster listings.
ITEM_FIELDS = ["id", "type", "name", "sha1", "size"]


@dataclass(frozen=True)
class BoxItem:
    """A file or folder on Box, as returned by a folder listing."""

    item_id: str
    name: str
    item_type: str  # "file", "folder" or "web_link"
    sha1: str | None = None
    size_bytes: int | None = None


def _type_name(raw_type: Any) -> str:
    """The Box SDK returns item types as enums; turn them into plain text."""
    return str(getattr(raw_type, "value", raw_type)).lower()


class BoxCloudDataSource(DataSource):
    """Reads files from Box through a local download cache."""

    kind = "box"

    def __init__(
        self,
        root_folder_ids: dict[str, str],
        cache: BoxDownloadCache,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        """
        Parameters
        ----------
        root_folder_ids : dict
            Root name -> Box folder id (the number at the end of the folder's
            Box URL), e.g. {"ephys_olympus2p": "123456789012"}.
        cache : BoxDownloadCache
            Where downloaded files are kept.
        client : a logged-in ``box_sdk_gen.BoxClient`` (optional)
        client_factory : function returning a logged-in client, called on
            first use (used when ``client`` is not given, so that creating
            this object never triggers a login by itself).
        """
        self.root_folder_ids = {name: str(fid) for name, fid in root_folder_ids.items() if fid}
        self.cache = cache
        self._client = client
        self._client_factory = client_factory
        # folder id -> list of BoxItem (filled on first listing of each folder)
        self._folder_listing_memory: dict[str, list[BoxItem]] = {}

    # ---------------------------------------------------------------------
    # Connection
    # ---------------------------------------------------------------------
    @property
    def client(self):
        """The Box client; logs in on first use if needed."""
        if self._client is None:
            if self._client_factory is None:
                raise RuntimeError("BoxCloudDataSource has neither a client nor a client_factory.")
            self._client = self._client_factory()
        return self._client

    def root_names(self) -> list[str]:
        return list(self.root_folder_ids)

    # ---------------------------------------------------------------------
    # Folder listings and path resolution
    # ---------------------------------------------------------------------
    def list_folder(self, folder_id: str) -> list[BoxItem]:
        """All items of one Box folder (every page), remembered for this session."""
        if folder_id in self._folder_listing_memory:
            return self._folder_listing_memory[folder_id]

        items: list[BoxItem] = []
        offset = 0
        while True:
            page = self.client.folders.get_folder_items(
                folder_id, fields=ITEM_FIELDS, offset=offset, limit=ITEMS_PER_PAGE
            )
            for entry in page.entries or []:
                items.append(
                    BoxItem(
                        item_id=str(entry.id),
                        name=str(entry.name),
                        item_type=_type_name(entry.type),
                        # The SDK calls the checksum attribute "sha_1" (API field "sha1").
                        sha1=getattr(entry, "sha_1", None),
                        size_bytes=getattr(entry, "size", None),
                    )
                )
            offset += ITEMS_PER_PAGE
            total = getattr(page, "total_count", None)
            # Stop when Box says we have everything, or when a page is not full.
            if not page.entries or (total is not None and offset >= total) or len(page.entries) < ITEMS_PER_PAGE:
                break

        self._folder_listing_memory[folder_id] = items
        return items

    def forget_folder_listings(self) -> None:
        """Forget remembered listings (use after new files were uploaded to Box)."""
        self._folder_listing_memory.clear()

    def _find_child(self, folder_id: str, name: str, wanted_type: str | None) -> BoxItem | None:
        """Find an item by name in a folder: exact match first, then ignoring case."""
        children = [
            item for item in self.list_folder(folder_id)
            if wanted_type is None or item.item_type == wanted_type
        ]
        for item in children:
            if item.name == name:
                return item
        case_insensitive = [item for item in children if item.name.lower() == name.lower()]
        if len(case_insensitive) == 1:
            logger.info("Using '%s' on Box for requested name '%s'.", case_insensitive[0].name, name)
            return case_insensitive[0]
        return None

    def _resolve_folder(self, root: str, relative_folder: str) -> str:
        """Box folder id of ``relative_folder`` inside a root."""
        self._check_root(root)
        folder_id = self.root_folder_ids[root]
        walked = []
        for part in [p for p in normalize_relative_path(relative_folder).split("/") if p]:
            child = self._find_child(folder_id, part, "folder")
            if child is None:
                raise FileNotFoundError(
                    f"Folder '{part}' not found on Box in '{root}/{'/'.join(walked)}'."
                )
            folder_id = child.item_id
            walked.append(child.name)
        return folder_id

    def _resolve_file(self, root: str, relative_path: str | Path) -> tuple[BoxItem, str]:
        """(BoxItem of the file, id of the folder containing it)."""
        clean_path = normalize_relative_path(relative_path)
        folder_part, _, file_name = clean_path.rpartition("/")
        folder_id = self._resolve_folder(root, folder_part)
        item = self._find_child(folder_id, file_name, "file")
        if item is None:
            raise FileNotFoundError(f"File '{clean_path}' not found on Box in root '{root}'.")
        return item, folder_id

    # ---------------------------------------------------------------------
    # DataSource interface
    # ---------------------------------------------------------------------
    def file_exists(self, root: str, relative_path: str | Path) -> bool:
        try:
            self._resolve_file(root, relative_path)
            return True
        except (FileNotFoundError, KeyError):
            return False

    def _download_function(self, file_id: str) -> Callable[[BinaryIO], None]:
        """A function that streams one Box file into an open file handle."""

        def download_into(handle: BinaryIO) -> None:
            stream = self.client.downloads.download_file(file_id)
            shutil.copyfileobj(stream, handle, COPY_BLOCK_SIZE)

        return download_into

    def fetch_file(self, root: str, relative_path: str | Path) -> FetchedFile:
        """Return a local copy of a Box file, downloading it only if needed."""
        clean_path = normalize_relative_path(relative_path)
        item, parent_folder_id = self._resolve_file(root, clean_path)
        box_path = f"{root}/{clean_path}"

        local_path = self.cache.lookup(item.item_id, item.sha1)
        if local_path is None:
            size_text = f" ({item.size_bytes / 1e6:.1f} MB)" if item.size_bytes else ""
            logger.info("Downloading %s from Box%s ...", box_path, size_text)
            local_path = self.cache.save_download(
                item.item_id, item.name, self._download_function(item.item_id), item.sha1, box_path
            )
            if item.name.lower().endswith(".oif"):
                self._download_oif_companion_folder(item, parent_folder_id)
        else:
            logger.info("Using cached copy of %s", box_path)

        return FetchedFile(
            local_path=local_path,
            original_file_name=item.name,
            source_description=f"box: {box_path} (file id {item.item_id}, sha1 {(item.sha1 or '?')[:12]})",
        )

    def _download_oif_companion_folder(self, oif_item: BoxItem, parent_folder_id: str) -> None:
        """Download "<name>.oif.files/" (all files, recursively) next to the cached .oif."""
        companion = self._find_child(parent_folder_id, oif_item.name + ".files", "folder")
        if companion is None:
            logger.warning("Companion folder %s.files not found on Box.", oif_item.name)
            return

        def download_folder(folder_id: str, relative_prefix: str) -> None:
            for child in self.list_folder(folder_id):
                child_relative = f"{relative_prefix}/{child.name}"
                if child.item_type == "folder":
                    download_folder(child.item_id, child_relative)
                elif child.item_type == "file":
                    self.cache.save_companion_file(
                        oif_item.item_id, child_relative, self._download_function(child.item_id)
                    )

        download_folder(companion.item_id, companion.name)

    def list_files(
        self, root: str, folder: str | Path = "", pattern: str = "*", recursive: bool = False
    ) -> list[str]:
        """Relative paths of matching files on Box, sorted alphabetically."""
        start_folder = normalize_relative_path(folder)
        found: list[str] = []

        def walk(folder_id: str, prefix: str) -> None:
            for item in self.list_folder(folder_id):
                item_path = f"{prefix}/{item.name}" if prefix else item.name
                if item.item_type == "file" and matches_pattern(item.name, pattern):
                    found.append(item_path)
                elif item.item_type == "folder" and recursive:
                    walk(item.item_id, item_path)

        walk(self._resolve_folder(root, start_folder), start_folder)
        return sorted(found)

    def list_subfolders(self, root: str, folder: str | Path = "") -> list[str]:
        """Names of the sub-folders of a folder (e.g. all experiment dates)."""
        folder_id = self._resolve_folder(root, normalize_relative_path(folder))
        return sorted(item.name for item in self.list_folder(folder_id) if item.item_type == "folder")

    def __repr__(self) -> str:
        return f"BoxCloudDataSource(roots={self.root_folder_ids}, cache={self.cache})"
