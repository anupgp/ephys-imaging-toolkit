"""
box_download_cache.py
=====================

Keeps files downloaded from Box on local disk, so each file is downloaded
only ONCE (and again only if it changed on Box).

Why a cache instead of temporary files?
---------------------------------------
The old ``from_box`` methods wrote every download to a new temporary file
(``delete=False``) that was never removed, so /tmp slowly filled with copies
of the same .oib/.abf files, and every re-run downloaded everything again.

Layout on disk
--------------
::

    <cache_folder>/box_files/<box file id>/<original file name>
    <cache_folder>/box_files/<box file id>/box_file_info.json   <- bookkeeping

Using the Box *file id* as folder name means two files with the same name in
different Box folders never collide, and keeping the original file name means
readers see the correct extension.

How "is my copy up to date?" is decided
---------------------------------------
Box reports a SHA-1 checksum for every file. After downloading, the checksum
of the local copy is computed and must match; it is saved in
``box_file_info.json``. Next time, if Box still reports the same SHA-1, the
cached copy is used without downloading.

Size limit
----------
When the cache grows beyond ``size_limit_bytes``, the least recently used
files are deleted until it fits again (the file just requested is never
deleted). Only folders inside ``<cache_folder>/box_files`` are ever removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import BinaryIO, Callable

logger = logging.getLogger(__name__)

INFO_FILE_NAME = "box_file_info.json"
# Read/write in 8 MB blocks: large enough to be fast, small enough for low memory use.
COPY_BLOCK_SIZE = 8 * 1024 * 1024


def sha1_of_file(path: Path) -> str:
    """SHA-1 checksum of a file (hex string), computed block by block."""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while block := handle.read(COPY_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


class BoxDownloadCache:
    """Local on-disk cache of Box files; see module docstring."""

    def __init__(self, cache_folder: str | Path, size_limit_bytes: int | None = None):
        """
        Parameters
        ----------
        cache_folder : folder for the cache (created if needed),
            e.g. "~/.cache/ephys_imaging_data_loader"
        size_limit_bytes : maximum total size; None means unlimited.
        """
        self.cache_folder = Path(cache_folder).expanduser()
        self.files_folder = self.cache_folder / "box_files"
        self.files_folder.mkdir(parents=True, exist_ok=True)
        self.size_limit_bytes = size_limit_bytes

    # ---------------------------------------------------------------------
    # Paths and bookkeeping
    # ---------------------------------------------------------------------
    def entry_folder(self, file_id: str) -> Path:
        """Folder that holds one cached Box file (and its info file)."""
        # Box ids are digits; refuse anything else so a bad id cannot escape the cache.
        if not str(file_id).isalnum():
            raise ValueError(f"Unexpected Box file id: {file_id!r}")
        return self.files_folder / str(file_id)

    def _read_info(self, file_id: str) -> dict | None:
        info_file = self.entry_folder(file_id) / INFO_FILE_NAME
        if not info_file.is_file():
            return None
        try:
            return json.loads(info_file.read_text())
        except json.JSONDecodeError:
            return None

    def _write_info(self, file_id: str, info: dict) -> None:
        info_file = self.entry_folder(file_id) / INFO_FILE_NAME
        temporary = info_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(info, indent=2))
        temporary.replace(info_file)

    # ---------------------------------------------------------------------
    # Main operations
    # ---------------------------------------------------------------------
    def lookup(self, file_id: str, expected_sha1: str | None) -> Path | None:
        """
        Return the cached copy if it exists and matches ``expected_sha1``.

        If Box did not report a SHA-1 (``expected_sha1`` is None), any complete
        cached copy is accepted.
        """
        info = self._read_info(file_id)
        if not info or not info.get("complete"):
            return None
        cached_path = self.entry_folder(file_id) / info["file_name"]
        if not cached_path.is_file():
            return None
        if expected_sha1 and info.get("sha1") != expected_sha1:
            logger.info("Box file %s changed since it was cached; downloading again.", info["file_name"])
            return None
        # Remember that it was used now (for least-recently-used cleanup).
        info["last_used"] = time.time()
        self._write_info(file_id, info)
        return cached_path

    def save_download(
        self,
        file_id: str,
        file_name: str,
        download_into: Callable[[BinaryIO], None],
        expected_sha1: str | None,
        box_path: str = "",
    ) -> Path:
        """
        Download a file into the cache and verify it.

        Parameters
        ----------
        file_id, file_name : Box id and real name of the file
        download_into : function that writes the file's bytes into an open
            binary file handle (provided by BoxCloudDataSource)
        expected_sha1 : checksum reported by Box; the download is rejected if
            the local copy does not match
        box_path : "root/relative/path" text, stored for your information
        """
        entry = self.entry_folder(file_id)
        # Start from a clean entry folder, so no stale companion files remain.
        if entry.exists():
            shutil.rmtree(entry)
        entry.mkdir(parents=True)

        final_path = entry / file_name
        partial_path = entry / (file_name + ".part")  # renamed only when complete
        with open(partial_path, "wb") as handle:
            download_into(handle)

        actual_sha1 = sha1_of_file(partial_path)
        if expected_sha1 and actual_sha1 != expected_sha1:
            partial_path.unlink(missing_ok=True)
            raise IOError(
                f"Download of {file_name} is corrupt (SHA-1 {actual_sha1} != {expected_sha1} "
                "reported by Box). Please try again."
            )
        partial_path.replace(final_path)

        self._write_info(
            file_id,
            {
                "file_name": file_name,
                "box_path": box_path,
                "sha1": actual_sha1,
                "size_bytes": final_path.stat().st_size,
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_used": time.time(),
                "complete": True,
            },
        )
        self.enforce_size_limit(keep_file_id=file_id)
        return final_path

    def save_companion_file(
        self, file_id: str, relative_name: str, download_into: Callable[[BinaryIO], None]
    ) -> Path:
        """
        Store an extra file next to a cached file (used for the
        "<name>.oif.files/..." folder that Olympus .oif files need).
        """
        target = self.entry_folder(file_id) / relative_name
        if self.entry_folder(file_id).resolve() not in target.resolve().parents:
            raise ValueError(f"Companion path escapes the cache entry: {relative_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            download_into(handle)
        return target

    # ---------------------------------------------------------------------
    # Housekeeping
    # ---------------------------------------------------------------------
    @staticmethod
    def _folder_size(folder: Path) -> int:
        return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())

    def total_size_bytes(self) -> int:
        """Current size of all cached files."""
        return self._folder_size(self.files_folder)

    def enforce_size_limit(self, keep_file_id: str | None = None) -> list[str]:
        """
        Delete least-recently-used entries until the cache fits its size limit.

        Returns the names of removed files (for logging / tests).
        """
        if self.size_limit_bytes is None:
            return []

        entries = []
        for entry in self.files_folder.iterdir():
            if not entry.is_dir():
                continue
            info = self._read_info(entry.name) or {}
            entries.append((info.get("last_used", 0.0), entry, info.get("file_name", entry.name)))

        total = sum(self._folder_size(entry) for _, entry, _ in entries)
        removed = []
        for _, entry, file_name in sorted(entries):  # oldest "last_used" first
            if total <= self.size_limit_bytes:
                break
            if entry.name == str(keep_file_id):
                continue
            size = self._folder_size(entry)
            shutil.rmtree(entry)
            total -= size
            removed.append(file_name)
            logger.info("Cache full: removed %s (%.1f MB)", file_name, size / 1e6)
        return removed

    def clear(self) -> None:
        """Delete every cached file (they will be downloaded again when needed)."""
        shutil.rmtree(self.files_folder, ignore_errors=True)
        self.files_folder.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        limit = "unlimited" if self.size_limit_bytes is None else f"{self.size_limit_bytes / 1e9:.1f} GB"
        return f"BoxDownloadCache('{self.cache_folder}', limit={limit})"
