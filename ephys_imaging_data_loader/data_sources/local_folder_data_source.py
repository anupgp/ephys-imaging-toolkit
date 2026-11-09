"""
local_folder_data_source.py
===========================

Data source for files that are already on this computer (internal disk,
external drive, network share, or a folder synced from Box).

Example
-------
>>> source = LocalFolderDataSource({"ephys_olympus2p": "/data/ephys", "imaging_olympus2p": "/data/img"})
>>> source.list_files("ephys_olympus2p", "20260322/c1", "*.abf")
['20260322/c1/26322000.abf', '20260322/c1/26322001.abf', ...]
>>> fetched = source.fetch_file("ephys_olympus2p", "20260322/c1/26322010.abf")
>>> fetched.local_path
PosixPath('/data/ephys/20260322/c1/26322010.abf')
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base_data_source import DataSource, FetchedFile, matches_pattern, normalize_relative_path

logger = logging.getLogger(__name__)


class LocalFolderDataSource(DataSource):
    """Reads files from local folders, one folder per named root."""

    kind = "local"

    def __init__(self, root_folders: dict[str, str | Path]):
        """
        Parameters
        ----------
        root_folders : dict
            Root name -> local folder, e.g. {"ephys_olympus2p": "/data/ephys"}.
            Roots whose folder is empty/None are ignored.
        """
        self.root_folders: dict[str, Path] = {
            name: Path(folder).expanduser() for name, folder in root_folders.items() if folder
        }

    def root_names(self) -> list[str]:
        return list(self.root_folders)

    def root_folder(self, root: str) -> Path:
        """The local folder behind a root name."""
        self._check_root(root)
        return self.root_folders[root]

    def _full_path(self, root: str, relative_path: str | Path) -> Path:
        """
        Absolute path of a file inside a root.

        If the exact path does not exist, each part is matched ignoring
        upper/lower case (folders are named "c1" on some days and "C1" on
        others; the Box source does the same). Only an unambiguous match is used.
        """
        exact = self.root_folder(root) / normalize_relative_path(relative_path)
        if exact.exists():
            return exact

        current = self.root_folder(root)
        for part in [p for p in normalize_relative_path(relative_path).split("/") if p]:
            candidate = current / part
            if not candidate.exists() and current.is_dir():
                matches = [child for child in current.iterdir() if child.name.lower() == part.lower()]
                if len(matches) == 1:
                    candidate = matches[0]
            current = candidate
        return current

    def file_exists(self, root: str, relative_path: str | Path) -> bool:
        if root not in self.root_folders:
            return False
        return self._full_path(root, relative_path).is_file()

    def fetch_file(self, root: str, relative_path: str | Path) -> FetchedFile:
        """Nothing to download: just check the file is there and return its path."""
        full_path = self._full_path(root, relative_path)
        if not full_path.is_file():
            raise FileNotFoundError(
                f"'{normalize_relative_path(relative_path)}' not found in local root "
                f"'{root}' ({self.root_folders[root]})."
            )
        # Olympus .oif files need their companion folder "<name>.oif.files".
        if full_path.suffix.lower() == ".oif":
            companion = full_path.with_name(full_path.name + ".files")
            if not companion.is_dir():
                logger.warning("Companion folder %s is missing; the .oif cannot be read.", companion)
        return FetchedFile(
            local_path=full_path,
            original_file_name=full_path.name,
            source_description=f"local: {full_path}",
        )

    def list_files(
        self, root: str, folder: str | Path = "", pattern: str = "*", recursive: bool = False
    ) -> list[str]:
        """Relative paths of matching files, sorted alphabetically."""
        root_folder = self.root_folder(root)
        search_folder = self._full_path(root, folder)  # case-insensitive fallback included
        if not search_folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {search_folder}")
        candidates = search_folder.rglob("*") if recursive else search_folder.iterdir()
        return sorted(
            path.relative_to(root_folder).as_posix()
            for path in candidates
            if path.is_file() and matches_pattern(path.name, pattern)
        )

    def __repr__(self) -> str:
        return f"LocalFolderDataSource(roots={ {k: str(v) for k, v in self.root_folders.items()} })"
