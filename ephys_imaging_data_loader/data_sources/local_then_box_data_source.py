"""
local_then_box_data_source.py
=============================

The "auto" data source: use the LOCAL copy of a file when it exists, and
fetch it from BOX only when it does not.

This is convenient when part of the data has been copied to the computer
(fast, works offline) and the rest is only on Box. Your code does not need
to know which is which.

Box is contacted only when a file is missing locally, so working entirely
from local files never triggers a Box login.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base_data_source import DataSource, FetchedFile
from .box_cloud_data_source import BoxCloudDataSource
from .local_folder_data_source import LocalFolderDataSource

logger = logging.getLogger(__name__)


class LocalThenBoxDataSource(DataSource):
    """Local first, Box as fallback; see module docstring."""

    kind = "auto"

    def __init__(self, local_source: LocalFolderDataSource, box_source: BoxCloudDataSource | None):
        self.local_source = local_source
        self.box_source = box_source  # may be None when Box is not configured

    def root_names(self) -> list[str]:
        names = list(self.local_source.root_names())
        if self.box_source is not None:
            names += [name for name in self.box_source.root_names() if name not in names]
        return names

    def _box_available_for(self, root: str) -> bool:
        return self.box_source is not None and root in self.box_source.root_names()

    def file_exists(self, root: str, relative_path: str | Path) -> bool:
        if self.local_source.file_exists(root, relative_path):
            return True
        return self._box_available_for(root) and self.box_source.file_exists(root, relative_path)

    def fetch_file(self, root: str, relative_path: str | Path) -> FetchedFile:
        self._check_root(root)
        # 1) Local copy? Use it (fast, offline).
        if self.local_source.file_exists(root, relative_path):
            return self.local_source.fetch_file(root, relative_path)
        # 2) Otherwise Box, if this root is configured there.
        if self._box_available_for(root):
            logger.info("'%s' not found locally; fetching from Box.", relative_path)
            return self.box_source.fetch_file(root, relative_path)
        raise FileNotFoundError(
            f"'{relative_path}' not found locally in root '{root}', and no Box "
            "folder is configured for this root."
        )

    def list_files(
        self, root: str, folder: str | Path = "", pattern: str = "*", recursive: bool = False
    ) -> list[str]:
        """
        Files from both places, merged (each relative path listed once).

        If the folder exists only locally or only on Box, that side is used.
        """
        self._check_root(root)
        found: set[str] = set()
        local_error: Exception | None = None
        if root in self.local_source.root_names():
            try:
                found.update(self.local_source.list_files(root, folder, pattern, recursive))
            except FileNotFoundError as error:
                local_error = error
        if self._box_available_for(root):
            found.update(self.box_source.list_files(root, folder, pattern, recursive))
        elif local_error is not None:
            raise local_error
        return sorted(found)

    def __repr__(self) -> str:
        return f"LocalThenBoxDataSource(local={self.local_source}, box={self.box_source})"
