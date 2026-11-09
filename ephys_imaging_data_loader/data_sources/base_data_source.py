"""
base_data_source.py
===================

The common interface shared by every place data can come from.

A *data source* answers two questions:

1. "Which files are in this folder?"          -> ``list_files(root, folder, pattern)``
2. "Give me this file as a path on my disk."  -> ``fetch_file(root, relative_path)``

Files are addressed by a **root name** plus a **relative path**, for example::

    root          = "ephys_olympus2p"
    relative_path = "20260322/c1/26322010.abf"

The root name is defined in the settings file and maps to a local folder
and/or a Box folder. Because code only ever uses (root, relative path), the
same analysis script works unchanged whether the data are on this computer
or in the cloud — only the chosen data source differs.

Implementations
---------------
- ``LocalFolderDataSource``    files already on this computer
- ``BoxCloudDataSource``       files on Box, downloaded once into a cache
- ``LocalThenBoxDataSource``   "auto": local copy if present, else Box
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class FetchedFile:
    """
    A file that is now available on local disk.

    Attributes
    ----------
    local_path : Path
        Where the file can be read (the original, or the cached Box download).
    original_file_name : str
        The file's real name (e.g. "26322010.abf").
    source_description : str
        Human-readable provenance, stored inside the loaded recording, e.g.
        "local: /data/.../26322010.abf" or "box: ephys_olympus2p/... (file id ...)".
    """

    local_path: Path
    original_file_name: str
    source_description: str


def normalize_relative_path(relative_path: str | Path) -> str:
    """
    Clean up a relative path so all sources understand it the same way.

    - backslashes (Windows) become forward slashes
    - leading/trailing slashes are removed ("/20260114/c1/x.abf" is accepted,
      as used in the old notebooks)
    - ".." is refused, so a path can never point outside its root folder
    """
    text = str(relative_path).replace("\\", "/").strip().strip("/")
    parts = [part for part in PurePosixPath(text).parts if part not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"'..' is not allowed in data paths: {relative_path}")
    return "/".join(parts)


def matches_pattern(file_name: str, pattern: str) -> bool:
    """
    Case-insensitive wildcard match, e.g. "*.abf" matches "26322010.ABF".

    Several patterns can be given separated by ";", e.g. "*.oib;*.oif".
    """
    name = file_name.lower()
    return any(fnmatch.fnmatch(name, single.strip().lower()) for single in pattern.split(";"))


class DataSource(ABC):
    """Abstract base class; see module docstring."""

    #: short label shown in messages: "local", "box" or "auto"
    kind: str = "abstract"

    @abstractmethod
    def root_names(self) -> list[str]:
        """Names of the roots this source knows about."""

    @abstractmethod
    def fetch_file(self, root: str, relative_path: str | Path) -> FetchedFile:
        """Make one file available on local disk and say where it is."""

    @abstractmethod
    def list_files(
        self, root: str, folder: str | Path = "", pattern: str = "*", recursive: bool = False
    ) -> list[str]:
        """
        Relative paths (from the root) of files in ``folder`` matching ``pattern``.

        ``recursive=True`` also searches all sub-folders.
        """

    @abstractmethod
    def file_exists(self, root: str, relative_path: str | Path) -> bool:
        """True if the file exists in this source (does not download it)."""

    def _check_root(self, root: str) -> None:
        """Raise a helpful error for unknown root names."""
        if root not in self.root_names():
            raise KeyError(
                f"Unknown data root '{root}' for the {self.kind} data source. "
                f"Known roots: {self.root_names()} (defined in data_source_settings.toml)"
            )
