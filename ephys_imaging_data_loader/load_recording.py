"""
load_recording.py
=================

The one function most users need: ``load_recording(...)``.

It combines the two layers of the package:

    data source  (where is the file?  local folder / Box / auto)
         |
         v   a file on local disk
    file reader  (what format?  .abf / .czi / .oib / .oif)
         |
         v
    EphysRecording or ImageRecording

Three ways to call it
---------------------
1. A plain local file path (no settings needed)::

       rec = load_recording("/path/to/26322010.abf")

2. Root name + relative path, choosing LOCAL or CLOUD with ``mode``::

       rec = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="local")
       rec = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="box")
       rec = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="auto")
       rec = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p")  # default_mode

3. With a data source you created yourself (e.g. to reuse one Box login)::

       box = open_data_source("box")
       rec = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", source=box)
"""

from __future__ import annotations

from pathlib import Path

from .data_sources.base_data_source import DataSource
from .data_sources.data_source_settings import find_settings_file, open_data_source
from .file_readers.choose_reader_by_file_type import read_recording_file
from .recording_data_containers import EphysRecording, ImageRecording

# Data sources opened by mode, remembered for the whole Python session so that
# repeated calls reuse the same Box login, folder listings and cache.
_opened_data_sources: dict[tuple[str | None, str | None], DataSource] = {}


def get_data_source(mode: str | None = None, settings_file: str | Path | None = None) -> DataSource:
    """
    The data source for ``mode`` ("local" / "box" / "auto" / None = default),
    created once per session and then reused.
    """
    settings_path = find_settings_file(settings_file)
    key = (mode, str(settings_path) if settings_path else None)
    if key not in _opened_data_sources:
        _opened_data_sources[key] = open_data_source(mode, settings_path)
    return _opened_data_sources[key]


def load_recording(
    path: str | Path,
    root: str | None = None,
    mode: str | None = None,
    source: DataSource | None = None,
    settings_file: str | Path | None = None,
    load_data_now: bool = False,
) -> EphysRecording | ImageRecording:
    """
    Load one recording (ephys or imaging) from a local file, a local root
    folder, or Box.

    Parameters
    ----------
    path : str or Path
        Either a full local file path (when ``root`` is not given), or a path
        relative to the root, e.g. "20260322/c1/26322010.abf".
    root : str, optional
        Name of a data root from the settings file, e.g. "ephys_olympus2p".
    mode : "local", "box" or "auto", optional
        Where to load from (only used with ``root``). None uses the
        settings file's ``default_mode``.
    source : DataSource, optional
        Use this data source instead of building one from ``mode``.
    settings_file : optional path to a settings file.
    load_data_now : bool
        Read samples/pixels immediately (default: on first use of ``.data``).

    Returns
    -------
    EphysRecording for .abf files, ImageRecording for .czi / .oib / .oif files.
    """
    # --- Case 1: a plain local file -------------------------------------------------
    # (root="" is treated like no root: the path is a full local path)
    if not root and source is None:
        if mode not in (None, "local"):
            raise ValueError("To load from Box, give a root name, e.g. root='ephys_olympus2p'.")
        return read_recording_file(Path(path).expanduser(), load_data_now=load_data_now)

    # --- Case 2/3: root + relative path through a data source ---------------------
    if root is None:
        raise ValueError("A root name is required when a data source is used.")
    data_source = source or get_data_source(mode, settings_file)
    fetched = data_source.fetch_file(root, path)
    return read_recording_file(
        fetched.local_path,
        source_description=fetched.source_description,
        original_file_name=fetched.original_file_name,
        load_data_now=load_data_now,
    )


def list_recording_files(
    root: str,
    folder: str | Path = "",
    pattern: str = "*.abf;*.czi;*.oib;*.oif",
    recursive: bool = False,
    mode: str | None = None,
    source: DataSource | None = None,
    settings_file: str | Path | None = None,
) -> list[str]:
    """
    List recording files (relative paths) in a folder of a data root.

    Example
    -------
    >>> list_recording_files("ephys_olympus2p", "20260322/c1", "*.abf", mode="local")
    ['20260322/c1/26322000.abf', '20260322/c1/26322001.abf', ...]
    """
    data_source = source or get_data_source(mode, settings_file)
    return data_source.list_files(root, folder, pattern, recursive)
