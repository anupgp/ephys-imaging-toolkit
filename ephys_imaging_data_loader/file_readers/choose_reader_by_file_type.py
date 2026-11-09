"""
choose_reader_by_file_type.py
=============================

Picks the right file reader from the file extension, so the rest of the code
can simply say "read this file" without knowing its format.

To support a new format later (e.g. TIFF exports), write a reader function
that returns an ``EphysRecording`` or ``ImageRecording`` and add its extension
to ``READER_FOR_EXTENSION`` below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..recording_data_containers import EphysRecording, ImageRecording
from .axon_abf_file_reader import read_axon_abf_file
from .olympus_oib_oif_file_reader import read_olympus_oib_oif_file
from .zeiss_czi_file_reader import read_zeiss_czi_file

# File extension (lower case) -> function that reads that format.
READER_FOR_EXTENSION: dict[str, Callable[..., EphysRecording | ImageRecording]] = {
    ".abf": read_axon_abf_file,          # Axon pClamp patch-clamp recordings
    ".czi": read_zeiss_czi_file,         # Zeiss ZEN (LSM880) images / line scans
    ".oib": read_olympus_oib_oif_file,   # Olympus FluoView, single-file format
    ".oif": read_olympus_oib_oif_file,   # Olympus FluoView, file + ".oif.files" folder
}

SUPPORTED_EXTENSIONS = tuple(READER_FOR_EXTENSION)


def is_supported_file(file_name: str | Path) -> bool:
    """True if a reader exists for this file's extension."""
    return Path(file_name).suffix.lower() in READER_FOR_EXTENSION


def read_recording_file(
    file_path: str | Path,
    source_description: str | None = None,
    original_file_name: str | None = None,
    load_data_now: bool = False,
) -> EphysRecording | ImageRecording:
    """
    Read any supported file from local disk, choosing the reader automatically.

    The extension of ``original_file_name`` is used if given (useful when the
    local copy has a different name), otherwise that of ``file_path``.
    """
    name_for_type = Path(original_file_name or file_path)
    extension = name_for_type.suffix.lower()
    reader = READER_FOR_EXTENSION.get(extension)
    if reader is None:
        raise ValueError(
            f"No reader for '{name_for_type.name}'. Supported file types: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
    return reader(
        file_path,
        source_description=source_description,
        original_file_name=original_file_name,
        load_data_now=load_data_now,
    )
