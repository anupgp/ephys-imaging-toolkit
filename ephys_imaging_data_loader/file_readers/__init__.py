"""
file_readers
============

One reader per file format. Every reader takes a LOCAL file path and returns
one of the standard containers (``EphysRecording`` / ``ImageRecording``).

Readers never download anything; getting the file onto the local disk is the
job of the ``data_sources`` package. This separation means the same reader is
used for local files and for files fetched from Box.

- ``axon_abf_file_reader``          .abf  (Axon pClamp)
- ``axon_abf_protocol_decoder``     helper: epochs + digital stimulus pulses
- ``olympus_oib_oif_file_reader``   .oib / .oif (Olympus FluoView)
- ``zeiss_czi_file_reader``         .czi (Zeiss ZEN)
- ``choose_reader_by_file_type``    picks one of the above by extension
"""

from .axon_abf_file_reader import read_axon_abf_file
from .choose_reader_by_file_type import (
    SUPPORTED_EXTENSIONS,
    is_supported_file,
    read_recording_file,
)
from .olympus_oib_oif_file_reader import read_olympus_oib_oif_file
from .zeiss_czi_file_reader import read_zeiss_czi_file

__all__ = [
    "read_axon_abf_file",
    "read_olympus_oib_oif_file",
    "read_zeiss_czi_file",
    "read_recording_file",
    "is_supported_file",
    "SUPPORTED_EXTENSIONS",
]
