"""
axon_abf_file_reader.py
=======================

Reads Axon Binary Format files (.abf, written by pClamp/Clampex) into an
``EphysRecording``.

What is read
------------
* signals of every channel for every sweep (lazily — only when ``.data`` is used)
* sampling rate, channel names and units, recording date/time
* start time of every sweep (seconds from the start of the file)
* the stimulation protocol (epochs and digital TTL pulses), decoded by
  ``axon_abf_protocol_decoder.py``

The reading is done with the ``neo`` library (``neo.io.AxonIO``), which
handles both ABF1 and ABF2 files and converts the stored 16-bit integers into
physical units (pA, mV, V ...) using the gains stored in the file.

Example
-------
>>> from ephys_imaging_data_loader.file_readers.axon_abf_file_reader import read_axon_abf_file
>>> recording = read_axon_abf_file("sample_data/ephys_data/20260322/c1/26322010.abf")
>>> print(recording.summary())
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from ..recording_data_containers.ephys_recording import EphysRecording
from .axon_abf_protocol_decoder import _clean_text, decode_protocol, describe_acquisition_mode

logger = logging.getLogger(__name__)


def _open_axon_reader(file_path: Path):
    """
    Open an .abf file with neo and parse its header.

    neo prints some harmless warnings for older ABF files; they are silenced
    here so they do not flood notebooks.
    """
    # Imported here so that importing this package does not require neo
    # unless ephys files are actually read.
    from neo.io import AxonIO

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = AxonIO(filename=str(file_path))
    return reader


def _read_all_samples(file_path: Path) -> np.ndarray:
    """
    Read every sample of an .abf file into an array (sweeps, channels, samples).

    This is the function stored in ``EphysRecording._data_loader``; it is only
    called when ``recording.data`` is used for the first time.

    Implementation note: neo's low-level "raw" interface is used
    (``get_analogsignal_chunk`` + ``rescale_signal_raw_to_float``) instead of
    ``read_block()``. It skips creating neo/quantities objects for every sweep
    and is therefore faster and uses less memory.
    """
    reader = _open_axon_reader(file_path)
    n_sweeps = reader.segment_count(block_index=0)

    sweeps = []
    for sweep_index in range(n_sweeps):
        # Raw integers as stored in the file: shape (n_samples, n_channels)
        raw_samples = reader.get_analogsignal_chunk(
            block_index=0, seg_index=sweep_index, stream_index=0
        )
        # Convert to physical units (pA, mV, ...) with the gains from the header.
        physical_samples = reader.rescale_signal_raw_to_float(
            raw_samples, dtype="float64", stream_index=0
        )
        sweeps.append(physical_samples.T)  # -> (n_channels, n_samples)

    return np.stack(sweeps)  # -> (n_sweeps, n_channels, n_samples)


def read_axon_abf_file(
    file_path: str | Path,
    source_description: str | None = None,
    original_file_name: str | None = None,
    load_data_now: bool = False,
) -> EphysRecording:
    """
    Read an Axon .abf file into an ``EphysRecording``.

    Parameters
    ----------
    file_path : str or Path
        Path of the .abf file on local disk (for Box data this is the cached copy).
    source_description : str, optional
        Where the file came from, stored in the recording for provenance.
        Defaults to "local: <file_path>".
    original_file_name : str, optional
        The real file name if ``file_path`` is a renamed copy.
    load_data_now : bool
        If True, read all samples immediately instead of on first use.

    Returns
    -------
    EphysRecording
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"ABF file not found: {file_path}")

    reader = _open_axon_reader(file_path)
    header = reader.header

    # ---- Channel layout ------------------------------------------------------
    # All channels of an .abf file share one "stream" (same sampling clock).
    streams = header["signal_streams"]
    if len(streams) > 1:
        logger.warning(
            "%s has %d signal streams; only the first one is read.",
            file_path.name,
            len(streams),
        )
    first_stream_id = streams[0]["id"]
    stream_channels = header["signal_channels"][
        header["signal_channels"]["stream_id"] == first_stream_id
    ]
    channel_names = [str(name) for name in stream_channels["name"]]
    channel_units = [str(unit) for unit in stream_channels["units"]]

    # ---- Sweep layout -----------------------------------------------------------
    n_sweeps = reader.segment_count(block_index=0)
    sampling_rate_hz = float(reader.get_signal_sampling_rate(stream_index=0))
    n_samples_per_sweep = int(
        reader.get_signal_size(block_index=0, seg_index=0, stream_index=0)
    )
    # When did each sweep start, relative to the start of the file?
    sweep_start_times_s = np.array(
        [reader.segment_t_start(block_index=0, seg_index=s) for s in range(n_sweeps)],
        dtype=float,
    )

    # ---- Header information -----------------------------------------------------
    axon_info = reader._axon_info  # neo's parsed ABF header (a plain dictionary)
    recording_datetime = reader.raw_annotations["blocks"][0].get("rec_datetime")

    # Protocol path is stored with Windows backslashes, e.g.
    # "D:\DATA\...\protocols\example_protocol.pro" -> keep only the file name.
    protocol_path = _clean_text(axon_info.get("sProtocolPath", ""))
    protocol_name = protocol_path.replace("\\", "/").split("/")[-1] if protocol_path else ""

    protocol_epochs, stimulus_pulses = decode_protocol(
        axon_info,
        n_sweeps=n_sweeps,
        n_samples_per_sweep=n_samples_per_sweep,
        sampling_rate_hz=sampling_rate_hz,
    )

    # A few extra header values that are often useful when checking data.
    header_metadata = {
        "abf_file_version": axon_info.get("fFileVersionNumber"),
        "protocol_path": protocol_path,
        "adc_channel_names_raw": [
            _clean_text(adc.get("ADCChNames", "")) for adc in axon_info.get("listADCInfo", [])
        ],
        "telegraph_additional_gain": [
            adc.get("fTelegraphAdditGain") for adc in axon_info.get("listADCInfo", [])
        ],
        "sweep_start_to_start_s": axon_info.get("protocol", {}).get("fEpisodeStartToStart"),
    }

    recording = EphysRecording(
        file_name=original_file_name or file_path.name,
        source_description=source_description or f"local: {file_path}",
        local_file_path=file_path,
        recording_datetime=recording_datetime,
        sampling_rate_hz=sampling_rate_hz,
        channel_names=channel_names,
        channel_units=channel_units,
        n_sweeps=n_sweeps,
        n_samples_per_sweep=n_samples_per_sweep,
        sweep_start_times_s=sweep_start_times_s,
        acquisition_mode=describe_acquisition_mode(axon_info),
        protocol_name=protocol_name,
        protocol_epochs=protocol_epochs,
        stimulus_pulses=stimulus_pulses,
        header_metadata=header_metadata,
        # The samples themselves are read later, only when needed:
        _data_loader=lambda: _read_all_samples(file_path),
    )

    if load_data_now:
        _ = recording.data  # triggers the lazy loader
    return recording
