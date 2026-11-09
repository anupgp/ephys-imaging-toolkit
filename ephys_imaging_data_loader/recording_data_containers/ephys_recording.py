"""
ephys_recording.py
==================

Defines ``EphysRecording``: ONE standard container for a patch-clamp recording,
no matter which file format or storage location (local disk / Box) it came from.

Why a separate container?
-------------------------
The older code (``ephys_class.py``, ``ephys_class_new.py``) mixed four jobs in
one class: downloading, file parsing, analysis and plotting. Here, a *file
reader* (see ``file_readers/axon_abf_file_reader.py``) fills in an
``EphysRecording``, and every analysis or plotting function only needs to know
this one container. If a new file format is added later, only a new reader is
needed — analysis code keeps working unchanged.

Array layout
------------
``recording.data`` is a NumPy array with shape::

    (n_sweeps, n_channels, n_samples_per_sweep)

- sweep   = one episode / trial of the protocol (Clampex "sweep")
- channel = one recorded input (e.g. "ImLEFT" current, "IN7" photodiode ...)
- sample  = one time point; time of sample i is ``i / sampling_rate_hz`` seconds

Units are always given explicitly (``channel_units``) and variable names carry
units where it matters (``_hz``, ``_s``, ``_ms``).

Lazy loading
------------
Reading the header of an .abf file is fast, but reading all samples can take a
moment for large files. The reader therefore gives the container a small
function (``_data_loader``) that reads the samples; it is only called the first
time ``recording.data`` is accessed, and the result is kept in memory.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class EphysRecording:
    """
    A patch-clamp recording: signals of all channels for all sweeps, plus the
    information needed to interpret them (sampling rate, units, protocol).

    Create it with ``load_recording(...)`` (recommended) or with a file reader;
    you normally do not construct it by hand.

    Attributes
    ----------
    file_name : str
        Original file name, e.g. "26322010.abf" (never a temporary name).
    source_description : str
        Where the file came from, e.g. "local: /path/to/file.abf" or
        "box: ephys_olympus2p/20260322/c1/26322010.abf (file id 123...)".
        Stored so every figure / table can be traced back to its raw file.
    local_file_path : Path
        The file on disk that was actually read (for Box data: the cached copy).
    recording_datetime : datetime.datetime | None
        Date and time when the recording started (from the file header).
    sampling_rate_hz : float
        Samples per second, per channel (e.g. 20000.0).
    channel_names : list[str]
        Names of the recorded channels, in the same order as axis 1 of ``data``.
    channel_units : list[str]
        Physical unit of each channel (e.g. "pA", "mV", "V").
    n_sweeps : int
        Number of sweeps (episodes) in the file.
    n_samples_per_sweep : int
        Number of samples in each sweep.
    sweep_start_times_s : np.ndarray
        Start time of each sweep, in seconds from the start of the file.
        Useful to align sweeps with imaging data that was triggered per sweep.
    acquisition_mode : str
        Human-readable acquisition mode, e.g. "episodic stimulation" or "gap-free".
    protocol_name : str
        Name of the Clampex protocol file used, e.g. "example_protocol.pro".
        Very handy for grouping recordings by experiment type.
    protocol_epochs : pandas.DataFrame
        The stimulation protocol, one row per (sweep, output, epoch), with
        start time, duration, level and digital output bits. See
        ``axon_abf_protocol_decoder.py`` for the column meanings.
    stimulus_pulses : pandas.DataFrame
        Every digital output pulse sent by the amplifier/digitizer during the
        recording, one row per pulse, with columns
        ``sweep, digital_output_bit, onset_s, duration_s``.
        These are the TTL pulses that typically trigger stimulators, uncaging
        lasers or the microscope.
    bad_sweeps : set[int]
        Sweep indices to exclude from analysis (you can add to this set).
    header_metadata : dict
        Extra header information from the file (file version, protocol path,
        telegraph gains ...). Kept for completeness and debugging.
    """

    # --- identity / provenance ------------------------------------------------
    file_name: str
    source_description: str
    local_file_path: Path

    # --- recording layout ----------------------------------------------------
    recording_datetime: datetime.datetime | None
    sampling_rate_hz: float
    channel_names: list[str]
    channel_units: list[str]
    n_sweeps: int
    n_samples_per_sweep: int
    sweep_start_times_s: np.ndarray

    # --- protocol -------------------------------------------------------------
    acquisition_mode: str = "unknown"
    protocol_name: str = ""
    protocol_epochs: pd.DataFrame = field(default_factory=pd.DataFrame)
    stimulus_pulses: pd.DataFrame = field(default_factory=pd.DataFrame)

    # --- bookkeeping ----------------------------------------------------------
    bad_sweeps: set[int] = field(default_factory=set)
    header_metadata: dict[str, Any] = field(default_factory=dict)

    # --- lazy data loading (filled in by the reader; not shown in repr) ------
    _data_loader: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _data: np.ndarray | None = field(default=None, repr=False)

    # =========================================================================
    # Access to the raw signals
    # =========================================================================
    @property
    def data(self) -> np.ndarray:
        """
        All samples as an array of shape (n_sweeps, n_channels, n_samples).

        The samples are read from the file the first time this property is
        used, then kept in memory, so later calls are instant.
        """
        if self._data is None:
            if self._data_loader is None:
                raise RuntimeError(
                    f"No data and no data loader available for {self.file_name}."
                )
            self._data = self._data_loader()
        return self._data

    @property
    def is_data_loaded(self) -> bool:
        """True once the samples have been read into memory."""
        return self._data is not None

    @property
    def time_s(self) -> np.ndarray:
        """
        Time axis of one sweep in seconds, starting at 0.

        Computed as sample_index / sampling_rate_hz (the old ``export_to_csv``
        multiplied by the sampling rate instead of dividing — fixed here).
        """
        return np.arange(self.n_samples_per_sweep) / self.sampling_rate_hz

    @property
    def sweep_duration_s(self) -> float:
        """Duration of one sweep in seconds."""
        return self.n_samples_per_sweep / self.sampling_rate_hz

    @property
    def good_sweeps(self) -> list[int]:
        """Indices of all sweeps that are NOT listed in ``bad_sweeps``."""
        return [s for s in range(self.n_sweeps) if s not in self.bad_sweeps]

    def channel_index(self, channel: str | int) -> int:
        """
        Convert a channel name (e.g. "ImLEFT") or index (e.g. 0) into an index.

        Name matching ignores upper/lower case and spaces, because the same
        amplifier channel is sometimes written "Im LEFT" and sometimes "ImLEFT".
        """
        if isinstance(channel, (int, np.integer)):
            if not 0 <= channel < len(self.channel_names):
                raise IndexError(
                    f"Channel index {channel} out of range; "
                    f"this file has channels {self.channel_names}."
                )
            return int(channel)

        wanted = str(channel).replace(" ", "").lower()
        for index, name in enumerate(self.channel_names):
            if name.replace(" ", "").lower() == wanted:
                return index
        raise KeyError(
            f"Channel '{channel}' not found in {self.file_name}; "
            f"available channels: {self.channel_names}"
        )

    def channel_unit(self, channel: str | int) -> str:
        """Physical unit (e.g. "pA") of a channel given by name or index."""
        return self.channel_units[self.channel_index(channel)]

    def get_trace(self, channel: str | int, sweep: int) -> np.ndarray:
        """
        One 1-D trace: all samples of one channel in one sweep.

        Example
        -------
        >>> current_pA = recording.get_trace("ImLEFT", sweep=0)
        >>> plt.plot(recording.time_s, current_pA)
        """
        return self.data[sweep, self.channel_index(channel), :]

    def get_traces(
        self, channel: str | int, sweeps: list[int] | None = None
    ) -> np.ndarray:
        """
        A 2-D array (n_selected_sweeps, n_samples) for one channel.

        By default only the good sweeps (not in ``bad_sweeps``) are returned.
        """
        selected_sweeps = self.good_sweeps if sweeps is None else list(sweeps)
        return self.data[selected_sweeps, self.channel_index(channel), :]

    def mean_trace(
        self, channel: str | int, sweeps: list[int] | None = None
    ) -> np.ndarray:
        """Average trace over sweeps (good sweeps by default)."""
        return self.get_traces(channel, sweeps).mean(axis=0)

    # =========================================================================
    # Stimulus information
    # =========================================================================
    def stimulus_onsets_s(
        self, sweep: int = 0, digital_output_bit: int | None = None
    ) -> np.ndarray:
        """
        Onset times (seconds from sweep start) of the digital stimulus pulses.

        Parameters
        ----------
        sweep : int
            Which sweep to look at (protocols can change from sweep to sweep,
            e.g. the paired-pulse interval grows each sweep).
        digital_output_bit : int or None
            Only pulses on this digital output (0 = first TTL output).
            None returns pulses from all digital outputs, sorted in time.
        """
        pulses = self.stimulus_pulses
        if pulses.empty:
            return np.array([])
        selected = pulses[pulses["sweep"] == sweep]
        if digital_output_bit is not None:
            selected = selected[selected["digital_output_bit"] == digital_output_bit]
        return np.sort(selected["onset_s"].to_numpy())

    def pulse_summary_text(self, sweep: int = 0) -> str:
        """
        One line describing the digital stimulus pulses of a sweep, per output:

            "DO0: 2 pulses, mean interval 800.0 ms | DO1: 2 pulses, mean interval 1000.1 ms"

        "DO1" = digital output 1. The mean interval is the average time between
        consecutive pulse onsets on that output (omitted for a single pulse).
        Each output is listed separately because different outputs usually
        drive different devices (e.g. uncaging laser vs. camera trigger), and
        mixing them would give a meaningless interval.
        """
        pulses = self.stimulus_pulses
        if pulses.empty:
            return "no digital stimulus pulses"
        pulses = pulses[pulses["sweep"] == sweep]
        if pulses.empty:
            return f"no digital stimulus pulses in sweep {sweep}"

        parts = []
        for bit, output_pulses in pulses.groupby("digital_output_bit"):
            onsets_s = np.sort(output_pulses["onset_s"].to_numpy())
            text = f"DO{int(bit)}: {len(onsets_s)} pulse{'s' if len(onsets_s) != 1 else ''}"
            if len(onsets_s) > 1:
                text += f", mean interval {np.diff(onsets_s).mean() * 1000:.1f} ms"
            parts.append(text)
        return " | ".join(parts)

    def find_threshold_crossings_s(
        self, channel: str | int, sweep: int, threshold: float
    ) -> np.ndarray:
        """
        Times (s) where a recorded channel rises through ``threshold``.

        Use this for rigs where the stimulus trigger is *recorded* on an input
        channel (e.g. "trigger1" on the Zeiss rig, in volts) rather than being
        generated by the Clampex protocol.
        """
        trace = self.get_trace(channel, sweep)
        above = trace >= threshold
        # A rising crossing is a sample that is above threshold while the
        # previous sample was below it.
        rising_indices = np.flatnonzero(above[1:] & ~above[:-1]) + 1
        return rising_indices / self.sampling_rate_hz

    # =========================================================================
    # Export / display helpers
    # =========================================================================
    def to_dataframe(
        self,
        channels: list[str | int] | None = None,
        sweeps: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Long-format table with columns: sweep, time_s, <channel 1>, <channel 2>, ...

        Handy for saving to CSV or for plotting with seaborn.
        """
        channel_indices = (
            list(range(len(self.channel_names)))
            if channels is None
            else [self.channel_index(c) for c in channels]
        )
        selected_sweeps = self.good_sweeps if sweeps is None else list(sweeps)

        per_sweep_tables = []
        for sweep in selected_sweeps:
            table = pd.DataFrame(
                self.data[sweep, channel_indices, :].T,
                columns=[self.channel_names[i] for i in channel_indices],
            )
            table.insert(0, "time_s", self.time_s)
            table.insert(0, "sweep", sweep)
            per_sweep_tables.append(table)
        # One concat at the end is much faster than growing a table in a loop.
        return pd.concat(per_sweep_tables, ignore_index=True)

    def export_to_csv(
        self,
        output_path: str | Path,
        channels: list[str | int] | None = None,
        sweeps: list[int] | None = None,
    ) -> Path:
        """Save selected channels/sweeps to a CSV file and return its path."""
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(channels, sweeps).to_csv(output_path, index=False)
        return output_path

    def summary(self) -> str:
        """A short, readable multi-line description of the recording."""
        channels = ", ".join(
            f"{name} [{unit}]"
            for name, unit in zip(self.channel_names, self.channel_units)
        )
        n_pulses = 0 if self.stimulus_pulses.empty else len(self.stimulus_pulses)
        lines = [
            f"Ephys recording : {self.file_name}",
            f"  source        : {self.source_description}",
            f"  recorded at   : {self.recording_datetime}",
            f"  mode          : {self.acquisition_mode}",
            f"  protocol      : {self.protocol_name or 'n/a'}",
            f"  sweeps        : {self.n_sweeps} "
            f"({len(self.bad_sweeps)} marked bad)",
            f"  sweep length  : {self.sweep_duration_s:.3f} s "
            f"({self.n_samples_per_sweep} samples at {self.sampling_rate_hz:g} Hz)",
            f"  channels      : {channels}",
            f"  stim pulses   : {n_pulses} digital pulses in total",
        ]
        return "\n".join(lines)

    def show(self, **viewer_options: Any):
        """
        Open the interactive trace viewer (see ``interactive_viewers``).

        In a notebook, the viewer is displayed below the cell. Keyword
        arguments are passed to ``build_ephys_trace_viewer``.
        """
        # Imported here so that loading data never requires the plotting libraries.
        from ..interactive_viewers.ephys_trace_viewer import build_ephys_trace_viewer

        return build_ephys_trace_viewer(self, **viewer_options)

    @staticmethod
    def show_experiment_traces(expdate, neuronid, dendriteid, spineid=None, average_by=None, group=None,
                               catalog=None, **options):
        """
        Show ALL traces of one experiment sample from the experiment sheet
        (expdate + neuron + dendrite, optionally one spine), aligned to the first
        stimulus; ``average_by="isi"`` averages the sweeps with the same ISI.

        >>> EphysRecording.show_experiment_traces(20250125, 1, 1, spineid=3, average_by="isi")
        >>> EphysRecording.show_experiment_traces(20250125, 1, 1, spineid=3, average_by="isi", group="ctrl")

        Same as ``ExperimentCatalog.show_experiment_traces`` (see there for all
        options), using the sheet from the settings file unless ``catalog`` is given.
        """
        from ..experiment_catalog.ephys_experiment_catalog import get_default_catalog

        catalog = catalog or get_default_catalog()
        if catalog is None:
            raise RuntimeError("No experiment sheet configured ([catalog] in data_source_settings.toml).")
        return catalog.show_experiment_traces(expdate, neuronid, dendriteid, spineid, average_by=average_by,
                                              group=group, **options)

    def __repr__(self) -> str:
        return (
            f"EphysRecording('{self.file_name}', {self.n_sweeps} sweeps, "
            f"channels={self.channel_names}, {self.sampling_rate_hz:g} Hz)"
        )
