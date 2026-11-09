"""
image_recording.py
==================

Defines ``ImageRecording``: ONE standard container for microscope data, whether
it came from a Zeiss LSM (.czi) or an Olympus FluoView (.oib/.oif) file.

The key idea: always the same axis order
----------------------------------------
Every vendor stores its dimensions differently (Olympus e.g. "CYX" or "CZYX",
Zeiss e.g. "BVCTZYX0S"). The readers rearrange everything into ONE fixed
5-dimensional order::

    data.shape == (T, C, Z, Y, X)

    T = time points  (frames, or scanned lines for a line scan)
    C = channels     (detectors, e.g. green / red)
    Z = focal planes (z-stack slices)
    Y = image rows
    X = image columns (pixels along the scan line)

Dimensions that a file does not have are simply of length 1. For example:

    acquisition type          typical shape (T, C, Z, Y, X)
    ------------------------  -----------------------------
    single frame              (1,    2, 1,   512, 512)
    z-stack                   (1,    2, 173, 512, 512)
    single line scan          (1,    2, 1,   1,   128)
    line-scan time series     (5200, 2, 1,   1,   128)   <- T is time, one line per row

So ``data[:, 0, 0, 0, :]`` is always the (time x position) kymograph of
channel 0 for a line scan, and ``data[0, 0, 5]`` is always z-slice 5 of
channel 0 for a z-stack, regardless of the microscope.

Pixel values are kept at their ORIGINAL bit depth (e.g. 12-bit values stored
in uint16). The older Zeiss code converted to 8-bit when loading, which throws
away intensity information; here, conversion happens only for display.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# The fixed axis order used by every ImageRecording (see module docstring).
STANDARD_AXIS_ORDER = "TCZYX"


@dataclass
class ImageRecording:
    """
    A microscope acquisition (frame, z-stack, line scan or time series) with
    the metadata needed to interpret it physically.

    Attributes
    ----------
    file_name : str
        Original file name, e.g. "Image0004.oib".
    source_description : str
        Where the file came from (local path or Box root/path), for provenance.
    local_file_path : Path
        The file on disk that was read (for Box data: the cached copy).
    microscope : str
        e.g. "Zeiss LSM880" or "Olympus FV1000".
    acquisition_type : str
        One of "single frame", "z-stack", "time series", "z-stack time series",
        "single line scan", "line-scan time series", "camera snapshot".
    acquisition_datetime : datetime.datetime | None
        When the acquisition started (from the file metadata).
    shape_tczyx : tuple[int, int, int, int, int]
        Size of each dimension in the standard order (T, C, Z, Y, X).
    dtype : numpy dtype
        Pixel data type as stored in the file (e.g. uint8, uint16).
    bits_per_pixel : int | None
        Number of *meaningful* bits (e.g. 12 for a 12-bit detector stored in
        uint16). Useful to set display ranges (max value = 2**bits - 1).
    channel_names : list[str]
        Name of each channel (axis C), e.g. ["NDD R1", "NDD R2"].
    pixel_size_x_um, pixel_size_y_um : float | None
        Physical size of one pixel along X and Y, in micrometres.
    z_step_um : float | None
        Distance between z-slices in micrometres (None if not a z-stack).
    time_interval_s : float | None
        Time between consecutive time points (frame period, or line period for
        a line scan), in seconds.
    timestamps_s : np.ndarray | None
        Measured time of each time point, in seconds relative to the first one
        (Zeiss stores these; more exact than ``time_interval_s``).
    events : pandas.DataFrame
        Events recorded by the microscope software, e.g. the "stim" markers of
        Zeiss ZEN, with columns ``time_s, event_type, description``.
        ``time_s`` is relative to the first time point (same clock as
        ``timestamps_s``), so stimulus times can be drawn on a kymograph.
    uncaging_points : pandas.DataFrame
        Laser stimulation / uncaging targets (Olympus "Bleach" points), with
        columns ``point, x_px, y_px`` (+ vendor-specific extras). Pixel
        coordinates follow image convention: (0, 0) is the TOP-left corner.
    reference_image : np.ndarray | None
        An overview image stored inside the file (Zeiss "Image" attachment for
        line scans, Olympus reference image), in (C, Y, X) order.
    reference_image_pixel_size_um : float | None
        Pixel size of the reference image in micrometres.
    metadata : dict
        Selected, human-readable metadata (laser power, zoom, objective, ...).
    raw_metadata : Any
        The complete vendor metadata (XML string for Zeiss, dictionary of INI
        sections for Olympus) in case something not extracted is needed later.
    """

    # --- identity / provenance ------------------------------------------------
    file_name: str
    source_description: str
    local_file_path: Path
    microscope: str
    acquisition_type: str
    acquisition_datetime: datetime.datetime | None

    # --- array description -------------------------------------------------------
    shape_tczyx: tuple[int, int, int, int, int]
    dtype: np.dtype
    bits_per_pixel: int | None
    channel_names: list[str]

    # --- physical calibration --------------------------------------------------
    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None
    z_step_um: float | None = None
    time_interval_s: float | None = None
    timestamps_s: np.ndarray | None = None

    # --- experiment annotations ------------------------------------------------
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    uncaging_points: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_image: np.ndarray | None = field(default=None, repr=False)
    reference_image_pixel_size_um: float | None = None

    # --- metadata ----------------------------------------------------------------
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    raw_metadata: Any = field(default=None, repr=False)

    # --- lazy data loading (filled in by the reader) ---------------------------
    _data_loader: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _data: np.ndarray | None = field(default=None, repr=False)

    # =========================================================================
    # Access to the pixel data
    # =========================================================================
    @property
    def data(self) -> np.ndarray:
        """
        All pixels as a 5-D array in (T, C, Z, Y, X) order, original dtype.

        Read from the file on first access, then kept in memory.
        """
        if self._data is None:
            if self._data_loader is None:
                raise RuntimeError(
                    f"No data and no data loader available for {self.file_name}."
                )
            loaded = self._data_loader()
            # Safety check: every reader must deliver the standard 5-D layout.
            if loaded.ndim != 5:
                raise ValueError(
                    f"Reader returned a {loaded.ndim}-D array for {self.file_name}; "
                    "expected 5-D (T, C, Z, Y, X)."
                )
            # The pixels must have exactly the shape announced from the metadata;
            # otherwise axes were mislabelled (e.g. channels and z-slices swapped)
            # and every z/channel selection would silently be wrong.
            if tuple(loaded.shape) != tuple(self.shape_tczyx):
                raise ValueError(
                    f"{self.file_name}: pixel array has shape {tuple(loaded.shape)} (T, C, Z, Y, X) "
                    f"but the metadata says {tuple(self.shape_tczyx)} — axes were not identified "
                    "correctly; please report this file."
                )
            self._data = loaded
        return self._data

    @property
    def is_data_loaded(self) -> bool:
        """True once the pixels have been read into memory."""
        return self._data is not None

    # Convenient names for the size of each dimension ---------------------------
    @property
    def n_timepoints(self) -> int:
        return self.shape_tczyx[0]

    @property
    def n_channels(self) -> int:
        return self.shape_tczyx[1]

    @property
    def n_z_slices(self) -> int:
        return self.shape_tczyx[2]

    @property
    def height_px(self) -> int:
        return self.shape_tczyx[3]

    @property
    def width_px(self) -> int:
        return self.shape_tczyx[4]

    @property
    def sizes(self) -> dict[str, int]:
        """Size of each dimension by name, e.g. {'T': 1, 'C': 2, 'Z': 1, ...}."""
        return dict(zip(STANDARD_AXIS_ORDER, self.shape_tczyx))

    @property
    def is_line_scan(self) -> bool:
        """True for line scans (only one image row, Y == 1)."""
        return self.height_px == 1

    @property
    def max_possible_value(self) -> int | None:
        """Largest value the detector can produce (2**bits - 1), if known."""
        if self.bits_per_pixel:
            return 2**self.bits_per_pixel - 1
        if np.issubdtype(self.dtype, np.integer):
            return int(np.iinfo(self.dtype).max)
        return None

    def channel_index(self, channel: str | int) -> int:
        """Convert a channel name (e.g. "NDD R1") or index into an index."""
        if isinstance(channel, (int, np.integer)):
            if not 0 <= channel < self.n_channels:
                raise IndexError(
                    f"Channel index {channel} out of range (0..{self.n_channels - 1})."
                )
            return int(channel)
        if channel in self.channel_names:
            return self.channel_names.index(channel)
        raise KeyError(
            f"Channel '{channel}' not found; available: {self.channel_names}"
        )

    # =========================================================================
    # Common views of the data
    # =========================================================================
    def get_plane(self, time_index: int = 0, channel: str | int = 0, z_index: int = 0) -> np.ndarray:
        """One 2-D image (Y, X) at a given time point, channel and z-slice."""
        return self.data[time_index, self.channel_index(channel), z_index]

    def projection(
        self,
        channel: str | int = 0,
        method: str = "max",
        z_range: tuple[int, int] | None = None,
        time_index: int = 0,
        smoothing_sigma_px: float = 0.0,
    ) -> np.ndarray:
        """
        Project a z-stack onto one 2-D image (Y, X).

        Parameters
        ----------
        channel : name or index of the channel
        method : "max" (maximum intensity projection), "mean" or "sum"
        z_range : (first, last) slice indices to include, last EXCLUDED,
            e.g. (10, 50). None uses all slices.
        time_index : which time point to project (for z-stack time series)
        smoothing_sigma_px : if > 0, blur every slice with a Gaussian of this
            width (in pixels) BEFORE projecting. Two-photon slices are noisy,
            and a maximum projection keeps the brightest noise pixel of every
            column; light smoothing (sigma 1-2 px) gives a much cleaner image.
            (For display only — the stored data are never changed.)
        """
        stack = self.data[time_index, self.channel_index(channel)]  # (Z, Y, X)
        if z_range is not None:
            stack = stack[z_range[0] : z_range[1]]
            if stack.shape[0] == 0:
                raise ValueError(
                    f"z_range {tuple(z_range)} selects no slices; {self.file_name} has slices "
                    f"0..{self.n_z_slices - 1} (z_range is (first, last + 1))."
                )
        if smoothing_sigma_px > 0:
            from scipy.ndimage import gaussian_filter

            # sigma 0 along Z: smooth within each slice only, never across slices.
            stack = gaussian_filter(stack.astype(np.float32), sigma=(0, smoothing_sigma_px, smoothing_sigma_px))
        match method:
            case "max":
                return stack.max(axis=0)
            case "mean":
                return stack.mean(axis=0)
            case "sum":
                return stack.sum(axis=0)
            case _:
                raise ValueError("method must be 'max', 'mean' or 'sum'")

    def kymograph(self, channel: str | int = 0) -> np.ndarray:
        """
        For line scans: a 2-D array (time points, pixels along the line).

        Row i is the line scanned at time ``line_times_s[i]``.
        """
        if not self.is_line_scan:
            raise ValueError(f"{self.file_name} is not a line scan (Y = {self.height_px}).")
        return self.data[:, self.channel_index(channel), 0, 0, :]

    @property
    def line_times_s(self) -> np.ndarray:
        """
        Time of every time point in seconds, starting at 0.

        Uses the measured timestamps when available, otherwise
        ``index * time_interval_s``.
        """
        if self.timestamps_s is not None and len(self.timestamps_s) == self.n_timepoints:
            return self.timestamps_s
        interval = self.time_interval_s or 1.0
        return np.arange(self.n_timepoints) * interval

    def line_profile_over_time(
        self, channel: str | int = 0, x_range_px: tuple[int, int] | None = None
    ) -> np.ndarray:
        """
        Mean fluorescence along part of the scan line, for every time point.

        This is the basic "ROI time series" of a line scan: pick the pixels that
        cover e.g. a spine head (``x_range_px=(40, 52)``, last excluded) and get
        one value per scanned line.
        """
        kymo = self.kymograph(channel).astype(np.float64)
        if x_range_px is not None:
            kymo = kymo[:, x_range_px[0] : x_range_px[1]]
        return kymo.mean(axis=1)

    def event_times_s(self, description: str | None = "stim") -> np.ndarray:
        """
        Times of microscope events (relative to the first time point).

        By default only events whose description is "stim" (Zeiss ZEN markers).
        Pass ``description=None`` to get all events.
        """
        if self.events.empty:
            return np.array([])
        selected = self.events
        if description is not None:
            selected = selected[selected["description"] == description]
        return selected["time_s"].to_numpy()

    # =========================================================================
    # Display helpers
    # =========================================================================
    def summary(self) -> str:
        """A short, readable multi-line description of the acquisition."""

        def fmt(value: float | None, unit: str, digits: int = 4) -> str:
            return "n/a" if value is None else f"{value:.{digits}g} {unit}"

        lines = [
            f"Image recording : {self.file_name}",
            f"  source        : {self.source_description}",
            f"  microscope    : {self.microscope}",
            f"  type          : {self.acquisition_type}",
            f"  acquired at   : {self.acquisition_datetime}",
            f"  shape (TCZYX) : {self.shape_tczyx}  dtype={np.dtype(self.dtype).name}"
            f"  bits={self.bits_per_pixel}",
            f"  channels      : {self.channel_names}",
            f"  pixel size    : x={fmt(self.pixel_size_x_um, 'um')}, "
            f"y={fmt(self.pixel_size_y_um, 'um')}, z step={fmt(self.z_step_um, 'um')}",
            f"  time interval : {fmt(self.time_interval_s, 's')}",
            f"  events        : {len(self.events)}",
            f"  uncaging pts  : {len(self.uncaging_points)}",
            # Uncaging (photostimulation) laser, read from the file's own settings
            # (Olympus: enabled "Bleach Laser"; transmissivity in % of full power).
            f"  uncaging laser: "
            + (
                f"{self.metadata['uncaging_laser_power_percent']:g} % power"
                + (f" at {self.metadata.get('uncaging_laser_wavelength_nm')} nm"
                   if self.metadata.get("uncaging_laser_wavelength_nm") else "")
                + (f", {self.metadata['uncaging_duration_ms']:g} ms per target"
                   if self.metadata.get("uncaging_duration_ms") is not None else "")
                # several targets: illumination + travel time per target, and the total
                + (
                    f"\n  uncaging time : {self.metadata['uncaging_targets']} targets x "
                    f"({self.metadata['uncaging_duration_ms']:g} ms + {self.metadata['uncaging_move_time_ms']:g} ms move)"
                    f" = {self.metadata['uncaging_total_time_ms']:g} ms total"
                    if (self.metadata.get("uncaging_targets") or 0) > 1 else ""
                )
                if self.metadata.get("uncaging_laser_power_percent") is not None
                else "n/a"
            ),
            f"  imaging laser : "
            + (
                f"{self.metadata['laser_transmissivity_percent']:g} % power"
                + (f" at {self.metadata.get('laser_wavelength_nm')} nm" if self.metadata.get("laser_wavelength_nm") else "")
                if self.metadata.get("laser_transmissivity_percent") is not None
                else "n/a"
            ),
            f"  reference img : "
            + ("none" if self.reference_image is None else str(self.reference_image.shape)),
        ]
        return "\n".join(lines)

    def show(self, **viewer_options: Any):
        """
        Open the most suitable interactive viewer for this acquisition:
        a kymograph viewer for line-scan time series, an image/stack viewer
        otherwise. Keyword arguments are passed to that viewer.
        """
        if self.is_line_scan and self.n_timepoints > 1:
            from ..interactive_viewers.linescan_kymograph_viewer import (
                build_linescan_kymograph_viewer,
            )

            return build_linescan_kymograph_viewer(self, **viewer_options)

        from ..interactive_viewers.image_stack_viewer import build_image_stack_viewer

        return build_image_stack_viewer(self, **viewer_options)

    def __repr__(self) -> str:
        return (
            f"ImageRecording('{self.file_name}', {self.microscope}, "
            f"{self.acquisition_type}, TCZYX={self.shape_tczyx})"
        )
