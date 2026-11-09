"""
olympus_oib_oif_file_reader.py
==============================

Reads Olympus FluoView files into an ``ImageRecording``:

* ``.oib`` — a single file (a "compound document", like a zip) containing
  the images as TIFFs plus text settings files.
* ``.oif`` — a text settings file with a companion folder
  ``<name>.oif.files/`` holding the TIFFs. Both files must be present.

The reading is done with the ``oiffile`` library. This module then:

1. rearranges the pixel array into the standard (T, C, Z, Y, X) order,
2. converts the calibration (pixel size, z-step, frame interval) into
   micrometres / seconds,
3. extracts the laser-stimulation ("Bleach") / uncaging points,
4. reads the reference image stored alongside the main image, if present.

Olympus metadata structure (for reference)
------------------------------------------
The main settings file is an INI-style text file with sections such as::

    [Axis 0 Parameters Common]   X axis: MaxSize (pixels), StartPosition,
                                 EndPosition, PixUnit ("um")
    [Axis 1 Parameters Common]   Y axis (same keys)
    [Axis 3 Parameters Common]   Z axis: Interval (step, in PixUnit e.g. "nm")
    [Axis 4 Parameters Common]   T axis: Interval (in PixUnit e.g. "ms")
    [Reference Image Parameter]  WidthConvertValue (um/pixel), ValidBitCounts
    [Bleach GUI Parameters Common]
                                 "Number Of Point", "Point 0 Position X", ...
                                 = the uncaging / photostimulation targets
    [Channel 1 Parameters]       "CH Name", "DyeName", PMT voltage ...

``oiffile`` returns this file as a dictionary of dictionaries
(``OifFile.mainfile[section][key]``), already converted to numbers.
"""

from __future__ import annotations

import datetime
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..recording_data_containers.image_recording import STANDARD_AXIS_ORDER, ImageRecording

logger = logging.getLogger(__name__)

# Axis letters that oiffile may use, and what they mean in our standard order.
# (Olympus also knows "L" = lambda/spectral; it is treated like a channel.)
OLYMPUS_AXIS_TO_STANDARD_AXIS = {"T": "T", "C": "C", "L": "C", "Z": "Z", "Y": "Y", "X": "X"}


# =============================================================================
# Small helpers
# =============================================================================
def _open_olympus_file(file_path: Path):
    """Open an .oib/.oif file with oiffile, hiding harmless library warnings."""
    import oiffile  # imported here: only needed when Olympus files are read

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return oiffile.OifFile(str(file_path))


def _length_to_um(value: float, unit: str) -> float:
    """Convert a length in nm / um / mm into micrometres."""
    factors = {"nm": 1e-3, "um": 1.0, "µm": 1.0, "mm": 1e3}
    return float(value) * factors.get(str(unit).strip(), 1.0)


def _time_to_s(value: float, unit: str) -> float:
    """Convert a time in us / ms / s into seconds."""
    factors = {"us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}
    return float(value) * factors.get(str(unit).strip(), 1.0)


def _find_axis_section(main_settings: dict, axis_code: str) -> dict | None:
    """
    Return the "[Axis N Parameters Common]" section whose AxisCode is e.g. "X".

    Olympus numbers the axes (0 = X, 1 = Y, 3 = Z, 4 = T usually), but looking
    them up by AxisCode is safer than relying on the number.
    """
    for section_name, section in main_settings.items():
        if section_name.startswith("Axis ") and section_name.endswith("Parameters Common"):
            if isinstance(section, dict) and section.get("AxisCode") == axis_code:
                return section
    return None


def main_and_reference_series(olympus_file) -> tuple[int, int | None]:
    """
    Indices (main, reference) of the image series inside an Olympus file.

    The acquisition is stored in TIFFs named like "s_C001.tif" / "s_C001Z001.tif";
    the reference image (the overview on which stimulation points were set) in
    "s_C001-R001.tif". oiffile lists the series in no guaranteed order (the
    reference often comes FIRST), so they are identified by file name.
    """
    main, reference = None, None
    for index, series in enumerate(olympus_file.series):
        names = [str(name) for name in series]
        is_reference = all(re.search(r"-R\d+\.tif$", name, re.IGNORECASE) for name in names)
        if is_reference and reference is None:
            reference = index
        elif not is_reference and main is None:
            main = index
    if main is None:
        raise ValueError("No main image series found (only reference images).")
    return main, reference


def series_axes_and_shape(olympus_file, series_index: int) -> tuple[str, tuple[int, ...]]:
    """
    Axes and shape of the pixel array returned by ``OifFile.asarray(series_index)``.

    Why not simply ``OifFile.axes`` / ``OifFile.shape``? Those describe the
    settings file ("AxisOrder", e.g. "ZCYX"), and the oiffile documentation
    warns that they "might differ from the axes order of series". The array
    is built as:
      * outer axes : from the series, i.e. the TIFF file names
                     ("s_C001Z001.tif" -> "CZ", shape (1, 109)),
      * plane axes : the 2-D image inside each TIFF = the LAST TWO axes of the
                     settings AxisOrder: "YX" for frames and z-stacks, but
                     "TX" for line scans (each TIFF holds time x position).
    Using the settings axes directly swapped C and Z in z-stacks (109
    "channels" x 1 slice), which broke z-projections over a partial range.
    """
    series = olympus_file.series[series_index]
    settings_axes = str(olympus_file.axes)
    sizes = {}
    for index in range(8):
        section = olympus_file.mainfile.get(f"Axis {index} Parameters Common", {})
        if section.get("AxisCode"):
            sizes[section["AxisCode"]] = int(section.get("MaxSize") or 0)
    plane_axes = settings_axes[-2:] if len(settings_axes) >= 2 else "YX"
    axes = str(series.axes) + plane_axes
    shape = tuple(int(n) for n in series.shape) + tuple(sizes.get(axis, 0) for axis in plane_axes)
    return axes, shape


def reorder_to_tczyx(pixels: np.ndarray, olympus_axes: str) -> np.ndarray:
    """
    Rearrange an Olympus array (axes e.g. "CYX" or "TZCYX") into (T, C, Z, Y, X).

    Missing axes are added with length 1. Unknown axes of length 1 are dropped;
    unknown axes longer than 1 raise an error so data are never silently lost.
    """
    axes = list(olympus_axes.upper())

    # 1) Drop unknown axes of length 1, refuse unknown axes with real content.
    for position in reversed(range(len(axes))):
        if axes[position] not in OLYMPUS_AXIS_TO_STANDARD_AXIS:
            if pixels.shape[position] == 1:
                pixels = pixels.squeeze(axis=position)
                del axes[position]
            else:
                raise ValueError(
                    f"Unsupported Olympus axis '{axes[position]}' with length "
                    f"{pixels.shape[position]} (axes '{olympus_axes}')."
                )

    # 2) Rename to standard letters (e.g. L -> C).
    standard_axes = [OLYMPUS_AXIS_TO_STANDARD_AXIS[a] for a in axes]
    if len(set(standard_axes)) != len(standard_axes):
        raise ValueError(f"Cannot map Olympus axes '{olympus_axes}' uniquely to TCZYX.")

    # 3) Add every missing standard axis at the end, with length 1.
    for axis in STANDARD_AXIS_ORDER:
        if axis not in standard_axes:
            pixels = pixels[..., np.newaxis]
            standard_axes.append(axis)

    # 4) Put the axes in T, C, Z, Y, X order.
    new_order = [standard_axes.index(axis) for axis in STANDARD_AXIS_ORDER]
    return np.transpose(pixels, new_order)


def _classify_acquisition(shape_tczyx: tuple[int, ...]) -> str:
    """Name the acquisition type from the sizes of T, Z and Y."""
    n_t, _, n_z, n_y, _ = shape_tczyx
    if n_y == 1:
        return "line-scan time series" if n_t > 1 else "single line scan"
    if n_z > 1 and n_t > 1:
        return "z-stack time series"
    if n_z > 1:
        return "z-stack"
    if n_t > 1:
        return "time series"
    return "single frame"


# =============================================================================
# Metadata extraction
# =============================================================================
# -----------------------------------------------------------------------------
# Wrong objective in the FluoView settings
# -----------------------------------------------------------------------------
# With the 60x objective, field width x zoom = 209.5 um for a 512-px image.
# If FluoView was set to the wrong objective (5x), files store 2539.7 um x zoom
# instead, about 12x too large. Such files are rescaled to the 60x calibration:
# pixel size = 209.5 um / zoom / pixels.
FIELD_WIDTH_X_ZOOM_60X_UM = 209.5
WRONG_FIELD_WIDTH_X_ZOOM_UM = 2539.7
# The objective FluoView was set to is stored in the channel .pty files
# ("ObjectiveLens Name", "Magnification"). On this setup the 60x objective is
# always used, so a file set to another objective is rescaled.
TRUE_OBJECTIVE_MAGNIFICATION = 60.0


def _read_objective(olympus_file) -> tuple[str, float | None]:
    """(objective name, magnification) set in FluoView, from the main image's channel .pty files."""
    names, magnifications = set(), set()
    for name, text in _read_embedded_text_files(olympus_file, ".pty").items():
        if re.search(r"-R\d+\.pty$", name, re.I):          # the reference image's own settings
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("ObjectiveLens Name="):
                names.add(" ".join(line.split("=", 1)[1].strip('"').split()))
            elif line.startswith("Magnification="):
                try:
                    magnifications.add(float(line.split("=", 1)[1]))
                except ValueError:
                    pass
    return "; ".join(sorted(names)), (magnifications.pop() if len(magnifications) == 1 else None)


def _wrong_objective_correction(main_settings: dict, calibration: dict,
                                objective_magnification: float | None = None) -> tuple[float, str]:
    """
    (factor to multiply the stored pixel sizes by, note); (1.0, "") for correctly calibrated files.

    Decided from the objective stored in the file when it is known: 60x -> no
    change; 5x -> the measured ratio of the two calibrations (209.5 / 2539.7 um);
    any other objective -> magnification / 60. Without objective information the
    old test is used: field width x zoom of ~2540 um (5x calibration).
    """
    stored = calibration["pixel_size_x_um"]
    if stored is None:
        return 1.0, ""
    if objective_magnification:
        if objective_magnification == TRUE_OBJECTIVE_MAGNIFICATION:
            return 1.0, ""
        factor = (FIELD_WIDTH_X_ZOOM_60X_UM / WRONG_FIELD_WIDTH_X_ZOOM_UM if objective_magnification == 5
                  else objective_magnification / TRUE_OBJECTIVE_MAGNIFICATION)
        return factor, (f"pixel size corrected from {stored:.4f} to {stored * factor:.4f} um: FluoView was set to "
                        f"the {objective_magnification:g}x objective, but the 60x objective was used")
    zoom = float(main_settings.get("Acquisition Parameters Common", {}).get("ZoomValue") or 0)
    x_axis = _find_axis_section(main_settings, "X")
    n_pixels = int((x_axis or {}).get("MaxSize") or 0)
    if not zoom or not n_pixels:
        return 1.0, ""
    field_x_zoom = stored * n_pixels * zoom
    # normalised to 512 px, since the 60x constant is for 512-px images
    field_x_zoom_512 = field_x_zoom * 512 / n_pixels
    if abs(field_x_zoom_512 / WRONG_FIELD_WIDTH_X_ZOOM_UM - 1) > 0.03:
        return 1.0, ""
    factor = FIELD_WIDTH_X_ZOOM_60X_UM / field_x_zoom_512
    return factor, (f"pixel size corrected from {stored:.4f} to {stored * factor:.4f} um: the file's calibration "
                    f"is for the wrong objective (field x zoom {field_x_zoom_512:.0f} um instead of "
                    f"{FIELD_WIDTH_X_ZOOM_60X_UM:g} um for the 60x objective)")


def _read_calibration(main_settings: dict) -> dict[str, float | None]:
    """Pixel size (um), z-step (um) and frame interval (s) from the axis sections."""
    calibration: dict[str, float | None] = {
        "pixel_size_x_um": None,
        "pixel_size_y_um": None,
        "z_step_um": None,
        "time_interval_s": None,
    }

    # Pixel size = (end position - start position) / number of pixels
    for axis_code, key in (("X", "pixel_size_x_um"), ("Y", "pixel_size_y_um")):
        section = _find_axis_section(main_settings, axis_code)
        if section and int(section.get("MaxSize") or 0) > 1:  # a 1-pixel axis (line scan Y) has no pixel size
            span = float(section.get("EndPosition", 0)) - float(section.get("StartPosition", 0))
            calibration[key] = _length_to_um(abs(span) / float(section["MaxSize"]), section.get("PixUnit", "um"))

    # Fallback: the "Reference Image Parameter" section stores um/pixel directly.
    reference = main_settings.get("Reference Image Parameter", {})
    if calibration["pixel_size_x_um"] is None and reference.get("WidthConvertValue"):
        calibration["pixel_size_x_um"] = _length_to_um(reference["WidthConvertValue"], reference.get("WidthUnit", "um"))
    if calibration["pixel_size_y_um"] is None and reference.get("HeightConvertValue"):
        calibration["pixel_size_y_um"] = _length_to_um(reference["HeightConvertValue"], reference.get("HeightUnit", "um"))

    # Z step: only meaningful if more than one slice was acquired.
    z_section = _find_axis_section(main_settings, "Z")
    if z_section and int(z_section.get("MaxSize", 0) or 0) > 1 and z_section.get("Interval"):
        calibration["z_step_um"] = _length_to_um(z_section["Interval"], z_section.get("PixUnit", "nm"))

    # Frame interval: "Interval" if given, else the time span divided by frames.
    t_section = _find_axis_section(main_settings, "T")
    if t_section and int(t_section.get("MaxSize", 0) or 0) > 1:
        unit = t_section.get("PixUnit", "ms")
        if t_section.get("Interval"):
            calibration["time_interval_s"] = _time_to_s(t_section["Interval"], unit)
        else:
            span = float(t_section.get("EndPosition", 0)) - float(t_section.get("StartPosition", 0))
            calibration["time_interval_s"] = _time_to_s(span / (int(t_section["MaxSize"]) - 1), unit)

    return calibration


def _read_channel_names(main_settings: dict, n_channels: int) -> list[str]:
    """
    Channel names from "[Channel N Parameters]" (N starts at 1).

    The dye name is used when it was entered, otherwise the detector name
    (e.g. "RXD2" = non-descanned detector 2).
    """
    names = []
    for channel_number in range(1, n_channels + 1):
        section = main_settings.get(f"Channel {channel_number} Parameters", {})
        candidates = [str(section.get(key) or "").strip() for key in ("DyeName", "CH Name")]
        # FluoView writes the text "None" for unnamed channels.
        name = next((c for c in candidates if c and c.lower() != "none"), "")
        names.append(name or f"Ch{channel_number}")
    return names


def _read_acquisition_datetime(main_settings: dict) -> datetime.datetime | None:
    """Parse "ImageCaputreDate" (sic — Olympus' spelling) plus milliseconds."""
    acquisition = main_settings.get("Acquisition Parameters Common", {})
    date_text = acquisition.get("ImageCaputreDate")
    if not date_text:
        return None
    try:
        timestamp = datetime.datetime.strptime(str(date_text).strip("'\""), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        logger.warning("Could not parse Olympus date '%s'.", date_text)
        return None
    milliseconds = int(acquisition.get("ImageCaputreDate+MilliSec", 0) or 0)
    return timestamp + datetime.timedelta(milliseconds=milliseconds)


def _read_uncaging_points(main_settings: dict, image_width_px: int, image_height_px: int) -> pd.DataFrame:
    """
    Laser stimulation targets from "[Bleach GUI Parameters Common]".

    The positions are stored in the pixel grid of the bleach GUI image
    ("ImageWidth" x "ImageHeight"). If that grid differs from the acquired
    image size, the positions are rescaled to the acquired image.
    Coordinates: (0, 0) is the top-left pixel, y increases downwards.
    """
    columns = ["point", "x_px", "y_px", "n_pixels", "x_px_bleach_gui", "y_px_bleach_gui"]
    bleach = main_settings.get("Bleach GUI Parameters Common", {})
    n_points = int(bleach.get("Number Of Point", 0) or 0)
    if n_points == 0:
        return pd.DataFrame(columns=columns)

    gui_width = float(bleach.get("ImageWidth") or image_width_px)
    gui_height = float(bleach.get("ImageHeight") or image_height_px)
    scale_x = image_width_px / gui_width
    scale_y = image_height_px / gui_height

    rows = []
    for point in range(n_points):
        x_key, y_key = f"Point {point} Position X", f"Point {point} Position Y"
        if x_key not in bleach or y_key not in bleach:
            continue
        x_gui, y_gui = float(bleach[x_key]), float(bleach[y_key])
        rows.append(
            {
                "point": point,
                "x_px": x_gui * scale_x,
                "y_px": y_gui * scale_y,
                "n_pixels": bleach.get(f"Point {point} Number Of Pixel"),
                "x_px_bleach_gui": x_gui,
                "y_px_bleach_gui": y_gui,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _read_embedded_text_files(olympus_file, suffix: str) -> dict[str, str]:
    """
    Read small text files stored inside the .oib/.oif (e.g. ".roi" files).

    They are UTF-16 encoded on FluoView systems; UTF-8 is used as a fallback.
    """
    texts = {}
    for name in olympus_file.glob(f"*{suffix}"):
        raw = olympus_file.open_file(name).read()
        encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
        texts[name] = raw.decode(encoding, errors="ignore")
    return texts


def _select_metadata(main_settings: dict) -> dict[str, Any]:
    """A flat dictionary of the most useful acquisition settings."""
    acquisition = main_settings.get("Acquisition Parameters Common", {})
    bleach = main_settings.get("Bleach GUI Parameters Common", {})
    version = main_settings.get("Version Info", {})
    selected = {
        "system_name": version.get("SystemName"),
        "system_version": version.get("SystemVersion"),
        "scan_mode": acquisition.get("ScanMode"),
        "zoom": acquisition.get("ZoomValue"),
        "laser_wavelength_nm": acquisition.get("LaserWavelength01"),
        "laser_transmissivity_percent": acquisition.get("LaserTransmissivity01"),
        "pixel_sampling_clock": acquisition.get("SamplingClock"),
        "bleach_comments": bleach.get("Comments"),
        # Units of the activation time are not documented in the file; kept raw.
        "bleach_activation_time_per_point_raw": bleach.get("Comb 0 Activation Time Per Point"),
    }
    # Uncaging (photostimulation) laser = the ENABLED "Bleach Laser N parameters"
    # section (720 nm on the FV1000 rig). FluoView stores its transmissivity in
    # 0.1 % steps (e.g. 300 -> 30.0 %); this is the authoritative uncaging power.
    enabled_bleach_lasers = [
        values for section, values in main_settings.items()
        if section.startswith("Bleach Laser") and isinstance(values, dict) and values.get("Laser Enable") == 1
    ]
    if len(enabled_bleach_lasers) == 1 and enabled_bleach_lasers[0].get("LaserTransmissivity") is not None:
        selected["uncaging_laser_power_percent"] = float(enabled_bleach_lasers[0]["LaserTransmissivity"]) / 10
        selected["uncaging_laser_wavelength_nm"] = enabled_bleach_lasers[0].get("LaserWavelength")
    # Uncaging duration per target: "Comb 0 Activation Time Per Point" is in
    # MICROSECONDS (e.g. 500 -> 0.5 ms); the number of scanned pixels per point
    # scales with it.
    activation_us = bleach.get("Comb 0 Activation Time Per Point")
    n_targets = int(bleach.get("Number Of Point", 0) or 0)
    if activation_us not in (None, "") and n_targets > 0:
        selected["uncaging_duration_ms"] = float(activation_us) / 1000
        selected["uncaging_targets"] = n_targets
        # Travel time of the scan mirrors to each target ("Point Move Time Point",
        # in NANOSECONDS). It equals the target's
        # "Number Of Move Pixel" x pixel time (1 / SamplingClock = 20 us at 50 kHz),
        # is the same for every target of an image, is listed for EVERY target
        # (including the first), and is 0 for single-target images.
        move_ns = float(bleach.get("Point Move Time Point") or 0)
        selected["uncaging_move_time_ms"] = move_ns / 1e6
        # Whole multi-target activation, from the trigger to the end of the last
        # target: every target = move + illumination. Computed from the settings
        # (the rig records the trigger, not the light, so it cannot be measured).
        selected["uncaging_total_time_ms"] = n_targets * (selected["uncaging_duration_ms"] + move_ns / 1e6)
    # Detector settings of each channel (PMT voltage/gain/offset).
    for section_name, section in main_settings.items():
        if section_name.startswith("Channel ") and section_name.endswith(" Parameters"):
            number = section_name.split()[1]
            selected[f"channel_{number}_pmt_voltage"] = section.get("AnalogPMTVoltage")
            selected[f"channel_{number}_pmt_gain"] = section.get("AnalogPMTGain")
            selected[f"channel_{number}_pmt_offset"] = section.get("AnalogPMTOffset")
    return selected


# =============================================================================
# Main entry point
# =============================================================================
def read_olympus_oib_oif_file(
    file_path: str | Path,
    source_description: str | None = None,
    original_file_name: str | None = None,
    load_data_now: bool = False,
) -> ImageRecording:
    """
    Read an Olympus FluoView .oib or .oif file into an ``ImageRecording``.

    Parameters
    ----------
    file_path : str or Path
        The .oib file, or the .oif file (its "<name>.oif.files" folder must
        be next to it).
    source_description : str, optional
        Provenance text stored in the recording (defaults to "local: <path>").
    original_file_name : str, optional
        The real file name if ``file_path`` is a renamed copy.
    load_data_now : bool
        Read the pixels immediately instead of on first use.
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Olympus file not found: {file_path}")

    olympus_file = _open_olympus_file(file_path)
    try:
        main_settings: dict = dict(olympus_file.mainfile)
        # Which series is the acquisition and which the reference image, and the
        # axes/shape of the acquisition's PIXEL ARRAY (see the two helper functions).
        main_series, reference_series = main_and_reference_series(olympus_file)
        olympus_axes, olympus_shape = series_axes_and_shape(olympus_file, main_series)
        pixel_dtype = np.dtype(olympus_file.dtype)

        # Work out the standard (T, C, Z, Y, X) shape WITHOUT reading pixels:
        # np.broadcast_to creates a zero-memory "fake" array of the right shape,
        # which reorder_to_tczyx can rearrange just like real data.
        shape_placeholder = np.broadcast_to(np.zeros((), dtype=np.uint8), olympus_shape)
        shape_tczyx = tuple(int(n) for n in reorder_to_tczyx(shape_placeholder, olympus_axes).shape)

        # Reference image ("-R001" TIFFs), if the file has one.
        reference_image = None
        if reference_series is not None:
            try:
                reference_raw = olympus_file.asarray(reference_series)
                reference_image = reference_raw.reshape((-1,) + reference_raw.shape[-2:])  # (C, Y, X)
            except Exception as error:  # a broken reference must not block the main image
                logger.warning("Could not read reference image of %s: %s", file_path.name, error)

        roi_texts = _read_embedded_text_files(olympus_file, ".roi")
        try:
            objective_name, objective_magnification = _read_objective(olympus_file)
        except Exception as error:  # objective info is optional
            logger.warning("Could not read the objective of %s: %s", file_path.name, error)
            objective_name, objective_magnification = "", None
    finally:
        olympus_file.close()

    calibration = _read_calibration(main_settings)
    reference_section = main_settings.get("Reference Image Parameter", {})
    bits = reference_section.get("ValidBitCounts")
    calibration_factor, calibration_note = _wrong_objective_correction(main_settings, calibration,
                                                                       objective_magnification)
    for key in ("pixel_size_x_um", "pixel_size_y_um"):
        if calibration[key] is not None:
            calibration[key] *= calibration_factor

    metadata = _select_metadata(main_settings)
    # objective as set in FluoView (the 60x was always used; see above)
    metadata["objective_in_file"] = objective_name or None
    metadata["objective_magnification_in_file"] = objective_magnification
    if calibration_note:
        metadata["pixel_size_note"] = calibration_note
    metadata["olympus_axes"] = olympus_axes  # axes of the pixel array (from the image series)
    metadata["olympus_settings_axes"] = str(main_settings.get("Axis Parameter Common", {}).get("AxisOrder", ""))[::-1]
    metadata["roi_files"] = roi_texts  # raw text of the embedded .roi files

    def load_pixels() -> np.ndarray:
        """Read all pixels and rearrange them to (T, C, Z, Y, X)."""
        reader = _open_olympus_file(file_path)
        try:
            series_index, _ = main_and_reference_series(reader)
            axes, _ = series_axes_and_shape(reader, series_index)
            return reorder_to_tczyx(reader.asarray(series_index), axes)
        finally:
            reader.close()

    recording = ImageRecording(
        file_name=original_file_name or file_path.name,
        source_description=source_description or f"local: {file_path}",
        local_file_path=file_path,
        microscope=f"Olympus {main_settings.get('Acquisition Parameters Common', {}).get('Acquisition Device', 'FluoView')}",
        acquisition_type=_classify_acquisition(shape_tczyx),
        acquisition_datetime=_read_acquisition_datetime(main_settings),
        shape_tczyx=shape_tczyx,
        dtype=pixel_dtype,
        bits_per_pixel=int(bits) if bits else None,
        channel_names=_read_channel_names(main_settings, shape_tczyx[1]),
        pixel_size_x_um=calibration["pixel_size_x_um"],
        pixel_size_y_um=calibration["pixel_size_y_um"],
        z_step_um=calibration["z_step_um"],
        time_interval_s=calibration["time_interval_s"],
        timestamps_s=None,  # Olympus files do not store per-frame timestamps
        # Stimulation points are set on a 2-D image. For frames/stacks that is the
        # acquired image; for line scans (Y = 1) it is the reference image, so
        # their coordinates are kept in the reference-image grid (no rescaling).
        uncaging_points=(
            _read_uncaging_points(main_settings, shape_tczyx[4], shape_tczyx[3])
            if shape_tczyx[3] > 1
            else _read_uncaging_points(
                main_settings,
                int(reference_section.get("ImageWidth") or 512),
                int(reference_section.get("ImageHeight") or 512),
            )
        ),
        reference_image=reference_image,
        # The reference image has its own calibration (for line scans the main
        # image's X calibration describes the scan line, not the overview).
        reference_image_pixel_size_um=(
            _length_to_um(reference_section["WidthConvertValue"], reference_section.get("WidthUnit", "um"))
            * calibration_factor
            if reference_section.get("WidthConvertValue") else calibration["pixel_size_x_um"]
        ),
        metadata=metadata,
        raw_metadata=main_settings,
        _data_loader=load_pixels,
    )

    if load_data_now:
        _ = recording.data
    return recording
