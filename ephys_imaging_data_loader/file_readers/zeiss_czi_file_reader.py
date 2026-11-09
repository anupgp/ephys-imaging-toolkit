"""
zeiss_czi_file_reader.py
========================

Reads Carl Zeiss Image files (.czi, written by ZEN on the LSM880) into an
``ImageRecording``.

Supported acquisitions (all found in the sample data)
-----------------------------------------------------
    type                     Zeiss dimensions         -> standard (T, C, Z, Y, X)
    -----------------------  -----------------------  -------------------------
    single line scan         C=2, X=128               (1,    2, 1,   1,   128)
    line-scan time series    C=2, T=5200, X=128       (5200, 2, 1,   1,   128)
    single frame             C=2, Y=512, X=512        (1,    2, 1,   512, 512)
    z-stack                  C=2, Z=173, Y=512, X=512 (1,    2, 173, 512, 512)
    camera snapshot          C=1, Y=1038, X=1388      (1,    1, 1,   1038, 1388)

What else is read
-----------------
* Timestamps  ("TimeStamps" attachment): the measured time of every line /
  frame. For line scans these are exact line times (e.g. every 1.0176 ms).
* Events      ("EventList" attachment): markers such as "stim", recorded by
  ZEN when a trigger arrived. Converted to seconds relative to the first line,
  so they can be drawn directly on the kymograph.
* Reference image ("Image" attachment): the overview frame that ZEN stores
  with single line scans.
* Calibration from the XML metadata: pixel size (the file stores metres; we
  convert to micrometres), z-step, zoom, scan rotation, objective ...

Notes on the czifile library version
------------------------------------
czifile 2026.x changed its interface: arrays are "squeezed" (length-1 axes
removed) by default, and ``CziFile.axes`` no longer exists. The old
``zeissImageClasses.py`` indexed an 8-D array (``img_raw[0,0,:,0,0,:,:,0]``)
and therefore no longer works. This reader opens files with
``squeeze=False`` and rearranges axes by NAME, so it does not depend on the
number or order of axes in the file.

Pixel values keep their original bit depth (the old code converted to 8-bit).
"""

from __future__ import annotations

import datetime
import io
import logging
import re
import warnings
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..recording_data_containers.image_recording import STANDARD_AXIS_ORDER, ImageRecording

logger = logging.getLogger(__name__)

METRES_TO_MICROMETRES = 1e6
SECONDS_PER_DAY = 86400.0


# =============================================================================
# Opening files and rearranging axes
# =============================================================================
def _open_czi(file_or_bytes: Path | io.BytesIO):
    """Open a .czi file (or an embedded .czi in memory) without squeezing axes."""
    import czifile  # imported here: only needed when Zeiss files are read

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return czifile.CziFile(file_or_bytes if isinstance(file_or_bytes, io.BytesIO) else str(file_or_bytes), squeeze=False)


def _first_scene(czi):
    """The first scene of the file (LSM data normally contains exactly one)."""
    scenes = czi.scenes
    if len(scenes) > 1:
        logger.warning("%s has %d scenes; only the first is read.", czi.filepath, len(scenes))
    return next(iter(scenes.values()))


def standard_shape_from_zeiss_sizes(sizes: dict[str, int]) -> tuple[int, int, int, int, int]:
    """
    Convert Zeiss dimension sizes (e.g. {'B':1,'V':1,'C':2,'T':5200,...}) into
    the standard (T, C, Z, Y, X) shape.

    Zeiss-specific dimensions (B = block, V = view, H = phase, S = RGB samples
    ...) must have length 1, except S (RGB samples) which is folded into the
    channel axis when there is only one channel (colour camera images).
    """
    sizes = dict(sizes)
    samples = sizes.pop("S", 1)
    for name, size in sizes.items():
        if name not in STANDARD_AXIS_ORDER and size != 1:
            raise ValueError(f"Unsupported Zeiss dimension '{name}' of length {size}.")
    n_channels = sizes.get("C", 1)
    if samples > 1:
        if n_channels != 1:
            raise ValueError("RGB samples together with several channels are not supported.")
        n_channels = samples
    return (sizes.get("T", 1), n_channels, sizes.get("Z", 1), sizes.get("Y", 1), sizes.get("X", 1))


def reorder_zeiss_array_to_tczyx(pixels: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    """
    Rearrange an unsqueezed Zeiss array with named ``dims`` into (T, C, Z, Y, X).

    Example: dims ('B','V','C','T','Z','Y','X','S') with shape
    (1, 1, 2, 5200, 1, 1, 128, 1)  ->  (5200, 2, 1, 1, 128)
    """
    dims = list(dims)

    # RGB camera images: move the sample axis S into the channel axis C.
    if "S" in dims and pixels.shape[dims.index("S")] > 1:
        sample_axis = dims.index("S")
        channel_axis = dims.index("C")
        pixels = np.moveaxis(pixels, sample_axis, channel_axis + 1)
        dims.insert(channel_axis + 1, dims.pop(sample_axis))
        # Merge C (length 1) and S into one channel axis.
        new_shape = list(pixels.shape)
        new_shape[channel_axis : channel_axis + 2] = [new_shape[channel_axis] * new_shape[channel_axis + 1]]
        pixels = pixels.reshape(new_shape)
        del dims[channel_axis + 1]

    # Remove every non-standard axis (all have length 1 at this point).
    for position in reversed(range(len(dims))):
        if dims[position] not in STANDARD_AXIS_ORDER:
            pixels = pixels.squeeze(axis=position)
            del dims[position]

    # Add missing standard axes, then put them in T, C, Z, Y, X order.
    for axis in STANDARD_AXIS_ORDER:
        if axis not in dims:
            pixels = pixels[..., np.newaxis]
            dims.append(axis)
    return np.transpose(pixels, [dims.index(axis) for axis in STANDARD_AXIS_ORDER])


# =============================================================================
# Metadata helpers (the metadata is one big XML document)
# =============================================================================
def _xml_text(root: ElementTree.Element, path: str) -> str | None:
    """Text of the first element matching ``path`` (ElementTree syntax), or None."""
    element = root.find(path)
    return element.text.strip() if element is not None and element.text else None


def _xml_float(root: ElementTree.Element, path: str) -> float | None:
    """Like ``_xml_text`` but converted to float (None if missing/invalid)."""
    text = _xml_text(root, path)
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _scaling_um(root: ElementTree.Element, axis: str) -> float | None:
    """
    Pixel size along an axis in micrometres.

    Zeiss stores <Scaling><Items><Distance Id="X"><Value> in METRES
    (e.g. 4.075e-07 m = 0.4075 um). A value of 0 means "not calibrated".
    """
    for distance in root.findall(".//Scaling/Items/Distance"):
        if distance.get("Id") == axis:
            value = _xml_float(distance, "Value")
            if value:  # skips None and 0
                return value * METRES_TO_MICROMETRES
    return None


def _parse_zeiss_datetime(root: ElementTree.Element) -> datetime.datetime | None:
    """
    Acquisition date. ZEN writes it to Information/Image/AcquisitionDateAndTime
    in newer versions and to Information/Document/CreationDate in older ones.
    """
    for path in (".//Information/Image/AcquisitionDateAndTime", ".//Information/Document/CreationDate"):
        text = _xml_text(root, path)
        if not text:
            continue
        # Python accepts at most 6 fractional digits; Zeiss sometimes writes 7.
        text = re.sub(r"(\.\d{6})\d+", r"\1", text).replace("Z", "+00:00")
        try:
            return datetime.datetime.fromisoformat(text)
        except ValueError:
            logger.warning("Could not parse Zeiss date '%s'.", text)
    return None


def _acquisition_modes(root: ElementTree.Element) -> list[str]:
    """All <AcquisitionMode> values, e.g. ['LaserScanningConfocalMicroscopy', 'Line']."""
    return [element.text.strip() for element in root.iter("AcquisitionMode") if element.text]


def _classify_acquisition(shape_tczyx: tuple[int, ...], modes: list[str]) -> str:
    """Name the acquisition type from its shape and the ZEN acquisition mode."""
    n_t, _, n_z, n_y, _ = shape_tczyx
    if "WideField" in modes:
        return "camera snapshot"
    if n_y == 1:
        return "line-scan time series" if n_t > 1 else "single line scan"
    if n_z > 1 and n_t > 1:
        return "z-stack time series"
    if n_z > 1:
        return "z-stack"
    if n_t > 1:
        return "time series"
    return "single frame"


def _scaled(value: float | None, factor: float) -> float | None:
    """value * factor, or None when the value is missing."""
    return None if value is None else value * factor


def _scan_geometry(root: ElementTree.Element) -> dict[str, float | None]:
    """Zoom, rotation (deg) and scanner offset (um) of a scan: where its field lies in the sample."""
    return {
        "zoom": _xml_float(root, ".//LaserScanInfo/ZoomX"),
        "scan_rotation_deg": _xml_float(root, ".//LaserScanInfo/SampleRotation"),
        "scan_offset_x_um": (_xml_float(root, ".//LaserScanInfo/SampleOffsetX") or 0.0) * METRES_TO_MICROMETRES,
        "scan_offset_y_um": (_xml_float(root, ".//LaserScanInfo/SampleOffsetY") or 0.0) * METRES_TO_MICROMETRES,
    }


def _select_metadata(root: ElementTree.Element) -> dict[str, Any]:
    """A flat dictionary of the most useful acquisition settings."""
    return {
        "acquisition_modes": _acquisition_modes(root),
        "zoom": _xml_float(root, ".//LaserScanInfo/ZoomX"),
        "pixel_dwell_time_s": _xml_float(root, ".//LaserScanInfo/PixelTime"),
        "line_time_s": _xml_float(root, ".//LaserScanInfo/LineTime"),
        "frame_time_s": _xml_float(root, ".//LaserScanInfo/FrameTime"),
        # Scan geometry: needed to locate the scanned line on the reference image.
        "scan_rotation_deg": _xml_float(root, ".//LaserScanInfo/SampleRotation"),
        "scan_offset_x_um": (_xml_float(root, ".//LaserScanInfo/SampleOffsetX") or 0.0) * METRES_TO_MICROMETRES,
        "scan_offset_y_um": (_xml_float(root, ".//LaserScanInfo/SampleOffsetY") or 0.0) * METRES_TO_MICROMETRES,
        # Focus (um): two files with the same focus, rotation and objective image
        # the same field, so one file's reference image can serve the other.
        "focus_offset_z_um": _scaled(_xml_float(root, ".//AcquisitionModeSetup/OffsetZ"), METRES_TO_MICROMETRES),
        "reference_z_um": _scaled(_xml_float(root, ".//AcquisitionModeSetup/ReferenceZ"), METRES_TO_MICROMETRES),
        "objective_model": _xml_text(root, ".//Objectives/Objective/Manufacturer/Model"),
        "objective_na": _xml_float(root, ".//Objectives/Objective/LensNA"),
        "objective_magnification": _xml_float(root, ".//Objectives/Objective/NominalMagnification"),
        "objective_immersion": _xml_text(root, ".//Objectives/Objective/Immersion"),
    }


# =============================================================================
# Attachments: timestamps, events, reference image
# =============================================================================
def _timestamps_to_relative_seconds(
    raw_timestamps: np.ndarray | None, expected_interval_s: float | None
) -> tuple[np.ndarray | None, float, float]:
    """
    Convert the raw TimeStamps attachment into seconds from the first time point.

    ZEN stores either seconds, or "OLE dates" (days since 1899-12-30), depending
    on the version. We decide by comparing the spacing of the timestamps with
    the frame/line time from the metadata.

    Returns (relative_seconds, first_raw_value, seconds_per_raw_unit), where the
    last two are needed to convert event times with the same clock.
    """
    if raw_timestamps is None or len(raw_timestamps) == 0:
        return None, 0.0, 1.0

    raw = np.asarray(raw_timestamps, dtype=np.float64)
    seconds_per_unit = 1.0
    if len(raw) > 1 and expected_interval_s:
        median_spacing = float(np.median(np.diff(raw)))
        # If the spacing only matches the expected interval after converting
        # days -> seconds, the file uses OLE dates.
        if abs(median_spacing * SECONDS_PER_DAY - expected_interval_s) < abs(median_spacing - expected_interval_s):
            seconds_per_unit = SECONDS_PER_DAY
    return (raw - raw[0]) * seconds_per_unit, float(raw[0]), seconds_per_unit


def _read_attachments(czi) -> dict[str, Any]:
    """Collect the TimeStamps, EventList and Image attachments of a file."""
    found: dict[str, Any] = {"timestamps": None, "events": [], "reference_czi_bytes": None}
    for attachment in czi.attachments():
        name = attachment.attachment_entry.name
        try:
            if name == "TimeStamps":
                found["timestamps"] = np.asarray(attachment.data())
            elif name == "EventList":
                found["events"] = [
                    {
                        "time_raw": float(event.time),
                        "event_type": getattr(event.event_type, "name", str(event.event_type)),
                        "description": str(event.description),
                    }
                    for event in attachment.data()
                ]
            elif name == "Image":
                # The overview image is itself a small .czi file inside the file.
                found["reference_czi_bytes"] = attachment.data(raw=True)
        except Exception as error:  # a damaged attachment must not block the main data
            logger.warning("Could not read attachment '%s': %s", name, error)
    return found


def _read_reference_image(czi_bytes: bytes) -> tuple[np.ndarray, float | None, dict[str, float | None]]:
    """
    Decode the embedded overview image -> ((C, Y, X) array, pixel size in um,
    its scan geometry). ZEN stores it with single line scans: the frame on which
    the line was placed (same zoom and rotation as the line, 512 x 512 px).
    """
    with _open_czi(io.BytesIO(czi_bytes)) as reference_czi:
        scene = _first_scene(reference_czi)
        pixels = reorder_zeiss_array_to_tczyx(scene.asarray(), scene.dims)
        root = ElementTree.fromstring(reference_czi.metadata())
    return pixels[0, :, 0], _scaling_um(root, "X"), _scan_geometry(root)


# =============================================================================
# Main entry point
# =============================================================================
def read_zeiss_czi_file(
    file_path: str | Path,
    source_description: str | None = None,
    original_file_name: str | None = None,
    load_data_now: bool = False,
) -> ImageRecording:
    """
    Read a Zeiss .czi file into an ``ImageRecording``.

    Parameters
    ----------
    file_path : str or Path
        The .czi file on local disk.
    source_description : str, optional
        Provenance text stored in the recording (defaults to "local: <path>").
    original_file_name : str, optional
        The real file name if ``file_path`` is a renamed copy.
    load_data_now : bool
        Read the pixels immediately instead of on first use.
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Zeiss file not found: {file_path}")

    with _open_czi(file_path) as czi:
        scene = _first_scene(czi)
        zeiss_sizes = dict(scene.sizes)
        pixel_dtype = np.dtype(scene.dtype)
        channel_names = [str(name) if name else f"Ch{i + 1}" for i, name in enumerate(getattr(scene, "channels", []) or [])]
        metadata_xml = czi.metadata()
        attachments = _read_attachments(czi)

    root = ElementTree.fromstring(metadata_xml)
    shape_tczyx = standard_shape_from_zeiss_sizes(zeiss_sizes)
    if len(channel_names) != shape_tczyx[1]:
        channel_names = [f"Ch{i + 1}" for i in range(shape_tczyx[1])]

    metadata = _select_metadata(root)
    metadata["zeiss_dimension_sizes"] = zeiss_sizes

    # ---- Time axis -------------------------------------------------------------
    timestamps_s, first_raw_time, seconds_per_raw_unit = _timestamps_to_relative_seconds(
        attachments["timestamps"], metadata["frame_time_s"]
    )
    if timestamps_s is not None:
        # Zeiss sometimes stores more timestamps than acquired time points.
        timestamps_s = timestamps_s[: shape_tczyx[0]]
    time_interval_s = None
    if shape_tczyx[0] > 1:
        if timestamps_s is not None and len(timestamps_s) > 1:
            time_interval_s = float(np.median(np.diff(timestamps_s)))
        else:
            time_interval_s = metadata["frame_time_s"]

    # ---- Events, in seconds relative to the first time point ----------------------
    events = pd.DataFrame(
        [
            {
                "time_s": (event["time_raw"] - first_raw_time) * seconds_per_raw_unit,
                "event_type": event["event_type"],
                "description": event["description"],
            }
            for event in attachments["events"]
        ],
        columns=["time_s", "event_type", "description"],
    )

    # ---- Reference (overview) image -------------------------------------------------
    reference_image, reference_pixel_size_um = None, None
    if attachments["reference_czi_bytes"] is not None:
        try:
            reference_image, reference_pixel_size_um, metadata["reference_image_geometry"] = _read_reference_image(
                attachments["reference_czi_bytes"])
        except Exception as error:
            logger.warning("Could not decode reference image of %s: %s", file_path.name, error)

    bits = _xml_float(root, ".//Information/Image/ComponentBitCount")
    z_step_um = _scaling_um(root, "Z") if shape_tczyx[2] > 1 else None

    def load_pixels() -> np.ndarray:
        """Read all pixels and rearrange them to (T, C, Z, Y, X)."""
        with _open_czi(file_path) as czi_for_pixels:
            data_scene = _first_scene(czi_for_pixels)
            return reorder_zeiss_array_to_tczyx(data_scene.asarray(), data_scene.dims)

    recording = ImageRecording(
        file_name=original_file_name or file_path.name,
        source_description=source_description or f"local: {file_path}",
        local_file_path=file_path,
        microscope="Zeiss " + ("camera" if "WideField" in metadata["acquisition_modes"] else "LSM"),
        acquisition_type=_classify_acquisition(shape_tczyx, metadata["acquisition_modes"]),
        acquisition_datetime=_parse_zeiss_datetime(root),
        shape_tczyx=shape_tczyx,
        dtype=pixel_dtype,
        bits_per_pixel=int(bits) if bits else None,
        channel_names=channel_names,
        pixel_size_x_um=_scaling_um(root, "X"),
        pixel_size_y_um=_scaling_um(root, "Y"),
        z_step_um=z_step_um,
        time_interval_s=time_interval_s,
        timestamps_s=timestamps_s,
        events=events,
        reference_image=reference_image,
        reference_image_pixel_size_um=reference_pixel_size_um,
        metadata=metadata,
        raw_metadata=metadata_xml,
        _data_loader=load_pixels,
    )

    if load_data_now:
        _ = recording.data
    return recording
