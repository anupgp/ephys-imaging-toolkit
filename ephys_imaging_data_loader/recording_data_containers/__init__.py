"""
recording_data_containers
=========================

The two standard containers that every file reader produces:

- ``EphysRecording``  — patch-clamp signals, shape (sweeps, channels, samples)
- ``ImageRecording``  — microscope pixels, always shape (T, C, Z, Y, X)

Analysis and plotting code should only depend on these two classes, never on
a specific file format or microscope vendor.
"""

from .ephys_recording import EphysRecording
from .image_recording import STANDARD_AXIS_ORDER, ImageRecording

__all__ = ["EphysRecording", "ImageRecording", "STANDARD_AXIS_ORDER"]
