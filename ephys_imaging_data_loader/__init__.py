"""
ephys_imaging_data_loader
=========================

Load and view patch-clamp and two-photon imaging data with ONE interface,
from local folders or from Box.

Supported files
---------------
    .abf          Axon pClamp (patch-clamp)        -> EphysRecording
    .czi          Zeiss ZEN / LSM880               -> ImageRecording
    .oib / .oif   Olympus FluoView FV1000          -> ImageRecording

Quick start
-----------
>>> from ephys_imaging_data_loader import load_recording, list_recording_files
>>>
>>> # a local file
>>> ephys = load_recording("/path/to/26322010.abf")
>>>
>>> # the same file by root + relative path, from local disk or from Box
>>> ephys = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="local")
>>> ephys = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="box")
>>>
>>> print(ephys.summary())
>>> ephys.show()          # interactive viewer in a notebook
>>>
>>> # choose recordings from the experiment sheet
>>> catalog = ExperimentCatalog.from_settings()
>>> catalog.browse()                                             # interactive chooser
>>> catalog.show_traces_by_isi(20260322, neuronid=1, dendriteid=1, spineid=0)
>>> catalog.show_experiment_traces(20250125, 1, 1, spineid=3, average_by="isi")   # all traces, mean per ISI
>>> catalog.show_epsp_deconvolution(20260322, 1, 1, 0, group="pre-apv")            # separate summated EPSPs

Package layout
--------------
    recording_data_containers/  EphysRecording, ImageRecording (standard containers)
    file_readers/               one reader per file format
    data_sources/               local folders, Box + download cache, settings file
    experiment_catalog/         choose/load recordings via the experiment sheet (CSV); traces by ISI
    epsp_analysis/              EPSP model (double exponential + sag); summated-EPSP deconvolution
    interactive_viewers/        Bokeh/Panel viewers and static matplotlib figures
    load_recording.py           load_recording(), list_recording_files()
"""

from .data_sources import (
    BoxCloudDataSource,
    BoxDownloadCache,
    LocalFolderDataSource,
    LocalThenBoxDataSource,
    load_settings,
    open_data_source,
)
from .experiment_catalog import ExperimentCatalog
from .file_readers import read_recording_file
from .load_recording import get_data_source, list_recording_files, load_recording
from .recording_data_containers import EphysRecording, ImageRecording

__version__ = "0.1.0"

__all__ = [
    "load_recording",
    "list_recording_files",
    "get_data_source",
    "open_data_source",
    "load_settings",
    "ExperimentCatalog",
    "read_recording_file",
    "EphysRecording",
    "ImageRecording",
    "LocalFolderDataSource",
    "BoxCloudDataSource",
    "BoxDownloadCache",
    "LocalThenBoxDataSource",
]
