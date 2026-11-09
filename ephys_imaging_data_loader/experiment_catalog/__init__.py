"""
experiment_catalog
==================

Choose recordings through the experiment sheet (the experiment sheet (CSV))
instead of by file name.

- ``ExperimentCatalog``           read / select / load rows; ``browse()`` for an
                                  interactive chooser; traces grouped by ISI;
                                  all traces of one experiment sample (show_experiment_traces)
- ``find_stimulus_onsets``        when was the stimulus delivered in a sweep?
- ``collect_traces_by_isi`` /
  ``TracesByIsi``                 aligned traces of one spine, per ISI
- ``build_ephys_file_name`` /
  ``parse_bad_sweeps``            how sheet columns become file names / sweep lists
"""

from .catalog_file_names import build_ephys_file_name, parse_bad_sweeps
from .ephys_experiment_catalog import ExperimentCatalog
from .stimulus_onset_detection import StimulusDetectionError, StimulusOnsets, find_stimulus_onsets
from .experiment_traces import ExperimentTraces, collect_experiment_traces
from .traces_by_isi import IsiTraceSet, TracesByIsi, collect_traces_by_isi, default_time_window_ms

__all__ = [
    "ExperimentCatalog",
    "find_stimulus_onsets",
    "StimulusOnsets",
    "StimulusDetectionError",
    "collect_traces_by_isi",
    "collect_experiment_traces",
    "ExperimentTraces",
    "TracesByIsi",
    "IsiTraceSet",
    "default_time_window_ms",
    "build_ephys_file_name",
    "parse_bad_sweeps",
]
