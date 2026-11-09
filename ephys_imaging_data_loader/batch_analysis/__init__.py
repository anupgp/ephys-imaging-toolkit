"""
batch_analysis
==============

Analysis of many experiment "sets" (expdate + neuronid + dendriteid + group) in
one run, each set in its own results folder, plus one combined table.

- ``set_selection``   which sets: one, a list, a text file, or all in the sheet
- ``batch_runner``    per-set folder (one CSV + figures) and the batch run
- ``__main__``        the same from a terminal:
                      python -m ephys_imaging_data_loader.batch_analysis --help

Usually used through ``ExperimentCatalog.analyse_set(...)`` and
``ExperimentCatalog.run_batch_analysis(...)``.
"""

from .batch_runner import analyse_set_to_folder, run_batch_analysis
from .set_selection import SetKey, parse_set_text, read_set_file, resolve_sets

__all__ = ["SetKey", "parse_set_text", "read_set_file", "resolve_sets", "analyse_set_to_folder",
           "run_batch_analysis"]
