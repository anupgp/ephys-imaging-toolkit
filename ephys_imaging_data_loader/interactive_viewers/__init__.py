"""
interactive_viewers
===================

Viewers take a recording container and show it; they never read files.

Interactive (Bokeh + Panel widgets; display in notebooks, or ``.show()`` in a browser):
- ``build_ephys_trace_viewer``         stacked channels, sweeps, mean, stimulus pulses
- ``build_image_stack_viewer``         frames / z-stacks, projections, uncaging targets
- ``build_linescan_kymograph_viewer``  line-scan kymograph + ΔF/F₀ trace + stim events

Experiment sheet (see ``experiment_catalog``):
- ``build_traces_by_isi_viewer``       one spine, one panel per ISI, averaged sweeps
- ``build_experiment_traces_viewer``   all traces of expdate/neuron/dendrite[/spine], optional averaging
- ``build_epsp_deconvolution_figure``  summated-EPSP deconvolution: template, per-ISI fits, summary
- ``ExperimentCatalogBrowser``         dropdowns + table chooser (import it from
                                       ``interactive_viewers.experiment_catalog_browser``;
                                       usually via ``catalog.browse()``)

Static files (matplotlib; .png / .svg / .pdf):
- ``save_ephys_traces_figure``, ``save_image_figure``, ``save_kymograph_figure``,
  ``save_traces_by_isi_figure``

Usually you do not need to import these: ``recording.show()`` picks the
right interactive viewer automatically.
"""

from .ephys_trace_viewer import build_ephys_trace_figure, build_ephys_trace_viewer
from .image_stack_viewer import build_image_figure, build_image_stack_viewer
from .linescan_kymograph_viewer import build_kymograph_figure, build_linescan_kymograph_viewer
from .save_static_figure import save_ephys_traces_figure, save_image_figure, save_kymograph_figure
from .epsp_deconvolution_figure import build_epsp_deconvolution_figure, save_epsp_deconvolution_figures
from .experiment_traces_viewer import build_experiment_traces_viewer, save_experiment_traces_figure
from .traces_by_isi_viewer import build_traces_by_isi_figure, build_traces_by_isi_viewer, save_traces_by_isi_figure

__all__ = [
    "build_ephys_trace_viewer",
    "build_ephys_trace_figure",
    "build_image_stack_viewer",
    "build_image_figure",
    "build_linescan_kymograph_viewer",
    "build_kymograph_figure",
    "save_ephys_traces_figure",
    "save_image_figure",
    "save_kymograph_figure",
    "build_traces_by_isi_viewer",
    "build_traces_by_isi_figure",
    "save_traces_by_isi_figure",
    "build_experiment_traces_viewer",
    "save_experiment_traces_figure",
    "build_epsp_deconvolution_figure",
    "save_epsp_deconvolution_figures",
]
