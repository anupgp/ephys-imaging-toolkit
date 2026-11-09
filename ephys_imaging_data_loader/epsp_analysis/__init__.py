"""
epsp_analysis
=============

Analysis of EPSPs recorded in current clamp.

- ``epsp_waveform_model``           double-exponential EPSP + sag model; fit; peak,
                                    20-80 % rise time, decay tau
- ``summated_epsp_deconvolution``   template from isolated EPSPs; separation of
                                    summated EPSPs (paired pulses or trains) for every ISI

Usually used through ``ExperimentCatalog.show_epsp_deconvolution(...)``.
"""

from .epsp_waveform_model import EpspMeasures, EpspParameters, epsp_waveform, fit_epsp, measure_epsp
from .summated_epsp_deconvolution import (
    EpspTemplate,
    MultiEpspFit,
    PairedEpspFit,
    SummatedEpspAnalysis,
    analyse_summated_epsps,
    build_epsp_template,
    deconvolve_summated_epsps,
    epsp_deconvolution_table,
    fit_epsp_train,
)

__all__ = [
    "epsp_waveform",
    "EpspParameters",
    "EpspMeasures",
    "fit_epsp",
    "measure_epsp",
    "EpspTemplate",
    "MultiEpspFit",
    "PairedEpspFit",
    "SummatedEpspAnalysis",
    "build_epsp_template",
    "analyse_summated_epsps",
    "deconvolve_summated_epsps",
    "epsp_deconvolution_table",
    "fit_epsp_train",
]
