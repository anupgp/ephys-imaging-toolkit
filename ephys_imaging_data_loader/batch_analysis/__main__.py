"""
Batch analysis from a terminal.

Examples
--------
    # every set in the sheet
    python -m ephys_imaging_data_loader.batch_analysis --output results

    # some sets, given directly (group optional) and/or in text files
    python -m ephys_imaging_data_loader.batch_analysis --sets "20260322,1,1" "20270313,1,1,ctrl" --output results
    python -m ephys_imaging_data_loader.batch_analysis --sets my_sets.txt --output results --overwrite

    # only some experiment types
    python -m ephys_imaging_data_loader.batch_analysis --expname photostim_5spines_apv_ca1apical_epsp

The data source (local / box / auto) and the sheet come from the settings file
(data_source_settings.toml), as for ``ExperimentCatalog.from_settings``.
"""

from __future__ import annotations

import argparse

from ..experiment_catalog.ephys_experiment_catalog import ExperimentCatalog
from .batch_runner import run_batch_analysis


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ephys_imaging_data_loader.batch_analysis",
                                     description="Deconvolution + spatial summation for many sets, one folder each.")
    parser.add_argument("--sets", nargs="*", default=None,
                        help="sets 'expdate,neuronid,dendriteid[,group]' and/or text files (one set per line); "
                             "default: every set in the sheet")
    parser.add_argument("--output", default="set_analysis_results", help="results folder")
    parser.add_argument("--expname", nargs="*", default=None, help="only these experiment names")
    parser.add_argument("--overwrite", action="store_true", help="analyse again sets that already have results")
    parser.add_argument("--threshold-isi", type=float, default=200.0, help="ISI (ms) from which responses are isolated")
    parser.add_argument("--summation-isi", type=float, default=1000.0, help="ISI (ms) used for the linear sum")
    parser.add_argument("--no-uncaging-offset", action="store_true", help="do not shift spines by their uncaging time")
    parser.add_argument("--keep-unselected", action="store_true", help="also use files with unselect = 1")
    parser.add_argument("--keep-bad-sweeps", action="store_true", help="also use the sweeps listed in badsweeps")
    parser.add_argument("--mode", default=None, help="data source mode: local, box or auto (default: settings file)")
    parser.add_argument("--settings", default=None, help="settings file (default: data_source_settings.toml)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    catalog = ExperimentCatalog.from_settings(mode=args.mode, settings_file=args.settings)
    run_batch_analysis(
        catalog, args.sets or None, args.output, expnames=args.expname, overwrite=args.overwrite,
        remove_unselected=not args.keep_unselected, remove_bad_sweeps=not args.keep_bad_sweeps, dpi=args.dpi,
        threshold_isi_ms=args.threshold_isi, summation_isi_ms=args.summation_isi,
        uncaging_offset=not args.no_uncaging_offset,
    )


if __name__ == "__main__":
    main()
