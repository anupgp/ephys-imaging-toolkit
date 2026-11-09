"""
batch_runner.py
===============

Runs the per-set analysis (``epsp_analysis.unitary_cluster_summation``: the
deconvolution of every single spine and cluster over its ISIs + the spatial
summation of every cluster) for one set, a list of sets, a text file of sets,
or every set in the sheet, and saves everything.

Output layout
-------------
    <output_dir>/
        all_sets.csv                       the rows of EVERY set of this run (one table), + sheet_* columns
                                           (region, area, side, age, temp, accessres ... of the files used;
                                           see sheet_columns.py)
        all_sets_summation.png             measured vs linear sum for all sets
        batch_log.csv                      per set: done / skipped / failed, time, message
        20260322_1_1_post-apv/             one folder per set
            20260322_1_1_post-apv.csv          ONE table: every input x ISI x stimulus
                                               (stimid; deconvolution + 1st response + linear
                                               sum + cluster spread + laser power), fixed columns
            20260322_1_1_post-apv_summation.png        clusters vs linear sum, spine map, PPR vs ISI
            20260322_1_1_post-apv_deconvolution_cluster_all5.png   one per input with >= 2 ISIs
            20260322_1_1_post-apv_notes.txt            warnings and assumptions

Use
---
    >>> catalog.run_batch_analysis()                                   # every set in the sheet
    >>> catalog.run_batch_analysis((20260322, 1, 1))                   # all groups of one dendrite
    >>> catalog.run_batch_analysis([(20260322, 1, 1, "post-apv"), "20270313 1 1"])
    >>> catalog.run_batch_analysis("my_sets.txt", output_dir="results_2026-10-15")

or from a terminal (see ``__main__.py``):

    python -m ephys_imaging_data_loader.batch_analysis --sets my_sets.txt --output results

A set whose folder already holds its CSV is not analysed again (its CSV is
still read into all_sets.csv) unless ``overwrite=True``. A set that fails is
written to batch_log.csv and the run goes on.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import pandas as pd

from ..epsp_analysis.unitary_cluster_summation import analyse_set, conform_set_table, set_summation_table
from .sheet_columns import add_sheet_columns
from ..epsp_analysis.ppr_recovery import add_ppr_recovery_columns, ppr_recovery_figure
from .set_selection import SetKey, resolve_sets

logger = logging.getLogger(__name__)

ANALYSIS_OPTIONS = ("threshold_isi_ms", "summation_isi_ms", "template_events", "template_length_ms", "after_ms",
                    "uncaging_offset")


def image_on_spine_positions(catalog):
    """
    Function recording -> 1-based scan positions of its laser points that are on
    spines, read from the recording's image with ``uncaging_target_labels``
    (shrinking clusters: points still at their full-cluster position); None when
    the recording has no image or the image does not resolve it. Results are cached.
    """
    from ..experiment_catalog.uncaging_target_labels import label_uncaging_targets

    cache: dict = {}

    def positions(recording):
        row_label = recording.traces.catalog_row
        if row_label not in cache:
            cache[row_label] = None
            if pd.notna(recording.value("ophysfile")):
                labels = label_uncaging_targets(catalog.load_imaging(row_label), catalog)
                if labels.resolved:
                    cache[row_label] = [i + 1 for i, on in enumerate(labels.on_spine) if on]
        return cache[row_label]

    return positions


def analyse_set_to_folder(
    catalog,
    key: SetKey,
    output_dir: str | Path = "set_analysis_results",
    remove_unselected: bool = True,
    remove_bad_sweeps: bool = True,
    deconvolution_figures: str = "multi_isi",
    dpi: int = 150,
    use_images: bool = True,
    **analysis_options,
) -> tuple[pd.DataFrame, list[Path]]:
    """
    Analyse one set and write its folder (see module docstring). Returns (table, written files).

    ``deconvolution_figures``: "multi_isi" (inputs recorded at >= 2 ISIs), "all" or "none".
    ``use_images``: read from each recording's image which laser points were on
    spines (timing of shrinking clusters), measure the cluster spread and draw
    the spine-map panel; False = assume spine k = point k, no spread, no image panel.
    ``analysis_options``: threshold_isi_ms (200), summation_isi_ms (1000), after_ms (300),
    template_events, template_length_ms, uncaging_offset (True).
    """
    from ..interactive_viewers.epsp_deconvolution_figure import save_epsp_deconvolution_figures
    from ..experiment_catalog.set_spine_map_image import cluster_spreads, spine_map_for_set
    from ..interactive_viewers.spine_summation_figure import build_set_summation_figure, save_figure

    after_ms = analysis_options.get("after_ms", 300.0)
    data = catalog.experiment_traces(key.expdate, key.neuronid, key.dendriteid, group=key.group, after_ms=after_ms,
                                     remove_unselected=remove_unselected, remove_bad_sweeps=remove_bad_sweeps)
    data.criteria.update(expdate=key.expdate, neuronid=key.neuronid, dendriteid=key.dendriteid)
    if use_images:
        analysis_options.setdefault("on_spine_positions", image_on_spine_positions(catalog))
    analyses = analyse_set(data, **analysis_options)
    image_cache: dict = {}                          # images loaded once for the spine map and the spreads
    spine_maps = {}
    image_notes = []
    for analysis in analyses:
        analysis.set_info["group"] = key.group      # the sheet's group as asked for
        if use_images:                              # never stops the analysis
            try:                                    # spread of every cluster's spines (µm)
                analysis.cluster_spread = cluster_spreads(catalog, analysis, image_cache)
            except Exception as error:
                image_notes.append(f"cluster spread not measured ({error})")
            try:                                    # image of the dendrite with the single spines' ids
                spine_maps[analysis.unit] = spine_map_for_set(catalog, analysis, image_cache)
            except Exception as error:
                image_notes.append(f"spine map image not drawn ({error})")

    folder = Path(output_dir) / key.name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{key.name}*"):          # files of an earlier run of this set
        old.unlink()
    written = []
    table = add_ppr_recovery_columns(set_summation_table(analyses))     # + ppr_fit_* (PPR vs ISI fit per input)
    table["set"] = key.name
    csv_path = folder / f"{key.name}.csv"
    table.to_csv(csv_path, index=False)
    written.append(csv_path)

    several_units = len(analyses) > 1
    notes = list(data.warnings) + image_notes
    path = save_figure(ppr_recovery_figure(table, f"{key.name}: paired-pulse ratio against ISI, fit per input"),
                       folder / f"{key.name}_ppr_fit.png", dpi)
    if path:
        written.append(path)
    for analysis in analyses:
        suffix = f"_{analysis.unit}" if several_units else ""
        path = save_figure(build_set_summation_figure(analysis, spine_map=spine_maps.get(analysis.unit)),
                           folder / f"{key.name}_summation{suffix}.png", dpi)
        if path:
            written.append(path)
        for spine_input in analysis.inputs:
            deconvolution = analysis.deconvolutions.get(spine_input.label)
            n_isis = len({fit.isi_ms for fit in deconvolution.fits}) if deconvolution else 0
            wanted = {"all": n_isis >= 1, "multi_isi": n_isis >= 2, "none": False}[deconvolution_figures]
            if wanted:
                name = f"{key.name}_deconvolution_{spine_input.file_label}{suffix}.png"
                written += save_epsp_deconvolution_figures([deconvolution], folder / name, dpi=dpi)
            notes += [f"{spine_input.label}: {note}" for note in spine_input.notes]
    notes_path = folder / f"{key.name}_notes.txt"
    notes_path.write_text("\n".join(dict.fromkeys(notes)) + "\n" if notes else "No warnings.\n")
    written.append(notes_path)
    return table, written


def run_batch_analysis(
    catalog,
    sets=None,
    output_dir: str | Path = "set_analysis_results",
    expnames: list[str] | None = None,
    overwrite: bool = False,
    combined_name: str = "all_sets",
    remove_unselected: bool = True,
    remove_bad_sweeps: bool = True,
    deconvolution_figures: str = "multi_isi",
    dpi: int = 150,
    verbose: bool = True,
    use_images: bool = True,
    **analysis_options,
) -> pd.DataFrame:
    """
    Analyse many sets (see module docstring) and return the combined table.

    ``sets``: None (all sets in the sheet), one set, a list of sets or a text-file
    path — see ``set_selection.resolve_sets``. ``expnames``: only rows of these
    experiment names when looking up sets / groups.
    """
    from ..interactive_viewers.spine_summation_figure import build_batch_summation_figure, save_figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = resolve_sets(catalog, sets, expnames)
    tables, log = [], []
    for number, key in enumerate(keys, start=1):
        started = time.time()
        csv_path = output_dir / key.name / f"{key.name}.csv"
        entry = {"set": key.name, "expdate": key.expdate, "neuronid": key.neuronid, "dendriteid": key.dendriteid,
                 "group": key.group}
        try:
            if csv_path.exists() and not overwrite:
                table = pd.read_csv(csv_path, dtype={"expdate": str})
                entry.update(status="skipped (already analysed)", message="")
            else:
                table, _ = analyse_set_to_folder(catalog, key, output_dir, remove_unselected, remove_bad_sweeps,
                                                 deconvolution_figures, dpi, use_images, **analysis_options)
                entry.update(status="done", message="")
            tables.append(table)
            entry.update(n_rows=len(table),
                         n_clusters_with_linear_sum=int(table.get("linear_sum_peak", pd.Series(dtype=float)).notna().sum()))
        except Exception as error:     # one set never stops the batch
            entry.update(status="failed", message=f"{type(error).__name__}: {error}")
            logger.debug(traceback.format_exc())
        entry["seconds"] = round(time.time() - started, 1)
        log.append(entry)
        if verbose:
            print(f"[{number}/{len(keys)}] {key.name}: {entry['status']}"
                  + (f" — {entry['message']}" if entry["message"] else ""), flush=True)

    # same columns and order for every set (CSVs of older runs: their extra columns go last)
    combined = conform_set_table(pd.concat(tables, ignore_index=True) if tables else pd.DataFrame())
    if not combined.empty and "files" in combined:     # + sheet facts (region, area, age, temp ...) per row
        combined = add_sheet_columns(combined, catalog.table)
    combined.to_csv(output_dir / f"{combined_name}.csv", index=False)
    pd.DataFrame(log).to_csv(output_dir / "batch_log.csv", index=False)
    if not combined.empty:
        save_figure(build_batch_summation_figure(combined), output_dir / f"{combined_name}_summation.png", dpi)
    if verbose:
        counts = pd.Series([e["status"].split(" ")[0] for e in log]).value_counts().to_dict()
        print(f"{len(keys)} sets: {counts}. Combined table: {(output_dir / f'{combined_name}.csv').resolve()}")
    return combined
