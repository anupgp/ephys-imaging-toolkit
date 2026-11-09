"""
population_analysis
===================

Analysis ACROSS sets, from the combined results CSV of a batch run
(``batch_analysis.run_batch_analysis``). Three layers, each usable on its own:

    results_tables.py           CSV -> tidy tables, one value per set (the experimental unit)
    population_statistics.py    tests on those tables; every test = one row of one table
    population_figures.py       draw_*(axes, table) panels and *_figure() layouts

A driver script per experiment (e.g. codes/new_analysis/population_analysis_photostim_5spines.py)
chooses the CSV, the selections and the figures, and writes figures + statistics +
the tables behind them into one output folder.
"""

from .population_figures import measured_vs_linear_sum_figure, ppr_vs_isi_figure, pre_post_figure
from .population_statistics import (compare_to_one, paired_test, ppr_statistics, pre_post_statistics,
                                    statistics_table, summation_statistics, two_group_test)
from .results_tables import (cluster_summary, flag_suspect_responses, load_results, paired_pulse_table,
                             pre_post_pairs, summation_table)
