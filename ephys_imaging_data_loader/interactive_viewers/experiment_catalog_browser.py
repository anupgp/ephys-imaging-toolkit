"""
experiment_catalog_browser.py
=============================

An interactive chooser for recordings listed in the experiment sheet.

Layout
------
    [experiment] [date] [neuron] [dendrite] [spine] [group] [unselect filter]
    ┌──────────────────────── table of matching rows ────────────────────────┐
    │ click a row to select it                                               │
    └────────────────────────────────────────────────────────────────────────┘
    [Show selected ephys file] [Show traces by ISI for this spine] [Show imaging file]
    status line
    ── the chosen viewer appears here ──

* The dropdowns narrow down step by step: choosing a date only offers the
  neurons recorded that day, and so on. "(any)" means no filter.
* The unselect filter is off by default ("ignore column"); choose
  "unselect = 0" or "unselect = 1" to keep only those rows.
* "Show traces by ISI" needs date, neuron, dendrite and spine to be chosen;
  it respects the experiment, group and unselect filters.

Usage
-----
>>> catalog = ExperimentCatalog.from_settings()
>>> browser = catalog.browse()      # displays in a notebook cell
>>> browser.selected_rows            # the clicked row(s) as a DataFrame
"""

from __future__ import annotations

import pandas as pd
import panel as pn

from ..experiment_catalog.ephys_experiment_catalog import DISPLAY_COLUMNS, ExperimentCatalog

ANY = "(any)"
# Dropdowns, in the order in which they narrow each other down.
FILTER_COLUMNS = ["expname", "expdate", "neuronid", "dendriteid", "spineid", "group"]
FILTER_LABELS = {
    "expname": "Experiment", "expdate": "Date", "neuronid": "Neuron",
    "dendriteid": "Dendrite", "spineid": "Spine", "group": "Group",
}
UNSELECT_CHOICES = {"ignore column": "ignore", "unselect = 0": 0, "unselect = 1": 1}


class ExperimentCatalogBrowser(pn.viewable.Viewer):
    """Dropdowns + table + buttons to pick and view recordings (see module docstring)."""

    def __init__(self, catalog: ExperimentCatalog, unselect=None, page_size: int = 12, **params):
        super().__init__(**params)
        pn.extension("tabulator")
        self.catalog = catalog
        self._updating = False  # guard: avoid reacting to our own option updates

        # ---- filter widgets ---------------------------------------------------------
        self.filters: dict[str, pn.widgets.Select] = {
            column: pn.widgets.Select(label=FILTER_LABELS[column], options=[ANY], value=ANY, width=160 if column != "expname" else 330)
            for column in FILTER_COLUMNS
        }
        self.unselect_filter = pn.widgets.Select(
            label="unselect filter",
            options=UNSELECT_CHOICES,
            value="ignore" if unselect is None else unselect,
            width=140,
        )
        for widget in [*self.filters.values(), self.unselect_filter]:
            widget.param.watch(self._on_filter_change, "value")

        # ---- table and buttons -----------------------------------------------------
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=DISPLAY_COLUMNS),
            selectable=1,
            disabled=True,  # read-only view of the sheet
            pagination="local",
            page_size=page_size,
            show_index=True,
            sizing_mode="stretch_width",
        )
        self.show_file_button = pn.widgets.Button(label="Show selected ephys file", color="primary")
        self.show_isi_button = pn.widgets.Button(label="Show traces by ISI for this spine", color="success")
        self.show_image_button = pn.widgets.Button(label="Show imaging file of selected row")
        self.show_file_button.on_click(self._show_selected_file)
        self.show_isi_button.on_click(self._show_traces_by_isi)
        self.show_image_button.on_click(self._show_selected_image)

        self.status = pn.pane.Markdown("")
        self.output = pn.Column(sizing_mode="stretch_width")

        self._refresh_options(start_index=0)
        self._refresh_table()

    # =========================================================================
    # Filters
    # =========================================================================
    def _criteria(self, up_to: int | None = None) -> dict:
        """Current dropdown choices (only those before position ``up_to``)."""
        columns = FILTER_COLUMNS if up_to is None else FILTER_COLUMNS[:up_to]
        criteria = {column: self.filters[column].value for column in columns if self.filters[column].value != ANY}
        unselect = self.unselect_filter.value
        criteria["unselect"] = None if unselect == "ignore" else unselect
        return criteria

    def _refresh_options(self, start_index: int, keep_choices: bool = False) -> None:
        """
        Re-compute the choices of every dropdown from ``start_index`` on.

        A choice that is no longer possible is normally reset to "(any)" (e.g. a
        neuron from a previously chosen date). With ``keep_choices=True`` (used
        when only the unselect filter changes) the user's choices are kept even
        if nothing matches, so the table simply shows 0 rows and switching the
        filter back restores the previous view.
        """
        self._updating = True
        try:
            for index in range(start_index, len(FILTER_COLUMNS)):
                column = FILTER_COLUMNS[index]
                values = self.catalog.distinct(column, **self._criteria(up_to=index))
                options = [ANY] + [str(v) if column in ("expname", "expdate", "group") else int(v) for v in values]
                widget = self.filters[column]
                current = widget.value
                if keep_choices and current not in options:
                    options.append(current)
                widget.options = options
                widget.value = current if current in options else ANY
        finally:
            self._updating = False

    def _refresh_table(self) -> None:
        rows = self.catalog.select(**self._criteria())
        self.table.value = rows[[c for c in DISPLAY_COLUMNS if c in rows.columns]]
        self.status.object = f"{len(rows)} matching recording(s)."

    def _on_filter_change(self, event) -> None:
        if self._updating:
            return
        changed = event.obj
        if changed is self.unselect_filter:
            # Unselect filter: update all option lists but keep the user's choices.
            self._refresh_options(start_index=0, keep_choices=True)
        else:
            # A dropdown change narrows the dropdowns AFTER the changed one.
            start = next((i + 1 for i, c in enumerate(FILTER_COLUMNS) if self.filters[c] is changed), 0)
            self._refresh_options(start_index=start)
        self._refresh_table()

    # =========================================================================
    # Selection and actions
    # =========================================================================
    @property
    def selected_rows(self) -> pd.DataFrame:
        """The rows clicked in the table (full sheet columns)."""
        if not self.table.selection:
            return self.catalog.table.iloc[0:0]
        labels = self.table.value.index[self.table.selection]
        return self.catalog.table.loc[labels]

    def _selected_label(self) -> int | None:
        rows = self.selected_rows
        if rows.empty:
            self.status.object = "⚠ Click a row in the table first."
            return None
        return int(rows.index[0])

    def _run(self, description: str, action) -> None:
        """Run an action, showing progress and turning errors into a message."""
        self.status.object = f"⏳ {description} …"
        try:
            self.output.objects = [action()]
            self.status.object = f"✔ {description}"
        except Exception as error:  # show the problem instead of a traceback
            self.status.object = f"⚠ {description} failed: {error}"

    def _show_selected_file(self, _event=None) -> None:
        label = self._selected_label()
        if label is not None:
            name = self.catalog.row(label)["ephys_file_name"]
            self._run(f"Loading {name}", lambda: self.catalog.load_ephys(label).show())

    def _show_selected_image(self, _event=None) -> None:
        label = self._selected_label()
        if label is not None:
            self._run("Loading imaging file", lambda: self.catalog.load_imaging(label).show())

    def _show_traces_by_isi(self, _event=None) -> None:
        criteria = self._criteria()
        required = ["expdate", "neuronid", "dendriteid", "spineid"]
        missing = [FILTER_LABELS[c] for c in required if c not in criteria]
        if missing:
            self.status.object = f"⚠ Choose {', '.join(missing)} first (dropdowns above)."
            return
        self._run(
            "Collecting traces by ISI",
            lambda: self.catalog.show_traces_by_isi(
                criteria["expdate"], criteria["neuronid"], criteria["dendriteid"], criteria["spineid"],
                expname=criteria.get("expname"), groups=criteria.get("group"), unselect=criteria.get("unselect"),
            ),
        )

    # =========================================================================
    # Display
    # =========================================================================
    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown(f"### Experiment catalog — `{self.catalog.csv_path}` ({len(self.catalog)} rows)"),
            pn.Row(*[self.filters[c] for c in FILTER_COLUMNS[:1]], *[self.filters[c] for c in FILTER_COLUMNS[1:]], self.unselect_filter),
            self.table,
            pn.Row(self.show_file_button, self.show_isi_button, self.show_image_button),
            self.status,
            self.output,
            sizing_mode="stretch_width",
        )
