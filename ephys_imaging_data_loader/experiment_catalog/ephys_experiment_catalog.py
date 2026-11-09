"""
ephys_experiment_catalog.py
===========================

``ExperimentCatalog``: choose and load recordings using the experiment sheet
(the experiment sheet (CSV)) instead of typing file names.

The sheet has one row per ephys recording (file) with the experiment name,
date, neuron / dendrite / spine ids, the file id, response and stimulus
channels, ISI, group (e.g. pre-apv / post-apv), bad sweeps and more.

Typical use
-----------
>>> from ephys_imaging_data_loader import ExperimentCatalog
>>> catalog = ExperimentCatalog.from_settings()              # sheet + roots from settings
>>> catalog.browse()                                          # interactive chooser
>>> rows = catalog.select(expdate=20260322, neuronid=1, dendriteid=1, spineid=0)
>>> recording = catalog.load_ephys(rows.index[0])             # bad sweeps already applied
>>> catalog.show_traces_by_isi(20260322, neuronid=1, dendriteid=1, spineid=0, isis=[30, 100, 1000])

The sheet is only READ; it is never modified. Values are cleaned in memory:
ids become whole numbers, "badsweeps" becomes a list, and the ephys file name
is built from ``ephysfile_prefix`` + ``ephysid`` (see ``catalog_file_names.py``).

Bad sweeps
----------
The sheet lists bad sweeps like "1;2;3;4". Clampex numbers sweeps from 1, and
the sheet was checked against the files (e.g. "1;2;3;4" for 4-sweep files), so
by default the numbers are read as "counting from 1" (``badsweeps_numbering=
"from_1"``) and converted to 0-based Python indices. Numbers that do not exist
in the file (e.g. 0, or 4 in a 3-sweep file) are ignored with a warning.

The "unselect" column
---------------------
It is NOT used unless you ask: ``select(..., unselect=0)`` keeps rows with
unselect == 0, ``unselect=[0, 1]`` keeps both, and ``unselect=None`` (default)
ignores the column. ``experiment_traces(..., remove_unselected=True)`` leaves out
rows with unselect == 1 (empty or 0 kept); the EPSP deconvolution does this by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data_sources.base_data_source import DataSource
from ..data_sources.data_source_settings import load_settings
from ..file_readers.choose_reader_by_file_type import read_recording_file
from ..recording_data_containers import EphysRecording, ImageRecording
from .catalog_file_names import build_ephys_file_name, build_ophys_file_names, parse_bad_sweeps
from .experiment_traces import ExperimentTraces, collect_experiment_traces
from .traces_by_isi import TracesByIsi, collect_traces_by_isi

logger = logging.getLogger(__name__)

INTEGER_COLUMNS = ["neuronid", "dendriteid", "spineid", "ephysid", "ophysid", "unselect", "clustersize"]
NUMBER_COLUMNS = ["isi", "stimpower", "stimpower_from_ophysfile", "stimpower_laser_metadata", "stimdur", "accessres", "temp"]
# Columns shown by default in tables (e.g. in the browser).
DISPLAY_COLUMNS = [
    "expname", "expdate", "neuronid", "dendriteid", "spineid", "group", "isi", "ephys_file_name",
    "ophysfile", "chan_resp", "chan_stim", "patchclamp", "stim_power_percent", "badsweeps", "unselect",
]


_default_catalogs: dict[str, "ExperimentCatalog | None"] = {}


def get_default_catalog(settings_file: str | Path | None = None) -> "ExperimentCatalog | None":
    """
    The catalog from the settings file, read once per session; None if no
    [catalog] is configured or the sheet cannot be read. Used to add sheet
    details (area, side, age ...) to viewer titles of any loaded recording.
    """
    key = str(settings_file or "")
    if key not in _default_catalogs:
        try:
            _default_catalogs[key] = ExperimentCatalog.from_settings(settings_file=settings_file)
        except Exception as error:  # no settings / no catalog: titles simply omit sheet details
            logger.debug("No default experiment catalog: %s", error)
            _default_catalogs[key] = None
    return _default_catalogs[key]


def _spine_text(value) -> str:
    """Sheet id (spineid, dendriteid ...) as text: 12345, '?' when empty."""
    return str(int(value)) if pd.notna(value) else "?"


def _as_list(value) -> list | None:
    """None -> None, scalar -> [scalar], list/tuple/set -> list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return list(value)
    return [value]


class ExperimentCatalog:
    """The experiment sheet, cleaned, with functions to select, load and plot rows."""

    def __init__(
        self,
        table: pd.DataFrame,
        ephys_root: str,
        imaging_root: str | None = None,
        source: DataSource | None = None,
        mode: str | None = None,
        settings_file: str | Path | None = None,
        badsweeps_numbering: str = "from_1",
        csv_path: str | Path | None = None,
    ):
        """
        Usually created with ``from_settings()`` or ``from_csv()``.

        Parameters
        ----------
        table : the raw sheet as read by pandas (all columns as text)
        ephys_root / imaging_root : data roots (from the settings) that contain
            "<expdate>/c<neuronid>/<file>"
        source / mode / settings_file : where files are loaded from, exactly as
            in ``load_recording`` ("local", "box", "auto" or a DataSource)
        badsweeps_numbering : "from_1" (Clampex style, default) or "from_0"
        """
        if badsweeps_numbering not in ("from_1", "from_0"):
            raise ValueError("badsweeps_numbering must be 'from_1' or 'from_0'")
        self.ephys_root = ephys_root
        self.imaging_root = imaging_root
        self.badsweeps_numbering = badsweeps_numbering
        self.csv_path = Path(csv_path) if csv_path else None
        self._source = source
        self._mode = mode
        self._settings_file = settings_file
        self.table = self._clean(table)

    # =========================================================================
    # Construction
    # =========================================================================
    @classmethod
    def from_csv(cls, csv_path: str | Path, ephys_root: str, imaging_root: str | None = None, **options) -> "ExperimentCatalog":
        """Read the sheet from a CSV file (read-only)."""
        csv_path = Path(csv_path).expanduser()
        # Everything as text first; types are fixed in _clean (ids like "007" stay intact).
        table = pd.read_csv(csv_path, dtype=str, skipinitialspace=True)
        return cls(table, ephys_root, imaging_root, csv_path=csv_path, **options)

    @classmethod
    def from_settings(cls, mode: str | None = None, settings_file: str | Path | None = None, **options) -> "ExperimentCatalog":
        """Use the [catalog] section of data_source_settings.toml."""
        settings = load_settings(settings_file)
        if not settings.catalog_csv:
            raise ValueError(
                "No [catalog] experiment_details_csv in the settings file "
                f"({settings.settings_file}); see data_source_settings.example.toml."
            )
        return cls.from_csv(
            settings.catalog_csv,
            ephys_root=settings.catalog_ephys_root,
            imaging_root=settings.catalog_imaging_root or None,
            mode=mode,
            settings_file=settings.settings_file,
            **options,
        )

    def _clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Tidy types and add derived columns (in memory only)."""
        table = raw.copy()
        table.columns = [str(c).strip() for c in table.columns]
        for column in table.columns:
            # pandas >= 3 reads text as its "str" dtype, older versions as object.
            if pd.api.types.is_string_dtype(table[column]) or table[column].dtype == object:
                table[column] = table[column].str.strip()

        # Date as "YYYYMMDD" text (tolerates "20260322.0").
        table["expdate"] = table["expdate"].str.replace(r"\.0$", "", regex=True)
        for column in INTEGER_COLUMNS:
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce").round().astype("Int64")
        for column in NUMBER_COLUMNS:
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce")

        # Ephys file name from prefix + id (or inferred from the date).
        names = [
            build_ephys_file_name(row.get("ephysfile_prefix"), row.get("ephysid"), row.get("expdate"))
            for row in table.to_dict("records")
        ]
        table["ephys_file_name"] = [name for name, _ in names]
        table["ephys_file_name_source"] = [how for _, how in names]
        table["badsweeps_list"] = table.get("badsweeps", pd.Series(index=table.index, dtype=object)).map(parse_bad_sweeps)

        # Imaging file names. Newer sheets: "ophysfile_prefix" ("Imagexxxx") +
        # "ophysid" (one id or "12;13"). Older sheets had an "ophysfile" column
        # with the names themselves. Either way, an "ophysfile" text column
        # ("Image0012.oib;Image0013.oib") is provided for the rest of the code.
        if "ophysfile_prefix" in table.columns and "ophysid" in raw.columns:
            names = [
                build_ophys_file_names(prefix, ids)
                for prefix, ids in zip(table["ophysfile_prefix"], raw["ophysid"].astype("string").str.strip())
            ]
            table["ophysfile"] = [";".join(n) if n else pd.NA for n in names]

        # Uncaging power actually used: the value read from the imaging file's laser
        # metadata (column "stimpower_laser_metadata") is authoritative; the typed-in
        # "stimpower" is only used where the metadata value is missing.
        # (column name used since 6 Oct 2026: "stimpower_from_ophysfile"; earlier "stimpower_laser_metadata")
        power_columns = [c for c in ("stimpower_from_ophysfile", "stimpower_laser_metadata", "stimpower") if c in table.columns]
        if power_columns:
            table["stim_power_percent"] = table[power_columns[0]]
            for fallback in power_columns[1:]:
                table["stim_power_percent"] = table["stim_power_percent"].fillna(table[fallback])

        # Channels to use: the file-verified columns when present (added to the sheet
        # after checking every .abf file), else the typed-in chan_resp / chan_stim.
        for derived, verified, typed in (("response_channel", "chan_resp_verified", "chan_resp"),
                                         ("stimulus_channel", "chan_stim_verified", "chan_stim")):
            columns = [c for c in (verified, typed) if c in table.columns]
            if columns:
                table[derived] = table[columns[0]]
                for fallback in columns[1:]:
                    table[derived] = table[derived].fillna(table[fallback])

        # Age on the day of the experiment, in days (postnatal day), from the date of birth.
        if "dob" in table.columns:
            experiment_day = pd.to_datetime(table["expdate"], format="%Y%m%d", errors="coerce")
            birth_day = pd.to_datetime(table["dob"].str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
            table["age_days"] = (experiment_day - birth_day).dt.days.astype("Int64")
        return table

    def rows_for_file(self, file_name: str) -> pd.DataFrame:
        """
        Sheet rows describing an ephys file, found by name (e.g. "26322023.abf"
        or "26322023"; case is ignored). Clampex names contain the date, so a
        name identifies one recording.
        """
        stem = Path(str(file_name)).stem.lower()
        return self.table[self.table["ephys_file_name"].str.lower() == stem]

    # =========================================================================
    # Selecting rows
    # =========================================================================
    def select(
        self,
        expname=None,
        expdate=None,
        neuronid=None,
        dendriteid=None,
        spineid=None,
        isi=None,
        group=None,
        unselect=None,
        only_with_ephys_file: bool = True,
        **other_columns,
    ) -> pd.DataFrame:
        """
        Rows matching ALL given criteria. Each criterion is a value or a list
        of values; None means "any".

        Examples
        --------
        >>> catalog.select(expdate=20260322, neuronid=1, dendriteid=1, spineid=0)
        >>> catalog.select(expname="photostim_5spines_apv_ca1apical_epsp", isi=[30, 50])
        >>> catalog.select(expdate="20260322", group="post-apv", unselect=0)
        >>> catalog.select(area="ca1", side="apical")          # any other column

        ``unselect``: None = ignore the column; a value / list = keep only rows
        with that value (see module docstring).
        ``only_with_ephys_file``: drop rows whose file name cannot be built.
        """
        criteria = dict(
            expname=expname, expdate=expdate, neuronid=neuronid, dendriteid=dendriteid,
            spineid=spineid, isi=isi, group=group, unselect=unselect, **other_columns,
        )
        mask = pd.Series(True, index=self.table.index)
        for column, wanted in criteria.items():
            values = _as_list(wanted)
            if values is None:
                continue
            if column not in self.table.columns:
                raise KeyError(f"Column '{column}' is not in the sheet; columns: {list(self.table.columns)}")
            if column == "expdate":
                values = [str(v).replace(".0", "") for v in values]
            elif column in INTEGER_COLUMNS + NUMBER_COLUMNS:
                values = [float(v) for v in values]
                mask &= self.table[column].astype("Float64").isin(values).fillna(False)
                continue
            mask &= self.table[column].isin([str(v) for v in values])
        if only_with_ephys_file:
            mask &= self.table["ephys_file_name"].notna()
        return self.table[mask]

    def distinct(self, column: str, **criteria) -> list:
        """Sorted distinct values of a column among rows matching ``criteria``."""
        values = self.select(only_with_ephys_file=False, **criteria)[column].dropna().unique()
        try:
            return sorted(values, key=lambda v: float(v))
        except (TypeError, ValueError):
            return sorted(values, key=str)

    def row(self, row_label: int) -> pd.Series:
        """One row by its label (the number shown in the index / browser table)."""
        return self.table.loc[row_label]

    # =========================================================================
    # Loading files
    # =========================================================================
    def _loader_kwargs(self) -> dict[str, Any]:
        return {"source": self._source, "mode": self._mode, "settings_file": self._settings_file}

    def ephys_relative_path(self, row_label: int) -> str:
        """'<expdate>/c<neuronid>/<file>.abf' for a row."""
        row = self.row(row_label)
        if not row["ephys_file_name"]:
            raise ValueError(f"Row {row_label}: no ephys file name (ephysfile_prefix / ephysid empty).")
        return f"{row['expdate']}/c{int(row['neuronid'])}/{row['ephys_file_name']}.abf"

    def _fetch_and_read(self, root: str, relative_path: str):
        from ..load_recording import get_data_source

        source = self._source or get_data_source(self._mode, self._settings_file)
        fetched = source.fetch_file(root, relative_path)
        return read_recording_file(
            fetched.local_path,
            source_description=fetched.source_description,
            original_file_name=fetched.original_file_name,
        )

    def load_ephys(self, row_label: int, apply_bad_sweeps: bool = True) -> EphysRecording:
        """
        Load the .abf file of a row, with the sheet's bad sweeps applied
        (``apply_bad_sweeps=False``: the "badsweeps" column is ignored, all sweeps are good).

        The sheet row is stored in ``recording.header_metadata["catalog_row"]``
        and any problems (e.g. a bad-sweep number outside the file) in
        ``recording.header_metadata["catalog_warnings"]``.
        """
        row = self.row(row_label)
        relative_path = self.ephys_relative_path(row_label)
        try:
            recording = self._fetch_and_read(self.ephys_root, relative_path)
        except FileNotFoundError as error:
            if row["ephys_file_name_source"] != "inferred from date":
                raise
            # The name was guessed from the date in the OLD Clampex style
            # (e.g. "27409007"). Newer Clampex versions name files
            # "2027_04_09_0007", so try that before giving up.
            date = str(row["expdate"])
            new_style = f"{date[:4]}_{date[4:6]}_{date[6:8]}_{int(row['ephysid']):04d}"
            try:
                recording = self._fetch_and_read(
                    self.ephys_root, f"{date}/c{int(row['neuronid'])}/{new_style}.abf"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{error} (no ephysfile_prefix in the sheet; tried the date-based names "
                    f"{row['ephys_file_name']}.abf and {new_style}.abf)"
                ) from error

        # Convert the sheet's sweep numbers to 0-based indices.
        warnings = []
        offset = 1 if self.badsweeps_numbering == "from_1" else 0
        for number in (row["badsweeps_list"] if apply_bad_sweeps else []):
            index = number - offset
            if 0 <= index < recording.n_sweeps:
                recording.bad_sweeps.add(index)
            else:
                warnings.append(
                    f"{recording.file_name}: bad sweep '{number}' in the sheet does not exist "
                    f"(file has sweeps 1–{recording.n_sweeps} when counting from 1) — ignored."
                )
        for message in warnings:
            logger.warning(message)
        recording.header_metadata["catalog_row"] = {"row": int(row_label), **row.drop("badsweeps_list").to_dict()}
        recording.header_metadata["catalog_warnings"] = warnings
        return recording

    def imaging_file_names(self, row_label: int) -> list[str]:
        """
        The imaging file(s) of a row. The sheet writes them as "Image0005",
        "Image0005.oib", or several separated by ";" ("Image0006;Image0008").
        """
        value = self.row(row_label).get("ophysfile")
        if pd.isna(value) or not str(value).strip():
            return []
        return [name.strip() for name in str(value).split(";") if name.strip()]

    def load_imaging(self, row_label: int, which: int = 0) -> ImageRecording:
        """
        Load an imaging file of a row (``which`` = 0 for the first when the
        sheet lists several). A name without extension is tried as .oib, then .oif.
        """
        if not self.imaging_root:
            raise ValueError("No imaging_root configured in the [catalog] settings.")
        row = self.row(row_label)
        names = self.imaging_file_names(row_label)
        if not names:
            raise ValueError(f"Row {row_label} has no imaging file (ophysfile is empty).")
        if not 0 <= which < len(names):
            raise IndexError(f"Row {row_label} lists {len(names)} imaging file(s): {names}")
        name = names[which]
        folder = f"{row['expdate']}/c{int(row['neuronid'])}"
        candidates = [name] if Path(name).suffix.lower() in (".oib", ".oif") else [name + ".oib", name + ".oif"]
        last_error = None
        # 1) the cell folder of the row; 2) on some days the imaging folders were
        #    numbered differently from the ephys folders (e.g. ephys c1, images c3):
        #    then the date's other cell folders are searched, and an image found in
        #    exactly ONE of them is used (recorded in metadata["found_in_other_folder"]).
        def remember_row(image: ImageRecording) -> ImageRecording:
            # Remember the sheet row (used e.g. to label uncaging targets by spine id).
            image.metadata["catalog_row"] = {"row": int(row_label), **row.drop("badsweeps_list").to_dict()}
            return image

        for candidate in candidates:
            try:
                return remember_row(self._fetch_and_read(self.imaging_root, f"{folder}/{candidate}"))
            except FileNotFoundError as error:
                last_error = error

        # Not in the row's own folder: look in the date's other cell folders.
        from ..load_recording import get_data_source

        source = self._source or get_data_source(self._mode, self._settings_file)
        found = [
            f"{row['expdate']}/{other}/{candidate}"
            for other in self._other_cell_folders(row["expdate"], int(row["neuronid"]))
            for candidate in candidates
            if source.file_exists(self.imaging_root, f"{row['expdate']}/{other}/{candidate}")
        ]
        if len(found) == 1:
            image = remember_row(self._fetch_and_read(self.imaging_root, found[0]))
            image.metadata["found_in_other_folder"] = (
                f"{Path(found[0]).name} found in folder {found[0].split('/')[1]}, not c{int(row['neuronid'])}"
            )
            return image
        if len(found) > 1:
            raise FileNotFoundError(
                f"Imaging file for row {row_label} is not in {folder} and exists in several other "
                f"cell folders ({found}); not guessing which one."
            )
        raise FileNotFoundError(f"Imaging file for row {row_label} not found as {' or '.join(candidates)}: {last_error}")

    def rows_for_image(self, expdate, ophysid: int, neuronid: int | None = None) -> list[tuple[int, int]]:
        """
        (row label, position in that row's image list) of every sheet row listing
        image number ``ophysid`` (e.g. 15 = Image0015) on ``expdate`` (and neuron).
        """
        import re

        expdate = str(expdate).replace(".0", "")
        table = self.table[(self.table["expdate"] == expdate) & self.table["ophysfile"].notna()]
        if neuronid is not None:
            table = table[table["neuronid"] == int(neuronid)]
        found = []
        for label, value in table["ophysfile"].items():
            for position, name in enumerate(str(value).split(";")):
                digits = re.findall(r"\d+", Path(name.strip()).stem)
                if digits and int(digits[-1]) == int(ophysid):
                    found.append((int(label), position))
        return found

    def image_grid(self, images, columns: int = 4, panel_size_in: float = 3.2, title: str = ""):
        """
        Matplotlib figure with several images and their uncaging points (see
        interactive_viewers/image_grid_figure.py). ``images``:
        [[20270408, 31], [20270408, 32]]  (expdate, ophysid),
        [[20270408, 1, 31]]               (expdate, neuronid, ophysid) when a date has several cells,
        ["20270408 31", ...]  or  {20270408: [12, 15, 16, 17]}.
        """
        from ..interactive_viewers.image_grid_figure import build_image_grid_figure, parse_image_requests
        from .uncaging_target_labels import label_uncaging_targets

        panels = []
        for expdate, ophysid, neuronid in parse_image_requests(images):
            where = self.rows_for_image(expdate, ophysid, neuronid)
            heading = f"{expdate} · image {ophysid}" + (f" · cell {neuronid}" if neuronid is not None else "")
            neurons = {int(self.table.at[label, "neuronid"]) for label, _ in where}
            if not where:
                panels.append({"recording": None, "title": heading, "error": "not listed in the sheet"})
                continue
            if len(neurons) > 1:
                panels.append({"recording": None, "title": heading, "error": f"listed for cells {sorted(neurons)}: "
                               "give (expdate, neuronid, ophysid)"})
                continue
            rows = self.table.loc[[label for label, _ in where]]
            files = rows["ephys_file_name"].nunique()
            listed = " · ".join(dict.fromkeys(f"e{int(r.ephysid)} s{_spine_text(r.spineid)}"
                                              for r in rows.itertuples()))
            label, position = where[0]
            heading = (f"{expdate} c{neurons.pop()} d{_spine_text(rows['dendriteid'].iloc[0])} · "
                       f"{self.imaging_file_names(label)[position]}" + (" (shared)" if files > 1 else "")
                       + "\n" + listed)
            try:
                recording = self.load_imaging(label, position)
                found = label_uncaging_targets(recording, self)     # spine ids exactly as in the sheet
                labels = found.labels
                if not found.resolved:
                    heading += "\n⚠ which points were stimulated is unknown: not named"
                panels.append({"recording": recording, "title": heading, "labels": labels})
            except Exception as error:
                panels.append({"recording": None, "title": heading, "error": f"{type(error).__name__}: {error}"})
        return build_image_grid_figure(panels, columns, panel_size_in, title)

    def show_image_grid(self, images, columns: int = 4, path=None, dpi: int = 200, **options):
        """
        Show several images with their uncaging points in ONE figure (notebook),
        and save it when ``path`` is given (PNG / SVG / PDF). Returns nothing;
        ``catalog.image_grid(...)`` gives the matplotlib Figure itself.

        >>> catalog.show_image_grid([[20270408, 12], [20270408, 15], [20270408, 16]])
        >>> catalog.show_image_grid({20270401: range(9, 17)}, columns=4, path="20270401_images.png")

        Each panel: image, laser points with their scan number (P1, P2 ...), red
        circle + spine id for points on spines, grey cross for points moved off
        the dendrite; title: image name, and ephys id (e) + spineid (s) of every
        sheet row listing it ("(shared)" = listed for several ephys files).
        """
        from ..interactive_viewers.display_helpers import display_figure_in_notebook

        figure = self.image_grid(images, columns, **options)
        if path is not None:
            from ..interactive_viewers.image_grid_figure import save_figure

            print("Saved:", save_figure(figure, path, dpi).resolve())
        display_figure_in_notebook(figure)
        # nothing is returned, so the notebook does not try to show the figure a
        # second time; use catalog.image_grid(...) to get the Figure object

    def _other_cell_folders(self, expdate: str, neuron: int) -> list[str]:
        """Cell folders of a date (locally and/or on Box) other than c<neuron>."""
        from ..load_recording import get_data_source

        source = self._source or get_data_source(self._mode, self._settings_file)
        folders: set[str] = set()
        local = getattr(source, "local_source", source if hasattr(source, "root_folders") else None)
        if local is not None and self.imaging_root in getattr(local, "root_folders", {}):
            date_folder = local.root_folders[self.imaging_root] / str(expdate)
            if date_folder.is_dir():
                folders |= {p.name for p in date_folder.iterdir() if p.is_dir()}
        box = getattr(source, "box_source", source if hasattr(source, "list_subfolders") else None)
        if box is not None and self.imaging_root in box.root_names():
            try:
                folders |= set(box.list_subfolders(self.imaging_root, str(expdate)))
            except FileNotFoundError:
                pass
        return sorted(f for f in folders if f.lower() != f"c{neuron}")

    # =========================================================================
    # Traces grouped by inter-stimulus interval
    # =========================================================================
    def traces_by_isi(
        self,
        expdate,
        neuronid,
        dendriteid,
        spineid,
        isis: list[float] | None = None,
        groups: list[str] | None = None,
        expname=None,
        unselect=None,
        channel: str | None = None,
        digital_output_bit: int | str = "auto",
        stimulus_channel: str | None = None,
        stimulus_threshold: float | None = None,
        time_window_ms: tuple[float, float] | None = None,
    ) -> TracesByIsi:
        """
        Load and align all recordings of one spine, grouped by ISI.

        ``isis`` = list of ISIs in ms to include (default: all available).
        ``stimulus_channel`` = a recorded trigger channel name, or "catalog" to
        use each row's ``chan_stim``; default: automatic digital-output choice.
        See ``traces_by_isi.collect_traces_by_isi`` for the other options.
        """
        rows = self.select(
            expname=expname, expdate=expdate, neuronid=neuronid, dendriteid=dendriteid,
            spineid=spineid, isi=isis, group=groups, unselect=unselect,
        )
        if rows.empty:
            available = self.distinct("isi", expdate=expdate, neuronid=neuronid, dendriteid=dendriteid, spineid=spineid)
            raise LookupError(
                f"No rows for date {expdate}, neuron {neuronid}, dendrite {dendriteid}, spine {spineid}"
                + (f" with ISIs {isis}" if isis else "")
                + f". ISIs available for this spine: {available}"
            )
        return collect_traces_by_isi(
            rows, self.load_ephys, channel=channel, digital_output_bit=digital_output_bit,
            stimulus_channel=stimulus_channel, stimulus_threshold=stimulus_threshold,
            time_window_ms=time_window_ms,
        )

    def show_traces_by_isi(self, expdate, neuronid, dendriteid, spineid, isis=None, **options):
        """Interactive traces-by-ISI view (see ``traces_by_isi`` for options)."""
        from ..interactive_viewers.traces_by_isi_viewer import build_traces_by_isi_viewer

        viewer_keys = {"show_sweeps", "shared_y_axis", "columns"}
        viewer_options = {k: options.pop(k) for k in list(options) if k in viewer_keys}
        data = self.traces_by_isi(expdate, neuronid, dendriteid, spineid, isis=isis, **options)
        return build_traces_by_isi_viewer(data, **viewer_options)

    def save_traces_by_isi_figure(self, output_path, expdate, neuronid, dendriteid, spineid, isis=None, **options):
        """Static PNG/SVG/PDF of the traces-by-ISI view; returns the file path."""
        from ..interactive_viewers.traces_by_isi_viewer import save_traces_by_isi_figure

        figure_keys = {"show_sweeps", "shared_y_axis", "columns", "dpi", "panel_size_in"}
        figure_options = {k: options.pop(k) for k in list(options) if k in figure_keys}
        data = self.traces_by_isi(expdate, neuronid, dendriteid, spineid, isis=isis, **options)
        return save_traces_by_isi_figure(data, output_path, **figure_options)

    # =========================================================================
    # All traces of one experiment sample (expdate / neuron / dendrite [/ spine])
    # =========================================================================
    def experiment_traces(
        self,
        expdate,
        neuronid,
        dendriteid,
        spineid=None,
        expname=None,
        group: str | list[str] | None = None,
        isis: list[float] | None = None,
        unselect=None,
        channel: str | None = None,
        stimulus_channel: str | None = "catalog",
        groups: list[str] | None = None,
        time_window_ms: tuple[float, float] | None = None,
        before_ms: float = 50.0,
        after_ms: float = 150.0,
        remove_unselected: bool = False,
        remove_bad_sweeps: bool = True,
    ) -> ExperimentTraces:
        """
        Load every recording of one experiment sample, aligned to the first stimulus.

        ``spineid`` None = all spines of the dendrite. ``group`` = sheet group(s)
        to include, e.g. "ctrl" or ["ctrl", "ctz"] (``groups`` is accepted as well).
        ``expname`` / ``isis`` / ``unselect`` narrow the selection further (None = any).
        Window per file: ``before_ms`` before the first pulse to ``after_ms`` after the last.
        Files listed in several sheet rows (e.g. the same recording under two
        experiment names) are loaded once; a warning names the duplicates.

        ``remove_unselected=True``: leave out the files whose "unselect" column is 1
        (empty or 0 = kept); they are named in the warnings.
        ``remove_bad_sweeps``: True (default) = leave out the sweeps listed in the
        "badsweeps" column; False = use every sweep of every file.
        See ``experiment_traces.collect_experiment_traces`` for the other options.
        """
        wanted_groups = _as_list(group) or []
        wanted_groups += [g for g in (_as_list(groups) or []) if g not in wanted_groups]
        rows = self.select(
            expname=expname, expdate=expdate, neuronid=neuronid, dendriteid=dendriteid,
            spineid=spineid, isi=isis, group=wanted_groups or None, unselect=unselect,
        )
        notes = []
        if remove_unselected and not rows.empty:
            unselected = rows["unselect"].astype("Float64").eq(1).fillna(False).astype(bool)
            if unselected.any():
                left_out = rows.loc[unselected]
                names = ", ".join(f"{name}.abf ({group + ', ' if isinstance(group, str) else ''}ISI {isi:g} ms)"
                                  for name, group, isi in
                                  zip(left_out["ephys_file_name"], left_out["group"], left_out["isi"]))
                notes.append(f"{int(unselected.sum())} file(s) left out because unselect = 1: {names}.")
            rows = rows[~unselected]
        if rows.empty:
            place = dict(expdate=expdate, neuronid=neuronid, dendriteid=dendriteid)
            spines = [int(v) for v in self.distinct("spineid", **place)]
            available_groups = [str(v) for v in self.distinct("group", **place, spineid=spineid)]
            raise LookupError(
                f"No rows for date {expdate}, neuron {neuronid}, dendrite {dendriteid}"
                + (f", spine {spineid}" if spineid is not None else "")
                + (f", group {wanted_groups}" if wanted_groups else "")
                + (" (all remaining rows have unselect = 1)" if notes else "")
                + f". Spines in the sheet for this dendrite: {spines}; groups: {available_groups}"
            )
        duplicated = rows["ephys_file_name"].duplicated(keep="first")
        if duplicated.any():
            for name in rows.loc[duplicated, "ephys_file_name"].unique():
                listed = rows[rows["ephys_file_name"] == name]
                notes.append(f"{name}.abf is listed in {len(listed)} sheet rows "
                             f"({', '.join(dict.fromkeys(listed['expname'].astype(str)))}); loaded once.")
            rows = rows[~duplicated]
        if not remove_bad_sweeps:
            with_bad = rows["badsweeps_list"].map(len) > 0
            if with_bad.any():
                notes.append(f"Bad sweeps NOT removed: all sweeps used (sheet lists bad sweeps for "
                             f"{', '.join(rows.loc[with_bad, 'ephys_file_name'])}).")
        data = collect_experiment_traces(
            rows, lambda label: self.load_ephys(label, apply_bad_sweeps=remove_bad_sweeps),
            criteria=dict(expdate=expdate, neuronid=neuronid, dendriteid=dendriteid, spineid=spineid),
            channel=channel, stimulus_channel=stimulus_channel, time_window_ms=time_window_ms,
            before_ms=before_ms, after_ms=after_ms,
        )
        data.warnings[:0] = notes
        return data

    def show_experiment_traces(self, expdate, neuronid, dendriteid, spineid=None, average_by=None, group=None,
                               **options):
        """
        Interactive view of ALL traces of one experiment sample.

        >>> catalog.show_experiment_traces(20250125, 1, 1, spineid=3)                  # every sweep
        >>> catalog.show_experiment_traces(20250125, 1, 1, 3, average_by="isi")         # mean per ISI
        >>> catalog.show_experiment_traces(20250125, 1, 1, 3, average_by=["isi", "group"])
        >>> catalog.show_experiment_traces(20250125, 1, 1)          # all spines, one panel per spine
        >>> catalog.show_experiment_traces(20250125, 1, 1, 3, average_by="isi", group="ctz")    # only ctz
        >>> catalog.show_experiment_traces(20260322, 1, 1, 0, group=["pre-apv", "post-apv"])

        ``group`` = the sheet's group(s) to include (one value or a list; None = all groups).

        Viewer options: color_by ("isi"), panel_by ("auto" = per spine when several),
        show_sweeps, baseline_subtract (False), shared_y_axis, columns. Other options: see ``experiment_traces``.
        """
        from ..interactive_viewers.experiment_traces_viewer import build_experiment_traces_viewer

        viewer_keys = {"color_by", "panel_by", "show_sweeps", "baseline_subtract", "shared_y_axis", "columns"}
        viewer_options = {k: options.pop(k) for k in list(options) if k in viewer_keys}
        data = self.experiment_traces(expdate, neuronid, dendriteid, spineid, group=group, **options)
        return build_experiment_traces_viewer(data, average_by=average_by, **viewer_options)

    def save_experiment_traces_figure(self, output_path, expdate, neuronid, dendriteid, spineid=None,
                                      average_by=None, group=None, **options):
        """Static PNG/SVG/PDF of ``show_experiment_traces`` (same options, incl. ``group``); returns the file path."""
        from ..interactive_viewers.experiment_traces_viewer import save_experiment_traces_figure

        figure_keys = {"color_by", "panel_by", "show_sweeps", "baseline_subtract", "shared_y_axis", "columns",
                       "panel_size_in", "dpi"}
        figure_options = {k: options.pop(k) for k in list(options) if k in figure_keys}
        data = self.experiment_traces(expdate, neuronid, dendriteid, spineid, group=group, **options)
        return save_experiment_traces_figure(data, output_path, average_by=average_by, **figure_options)

    # =========================================================================
    # Summated EPSPs: template from isolated EPSPs, deconvolution per ISI
    # =========================================================================
    def deconvolve_summated_epsps(
        self,
        expdate,
        neuronid,
        dendriteid,
        spineid=None,
        group=None,
        threshold_isi_ms: float = 200.0,
        template_events: str = "both",
        template_length_ms: float | None = None,
        after_ms: float = 300.0,
        remove_unselected: bool = True,
        remove_bad_sweeps: bool = True,
        **selection,
    ):
        """
        Separate the 1st and 2nd EPSPs of every ISI of one experiment sample
        (selected like ``show_experiment_traces``; current-clamp recordings only).

        ``remove_unselected`` (default True): leave out files with unselect = 1 in the sheet.
        ``remove_bad_sweeps`` (default True): leave out the sweeps listed in "badsweeps".
        What was left out is listed in the warnings (shown on the figures).

        * template = average of the isolated 1st and 2nd EPSPs of sweeps with
          ISI >= ``threshold_isi_ms`` (``template_events`` "both" / "first" / "second"),
          fitted with a double-exponential EPSP + sag model;
        * every ISI: averaged trace = EPSP1 (template shape) + EPSP2 + ... + EPSPn (own
          amplitude, latency, rise and decay each; any number of pulses) -> absolute
          baseline, peak, 20-80 % rise and decay tau of every EPSP.
        ``after_ms``: how long after the 2nd stimulus is loaded and fitted (sag included).

        Returns one ``SummatedEpspAnalysis`` per group (``.table()`` gives the numbers).
        See ``epsp_analysis/summated_epsp_deconvolution.py`` for the method.
        """
        from ..epsp_analysis.summated_epsp_deconvolution import deconvolve_summated_epsps

        data = self.experiment_traces(expdate, neuronid, dendriteid, spineid, group=group, after_ms=after_ms,
                                      remove_unselected=remove_unselected, remove_bad_sweeps=remove_bad_sweeps,
                                      **selection)
        return deconvolve_summated_epsps(data, threshold_isi_ms, template_events, template_length_ms, after_ms)

    def show_epsp_deconvolution(self, expdate, neuronid, dendriteid, spineid=None, group=None, columns: int = 4,
                                csv_path=None, figure_path=None, dpi: int = 200, **options):
        """
        Figures of ``deconvolve_summated_epsps`` (one per group): template, summary,
        and per ISI the recorded trace vs the fitted EPSPs. The figures are
        displayed in a notebook, the figures and the table are SAVED, and the
        analyses are returned.

        >>> catalog.show_epsp_deconvolution(20260325, 1, 2, 0)
        ...     # saves 20260325_1_2_0_thr200.png and 20260325_1_2_0_thr200.csv
        >>> analyses = catalog.show_epsp_deconvolution(20260605, 1, 2, 0, threshold_isi_ms=400)
        >>> analyses[0].table()        # numbers per ISI
        >>> catalog.show_epsp_deconvolution(20260322, 1, 1, 0, remove_unselected=False, remove_bad_sweeps=False)

        ``remove_unselected`` (default True): leave out files with unselect = 1 in the sheet.
        ``remove_bad_sweeps`` (default True): leave out the sweeps listed in the "badsweeps" column.
        What was left out is listed in the warnings at the top of each figure.

        ``figure_path`` / ``csv_path``: where to save. Default (when not given):
        "<expdate>_<neuronid>_<dendriteid>_<spineid>_thr<threshold_isi_ms>.png" / ".csv"
        in the current folder (spineid "all" when no spine is given). With several
        groups every figure gets "_<group>" added; the CSV holds all groups.
        Any number of pulses per sweep (paired pulses, trains) is handled.
        """
        from ..epsp_analysis.summated_epsp_deconvolution import epsp_deconvolution_table
        from ..interactive_viewers.epsp_deconvolution_figure import build_epsp_deconvolution_figure

        analyses = self.deconvolve_summated_epsps(expdate, neuronid, dendriteid, spineid, group=group, **options)
        threshold = options.get("threshold_isi_ms", 200.0)
        default_stem = (f"{str(expdate).replace('.0', '')}_{neuronid}_{dendriteid}_"
                        f"{'all' if spineid is None else spineid}_thr{float(threshold):g}")
        figure_path = Path(figure_path or f"{default_stem}.png")
        csv_path = Path(csv_path or f"{default_stem}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        epsp_deconvolution_table(analyses).to_csv(csv_path, index=False)
        print(f"Table saved: {csv_path.resolve()}")
        from ..interactive_viewers.display_helpers import display_figure_in_notebook

        for analysis in analyses:     # shown as PNG (works in every notebook); nothing outside IPython
            display_figure_in_notebook(build_epsp_deconvolution_figure(analysis, columns))
        from ..interactive_viewers.epsp_deconvolution_figure import save_epsp_deconvolution_figures

        figure_path.parent.mkdir(parents=True, exist_ok=True)
        written = save_epsp_deconvolution_figures(analyses, figure_path, dpi=dpi, columns=columns)
        print("Figures saved:", ", ".join(str(Path(path).resolve()) for path in written))
        return analyses

    def save_epsp_deconvolution_figures(self, output_path, expdate, neuronid, dendriteid, spineid=None, group=None,
                                        dpi: int = 200, columns: int = 4, **options):
        """
        Save the ``show_epsp_deconvolution`` figures (one file per group) AND the
        table of all groups as '<name>.csv' next to them; returns all written paths.
        """
        from ..epsp_analysis.summated_epsp_deconvolution import epsp_deconvolution_table
        from ..interactive_viewers.epsp_deconvolution_figure import save_epsp_deconvolution_figures

        analyses = self.deconvolve_summated_epsps(expdate, neuronid, dendriteid, spineid, group=group, **options)
        written = save_epsp_deconvolution_figures(analyses, output_path, dpi=dpi, columns=columns)
        csv_path = Path(output_path).with_suffix(".csv")
        epsp_deconvolution_table(analyses).to_csv(csv_path, index=False)
        return written + [csv_path]

    # =========================================================================
    # Per-set analysis (deconvolution + spatial summation) and batch runs
    # =========================================================================
    def analyse_set(self, expdate, neuronid, dendriteid, group=None, output_dir="set_analysis_results",
                    show: bool = True, **options):
        """
        Analyse ONE set (expdate + neuron + dendrite + group; all groups when
        ``group`` is None) and write its folder: one CSV with every single spine
        and cluster x ISI (deconvolution, 1st response, linear sum, gain), the
        summation figure, deconvolution figures and notes. Returns the table.

        >>> catalog.analyse_set(20270313, 1, 1)
        >>> catalog.analyse_set(20260322, 1, 1, "post-apv", threshold_isi_ms=400)

        ``show``: also display the summation figure(s) in a notebook.
        Options: threshold_isi_ms (200), summation_isi_ms (1000), after_ms (300),
        uncaging_offset (True), remove_unselected (True), remove_bad_sweeps (True),
        deconvolution_figures ("multi_isi" / "all" / "none"), dpi.
        See ``epsp_analysis/unitary_cluster_summation.py`` and ``batch_analysis/batch_runner.py``.
        """
        from ..batch_analysis import analyse_set_to_folder, resolve_sets

        tables = []
        for key in resolve_sets(self, (expdate, neuronid, dendriteid) + ((group,) if group is not None else ())):
            table, written = analyse_set_to_folder(self, key, output_dir, **options)
            tables.append(table)
            print(f"{key.name}: {len(table)} rows -> {Path(output_dir).resolve() / key.name}")
            if show:
                try:
                    from IPython.display import Image, display
                    for path in written:
                        if path.name.endswith(".png") and "_summation" in path.name:
                            display(Image(filename=str(path)))
                except ImportError:
                    pass
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    def run_batch_analysis(self, sets=None, output_dir="set_analysis_results", **options):
        """
        ``analyse_set`` for many sets, each in its own folder, plus all_sets.csv
        (every row of the run), all_sets_summation.png and batch_log.csv.

        >>> catalog.run_batch_analysis()                                    # every set in the sheet
        >>> catalog.run_batch_analysis((20260322, 1, 1))                    # one dendrite, all groups
        >>> catalog.run_batch_analysis([(20260322, 1, 1, "post-apv"), "20270313 1 1"])
        >>> catalog.run_batch_analysis("my_sets.txt", overwrite=True)       # one set per line

        Options: expnames, overwrite (False: sets with results are read, not redone),
        and those of ``analyse_set``. See ``batch_analysis/batch_runner.py``.
        """
        from ..batch_analysis import run_batch_analysis

        return run_batch_analysis(self, sets, output_dir, **options)

    # =========================================================================
    # Checking the sheet's stimulus channel against the files
    # =========================================================================
    def check_stimulus_channels(self, **criteria) -> pd.DataFrame:
        """
        For every selected row: which recorded channel REALLY carries the
        trigger / light pulses, compared with the sheet's ``chan_stim``, and
        whether the stimulus interval in the file matches the sheet's ISI.

        Loads each file (from Box if needed), so select a few dates at a time, e.g.
        ``catalog.check_stimulus_channels(expdate=20260322)``.

        Columns: row, file, sheet_chan_stim, trigger_channels (all channels with
        trigger-like pulses), stimulus (identified output + confirming channel),
        pulses, ipi_ms (file), sheet_isi_ms, chan_stim_ok, isi_ok, problem.
        """
        from .stimulus_onset_detection import identify_stimulus, trigger_like_channels

        results = []
        for row_label, row in self.select(**criteria).iterrows():
            entry = {"row": row_label, "file": row["ephys_file_name"], "sheet_chan_stim": row.get("chan_stim"),
                     "sheet_isi_ms": row.get("isi")}
            try:
                recording = self.load_ephys(row_label)
                sweep = recording.good_sweeps[0] if recording.good_sweeps else 0
                response = row.get("response_channel") if pd.notna(row.get("response_channel")) else None
                triggers = [t.confirmed_by[0] for t in trigger_like_channels(recording, sweep, exclude=[response] if response else [])]
                stimulus = identify_stimulus(recording, sweep, response_channel=response)
                sheet_channel = str(row.get("chan_stim") or "").replace(" ", "").lower()
                entry.update(
                    trigger_channels="/".join(triggers) or "none",
                    stimulus=stimulus.source,
                    pulses=stimulus.n_pulses,
                    ipi_ms=None if stimulus.mean_interval_ms is None else round(stimulus.mean_interval_ms, 1),
                    chan_stim_ok=sheet_channel in [t.replace(" ", "").lower() for t in triggers],
                    isi_ok=(stimulus.mean_interval_ms is None and not row.get("isi"))
                    or (stimulus.mean_interval_ms is not None and pd.notna(row.get("isi"))
                        and abs(stimulus.mean_interval_ms - float(row["isi"])) <= max(2.0, 0.05 * float(row["isi"]))),
                    problem="",
                )
            except Exception as error:  # report, don't stop the whole check
                entry.update(problem=str(error))
            results.append(entry)
        return pd.DataFrame(results)

    # =========================================================================
    # Interactive chooser
    # =========================================================================
    def browse(self, **options):
        """Interactive chooser (dropdowns + table + buttons); see experiment_catalog_browser.py."""
        from ..interactive_viewers.experiment_catalog_browser import ExperimentCatalogBrowser

        return ExperimentCatalogBrowser(catalog=self, **options)

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return (
            f"ExperimentCatalog({len(self.table)} rows, {self.table['expdate'].nunique()} dates, "
            f"ephys root '{self.ephys_root}', sheet '{self.csv_path}')"
        )
